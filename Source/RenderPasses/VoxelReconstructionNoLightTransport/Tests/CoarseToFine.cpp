#include "../VoxelReconstructionNoLightTransport.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <chrono>

struct CoarseToFineTestAccess
{
    static void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    template<typename T>
    static T read(const std::vector<ref<Buffer>>& pages, uint32_t id)
    {
        T result;
        pages[id / SPARSE_POOL_PAGE_SIZE]->getBlob(&result, size_t(id % SPARSE_POOL_PAGE_SIZE) * sizeof(T), sizeof(T));
        return result;
    }

    static void checkGeometry(const GaussianEllipsoid& parent, const GaussianEllipsoid& child, float3 size, uint3 bits)
    {
        require(all(child.rotation == parent.rotation), "Refinement changed parent rotation");
        const float3 axes = exp(parent.logScale);
        const float3 childAxes = exp(child.logScale);
        const float3 ratio = childAxes / axes;
        require(all(abs(ratio - float3(ratio.x)) < float3(2e-5f)), "Refinement changed semi-axis proportions");
        require(
            all(child.center >= float3(0.01f - 2e-5f)) && all(child.center <= float3(0.99f + 2e-5f)),
            "Child center outside optimizer bounds"
        );
        const float4 q = parent.rotation / std::sqrt(dot(parent.rotation, parent.rotation));
        const float w = q.x, x = q.y, y = q.z, z = q.w;
        const float3 rows[3] = {
            float3(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            float3(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            float3(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))};
        const float3 displacement = (0.5f * (float3(bits) + child.center) - parent.center) * size;
        const float3 principal = (rows[0] * displacement.x + rows[1] * displacement.y + rows[2] * displacement.z) / axes;
        require(length(principal) + ratio.x <= 1.0f + 3e-5f, "Child protrudes outside parent ellipsoid");
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const float extent = length(rows[axis] * childAxes) / (0.5f * size[axis]);
            require(extent <= std::min(child.center[axis], 1.0f - child.center[axis]) + 3e-5f, "Child protrudes outside its own cell");
        }
        require(all(childAxes >= max(0.5f * size * 1e-4f, float3(1e-8f)) * 0.999f), "Child below optimizer minimum scale");
    }

    static void run(const ref<Device>& device)
    {
        auto ctx = device->getRenderContext();
        auto pass = VoxelReconstructionNoLightTransport::create(device, {});
        require(pass->mReconstructionMode == uint32_t(RECONSTRUCTION_MODE), "Configured reconstruction mode was not applied");
        pass->mReconstructionMode = 1u;
        pass->mCoarseToFine.targetResolution = 64u;
        pass->mCoarseToFine.totalIterations = 12u; // Linear level weights produce 2, 4, and 6 rounds.
        pass->mCoarseToFine.saveEachLevel = false; // Most scheduling checks must not write experiment checkpoints.
        pass->createUpdatePassResource(ctx);
        pass->createTopologyPassResource(ctx);
        pass->createDeletionPassResources();
        pass->mUpdatePass.mpComputePass->getRootVar(); // Test rebinding existing update resources at refinement.
        {
            const auto fixture = getProjectDirectory() / "Source/RenderPasses/VoxelReconstructionNoLightTransport/Tests/CoarseSeed.ply";
            require(pass->initializePointCloudVoxelData(ctx, fixture), "Mode1 coarse PLY initialization failed");
            require(
                pass->mVoxelResolution == 16u && pass->mGridResources.gridData.activeVoxelCount == 1u,
                "Mode1 initialized PLY at the target resolution instead of the coarse level"
            );
        }
        GridData grid = pass->makeVoxelGrid(16u);
        require(all(grid.voxelCount == uint3(16)), "Coarse initialization did not build a 16 grid");
        grid.voxelSize = float3(0.5f, 1.0f, 1.5f); // Exercise non-cubic world-space scale transfer.
        grid.activeVoxelCount = grid.solidVoxelCount = 1u;
        auto resources = pass->allocateSparseGrid(ctx, grid, 1u);
        auto block = pass->createSparseGridBlock(resources);
        pass->commitSparseGrid(std::move(resources), block, 16u);
        VoxelData parent = {};
        parent.occupied = 1u;
        parent.ellipsoid.center = float3(0.25f, 0.6f, 0.7f);
        // This scheduling fixture fully contains every child; selective
        // intersection and sparse allocation are covered separately below.
        parent.ellipsoid.logScale = float3(std::log(3.0f), std::log(2.5f), std::log(2.0f));
        parent.ellipsoid.rotation = float4(std::cos(0.3f), 0, 0, std::sin(0.3f));
        parent.radiance.coefficients[0] = float3(0.3f, 0.6f, 0.9f);
        parent.opacity.coefficients[0] = std::log(0.4f / 0.6f) / calcSH(0u, float3(0, 0, 1));
        const uint32_t cell = 3u + 16u * 4u + 256u * 5u;
        pass->uploadSparseBatch(ctx, block, 0u, &cell, 1u, &parent);
        pass->barrierSparseVoxels(ctx);
        ctx->submit(true);
        pass->mPointCloud.initialized = true;
        require(pass->prepareReconstruction(ctx), "Could not start mode1 from an existing coarse grid");
        require(pass->mOptimizerParams.maxIteration == 12u, "Level schedule did not use the configured total");
        pass->mOptimizerParams.isRunning = pass->mEnableReconstruction = true;

        pass->mOptimizerParams.currentIteration = 50u;
        pass->mRayMarchingPass.mSampleIndex = pass->mRayMarchingPass.mSpp;
        require(!pass->shouldCollectDeletionEvidence(), "Mode1 enabled deletion evidence");
        pass->mTopologySettings.candidateCount = 1u;
        pass->deleteAndCompactCandidates(ctx);
        pass->growNeighborVoxels(ctx);
        require(pass->mGridResources.gridData.activeVoxelCount == 1u, "Mode1 performed adaptive topology changes");
        pass->mTopologySettings.candidateCount = 0u;

        GeometryAdamState oldAdam = {};
        oldAdam.centerSteps = 19u;
        pass->mGridResources.adamPages[0]->setBlob(&oldAdam, 0, sizeof(oldAdam));
        pass->mOptimizerParams.currentIteration = 1u;
        pass->advanceCoarseToFine(ctx);
        require(pass->mVoxelResolution == 16u, "Refinement ran before a full level completed");
        const float beforeLr = pass->getEffectiveOpacityLearningRate();
        pass->mCoarseToFine.saveEachLevel = true;
        pass->mReconstructionNameTag =
            "level_checkpoint_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        const auto levelCheckpoint = pass->getDefaultReconstructionSavePath();
        struct CheckpointCleanup
        {
            std::filesystem::path file;
            ~CheckpointCleanup()
            {
                std::error_code error;
                std::filesystem::remove(file, error);
            }
        } checkpointCleanup{levelCheckpoint};
        pass->mOptimizerParams.currentIteration = 2u;
        pass->advanceCoarseToFine(ctx);
        require(std::filesystem::is_regular_file(levelCheckpoint), "Completed coarse level was not checkpointed before refinement");
        pass->mCoarseToFine.saveEachLevel = false;
        pass->mReconstructionNameTag.clear();
        require(
            pass->mVoxelResolution == 32u && pass->mGridResources.gridData.activeVoxelCount == 8u,
            "First level did not subdivide one parent into eight cells"
        );
        require(
            all(pass->mGridResources.gridData.gridMin == grid.gridMin) &&
                all(pass->mGridResources.gridData.voxelSize == grid.voxelSize * 0.5f),
            "Refinement moved the world grid bounds"
        );
        require(
            pass->mOptimizerParams.currentIteration == 2u && pass->getEffectiveOpacityLearningRate() == beforeLr,
            "Refinement reset the training clock or opacity schedule"
        );
        const auto bytes = ctx->readTextureSubresource(pass->mGridResources.indexPages[0].get(), 0);
        uint32_t occupied = 0u;
        for (size_t offset = 0; offset < bytes.size(); offset += sizeof(int32_t))
        {
            int32_t id;
            std::memcpy(&id, bytes.data() + offset, sizeof(id));
            if (id >= 0)
                ++occupied;
        }
        require(occupied == 8u, "Refinement added cells outside the coarse occupied cell");
        for (uint32_t id = 0u; id < 8u; ++id)
        {
            const auto child = read<VoxelData>(pass->mGridResources.voxelPages, id);
            const uint32_t x = 6u + (id & 1u), y = 8u + ((id >> 1u) & 1u), z = 10u + ((id >> 2u) & 1u);
            const uint32_t expectedCell = x + 32u * y + 1024u * z;
            require(read<uint32_t>(pass->mGridResources.cellIndexPages, id) == expectedCell, "Child spatial address is incorrect");
            int32_t mapped;
            std::memcpy(&mapped, bytes.data() + expectedCell * sizeof(int32_t), sizeof(mapped));
            require(mapped == int32_t(id), "Spatial map disagrees with compact child ID");
            require(child.occupied == 1u, "Child unoccupied");
            checkGeometry(parent.ellipsoid, child.ellipsoid, grid.voxelSize, uint3(id & 1u, (id >> 1u) & 1u, (id >> 2u) & 1u));
            require(std::memcmp(&child.radiance, &parent.radiance, sizeof(parent.radiance)) == 0, "Refinement reset radiance");
            require(
                std::memcmp(&child.opacity, &parent.opacity, sizeof(parent.opacity)) == 0, "Refinement did not fully inherit opacity SH"
            );
            const auto adam = read<GeometryAdamState>(pass->mGridResources.adamPages, id);
            const GeometryAdamState zero = {};
            require(std::memcmp(&adam, &zero, sizeof(adam)) == 0, "Refinement retained incompatible parent Adam moments");
            require(
                read<uint32_t>(pass->mGridResources.radianceAdamIndexPages, id) == 0xffffffffu,
                "Refinement retained parent radiance Adam mapping"
            );
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, id).growthBirthIterationPlusOne == 0u,
                "Resolution subdivision was incorrectly marked as neighbor growth"
            );
        }
        // Execute the real optimizer after resource replacement.
        GradRecord gradient = {};
        gradient.appearanceValid = 1u;
        gradient.radianceGrad.coefficients[0] = float3(1.0f);
        pass->mGridResources.gradPages[0]->setBlob(&gradient, 0, sizeof(gradient));
        auto update = pass->mUpdatePass.mpComputePass;
        auto cb = update->getRootVar()["CB"];
        cb["gSparseUpdateOffset"] = 0u;
        cb["gUseGradCountNormalize"] = true;
        cb["gGradScale"] = 1.0f;
        cb["gLrRadiance"] = 0.001f;
        cb["gLrOpacity"] = 0.0f;
        cb["gLrCenter"] = 0.0f;
        cb["gLrShape"] = 0.0f;
        cb["gLrRotation"] = 0.0f;
        cb["gEnableEllipsoidPruning"] = false;
        cb["gRadianceAdamCapacity"] = uint32_t(pass->mGridResources.radianceAdamPages.size()) * SPARSE_POOL_PAGE_SIZE;
        update->execute(ctx, uint3(8u, 1, 1));
        pass->barrierSparseVoxels(ctx);
        ctx->submit(true);
        require(
            read<VoxelData>(pass->mGridResources.voxelPages, 0).radiance.coefficients[0].x < parent.radiance.coefficients[0].x,
            "Optimizer did not update the refined grid"
        );
        std::cout << "PASS: mode1 gates topology; eight-child transfer, opacity, fresh Adam, production optimizer rebinding\n";

        pass->mOptimizerParams.currentIteration = 5u;
        pass->advanceCoarseToFine(ctx);
        require(pass->mVoxelResolution == 32u, "Second level did not receive its own optimization time");
        pass->mOptimizerParams.currentIteration = 6u;
        pass->advanceCoarseToFine(ctx);
        require(
            pass->mVoxelResolution == 64u && pass->mGridResources.gridData.activeVoxelCount > 8u &&
                pass->mGridResources.gridData.activeVoxelCount <= 64u,
            "Second refinement failed across pool pages"
        );
        pass->mRayMarchingPass.mSampleIndex = pass->mRayMarchingPass.mSpp;
        require(pass->shouldCollectDeletionEvidence(), "Mode1 target level did not enable deletion evidence");
        pass->mOptimizerParams.currentIteration = 11u;
        pass->advanceCoarseToFine(ctx);
        require(pass->mOptimizerParams.isRunning, "Training stopped before optimizing the target level");
        const uint32_t beforeTargetDeletion = pass->mGridResources.gridData.activeVoxelCount;
        TopologyEvidence targetCandidate = read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, beforeTargetDeletion - 1u);
        targetCandidate.packedCountsAndFlags |= kTopologyEvidenceDeletionCandidate;
        pass->mGridResources.topologyEvidencePages[(beforeTargetDeletion - 1u) / SPARSE_POOL_PAGE_SIZE]->setBlob(
            &targetCandidate, size_t((beforeTargetDeletion - 1u) % SPARSE_POOL_PAGE_SIZE) * sizeof(targetCandidate), sizeof(targetCandidate)
        );
        pass->mTopologySettings.candidateCount = 1u;
        pass->mOptimizerParams.currentIteration = 12u;
        pass->advanceCoarseToFine(ctx);
        require(
            !pass->mOptimizerParams.isRunning && pass->mSaveReconstructionRequested,
            "Target level completion did not stop and request a save"
        );
        require(
            pass->mGridResources.gridData.activeVoxelCount == beforeTargetDeletion - 1u,
            "Mode1 target level did not compact a confirmed deletion candidate"
        );
        std::cout << "PASS: weighted 16 -> 32 -> 64 schedule uses the exact total and enables deletion only at the target\n";

        auto oldPage = pass->mGridResources.voxelPages[0];
        const uint32_t beforeFailureCount = pass->mGridResources.gridData.activeVoxelCount;
        // Ensure the capacity fixture, independently of the fitted shapes, has eight children per parent.
        for (uint32_t id = 0; id < beforeFailureCount; ++id)
        {
            auto voxel = read<VoxelData>(pass->mGridResources.voxelPages, id);
            voxel.ellipsoid.center = float3(0.5f);
            voxel.ellipsoid.rotation = float4(1, 0, 0, 0);
            voxel.ellipsoid.logScale = log(pass->mGridResources.gridData.voxelSize);
            auto parentCellIndex = read<uint32_t>(pass->mGridResources.cellIndexPages, id);
            pass->uploadSparseBatch(ctx, pass->mpGridBlock, id, &parentCellIndex, 1u, &voxel);
        }
        pass->barrierSparseVoxels(ctx);
        pass->mCoarseToFine.targetResolution = 128u;
        pass->mOptimizerParams.isRunning = pass->mEnableReconstruction = true;
        pass->mSaveReconstructionRequested = false;
        pass->advanceCoarseToFine(ctx); // Eight children per parent exceed this test's 64-slot pool.
        require(
            !pass->mOptimizerParams.isRunning && pass->mVoxelResolution == 64u && pass->mGridResources.voxelPages[0] == oldPage &&
                pass->mGridResources.gridData.activeVoxelCount == beforeFailureCount,
            "Refinement failure damaged the last complete level"
        );
        require(!pass->mSaveReconstructionRequested, "Failed refinement was reported as successful completion");
        pass->mReconstructionMode = 0u;
        pass->mOptimizerParams.maxIteration = 123u;
        require(
            pass->prepareReconstruction(ctx) && pass->mOptimizerParams.maxIteration == 123u,
            "Mode0 restart acquired the pyramid iteration schedule"
        );
        require(pass->mGridResources.voxelPages[0] == oldPage, "Mode0 restart replaced an initialized grid");
        std::cout << "PASS: refinement capacity failure retains old grid; mode0 restart remains unchanged\n";
        checkIntersections(device);
    }

    static void checkSaveLoad(VoxelReconstructionNoLightTransport& source, RenderContext* ctx)
    {
        const auto grid = source.mGridResources.gridData;
        for (uint32_t mode : {0u, 1u})
        {
            source.mReconstructionMode = mode;
            const auto modeDirectory = source.getReconstructionModeDirectory();
            require(modeDirectory.filename() == (mode == 0u ? "mode0" : "mode1"), "Save path uses the wrong mode directory");
            const auto directory =
                modeDirectory / ("io_roundtrip_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(modeDirectory);
            require(std::filesystem::create_directory(directory), "Could not create an isolated IO test directory");
            struct Cleanup
            {
                std::filesystem::path file;
                std::filesystem::path directory;
                ~Cleanup()
                {
                    std::error_code error;
                    std::filesystem::remove(file, error);
                    std::filesystem::remove(directory, error);
                }
            } cleanup{directory / "roundtrip.bin", directory};
            source.saveSparseReconstruction(ctx, cleanup.file);
            auto loaded = VoxelReconstructionNoLightTransport::create(source.mpDevice, {});
            loaded->mReconstructionMode = mode;
            loaded->mUpdatePass.init();
            loaded->loadSparseReconstruction(ctx, cleanup.file);
            const auto& restored = loaded->mGridResources.gridData;
            require(
                all(restored.voxelCount == grid.voxelCount) && all(restored.voxelSize == grid.voxelSize) &&
                    all(restored.gridMin == grid.gridMin) && restored.activeVoxelCount == grid.activeVoxelCount,
                "Save/load changed refined grid metadata"
            );
            for (uint32_t id = 0u; id < grid.activeVoxelCount; ++id)
            {
                const auto before = read<VoxelData>(source.mGridResources.voxelPages, id);
                const auto after = read<VoxelData>(loaded->mGridResources.voxelPages, id);
                require(std::memcmp(&before, &after, sizeof(before)) == 0, "Save/load changed voxel geometry or appearance bits");
                require(
                    read<uint32_t>(source.mGridResources.cellIndexPages, id) == read<uint32_t>(loaded->mGridResources.cellIndexPages, id),
                    "Save/load changed sparse spatial addresses"
                );
            }
            require(
                ctx->readTextureSubresource(source.mGridResources.indexPages[0].get(), 0) ==
                    ctx->readTextureSubresource(loaded->mGridResources.indexPages[0].get(), 0),
                "Loaded index does not preserve empty cells"
            );
            require(
                !loaded->mOptimizerParams.isRunning && loaded->mOptimizerParams.currentIteration == 0u,
                "Loading did not stop/reset optimization"
            );
            loaded->mCoarseToFine.targetResolution = 64u;
            loaded->mCoarseToFine.totalIterations = 6u;
            require(
                loaded->prepareReconstruction(ctx) && loaded->mVoxelResolution == 32u,
                "Restart after loading discarded the saved resolution"
            );
            if (mode == 1u)
                require(loaded->mOptimizerParams.maxIteration == 6u, "Loaded mode1 schedule did not use the configured total");
        }
        source.mReconstructionMode = 1u;
        std::cout
            << "PASS: mode0/mode1 v3 save-load roundtrip preserves refined sparse cells and parameters; mode1 restarts at saved level\n";
    }

    static void checkIntersections(const ref<Device>& device)
    {
        auto ctx = device->getRenderContext();
        auto pass = VoxelReconstructionNoLightTransport::create(device, {});
        pass->mReconstructionMode = 1u;
        pass->mCoarseToFine.targetResolution = 32u;
        const auto check = [&](VoxelData parent, float3 size, uint32_t expectedMask, uint32_t parents = 1u)
        {
            GridData grid = {};
            grid.voxelCount = uint3(16u);
            grid.voxelSize = size;
            grid.activeVoxelCount = grid.solidVoxelCount = parents;
            auto resources = pass->allocateSparseGrid(ctx, grid, parents);
            auto block = pass->createSparseGridBlock(resources);
            pass->commitSparseGrid(std::move(resources), block, 16u);
            std::vector<uint32_t> cells(parents);
            std::vector<VoxelData> data(parents, parent);
            for (uint32_t i = 0; i < parents; ++i)
                cells[i] = i + 2u + 16u * 3u + 256u * 3u;
            pass->uploadSparseBatch(ctx, block, 0, cells.data(), parents, data.data());
            pass->barrierSparseVoxels(ctx);
            ctx->submit(true);
            if (expectedMask == 0u)
            {
                bool rejected = false;
                try
                {
                    pass->refineCoarseToFineGrid(ctx);
                }
                catch (const std::exception& error)
                {
                    rejected = std::string(error.what()).find("No eligible refinement children") != std::string::npos;
                }
                require(rejected, "Transparent parent was not filtered");
                require(
                    pass->mVoxelResolution == 16u && pass->mpGridBlock == block, "All-filtered refinement did not retain the previous grid"
                );
                return;
            }
            pass->refineCoarseToFineGrid(ctx);
            const auto bytes = ctx->readTextureSubresource(pass->mGridResources.indexPages[0].get(), 0);
            std::vector<bool> seen(pass->mGridResources.gridData.activeVoxelCount, false);
            uint32_t occupied = 0u;
            for (uint32_t p = 0; p < parents; ++p)
                for (uint32_t child = 0; child < 8u; ++child)
                {
                    uint32_t x = 2u * (p + 2u) + (child & 1u);
                    uint32_t y = 6u + ((child >> 1u) & 1u), z = 6u + ((child >> 2u) & 1u);
                    uint32_t cellIndex = x + 32u * y + 1024u * z;
                    int32_t id;
                    std::memcpy(&id, bytes.data() + cellIndex * sizeof(int32_t), sizeof(id));
                    if ((expectedMask & (1u << child)) == 0u)
                    {
                        require(id == -1, "A child outside the parent ellipsoid became occupied");
                        continue;
                    }
                    if (id < 0 || uint32_t(id) >= seen.size())
                        throw std::runtime_error(fmt::format(
                            "Missing fitted child {}; expected mask {}; parent axes ({}, {}, {})",
                            child,
                            expectedMask,
                            std::exp(parent.ellipsoid.logScale.x),
                            std::exp(parent.ellipsoid.logScale.y),
                            std::exp(parent.ellipsoid.logScale.z)
                        ));
                    require(!seen[id], "Multiple child cells share one compact ID");
                    seen[id] = true;
                    ++occupied;
                    require(
                        read<uint32_t>(pass->mGridResources.cellIndexPages, uint32_t(id)) == cellIndex,
                        "Filtered child has an incorrect reverse spatial index"
                    );
                    const auto inherited = read<VoxelData>(pass->mGridResources.voxelPages, uint32_t(id));
                    require(
                        inherited.occupied == 1u && std::memcmp(&inherited.radiance, &parent.radiance, sizeof(parent.radiance)) == 0,
                        "Filtered child did not inherit parent appearance"
                    );
                    require(
                        std::memcmp(&inherited.opacity, &parent.opacity, sizeof(parent.opacity)) == 0,
                        "Directional opacity SH was modified during refinement"
                    );
                    checkGeometry(parent.ellipsoid, inherited.ellipsoid, size, uint3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u));
                }
            require(occupied == seen.size(), "Compact refinement retained unused/rejected child slots");
            if (parents == 9u)
                checkSaveLoad(*pass, ctx);
        };
        VoxelData sphere = {};
        sphere.occupied = 1u;
        sphere.radiance.coefficients[0] = float3(0.2f, 0.4f, 0.6f);
        sphere.ellipsoid.rotation = float4(1, 0, 0, 0);
        sphere.ellipsoid.center = float3(0.25f);
        sphere.ellipsoid.logScale = float3(std::log(0.1f));
        check(sphere, float3(1), 0x01u);     // Ellipsoid wholly inside one child, with no corner inside it.
        check(sphere, float3(1), 0x01u, 9u); // Worst-case 72 > 64 slots; actual nine children must fit.
        sphere.opacity.coefficients[0] = std::log(0.001f / 0.999f) / calcSH(0u, float3(0, 0, 1));
        check(sphere, float3(1), 0u); // Transparent in every direction; rollback if all parents are rejected.
        pass->mCoarseToFine.parentOpacityThreshold = 0.0f;
        check(sphere, float3(1), 0x01u); // Zero explicitly disables filtering.
        pass->mCoarseToFine.parentOpacityThreshold = 0.01f;
#if SH_OPACITY_COUNT > 1
        sphere.opacity.coefficients[1] = 20.0f;
        check(sphere, float3(1), 0x01u); // Low DC does not imply transparency in every direction.
        sphere.opacity.coefficients[1] = 0.0f;
#endif
        sphere.opacity.coefficients[0] = 0.0f;
        sphere.ellipsoid.logScale = float3(std::log(0.249f));
        check(sphere, float3(1), 0x01u);
        sphere.ellipsoid.logScale = float3(std::log(0.25f));
        check(sphere, float3(1), 0x01u); // Tangency alone cannot contain a positive-volume child.
        sphere.ellipsoid.logScale = float3(std::log(0.26f));
        check(sphere, float3(1), 0x17u); // Face-interior overlap, even when child centers/corners miss.
        sphere.ellipsoid.logScale = float3(std::log(0.4f));
        check(sphere, float3(1), 0x7fu); // Edge overlaps but the diagonally opposite corner is outside.
        sphere.ellipsoid.logScale = float3(std::log(2.0f));
        check(sphere, float3(1), 0xffu); // All children wholly inside the parent volume.
        sphere.ellipsoid.logScale = float3(std::log(0.05f), std::log(0.1f), std::log(0.15f));
        check(sphere, float3(0.5f, 1.0f, 1.5f), 0x01u);
        sphere.ellipsoid.center = float3(0.25f, 0.75f, 0.25f);
        sphere.ellipsoid.rotation = float4(std::cos(3.14159265359f / 8.0f), 0, 0, std::sin(3.14159265359f / 8.0f));
        for (float width : {0.02f, 0.001f, 0.0001f})
        {
            sphere.ellipsoid.logScale = float3(std::log(0.6f), std::log(width), std::log(width));
            // Very thin edge slivers cannot hold centers inside [0.01,0.99]; keep the central child only.
            check(sphere, float3(1), width > 0.01f ? 0x0du : 0x04u);
        }
        sphere.ellipsoid.logScale = float3(std::log(1e-6f));
        check(sphere, float3(1), 0u); // Never inflate unrepresentable geometry to the optimizer minimum.
        std::cout << "PASS: refinement volume containment, face/edge/tangent overlap, rotated thin-shape rejection, non-cubic cells, "
                     "actual-count allocation; inherited rotation/ratios and parent/cell containment\n";
    }
};

void runCoarseToFineTests(const ref<Device>& device)
{
    CoarseToFineTestAccess::run(device);
}
