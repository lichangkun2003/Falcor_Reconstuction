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

    static void checkGeometry(const GaussianEllipsoid& parent, const GaussianEllipsoid& child, float3 size, uint3 bits,
        float overlap = 0.1f)
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
            require(extent <= std::min(child.center[axis], 1.0f - child.center[axis]) + overlap + 3e-5f,
                "Child exceeds the bounded face overlap");
        }
        require(all(childAxes >= max(0.5f * size * 1e-4f, float3(1e-8f)) * 0.999f), "Child below optimizer minimum scale");
    }

    static float attenuatedOpacityDC(const SphericalHarmonicsOpacity& parent, float opticalDepthScale)
    {
        const float y0 = calcSH(0u, float3(0, 0, 1));
        const double parentLogit = double(parent.coefficients[0]) * y0;
        const double parentAlpha = 1.0 / (1.0 + std::exp(-parentLogit));
        const double childAlpha = 1.0 - std::pow(1.0 - parentAlpha, opticalDepthScale);
        return float(std::log(childAlpha / (1.0 - childAlpha)) / y0);
    }

    static void checkAppearance(const VoxelData& parent, const VoxelData& child, float opticalDepthScale)
    {
        require(all(child.radiance.coefficients[0] == parent.radiance.coefficients[0]), "Refinement changed radiance DC");
        for (uint32_t coefficient = 1u; coefficient < SH_COUNT; ++coefficient)
            require(all(child.radiance.coefficients[coefficient] == float3(0)), "Refinement retained directional radiance");
        require(std::abs(child.opacity.coefficients[0] - attenuatedOpacityDC(parent.opacity, opticalDepthScale)) < 1e-4f,
            "Refinement opacity DC did not use optical-depth attenuation");
        for (uint32_t coefficient = 1u; coefficient < SH_OPACITY_COUNT; ++coefficient)
            require(child.opacity.coefficients[coefficient] == 0.0f, "Refinement retained directional opacity");
    }

    static void checkTrainingViewOrder(VoxelReconstructionNoLightTransport& pass)
    {
        const auto originalCameras = pass.mReferenceCameras;
        const auto originalOptimizer = pass.mOptimizerParams;
        const auto originalOrder = pass.mTrainingViewOrder;
        const auto originalRng = pass.mTrainingViewRng;
        pass.mTrainingViewRng.seed(42u);
        std::vector<uint32_t> previous;
        bool changed = false;
        for (uint32_t count : {1u, 8u, 8u, 8u, 3u})
        {
            pass.mReferenceCameras.resize(count);
            pass.shuffleTrainingViews();
            std::vector<bool> seen(count, false);
            const auto order = pass.mTrainingViewOrder;
            for (uint32_t position = 0u; position < count; ++position)
            {
                pass.mOptimizerParams.currentView = position;
                const uint32_t camera = pass.getTrainingViewIndex();
                require(camera < count && !seen[camera], "Shuffled iteration repeated or missed a camera");
                seen[camera] = true;
                for (uint32_t sample = 0u; sample < 8u; ++sample)
                    require(pass.getTrainingViewIndex() == camera, "Camera changed between SPP samples");
            }
            require(pass.mTrainingViewOrder == order, "Reading view IDs mutated the training permutation");
            if (previous.size() == count && previous != order)
                changed = true;
            previous = order;
        }
        require(changed, "Training order was not reshuffled between iterations");
        pass.mReferenceCameras = originalCameras;
        pass.mOptimizerParams = originalOptimizer;
        pass.mTrainingViewOrder = originalOrder;
        pass.mTrainingViewRng = originalRng;
        std::cout << "PASS: per-iteration shuffled camera permutation, stable SPP IDs, dataset resize\n";
    }

    static void checkLevelBudgets(VoxelReconstructionNoLightTransport& pass)
    {
        const auto original = pass.mCoarseToFine;
        pass.mCoarseToFine.scheduleStartResolution = 64u;
        pass.mCoarseToFine.targetResolution = 512u;
        pass.mCoarseToFine.totalIterations = 700u;
        const uint32_t weights[4] = COARSE_TO_FINE_LEVEL_WEIGHTS;
        const uint64_t weightSum = uint64_t(weights[0]) + weights[1] + weights[2] + weights[3];
        uint32_t expected[4];
        uint64_t cumulativeWeight = 0u, previousBudget = 0u;
        for (uint32_t index = 0u; index < 4u; ++index)
        {
            cumulativeWeight += weights[index];
            const uint64_t budget = 696u * cumulativeWeight / weightSum;
            expected[index] = uint32_t(1u + budget - previousBudget);
            previousBudget = budget;
        }
        for (uint32_t index = 0u; index < 4u; ++index)
            require(pass.coarseToFineLevelBudget(64u << index) == expected[index], "64-start experiment ratios are incorrect");
        for (uint32_t levels = 1u; levels <= 7u; ++levels)
        {
            pass.mCoarseToFine.scheduleStartResolution = 16u;
            pass.mCoarseToFine.targetResolution = 16u << (levels - 1u);
            for (uint32_t total : {levels, levels + 1u, 12u, 700u, 701u, 1000000u})
            {
                pass.mCoarseToFine.totalIterations = total;
                uint32_t sum = 0u;
                for (uint32_t index = 0u; index < levels; ++index)
                {
                    const uint32_t budget = pass.coarseToFineLevelBudget(16u << index);
                    require(budget >= 1u, "A level received no optimization time");
                    sum += budget;
                }
                require(sum == total, "Interpolated schedule did not preserve the exact iteration total");
            }
        }
        pass.mCoarseToFine.scheduleStartResolution = pass.mCoarseToFine.targetResolution = 512u;
        pass.mCoarseToFine.totalIterations = 700u;
        require(pass.coarseToFineLevelBudget(512u) == 700u, "Final-level restart did not receive the full total");
        pass.mCoarseToFine = original;
        std::cout << "PASS: coarse-biased level ratios, exact totals across 1-7 levels, minimum budgets, target-level restart\n";
    }

    static void run(const ref<Device>& device)
    {
        auto ctx = device->getRenderContext();
        auto pass = VoxelReconstructionNoLightTransport::create(device, {});
        require(pass->mReconstructionMode == uint32_t(RECONSTRUCTION_MODE), "Configured reconstruction mode was not applied");
        checkLevelBudgets(*pass);
        checkTrainingViewOrder(*pass);
        pass->mReconstructionMode = 1u;
        pass->mCoarseToFine.startResolution = 16u;
        pass->mCoarseToFine.targetResolution = 64u;
        pass->mCoarseToFine.totalIterations = 12u; // Interpolated weights produce 3, 3, and 6 rounds.
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
        const uint32_t firstBoundary = pass->coarseToFineLevelBudget(16u);
        const uint32_t secondBoundary = firstBoundary + pass->coarseToFineLevelBudget(32u);
        pass->mOptimizerParams.currentIteration = firstBoundary - 1u;
        pass->advanceCoarseToFine(ctx);
        require(pass->mVoxelResolution == 16u, "Extra coarse round was skipped");
        pass->mOptimizerParams.currentIteration = firstBoundary;
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
            pass->mOptimizerParams.currentIteration == firstBoundary && pass->getEffectiveOpacityLearningRate() == beforeLr,
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
            checkGeometry(parent.ellipsoid, child.ellipsoid, grid.voxelSize,
                uint3(id & 1u, (id >> 1u) & 1u, (id >> 2u) & 1u), pass->mCoarseToFine.childFaceOverlap);
            checkAppearance(parent, child, pass->mCoarseToFine.opacityOpticalDepthScale);
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
        cb["gGradientSampleCount"] = 1u;
        cb["gGradScale"] = 1.0f;
        cb["gLrRadiance"] = 0.001f;
        cb["gLrOpacity"] = 0.0f;
        cb["gLrCenter"] = 0.0f;
        cb["gLrShape"] = 0.0f;
        cb["gLrRotation"] = 0.0f;
        cb["gRadianceAdamCapacity"] = uint32_t(pass->mGridResources.radianceAdamPages.size()) * SPARSE_POOL_PAGE_SIZE;
        update->execute(ctx, uint3(8u, 1, 1));
        pass->barrierSparseVoxels(ctx);
        ctx->submit(true);
        require(
            read<VoxelData>(pass->mGridResources.voxelPages, 0).radiance.coefficients[0].x < parent.radiance.coefficients[0].x,
            "Optimizer did not update the refined grid"
        );
        std::cout << "PASS: mode1 gates topology; eight-child transfer, opacity, fresh Adam, production optimizer rebinding\n";

        pass->mOptimizerParams.currentIteration = secondBoundary - 1u;
        pass->advanceCoarseToFine(ctx);
        require(pass->mVoxelResolution == 32u, "Second level did not receive its own optimization time");
        pass->mOptimizerParams.currentIteration = secondBoundary;
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
        checkExperimentNaming(device);
    }

    static void checkExperimentNaming(const ref<Device>& device)
    {
        const auto directory = std::filesystem::temp_directory_path() /
            ("falcor_experiment_names_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(directory), "Could not create isolated naming directory");
        struct Cleanup
        {
            std::filesystem::path directory;
            ~Cleanup()
            {
                // Only this test's freshly created temporary directory is removed.
                std::error_code error;
                std::filesystem::remove_all(directory, error);
            }
        } cleanup{directory};
        auto ctx = device->getRenderContext();
        auto pass = VoxelReconstructionNoLightTransport::create(device, {});
        pass->mReconstructionOutputRoot = directory;
        pass->mReconstructionMode = 1u;
        pass->mCoarseToFine.startResolution = 16u;
        pass->mCoarseToFine.targetResolution = 32u;
        pass->mReconstructionNameTag = "naming";
        GridData grid = {};
        grid.voxelCount = uint3(16u);
        grid.voxelSize = float3(1.0f);
        grid.activeVoxelCount = grid.solidVoxelCount = 1u;
        auto resources = pass->allocateSparseGrid(ctx, grid, 1u);
        auto block = pass->createSparseGridBlock(resources);
        pass->commitSparseGrid(std::move(resources), block, 16u);
        VoxelData parent = {};
        parent.occupied = 1u;
        parent.ellipsoid.center = float3(0.5f);
        parent.ellipsoid.logScale = float3(std::log(2.0f));
        parent.ellipsoid.rotation = float4(1, 0, 0, 0);
        const uint32_t cell = 2u + 16u * 3u + 256u * 3u;
        pass->uploadSparseBatch(ctx, block, 0u, &cell, 1u, &parent);
        pass->barrierSparseVoxels(ctx);
        pass->mReduceLossPass.iterationLossHistory = {0.2f, 0.1f};
        const auto first = pass->getDefaultReconstructionSavePath();
        if (first.filename().string().find("_E0_naming_16_") == std::string::npos)
            throw std::runtime_error("Empty directory did not start at E0: " + first.string());
        require(pass->getDefaultReconstructionSavePath() == first, "Path preview consumed experiment numbers");
        pass->saveSparseReconstruction(ctx, first);
        require(std::filesystem::is_regular_file(first.parent_path() / "Loss" / (first.stem().string() + ".csv")),
            "Loss file does not share its bin's experiment and level name");
        pass->refineCoarseToFineGrid(ctx);
        const auto fine = pass->getDefaultReconstructionSavePath();
        require(fine.filename().string().find("_E0_naming_32_") != std::string::npos, "One run's levels did not share E0");
        pass->mPointCloud.initialized = true;
        pass->saveReconstruction(ctx);
        pass->refreshReconstructionFileList();
        require(pass->mViewedReconstructionPath == fine && pass->mSelectedExperimentLevel == 1u &&
            pass->mExperimentLevelResolutions == std::vector<uint32_t>({16u, 32u}),
            "Final save did not expose the current experiment's completed levels");
        // Loading any one level discovers the other checkpoints of the same experiment.
        pass->loadSparseReconstruction(ctx, fine);
        pass->refreshReconstructionFileList();
        require(pass->mExperimentLevelResolutions == std::vector<uint32_t>({16u, 32u}) &&
            pass->mSelectedExperimentLevel == 1u, "Loaded fine result did not discover/order its coarse checkpoint");
        const auto coarseFile = pass->mReconstructionFilePaths[pass->mExperimentLevelFileIndices[0u]];
        pass->loadSparseReconstruction(ctx, coarseFile);
        pass->refreshReconstructionFileList();
        require(pass->mVoxelResolution == 16u && pass->mSelectedExperimentLevel == 0u &&
            pass->mExperimentLevelResolutions.size() == 2u, "Switching coarse lost the experiment's fine level");
        const auto fineFile = pass->mReconstructionFilePaths[pass->mExperimentLevelFileIndices[1u]];
        pass->loadSparseReconstruction(ctx, fineFile);
        pass->refreshReconstructionFileList();
        require(pass->mVoxelResolution == 32u && pass->mSelectedExperimentLevel == 1u,
            "Could not return to the saved finest level");
        const auto viewed = pass->mViewedReconstructionPath;
        pass->loadReconstruction(ctx, directory / "missing.bin");
        pass->refreshReconstructionFileList();
        require(pass->mViewedReconstructionPath == viewed && pass->mSelectedExperimentLevel == 1u && pass->mVoxelResolution == 32u,
            "Failed level load changed the live grid/viewing experiment");
        pass->mReduceLossPass.iterationLossHistory = {0.2f, 0.1f};
        const auto repeated = pass->getDefaultReconstructionSavePath();
        require(repeated.filename().string().find("_E1_naming_32_") != std::string::npos, "Repeated save would overwrite E0");
        pass->saveSparseReconstruction(ctx, repeated);
        const auto name = first.filename().string();
        const auto base = name.substr(0u, name.find("_E0_") + 2u);
        const auto two = first.parent_path() / (base + "2_other_16.bin");
        const auto nine = first.parent_path() / (base + "9_other_16.bin");
        pass->saveSparseReconstruction(ctx, two);
        pass->saveSparseReconstruction(ctx, nine);
        require(std::filesystem::remove(nine), "Could not remove the isolated test bin to check CSV-only numbering");
        pass->resetPointCloudOptimization(ctx);
        const auto ten = pass->getDefaultReconstructionSavePath();
        require(ten.filename().string().find("_E10_") != std::string::npos, "Restart did not scan the maximum existing numeric index");
        pass->saveSparseReconstruction(ctx, ten);
        pass->refreshReconstructionFileList();
        const auto& files = pass->mReconstructionFilePaths;
        const auto twoPosition = std::find_if(files.begin(), files.end(), [&](const auto& path) { return path.filename() == two.filename(); });
        const auto tenPosition = std::find_if(files.begin(), files.end(), [&](const auto& path) { return path.filename() == ten.filename(); });
        require(twoPosition != files.end() && tenPosition != files.end() && twoPosition < tenPosition,
            "File list sorts E10 before E2");
        require(pass->mExperimentLevelResolutions == std::vector<uint32_t>({16u, 32u}),
            "Another E-numbered experiment leaked into the viewing levels");
        pass->mReconstructionMode = 0u;
        pass->refreshReconstructionFileList();
        require(pass->mExperimentLevelFileIndices.empty(), "Viewing levels leaked between mode directories");
        pass->mReconstructionMode = 1u;
        pass->refreshReconstructionFileList();
        require(pass->mExperimentLevelResolutions.size() == 2u, "Returning to mode1 did not restore the viewing group");
        const auto realViewed = pass->mViewedReconstructionPath;
        const auto modeDirectory = pass->getReconstructionModeDirectory();
        pass->mViewedReconstructionPath = modeDirectory / "ship_recon10_5_E0_tag_128_alpha_64_radiance_adam.bin";
        pass->mReconstructionFilePaths = {
            modeDirectory / "ship_recon10_5_E0_tag_128_alpha_512_radiance_adam.bin",
            pass->mViewedReconstructionPath,
            modeDirectory / "ship_recon10_5_E0_tag_128_alpha_128_radiance_adam.bin",
            modeDirectory / "ship_recon10_6_E0_tag_128_alpha_256_radiance_adam.bin",
            modeDirectory / "ship_recon10_5_E1_tag_128_alpha_256_radiance_adam.bin",
            modeDirectory / "ship_recon10_5_E0_other_256_radiance_adam.bin",
        };
        pass->refreshExperimentLevels();
        require(pass->mExperimentLevelResolutions == std::vector<uint32_t>({64u, 128u, 512u}) &&
            pass->mSelectedExperimentLevel == 0u, "Resolution-like name tags/dates mixed experiments or corrupted numeric sorting");
        pass->mViewedReconstructionPath = realViewed;
        pass->refreshReconstructionFileList();
        std::cout << "PASS: experiment-level discovery, numeric resolution order, coarse/fine load switching, failed-load retention, E/mode isolation\n";
        auto restarted = VoxelReconstructionNoLightTransport::create(device, {});
        restarted->mReconstructionOutputRoot = directory;
        restarted->mReconstructionMode = 1u;
        require(restarted->getDefaultReconstructionSavePath().filename().string().find("_E11_") != std::string::npos,
            "A new process/pass reused an existing experiment index");
        restarted->mReconstructionMode = 0u;
        require(restarted->getDefaultReconstructionSavePath().filename().string().find("_E0_") != std::string::npos,
            "Experiment numbering leaked between mode directories");
        std::cout << "PASS: experiment E numbering, shared level prefix, repeated-save protection, matching loss CSV, restart scan, numeric order\n";
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
        pass->mCoarseToFine.startResolution = 16u;
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
                    require(inherited.occupied == 1u, "Filtered child is not occupied");
                    checkAppearance(parent, inherited, pass->mCoarseToFine.opacityOpticalDepthScale);
                    checkGeometry(parent.ellipsoid, inherited.ellipsoid, size,
                        uint3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u), pass->mCoarseToFine.childFaceOverlap);
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
        for (uint32_t coefficient = 1u; coefficient < SH_COUNT; ++coefficient)
            sphere.radiance.coefficients[coefficient] = float(coefficient) * float3(0.3f, -0.2f, 0.1f);
        for (uint32_t coefficient = 1u; coefficient < SH_OPACITY_COUNT; ++coefficient)
            sphere.opacity.coefficients[coefficient] = coefficient % 2u ? float(coefficient) : -float(coefficient);
        for (float scale : {0.65f, 1.0f})
        {
            pass->mCoarseToFine.opacityOpticalDepthScale = scale;
            check(sphere, float3(1), 0x01u); // Clear all directional terms even with DC attenuation disabled.
        }
        pass->mCoarseToFine.opacityOpticalDepthScale = 0.65f;
        for (uint32_t coefficient = 1u; coefficient < SH_OPACITY_COUNT; ++coefficient)
            sphere.opacity.coefficients[coefficient] = 0.0f;
        check(sphere, float3(1), 0x01u);     // Ellipsoid wholly inside one child, with no corner inside it.
        check(sphere, float3(1), 0x01u, 9u); // Worst-case 72 > 64 slots; actual nine children must fit.
        sphere.opacity.coefficients[0] = std::log(0.001f / 0.999f) / calcSH(0u, float3(0, 0, 1));
        require(pass->mCoarseToFine.parentOpacityThreshold == 0.0f, "Intermediate opacity filtering is not disabled by default");
        check(sphere, float3(1), 0x01u); // Low-opacity parents survive by default until target-level deletion.
        pass->mCoarseToFine.parentOpacityThreshold = 0.01f;
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
        for (uint32_t id = 0u; id < 8u; ++id)
        {
            const auto child = read<VoxelData>(pass->mGridResources.voxelPages, id);
            require(all(abs(child.ellipsoid.center - float3(0.5f)) < float3(1e-5f)),
                "Fully interior child was pulled toward the parent center");
            require(all(abs(exp(child.ellipsoid.logScale) - float3(0.3f)) < float3(2e-5f)),
                "Interior child did not reach the fine-grid overlap size");
        }
        pass->mCoarseToFine.childFaceOverlap = 0.0f;
        check(sphere, float3(1), 0xffu);
        require(all(abs(exp(read<VoxelData>(pass->mGridResources.voxelPages, 0u).ellipsoid.logScale) - float3(0.25f)) < float3(2e-5f)),
            "Zero overlap did not restore cell-sized interior initialization");
        pass->mCoarseToFine.childFaceOverlap = 0.1f;
        sphere.ellipsoid.center = float3(0.5f);
        sphere.ellipsoid.logScale = float3(std::log(0.5f));
        // These centers are inside, but their boxes cross the parent's curved
        // boundary. They must use the inward-shifted boundary path.
        check(sphere, float3(1), 0xffu);
        for (uint32_t id = 0u; id < 8u; ++id)
        {
            const auto child = read<VoxelData>(pass->mGridResources.voxelPages, id);
            require(length(child.ellipsoid.center - float3(0.5f)) > 0.1f,
                "Boundary cell was misclassified using only its center");
            require(all(exp(child.ellipsoid.logScale) > float3(0.2f)),
                "Relaxed boundary fit still has the old strictly-inscribed size");
        }
        const auto overlapChild = read<VoxelData>(pass->mGridResources.voxelPages, 0u).ellipsoid;
        const float overlapCoverage = length((float3(1.0f, 0.5f, 0.5f) - overlapChild.center) * 0.5f / exp(overlapChild.logScale));
        require(overlapCoverage < 1.0f, "Refinement overlap did not cover a representative interior face gap");
        pass->mCoarseToFine.childFaceOverlap = 0.0f;
        check(sphere, float3(1), 0xffu);
        const auto strictChild = read<VoxelData>(pass->mGridResources.voxelPages, 0u).ellipsoid;
        const float strictCoverage = length((float3(1.0f, 0.5f, 0.5f) - strictChild.center) * 0.5f / exp(strictChild.logScale));
        require(strictCoverage > 1.0f, "Coverage regression fixture does not reproduce the strict-fit gap");
        pass->mCoarseToFine.childFaceOverlap = 0.1f;
        sphere.ellipsoid.center = float3(0.25f);
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
        std::cout << "PASS: refinement interior/boundary classification, bounded face overlap, inward boundary centers, "
                     "parent containment, face/edge/tangent overlap, thin-shape rejection, non-cubic cells, compact allocation, "
                     "DC-only appearance inheritance with and without opacity attenuation\n";
    }
};

void runCoarseToFineTests(const ref<Device>& device)
{
    CoarseToFineTestAccess::run(device);
}
