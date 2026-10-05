#include "../VoxelReconstructionNoLightTransport.h"
#include <cmath>
#include <cstring>
#include <iostream>

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

    static void run(const ref<Device>& device)
    {
        auto ctx = device->getRenderContext();
        auto pass = VoxelReconstructionNoLightTransport::create(device, {});
        require(pass->mReconstructionMode == 0u, "Fixed-resolution mode must remain the default");
        pass->mReconstructionMode = 1u;
        pass->mCoarseToFine.targetResolution = 64u;
        pass->mCoarseToFine.iterationsPerLevel = 2u;
        pass->createUpdatePassResource(ctx);
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
        parent.ellipsoid.logScale = float3(std::log(0.8f), std::log(0.5f), std::log(0.3f));
        parent.ellipsoid.rotation = float4(std::cos(0.3f), 0, 0, std::sin(0.3f));
        parent.radiance.coefficients[0] = float3(0.3f, 0.6f, 0.9f);
        parent.opacity.coefficients[0] = std::log(0.4f / 0.6f) / calcSH(0u, float3(0, 0, 1));
        const uint32_t cell = 3u + 16u * 4u + 256u * 5u;
        pass->uploadSparseBatch(ctx, block, 0u, &cell, 1u, &parent);
        pass->barrierSparseVoxels(ctx);
        ctx->submit(true);
        pass->mPointCloud.initialized = true;
        require(pass->prepareReconstruction(ctx), "Could not start mode1 from an existing coarse grid");
        require(pass->mOptimizerParams.maxIteration == 6u, "Level schedule omitted final-resolution optimization");
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
        pass->mOptimizerParams.currentIteration = 2u;
        pass->advanceCoarseToFine(ctx);
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
            require(
                child.occupied == 1u && all(child.ellipsoid.center == parent.ellipsoid.center) &&
                    all(child.ellipsoid.rotation == parent.ellipsoid.rotation),
                "Child lost local center or rotation"
            );
            require(
                all(abs(child.ellipsoid.logScale - parent.ellipsoid.logScale + float3(std::log(2.0f))) < float3(1e-6f)),
                "Child world semi-axes did not halve"
            );
            require(std::memcmp(&child.radiance, &parent.radiance, sizeof(parent.radiance)) == 0, "Refinement reset radiance");
            auto opacity = child.opacity;
            const float alpha = opacity.calcOpacity(float3(0, 0, 1));
            require(
                std::abs(1.0f - (1.0f - alpha) * (1.0f - alpha) - 0.4f) < 1e-5f,
                "Refinement did not preserve two-child constant-opacity transmittance"
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

        pass->mOptimizerParams.currentIteration = 3u;
        pass->advanceCoarseToFine(ctx);
        require(pass->mVoxelResolution == 32u, "Second level did not receive its own optimization time");
        pass->mOptimizerParams.currentIteration = 4u;
        pass->advanceCoarseToFine(ctx);
        require(
            pass->mVoxelResolution == 64u && pass->mGridResources.gridData.activeVoxelCount == 64u,
            "Second refinement failed across pool pages"
        );
        pass->mOptimizerParams.currentIteration = 5u;
        pass->advanceCoarseToFine(ctx);
        require(pass->mOptimizerParams.isRunning, "Training stopped before optimizing the target level");
        pass->mOptimizerParams.currentIteration = 6u;
        pass->advanceCoarseToFine(ctx);
        require(
            !pass->mOptimizerParams.isRunning && pass->mSaveReconstructionRequested,
            "Target level completion did not stop and request a save"
        );
        std::cout << "PASS: 16 -> 32 -> 64 schedule includes full optimization at the target resolution\n";

        auto oldPage = pass->mGridResources.voxelPages[0];
        pass->mCoarseToFine.targetResolution = 128u;
        pass->mOptimizerParams.isRunning = pass->mEnableReconstruction = true;
        pass->mSaveReconstructionRequested = false;
        pass->advanceCoarseToFine(ctx); // 512 children exceed this test's 64-slot pool.
        require(
            !pass->mOptimizerParams.isRunning && pass->mVoxelResolution == 64u && pass->mGridResources.voxelPages[0] == oldPage &&
                pass->mGridResources.gridData.activeVoxelCount == 64u,
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
    }
};

void runCoarseToFineTests(const ref<Device>& device)
{
    CoarseToFineTestAccess::run(device);
}
