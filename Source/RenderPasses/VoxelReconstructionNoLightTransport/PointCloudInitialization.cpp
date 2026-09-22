#include "VoxelReconstructionNoLightTransport.h"

#if RECON_MODE == RECON_MODE_POINT_CLOUD
#include "PointCloudLoader.h"

void VoxelReconstructionNoLightTransport::resetPointCloudOptimization(RenderContext* pRenderContext)
{
    mEnableReconstruction = false;
    mOptimizerParams.reset();
    mPointCloud.startRequested = false;
    mPointCloud.clearAccumulation = true;
    mFrameCount = 0;
    mRayMarchingPass.mFrameIndex = 0;
    mRayMarchingPass.mSampleIndex = 0;
    mRayMarchingPass.mOptionsChanged = true;
    mReduceLossPass.meanLoss = 0.f;
    mReduceLossPass.iterationLossSum = 0.f;
    mReduceLossPass.iterationLossCount = 0;
    mReduceLossPass.iterationLossHistory.clear();
    if (mpPathRecordBuffer) pRenderContext->clearUAV(mpPathRecordBuffer->getUAV().get(), uint4(0));
    if (mGradientPass.gradBuffer) pRenderContext->clearUAV(mGradientPass.gradBuffer->getUAV().get(), uint4(0));
}

bool VoxelReconstructionNoLightTransport::initializePointCloudVoxelData(RenderContext* pRenderContext)
{
    const auto path = std::filesystem::path(ReferenceImageDir) / "init_points.ply";
    try
    {
        // Read and validate before touching the active reconstruction.
        auto grid = mGridResources.gridData;
        auto points = PointCloudInitialization::load(path,
            {grid.voxelCount.x, grid.voxelCount.y, grid.voxelCount.z},
            {grid.gridMin.x, grid.gridMin.y, grid.gridMin.z},
            {grid.voxelSize.x, grid.voxelSize.y, grid.voxelSize.z});
        grid.solidVoxelCount = static_cast<uint32_t>(points.seeds.size());
        static_assert(sizeof(PointCloudInitialization::Seed) == sizeof(uint32_t));
        if (!mpInitializePointCloudPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary("RenderPasses/VoxelReconstructionNoLightTransport/Shader/InitializePointCloud.cs.slang").csEntry("main");
            mpInitializePointCloudPass = ComputePass::create(mpDevice, desc, getReconstructionDefines());
        }

        auto seeds = mpDevice->createStructuredBuffer(sizeof(PointCloudInitialization::Seed), grid.solidVoxelCount,
            ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, points.seeds.data());
        auto voxels = mpDevice->createStructuredBuffer(sizeof(VoxelData), grid.totalVoxelCount(),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock");
        auto block = ParameterBlock::create(mpDevice, reflector);
        auto gridVar = block->getRootVar();
        gridVar["gridDataBuffer"] = voxels;
        gridVar["vBuffer"] = mGridResources.vBuffer;
        gridVar["voxelCount"] = grid.voxelCount;
        gridVar["voxelSize"] = grid.voxelSize;
        gridVar["gridMin"] = grid.gridMin;
        gridVar["solidVoxelCount"] = grid.solidVoxelCount;
        auto var = mpInitializePointCloudPass->getRootVar();
        var["gGridDataParamBlock"] = block;
        var["gSeeds"] = seeds;
        var["CB"]["gSeedCount"] = grid.solidVoxelCount;
        pRenderContext->clearUAV(voxels->getUAV().get(), uint4(0));
        // Limit each dispatch to D3D12's 65535 thread groups in X.
        constexpr uint32_t batchSize = 65535u * 256u;
        for (uint32_t offset = 0; offset < grid.solidVoxelCount; )
        {
            const uint32_t batchCount = std::min(batchSize, grid.solidVoxelCount - offset);
            var["CB"]["gSeedOffset"] = offset;
            mpInitializePointCloudPass->execute(pRenderContext, uint3(batchCount, 1, 1));
            offset += batchCount;
        }
        pRenderContext->uavBarrier(voxels.get());
        pRenderContext->submit(true);
        var["gSeeds"].setBuffer(nullptr);

        // Commit only after successful initialization, then release references to the old grid.
        const auto rebindGrid = [&block](const auto& pass)
        {
            if (pass && pass->getVars()) pass->getRootVar()["gGridDataParamBlock"].setParameterBlock(block);
        };
        rebindGrid(mpInitializeDataPass);
        rebindGrid(mRayMarchingPass.mpFullScreenPass);
        rebindGrid(mGradientPass.mpComputePass);
        rebindGrid(mUpdatePass.mpComputePass);
        mGridResources.gridDataBuffer = voxels;
        mGridResources.gridData = grid;
        mpGridBlock = block;
        resetPointCloudOptimization(pRenderContext);
        mPointCloud.initialized = true;
        mLoadedReconstructionForViewing = false;
        const auto& stats = points.statistics;
        mPointCloud.status = fmt::format("PLY: {} points, {} occupied voxels; {} outside, {} invalid",
            stats.inputPoints, grid.solidVoxelCount, stats.outsidePoints, stats.invalidPoints);
        logInfo("Point-cloud initialization: {}. {}. Coordinates: NeRF (x,y,z) -> Falcor (x,z,-y).", path.string(), mPointCloud.status);
        return true;
    }
    catch (const std::exception& error)
    {
        // A malformed/missing point cloud must not start optimization on an empty or stale grid.
        mEnableReconstruction = false;
        mOptimizerParams.isRunning = false;
        mPointCloud.startRequested = false;
        mPointCloud.status = "Initialization failed: " + std::string(error.what());
        logError("Point-cloud initialization failed for {}; previous voxels retained: {}", path.string(), error.what());
        return false;
    }
}
#endif
