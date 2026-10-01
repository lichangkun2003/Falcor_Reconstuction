#include "VoxelReconstructionNoLightTransport.h"

#if RECON_MODE == RECON_MODE_POINT_CLOUD
#include "PointCloudLoader.h"

namespace
{
uint32_t sparsePageCount(uint32_t capacity)
{
    return std::max(1u, (capacity + SPARSE_POOL_PAGE_SIZE - 1u) / SPARSE_POOL_PAGE_SIZE);
}
}

VoxelReconstructionNoLightTransport::GridResources VoxelReconstructionNoLightTransport::allocateSparseGrid(
    RenderContext* pRenderContext, const GridData& requestedGrid, uint32_t capacity)
{
    GridResources resources;
    resources.gridData = requestedGrid;
    if (any(requestedGrid.voxelCount == uint3(0)) || any(requestedGrid.voxelCount > uint3(1024)))
        throw RuntimeError("Mode 1 compact storage supports voxel grids up to 1024 cells per axis.");
    resources.gridData.indexPageCount = uint3(
        (requestedGrid.voxelCount.x + SPARSE_INDEX_PAGE_EDGE - 1u) / SPARSE_INDEX_PAGE_EDGE,
        (requestedGrid.voxelCount.y + SPARSE_INDEX_PAGE_EDGE - 1u) / SPARSE_INDEX_PAGE_EDGE,
        (requestedGrid.voxelCount.z + SPARSE_INDEX_PAGE_EDGE - 1u) / SPARSE_INDEX_PAGE_EDGE
    );
    const uint32_t indexPageCount = resources.gridData.indexPageCount.x * resources.gridData.indexPageCount.y *
        resources.gridData.indexPageCount.z;
    if (indexPageCount > SPARSE_INDEX_MAX_PAGES)
        throw RuntimeError("Mode 1 compact spatial index requires more than eight 512^3 pages.");

    const uint32_t poolPageCount = sparsePageCount(capacity);
    if (poolPageCount > SPARSE_POOL_MAX_PAGES)
        throw RuntimeError("Mode 1 occupied voxel count exceeds the segmented parameter-pool limit.");
    resources.gridData.voxelCapacity = poolPageCount * SPARSE_POOL_PAGE_SIZE;

    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    for (uint32_t z = 0; z < resources.gridData.indexPageCount.z; ++z)
    for (uint32_t y = 0; y < resources.gridData.indexPageCount.y; ++y)
    for (uint32_t x = 0; x < resources.gridData.indexPageCount.x; ++x)
    {
        const uint32_t width = std::min<uint32_t>(SPARSE_INDEX_PAGE_EDGE, requestedGrid.voxelCount.x - x * SPARSE_INDEX_PAGE_EDGE);
        const uint32_t height = std::min<uint32_t>(SPARSE_INDEX_PAGE_EDGE, requestedGrid.voxelCount.y - y * SPARSE_INDEX_PAGE_EDGE);
        const uint32_t depth = std::min<uint32_t>(SPARSE_INDEX_PAGE_EDGE, requestedGrid.voxelCount.z - z * SPARSE_INDEX_PAGE_EDGE);
        auto page = mpDevice->createTexture3D(width, height, depth, ResourceFormat::R32Int, 1u, nullptr, flags);
        pRenderContext->clearUAV(page->getUAV().get(), uint4(0xffffffffu));
        resources.indexPages.push_back(std::move(page));
    }

    for (uint32_t page = 0; page < poolPageCount; ++page)
    {
        auto voxels = mpDevice->createStructuredBuffer(sizeof(VoxelData), SPARSE_POOL_PAGE_SIZE, flags);
        auto gradients = mpDevice->createStructuredBuffer(sizeof(GradRecord), SPARSE_POOL_PAGE_SIZE, flags);
        auto adam = mpDevice->createStructuredBuffer(sizeof(GeometryAdamState), SPARSE_POOL_PAGE_SIZE, flags);
        auto radianceIndex = mpDevice->createStructuredBuffer(sizeof(uint32_t), SPARSE_POOL_PAGE_SIZE, flags);
        auto cells = mpDevice->createStructuredBuffer(sizeof(uint32_t), SPARSE_POOL_PAGE_SIZE, flags);
        pRenderContext->clearUAV(voxels->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(gradients->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(adam->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(radianceIndex->getUAV().get(), uint4(0xffffffffu));
        pRenderContext->clearUAV(cells->getUAV().get(), uint4(0));
        resources.voxelPages.push_back(std::move(voxels));
        resources.gradPages.push_back(std::move(gradients));
        resources.adamPages.push_back(std::move(adam));
        resources.radianceAdamIndexPages.push_back(std::move(radianceIndex));
        resources.cellIndexPages.push_back(std::move(cells));
    }
    // Start with state for one eighth of the occupied cells. More pages are
    // added after an iteration if the counter shows they are needed.
    const uint32_t radiancePages = sparsePageCount(std::max(1u, capacity / 8u));
    for (uint32_t page = 0; page < radiancePages; ++page)
    {
        auto adam = mpDevice->createStructuredBuffer(sizeof(RadianceAdamState), SPARSE_POOL_PAGE_SIZE, flags);
        pRenderContext->clearUAV(adam->getUAV().get(), uint4(0));
        resources.radianceAdamPages.push_back(std::move(adam));
    }
    // [0] is the next free slot; [1] records whether any voxel ran out.
    resources.radianceAdamCounter = mpDevice->createStructuredBuffer(sizeof(uint32_t), 2u, flags);
    pRenderContext->clearUAV(resources.radianceAdamCounter->getUAV().get(), uint4(0));
    return resources;
}

ref<ParameterBlock> VoxelReconstructionNoLightTransport::createSparseGridBlock(const GridResources& resources)
{
    auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock");
    auto block = ParameterBlock::create(mpDevice, reflector);
    auto var = block->getRootVar();
    var["voxelCount"] = resources.gridData.voxelCount;
    var["voxelSize"] = resources.gridData.voxelSize;
    var["gridMin"] = resources.gridData.gridMin;
    var["solidVoxelCount"] = resources.gridData.solidVoxelCount;
    var["activeVoxelCount"] = resources.gridData.activeVoxelCount;
    var["voxelCapacity"] = resources.gridData.voxelCapacity;
    var["indexPageCount"] = resources.gridData.indexPageCount;
    for (uint32_t i = 0; i < resources.indexPages.size(); ++i) var["indexPages"][i] = resources.indexPages[i];
    for (uint32_t i = 0; i < resources.voxelPages.size(); ++i)
    {
        var["voxelPages"][i] = resources.voxelPages[i];
        var["gradPages"][i] = resources.gradPages[i];
        var["cellIndexPages"][i] = resources.cellIndexPages[i];
    }
    return block;
}

void VoxelReconstructionNoLightTransport::commitSparseGrid(
    GridResources&& resources, const ref<ParameterBlock>& block, uint32_t resolution)
{
    const auto bind = [&](const auto& pass)
    {
        if (pass && pass->getVars()) pass->getRootVar()["gGridDataParamBlock"].setParameterBlock(block);
    };
    bind(mpInitializeDataPass);
    bind(mpInitializePointCloudPass);
    bind(mpBuildSparseIndexPass);
    bind(mRayMarchingPass.mpFullScreenPass);
    bind(mGradientPass.mpComputePass);
    bind(mUpdatePass.mpComputePass);
    if (mUpdatePass.mpComputePass && mUpdatePass.mpComputePass->getVars())
    {
        auto var = mUpdatePass.mpComputePass->getRootVar()["gGeometryAdamPages"];
        const size_t pageCount = std::max(mGridResources.adamPages.size(), resources.adamPages.size());
        for (size_t page = 0; page < pageCount; ++page)
        {
            if (page < resources.adamPages.size()) var[uint32_t(page)] = resources.adamPages[page];
            else var[uint32_t(page)].setBuffer(nullptr);
        }
        auto radianceMap = mUpdatePass.mpComputePass->getRootVar()["gRadianceAdamIndexPages"];
        const size_t mapCount = std::max(mGridResources.radianceAdamIndexPages.size(), resources.radianceAdamIndexPages.size());
        for (size_t page = 0; page < mapCount; ++page)
        {
            if (page < resources.radianceAdamIndexPages.size()) radianceMap[uint32_t(page)] = resources.radianceAdamIndexPages[page];
            else radianceMap[uint32_t(page)].setBuffer(nullptr);
        }
        auto radianceState = mUpdatePass.mpComputePass->getRootVar()["gRadianceAdamPages"];
        const size_t stateCount = std::max(mGridResources.radianceAdamPages.size(), resources.radianceAdamPages.size());
        for (size_t page = 0; page < stateCount; ++page)
        {
            if (page < resources.radianceAdamPages.size()) radianceState[uint32_t(page)] = resources.radianceAdamPages[page];
            else radianceState[uint32_t(page)].setBuffer(nullptr);
        }
        mUpdatePass.mpComputePass->getRootVar()["gRadianceAdamCounter"] = resources.radianceAdamCounter;
    }
    mGridResources = std::move(resources);
    mpGridBlock = block;
    mVoxelResolution = resolution;
}

void VoxelReconstructionNoLightTransport::reserveSparseVoxelCapacity(
    RenderContext* pRenderContext, uint32_t minimumCapacity)
{
    if (minimumCapacity <= mGridResources.gridData.voxelCapacity) return;
    const uint32_t requiredPages = sparsePageCount(minimumCapacity);
    if (requiredPages > SPARSE_POOL_MAX_PAGES) throw RuntimeError("Mode 1 sparse pool capacity exceeded.");
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    GridResources next = mGridResources;
    while (next.voxelPages.size() < requiredPages)
    {
        auto voxels = mpDevice->createStructuredBuffer(sizeof(VoxelData), SPARSE_POOL_PAGE_SIZE, flags);
        auto gradients = mpDevice->createStructuredBuffer(sizeof(GradRecord), SPARSE_POOL_PAGE_SIZE, flags);
        auto adam = mpDevice->createStructuredBuffer(sizeof(GeometryAdamState), SPARSE_POOL_PAGE_SIZE, flags);
        auto radianceIndex = mpDevice->createStructuredBuffer(sizeof(uint32_t), SPARSE_POOL_PAGE_SIZE, flags);
        auto cells = mpDevice->createStructuredBuffer(sizeof(uint32_t), SPARSE_POOL_PAGE_SIZE, flags);
        pRenderContext->clearUAV(voxels->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(gradients->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(adam->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(radianceIndex->getUAV().get(), uint4(0xffffffffu));
        pRenderContext->clearUAV(cells->getUAV().get(), uint4(0));
        next.voxelPages.push_back(std::move(voxels));
        next.gradPages.push_back(std::move(gradients));
        next.adamPages.push_back(std::move(adam));
        next.radianceAdamIndexPages.push_back(std::move(radianceIndex));
        next.cellIndexPages.push_back(std::move(cells));
    }
    next.gridData.voxelCapacity = requiredPages * SPARSE_POOL_PAGE_SIZE;
    auto block = createSparseGridBlock(next);
    commitSparseGrid(std::move(next), block, mVoxelResolution);
}

void VoxelReconstructionNoLightTransport::reserveRadianceAdamCapacity(
    RenderContext* pRenderContext, uint32_t minimumCapacity)
{
    const uint32_t requiredPages = sparsePageCount(minimumCapacity);
    if (requiredPages <= mGridResources.radianceAdamPages.size()) return;
    if (requiredPages > SPARSE_POOL_MAX_PAGES)
        throw RuntimeError("Mode 1 radiance Adam pool capacity exceeded.");
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto var = mUpdatePass.mpComputePass->getRootVar()["gRadianceAdamPages"];
    while (mGridResources.radianceAdamPages.size() < requiredPages)
    {
        auto page = mpDevice->createStructuredBuffer(sizeof(RadianceAdamState), SPARSE_POOL_PAGE_SIZE, flags);
        pRenderContext->clearUAV(page->getUAV().get(), uint4(0));
        const uint32_t index = uint32_t(mGridResources.radianceAdamPages.size());
        var[index] = page;
        mGridResources.radianceAdamPages.push_back(std::move(page));
    }
}

void VoxelReconstructionNoLightTransport::clearSparseGradients(RenderContext* pRenderContext)
{
    for (const auto& page : mGridResources.gradPages) pRenderContext->clearUAV(page->getUAV().get(), uint4(0));
}

void VoxelReconstructionNoLightTransport::barrierSparseVoxels(RenderContext* pRenderContext)
{
    for (const auto& page : mGridResources.voxelPages) pRenderContext->uavBarrier(page.get());
}

void VoxelReconstructionNoLightTransport::barrierSparseGradients(RenderContext* pRenderContext)
{
    for (const auto& page : mGridResources.gradPages) pRenderContext->uavBarrier(page.get());
}

void VoxelReconstructionNoLightTransport::uploadSparseBatch(
    RenderContext* pRenderContext, const ref<ParameterBlock>& block, uint32_t offset,
    const uint32_t* cells, uint32_t count, const VoxelData* data)
{
    if (count == 0) return;
    if (!mpBuildSparseIndexPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/VoxelReconstructionNoLightTransport/Shader/BuildSparseIndex.cs.slang").csEntry("main");
        mpBuildSparseIndexPass = ComputePass::create(mpDevice, desc, getReconstructionDefines());
    }

    uint32_t copied = 0;
    while (copied < count)
    {
        const uint32_t id = offset + copied;
        const uint32_t page = id / SPARSE_POOL_PAGE_SIZE;
        const uint32_t inPage = id % SPARSE_POOL_PAGE_SIZE;
        const uint32_t chunk = std::min(count - copied, SPARSE_POOL_PAGE_SIZE - inPage);
        auto root = block->getRootVar();
        auto cellBuffer = root["cellIndexPages"][page].getBuffer();
        auto voxelBuffer = root["voxelPages"][page].getBuffer();
        pRenderContext->updateBuffer(cellBuffer.get(), cells + copied, size_t(inPage) * sizeof(uint32_t), size_t(chunk) * sizeof(uint32_t));
        if (data)
            pRenderContext->updateBuffer(voxelBuffer.get(), data + copied, size_t(inPage) * sizeof(VoxelData), size_t(chunk) * sizeof(VoxelData));
        copied += chunk;
    }

    auto var = mpBuildSparseIndexPass->getRootVar();
    var["gGridDataParamBlock"] = block;
    constexpr uint32_t dispatchLimit = 65535u * 256u;
    for (uint32_t dispatched = 0; dispatched < count; )
    {
        const uint32_t chunk = std::min(dispatchLimit, count - dispatched);
        var["CB"]["gVoxelOffset"] = offset + dispatched;
        var["CB"]["gVoxelCount"] = offset + count;
        mpBuildSparseIndexPass->execute(pRenderContext, uint3(chunk, 1, 1));
        dispatched += chunk;
    }
}

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
    clearSparseGradients(pRenderContext);
}

bool VoxelReconstructionNoLightTransport::initializePointCloudVoxelData(RenderContext* pRenderContext)
{
    const auto path = resolveReconstructionPath(ReferenceImageDir) / "point_cloud.ply";
    try
    {
        // Read and validate before touching the active reconstruction.
        auto grid = mGridResources.gridData;
        auto points = PointCloudInitialization::load(path,
            {grid.voxelCount.x, grid.voxelCount.y, grid.voxelCount.z},
            {grid.gridMin.x, grid.gridMin.y, grid.gridMin.z},
            {grid.voxelSize.x, grid.voxelSize.y, grid.voxelSize.z},
            double(mPointCloud.opacityThreshold));
        if (points.seeds.empty())
            throw RuntimeError("Point-cloud initialization produced no occupied voxels for the current grid.");
        const uint64_t occupiedCount = points.seeds.size();
        const uint64_t maximumCapacity = uint64_t(SPARSE_POOL_PAGE_SIZE) * SPARSE_POOL_MAX_PAGES;
        if (occupiedCount > maximumCapacity)
        {
            const double poolGiB = double(occupiedCount) *
                double(sizeof(VoxelData) + sizeof(GradRecord) + sizeof(GeometryAdamState) + 2u * sizeof(uint32_t)) /
                double(1ull << 30);
            throw RuntimeError(fmt::format(
                "Point cloud occupies {} voxels (pool limit {}). The voxel, gradient, geometry/opacity Adam, and index pools alone "
                "would need at least {:.1f} GiB. Increase Gaussian Opacity Threshold to reduce coverage, "
                "or use a lower grid resolution; raising the page limit alone may exhaust GPU memory.",
                occupiedCount, maximumCapacity, poolGiB));
        }
        grid.solidVoxelCount = static_cast<uint32_t>(occupiedCount);
        grid.activeVoxelCount = grid.solidVoxelCount;
        static_assert(sizeof(PointCloudInitialization::Seed) == sizeof(uint32_t));
        if (!mpInitializePointCloudPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary("RenderPasses/VoxelReconstructionNoLightTransport/Shader/InitializePointCloud.cs.slang").csEntry("main");
            mpInitializePointCloudPass = ComputePass::create(mpDevice, desc, getReconstructionDefines());
        }

        auto seeds = mpDevice->createStructuredBuffer(sizeof(PointCloudInitialization::Seed), grid.solidVoxelCount,
            ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, points.seeds.data());
        // Allocate only the pages needed by the seeds. Extra pages for future
        // growth can be added with reserveSparseVoxelCapacity() when needed.
        auto resources = allocateSparseGrid(pRenderContext, grid, grid.solidVoxelCount);
        auto block = createSparseGridBlock(resources);
        auto var = mpInitializePointCloudPass->getRootVar();
        var["gGridDataParamBlock"] = block;
        var["gSeeds"] = seeds;
        var["CB"]["gSeedCount"] = grid.solidVoxelCount;
        // Limit each dispatch to D3D12's 65535 thread groups in X.
        constexpr uint32_t batchSize = 65535u * 256u;
        for (uint32_t offset = 0; offset < grid.solidVoxelCount; )
        {
            const uint32_t batchCount = std::min(batchSize, grid.solidVoxelCount - offset);
            var["CB"]["gSeedOffset"] = offset;
            mpInitializePointCloudPass->execute(pRenderContext, uint3(batchCount, 1, 1));
            offset += batchCount;
        }
        for (const auto& page : resources.voxelPages) pRenderContext->uavBarrier(page.get());
        for (const auto& page : resources.indexPages) pRenderContext->uavBarrier(page.get());
        pRenderContext->submit(true);
        var["gSeeds"].setBuffer(nullptr);

        // Commit only after successful initialization, then release references to the old grid.
        commitSparseGrid(std::move(resources), block, mVoxelResolution);
        resetPointCloudOptimization(pRenderContext);
        mPointCloud.initialized = true;
        mLoadedReconstructionForViewing = false;
        const auto& stats = points.statistics;
        mPointCloud.status = fmt::format(
            "PLY: {} points, {} occupied voxels; {} outside, {} invalid, {} below alpha; "
            "{} blocks tested, {} full blocks skipped, {} boundary cells tested; {} cells added (max {})",
            stats.inputPoints, grid.solidVoxelCount, stats.outsidePoints, stats.invalidPoints,
            stats.droppedByOpacity, stats.testedBlocks, stats.skippedFullBlocks, stats.testedCells,
            stats.coveredCells, stats.maxCoveredCells);
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
