#include "VoxelReconstructionNoLightTransport.h"
#include <cmath>

namespace
{
bool isPowerOfTwo(uint32_t value)
{
    return value != 0u && (value & (value - 1u)) == 0u;
}
} // namespace

void VoxelReconstructionNoLightTransport::validateCoarseToFineSettings() const
{
    if (!isPowerOfTwo(mCoarseToFine.startResolution) || !isPowerOfTwo(mCoarseToFine.targetResolution) ||
        mCoarseToFine.startResolution > mCoarseToFine.targetResolution || mCoarseToFine.targetResolution > 1024u ||
        mCoarseToFine.iterationsPerLevel == 0u || mCoarseToFine.iterationsPerLevel > 100000u)
        throw RuntimeError("Coarse-to-fine requires power-of-two resolutions, start <= target <= 1024, and 1..100000 rounds per level.");
    if (!std::isfinite(mCoarseToFine.parentOpacityThreshold) || mCoarseToFine.parentOpacityThreshold < 0.0f ||
        mCoarseToFine.parentOpacityThreshold > 0.99f)
        throw RuntimeError("Refinement parent opacity threshold must be in [0, 0.99].");
}

uint32_t VoxelReconstructionNoLightTransport::coarseToFineRemainingLevels() const
{
    validateCoarseToFineSettings();
    uint32_t resolution =
        mPointCloud.initialized && !mCoarseToFine.initializationPending ? mVoxelResolution : mCoarseToFine.startResolution;
    if (!isPowerOfTwo(resolution) || resolution > mCoarseToFine.targetResolution)
        throw RuntimeError("Current grid is incompatible with the coarse-to-fine target. Use Init / Reset from PLY.");
    uint32_t levels = 1u;
    for (; resolution < mCoarseToFine.targetResolution; resolution *= 2u)
        ++levels;
    return levels;
}

bool VoxelReconstructionNoLightTransport::prepareReconstruction(RenderContext* pRenderContext)
{
    try
    {
        if (isCoarseToFine())
            validateCoarseToFineSettings();
        if ((!mPointCloud.initialized || mCoarseToFine.initializationPending) && !initializePointCloudVoxelData(pRenderContext))
            return false;
        const uint32_t levels = isCoarseToFine() ? coarseToFineRemainingLevels() : 0u;
        resetPointCloudOptimization(pRenderContext);
        mCoarseToFine.levelStartIteration = 0u;
        if (isCoarseToFine())
        {
            mOptimizerParams.maxIteration = levels * mCoarseToFine.iterationsPerLevel;
            mCoarseToFine.status = fmt::format(
                "Resolution {} -> {}; {} rounds per level, {} total remaining rounds",
                mVoxelResolution,
                mCoarseToFine.targetResolution,
                mCoarseToFine.iterationsPerLevel,
                mOptimizerParams.maxIteration
            );
            mTopologySettings.growthStatus = "Disabled in mode1";
            mTopologySettings.deletionStatus = "Disabled in mode1";
        }
        return true;
    }
    catch (const std::exception& error)
    {
        stopReconstruction();
        mCoarseToFine.status = std::string("Cannot start: ") + error.what();
        logError("{}", mCoarseToFine.status);
        return false;
    }
}

void VoxelReconstructionNoLightTransport::advanceCoarseToFine(RenderContext* pRenderContext)
{
    if (!isCoarseToFine() || !mOptimizerParams.isRunning ||
        mOptimizerParams.currentIteration - mCoarseToFine.levelStartIteration < mCoarseToFine.iterationsPerLevel)
        return;
    try
    {
        if (mVoxelResolution < mCoarseToFine.targetResolution)
        {
            if (mCoarseToFine.saveEachLevel)
            {
                const auto checkpoint = getDefaultReconstructionSavePath();
                try
                {
                    saveSparseReconstruction(pRenderContext, checkpoint);
                    mReconstructionIOStatus = "Saved level checkpoint: " + checkpoint.filename().string();
                    mReconstructionFileListDirty = true;
                }
                catch (const std::exception& error)
                {
                    mReconstructionIOStatus = std::string("Level checkpoint failed; refinement continues: ") + error.what();
                    logError("{}", mReconstructionIOStatus);
                }
            }
            refineCoarseToFineGrid(pRenderContext);
            mCoarseToFine.levelStartIteration = mOptimizerParams.currentIteration;
        }
        else
        {
            mCoarseToFine.status =
                fmt::format("Complete: resolution {}, {} total rounds", mVoxelResolution, mOptimizerParams.currentIteration);
            stopReconstruction();
            mSaveReconstructionRequested = true;
        }
    }
    catch (const std::exception& error)
    {
        stopReconstruction();
        mCoarseToFine.status = std::string("Refinement stopped; previous grid retained: ") + error.what();
        logError("{}", mCoarseToFine.status);
    }
}

void VoxelReconstructionNoLightTransport::refineCoarseToFineGrid(RenderContext* pRenderContext)
{
    validateCoarseToFineSettings();
    const uint32_t oldResolution = mVoxelResolution;
    const uint32_t nextResolution = oldResolution * 2u;
    if (!isCoarseToFine() || oldResolution == 0u || nextResolution > mCoarseToFine.targetResolution)
        throw RuntimeError("Invalid coarse-to-fine refinement level.");
    const uint32_t parents = mGridResources.gridData.activeVoxelCount;
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto ranges = mpDevice->createStructuredBuffer(sizeof(uint2), std::max(parents, 1u), flags);
    auto counter = mpDevice->createStructuredBuffer(sizeof(uint32_t), 1u, flags);
    const auto createPass = [&](const char* entry)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/VoxelReconstructionNoLightTransport/Shader/RefineGrid.cs.slang").csEntry(entry);
        return ComputePass::create(mpDevice, desc, getReconstructionDefines(), true);
    };
    auto countPass = createPass("countChildren");
    auto pass = createPass("main");
    const auto dispatch = [&](const ref<ComputePass>& stage)
    {
        auto var = stage->getRootVar();
        var["gCoarseGrid"] = mpGridBlock;
        var["gChildRanges"] = ranges;
        var["gChildCounter"] = counter;
        var["CB"]["gParentCount"] = parents;
        var["CB"]["gParentOpacityThreshold"] = mCoarseToFine.parentOpacityThreshold;
        constexpr uint32_t batchSize = 65535u * 256u;
        for (uint32_t offset = 0; offset < parents;)
        {
            const uint32_t count = std::min(batchSize, parents - offset);
            var["CB"]["gParentOffset"] = offset;
            stage->execute(pRenderContext, uint3(count, 1, 1));
            offset += count;
        }
    };
    pRenderContext->clearUAV(counter->getUAV().get(), uint4(0));
    barrierSparseVoxels(pRenderContext);
    dispatch(countPass);
    pRenderContext->uavBarrier(counter.get());
    pRenderContext->uavBarrier(ranges.get());
    pRenderContext->submit(true);
    uint32_t children = 0u;
    counter->getBlob(&children, 0, sizeof(children));
    // Check and allocate the actual intersecting population, not parents * 8.
    if (children == 0u)
        throw RuntimeError("No eligible refinement children. Check parent geometry or lower Refinement Parent Opacity Threshold.");
    if (children > uint64_t(SPARSE_POOL_PAGE_SIZE) * SPARSE_POOL_MAX_PAGES)
        throw RuntimeError(
            fmt::format("Refinement needs {} cells; pool limit is {}", children, uint64_t(SPARSE_POOL_PAGE_SIZE) * SPARSE_POOL_MAX_PAGES)
        );

    GridData grid = mGridResources.gridData;
    grid.voxelCount *= 2u;
    grid.voxelSize *= 0.5f;
    grid.activeVoxelCount = grid.solidVoxelCount = uint32_t(children);
    // Build the entire next level separately. Allocation/shader failures leave
    // the live grid, parameters, and optimizer state untouched.
    auto resources = allocateSparseGrid(pRenderContext, grid, uint32_t(children));
    auto block = createSparseGridBlock(resources);
    auto var = pass->getRootVar();
    var["gGridDataParamBlock"] = block;
    pRenderContext->clearUAV(counter->getUAV().get(), uint4(0));
    dispatch(pass);
    for (const auto& page : resources.voxelPages)
        pRenderContext->uavBarrier(page.get());
    for (const auto& page : resources.indexPages)
        pRenderContext->uavBarrier(page.get());
    for (const auto& page : resources.cellIndexPages)
        pRenderContext->uavBarrier(page.get());
    pRenderContext->uavBarrier(counter.get());
    pRenderContext->submit(true);
    uint32_t initialized = 0u;
    counter->getBlob(&initialized, 0, sizeof(initialized));
    if (initialized != children)
        throw RuntimeError("Refinement child count/initialization mismatch.");
    commitSparseGrid(std::move(resources), block, nextResolution);
    mGrowthCooldownPresent = false;
    mPointCloud.clearAccumulation = true;
    mRayMarchingPass.mSampleIndex = 0u;
    mRayMarchingPass.mOptionsChanged = true;
    if (mpPathRecordBuffer)
        pRenderContext->clearUAV(mpPathRecordBuffer->getUAV().get(), uint4(0));
    mTopologySettings.candidateCount = 0u;
    mCoarseToFine.status = fmt::format(
        "Iteration {}: refined {} -> {}, {} -> {} occupied cells",
        mOptimizerParams.currentIteration,
        oldResolution,
        nextResolution,
        parents,
        children
    );
    logInfo("{}", mCoarseToFine.status);
}

void VoxelReconstructionNoLightTransport::renderUICoarseToFine(Gui::Widgets& widget)
{
    if (!mEnableReconstruction && !mOptimizerParams.isRunning && !mPointCloud.startRequested)
    {
        static const Gui::DropdownList modes = {{0u, "mode0: fixed resolution"}, {1u, "mode1: coarse to fine"}};
        if (widget.dropdown("Reconstruction Mode", modes, mReconstructionMode))
        {
            mCoarseToFine.initializationPending = true;
            mReconstructionFileListDirty = true;
        }
        if (isCoarseToFine())
        {
            static const Gui::DropdownList resolutions = {
                {16u, "16"}, {32u, "32"}, {64u, "64"}, {128u, "128"}, {256u, "256"}, {512u, "512"}, {1024u, "1024"}};
            if (widget.dropdown("Coarse Start Resolution", resolutions, mCoarseToFine.startResolution))
            {
                mCoarseToFine.targetResolution = std::max(mCoarseToFine.targetResolution, mCoarseToFine.startResolution);
                mCoarseToFine.initializationPending = true;
            }
            if (widget.dropdown("Target Resolution", resolutions, mCoarseToFine.targetResolution))
            {
                mCoarseToFine.startResolution = std::min(mCoarseToFine.startResolution, mCoarseToFine.targetResolution);
                mCoarseToFine.initializationPending = true;
            }
            widget.var("Iterations Per Level", mCoarseToFine.iterationsPerLevel, 1u, 100000u, 1u);
            widget.var("Refinement Parent Opacity Threshold", mCoarseToFine.parentOpacityThreshold, 0.0f, 0.99f, 0.001f);
            widget.tooltip(
                "At refinement only: discard parents whose opacity SH upper bound is below this threshold. Zero disables filtering."
            );
            widget.checkbox("Save Each Completed Level", mCoarseToFine.saveEachLevel);
            widget.tooltip("Save the current mode1 grid before every refinement. The final target level is saved at completion as usual.");
        }
        if (mCoarseToFine.initializationPending)
            widget.text("Next Enable or Init / Reset reloads the PLY with the selected mode/resolution.");
    }
    if (isCoarseToFine())
    {
        widget.text("mode1: no regular deletion, pruning, or neighbor growth; low-opacity filtering only at refinement.");
        widget.text(fmt::format(
            "Current resolution: {}; target: {}; level rounds: {} / {}",
            mVoxelResolution,
            mCoarseToFine.targetResolution,
            mOptimizerParams.currentIteration - mCoarseToFine.levelStartIteration,
            mCoarseToFine.iterationsPerLevel
        ));
        widget.text(mCoarseToFine.status);
    }
}
