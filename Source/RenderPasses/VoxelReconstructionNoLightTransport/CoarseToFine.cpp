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
        mCoarseToFine.startResolution > mCoarseToFine.targetResolution || mCoarseToFine.targetResolution > 1024u)
        throw RuntimeError("Coarse-to-fine requires power-of-two resolutions with start <= target <= 1024.");
    uint32_t configuredLevels = 1u;
    for (uint32_t resolution = mCoarseToFine.startResolution; resolution < mCoarseToFine.targetResolution; resolution *= 2u)
        ++configuredLevels;
    if (mCoarseToFine.totalIterations < configuredLevels || mCoarseToFine.totalIterations > 1000000u)
        throw RuntimeError("Coarse-to-fine total iterations must be at least the configured level count and at most 1000000.");
    if (!std::isfinite(mCoarseToFine.parentOpacityThreshold) || mCoarseToFine.parentOpacityThreshold < 0.0f ||
        mCoarseToFine.parentOpacityThreshold > 0.99f)
        throw RuntimeError("Refinement parent opacity threshold must be in [0, 0.99].");
    if (!std::isfinite(mCoarseToFine.opacityOpticalDepthScale) || mCoarseToFine.opacityOpticalDepthScale <= 0.0f ||
        mCoarseToFine.opacityOpticalDepthScale > 1.0f)
        throw RuntimeError("Refinement opacity optical-depth scale must be in (0, 1].");
    if (!std::isfinite(mCoarseToFine.childFaceOverlap) || mCoarseToFine.childFaceOverlap < 0.0f ||
        mCoarseToFine.childFaceOverlap > 0.2f)
        throw RuntimeError("Refinement child face overlap must be in [0, 0.2] fine-voxel widths.");
    if (!std::isfinite(mCoarseToFine.coarseGrowthFacePenetration) || mCoarseToFine.coarseGrowthFacePenetration < 0.0f ||
        mCoarseToFine.coarseGrowthFacePenetration > float(GROWTH_MAX_FACE_PENETRATION_VOXELS) ||
        mCoarseToFine.coarseGrowthInterval == 0u)
        throw RuntimeError("Coarse growth requires a valid face penetration and a nonzero interval.");
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

uint32_t VoxelReconstructionNoLightTransport::coarseToFineLevelBudget(uint32_t resolution) const
{
    const uint32_t start = mCoarseToFine.scheduleStartResolution;
    if (!isPowerOfTwo(start) || !isPowerOfTwo(resolution) || resolution < start || resolution > mCoarseToFine.targetResolution)
        throw RuntimeError("Current resolution is outside the active coarse-to-fine schedule.");
    uint32_t levels = 1u;
    for (uint32_t value = start; value < mCoarseToFine.targetResolution; value *= 2u)
        ++levels;
    uint32_t index = 0u;
    for (uint32_t value = start; value < resolution; value *= 2u)
        ++index;
    if (mCoarseToFine.totalIterations < levels)
        throw RuntimeError("The iteration total is smaller than the active level count.");
    if (levels == 1u) return mCoarseToFine.totalIterations;
    constexpr uint64_t weights[4] = COARSE_TO_FINE_LEVEL_WEIGHTS;
    static_assert(weights[0] > 0u && weights[1] > 0u && weights[2] > 0u && weights[3] > 0u,
        "Coarse-to-fine level weights must be positive.");
    // Integer interpolation keeps exact totals without floating-point rounding.
    // Its common denominator (levels - 1) cancels when allocating rounds.
    uint64_t weightSum = 0u, previousWeight = 0u, cumulativeWeight = 0u;
    for (uint32_t level = 0u; level < levels; ++level)
    {
        const uint32_t position = 3u * level;
        const uint32_t segment = position / (levels - 1u);
        const uint32_t fraction = position % (levels - 1u);
        const uint64_t weight = weights[segment] * (levels - 1u - fraction) +
            weights[std::min(segment + 1u, 3u)] * fraction;
        weightSum += weight;
        if (level < index) previousWeight += weight;
        if (level <= index) cumulativeWeight += weight;
    }
    const uint64_t remainder = mCoarseToFine.totalIterations - levels;
    return 1u + uint32_t(remainder * cumulativeWeight / weightSum - remainder * previousWeight / weightSum);
}

bool VoxelReconstructionNoLightTransport::prepareReconstruction(RenderContext* pRenderContext)
{
    try
    {
        if (isCoarseToFine())
            validateCoarseToFineSettings();
        if ((!mPointCloud.initialized || mCoarseToFine.initializationPending) && !initializePointCloudVoxelData(pRenderContext))
            return false;
        if (isCoarseToFine())
            coarseToFineRemainingLevels();
        resetPointCloudOptimization(pRenderContext);
        mCoarseToFine.levelStartIteration = 0u;
        if (isCoarseToFine())
        {
            mCoarseToFine.scheduleStartResolution = mVoxelResolution;
            mCoarseToFine.levelIterationBudget = coarseToFineLevelBudget(mVoxelResolution);
            mOptimizerParams.maxIteration = mCoarseToFine.totalIterations;
            mCoarseToFine.status = fmt::format(
                "Resolution {} -> {}; current level {} rounds, {} total scheduled rounds",
                mVoxelResolution,
                mCoarseToFine.targetResolution,
                mCoarseToFine.levelIterationBudget,
                mCoarseToFine.totalIterations
            );
            mTopologySettings.growthStatus = mVoxelResolution == mCoarseToFine.targetResolution
                                                 ? "Disabled at the mode1 target level"
                                                 : "Coarse growth waiting for its scheduled boundary";
            mTopologySettings.deletionStatus = mVoxelResolution == mCoarseToFine.targetResolution ? "Enabled at the mode1 target level"
                                                                                                  : "Waiting for the mode1 target level";
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
    if (!isCoarseToFine() || !mOptimizerParams.isRunning)
        return;
    try
    {
        const uint32_t levelIteration = mOptimizerParams.currentIteration - mCoarseToFine.levelStartIteration;
        if (mVoxelResolution == mCoarseToFine.targetResolution)
        {
            evaluateDeletionEvidence(pRenderContext);
            const bool reachedLevelBudget = levelIteration >= mCoarseToFine.levelIterationBudget;
            const uint32_t evidenceInterval = std::max(1u, mTopologySettings.evidenceInterval);
            const uint32_t deletionInterval = std::max(1u, mTopologySettings.deletionInterval);
            const uint32_t firstDeletionIteration = getDeletionEvidenceStartIteration() + 2u * evidenceInterval;
            const bool periodicDeletionBoundary = mOptimizerParams.currentIteration >= firstDeletionIteration &&
                                                  (mOptimizerParams.currentIteration - firstDeletionIteration) % deletionInterval == 0u;
            const bool hadCandidatesAtBoundary = mTopologySettings.candidateCount > 0u;
            if (hadCandidatesAtBoundary && (periodicDeletionBoundary || reachedLevelBudget))
            {
                try
                {
                    deleteAndCompactCandidates(pRenderContext);
                }
                catch (const std::exception& error)
                {
                    mTopologySettings.deletionStatus = std::string("Automatic compaction failed: ") + error.what();
                    logError("{}", mTopologySettings.deletionStatus);
                }
            }
            if (reachedLevelBudget)
            {
                mCoarseToFine.status =
                    fmt::format("Complete: resolution {}, {} total rounds", mVoxelResolution, mOptimizerParams.currentIteration);
                stopReconstruction();
                if (!hadCandidatesAtBoundary)
                    mTopologySettings.deletionStatus = "Target-level training complete; no additional confirmed candidates";
                mSaveReconstructionRequested = true;
            }
            return;
        }
        if (levelIteration < mCoarseToFine.levelIterationBudget)
        {
            growNeighborVoxels(pRenderContext);
            return;
        }
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
            mCoarseToFine.levelIterationBudget = coarseToFineLevelBudget(mVoxelResolution);
            mTopologySettings.lastGrowthCount = 0u;
            mTopologySettings.lastGrowthPages = 0u;
            if (mVoxelResolution == mCoarseToFine.targetResolution)
            {
                mTopologySettings.growthStatus = "Disabled at the mode1 target level";
                mTopologySettings.deletionStatus = "Collecting deletion evidence at the mode1 target level";
            }
            else
            {
                mTopologySettings.growthStatus = "Coarse growth waiting for its scheduled boundary";
            }
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
        var["CB"]["gOpacityOpticalDepthScale"] = mCoarseToFine.opacityOpticalDepthScale;
        var["CB"]["gChildFaceOverlap"] = mCoarseToFine.childFaceOverlap;
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
            widget.var("Total Iterations", mCoarseToFine.totalIterations, 1u, 1000000u, 1u);
            widget.var("Refinement Parent Opacity Threshold", mCoarseToFine.parentOpacityThreshold, 0.0f, 0.99f, 0.001f);
            widget.tooltip(
                "At refinement only: discard parents whose opacity SH upper bound is below this threshold. Zero disables filtering."
            );
            widget.var("Refinement Opacity Optical-Depth Scale", mCoarseToFine.opacityOpticalDepthScale, 0.01f, 1.0f, 0.01f);
            widget.tooltip(
                "Attenuate child opacity at every refinement in optical-depth space. 1 keeps parent opacity; 0.65 reduces stacking."
            );
            widget.var("Refinement Child Face Overlap (voxels)", mCoarseToFine.childFaceOverlap, 0.0f, 0.2f, 0.01f);
            widget.tooltip("Maximum extent beyond each fine-cell face. Parent ellipsoid containment is retained. Zero disables overlap.");
            widget.checkbox("Enable Coarse Growth", mCoarseToFine.enableCoarseGrowth);
            widget.var(
                "Coarse Growth Face Penetration",
                mCoarseToFine.coarseGrowthFacePenetration,
                0.0f,
                float(GROWTH_MAX_FACE_PENETRATION_VOXELS),
                0.05f
            );
            widget.var("Coarse Growth Interval", mCoarseToFine.coarseGrowthInterval, 1u, 100u, 1u);
            widget.checkbox("Save Each Completed Level", mCoarseToFine.saveEachLevel);
            widget.tooltip("Save the current mode1 grid before every refinement. The final target level is saved at completion as usual.");
        }
        if (mCoarseToFine.initializationPending)
            widget.text("Next Enable or Init / Reset reloads the PLY with the selected mode/resolution.");
    }
    if (isCoarseToFine())
    {
        widget.text("mode1: coarse levels may grow; target-level deletion is enabled; pruning remains disabled.");
        widget.text(fmt::format(
            "Current resolution: {}; target: {}; level rounds: {} / {}; total: {} / {}",
            mVoxelResolution,
            mCoarseToFine.targetResolution,
            mOptimizerParams.currentIteration - mCoarseToFine.levelStartIteration,
            mCoarseToFine.levelIterationBudget,
            mOptimizerParams.currentIteration,
            mCoarseToFine.totalIterations
        ));
        widget.text(mCoarseToFine.status);
    }
}
