#include "VoxelReconstructionNoLightTransport.h"

namespace
{
constexpr uint32_t kDispatchItems = 65535u * 256u;
// GaussianEllipsoid (40 bytes), parent address, face, and two camera counts.
constexpr uint32_t kCandidateStride = 56u;
}

void VoxelReconstructionNoLightTransport::resetGrowthEvidence()
{
    mpGrowthCandidates = nullptr;
    mGrowthCandidateCount = mGrowthEvidenceViews = 0u;
    mGrowthEvidenceBoundary = mGrowthEvidenceResolution = 0u;
    mGrowthEvidenceSeenViews.clear();
    for (const auto& pass : {mpBuildGrowthCandidatesPass, mpEvaluateGrowthCandidatesPass,
         mpProposeEvidenceGrowthPass, mpInitializeEvidenceGrowthPass, mpClearEvidenceGrowthClaimsPass})
        if (pass && pass->getVars()) pass->getRootVar()["gGrowthCandidates"].setBuffer(nullptr);
}

void VoxelReconstructionNoLightTransport::beginGrowthEvidence(RenderContext* ctx, uint32_t boundary)
{
    resetGrowthEvidence();
    if (!mpBuildGrowthCandidatesPass) createGrowthPassResources();
    const uint32_t parents = mGridResources.gridData.activeVoxelCount;
    if (!parents) return;
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto counter = mpDevice->createStructuredBuffer(sizeof(uint32_t), 1u, flags);
    auto var = mpBuildGrowthCandidatesPass->getRootVar();
    var["gGridDataParamBlock"] = mpGridBlock;
    var["gGrowthCounter"] = counter;
    auto cb = var["CB"];
    cb["gParentCount"] = parents;
    cb["gCurrentIteration"] = boundary;
    cb["gGrowthWaitIterations"] = mTopologySettings.growthWaitIterations;
    cb["gFacePenetration"] = 0.0f;
    cb["gShrink"] = std::clamp(mTopologySettings.growthShrink, 0.01f, 0.99f);
    cb["gContactOffset"] = std::clamp(mTopologySettings.growthContactOffset, 0.01f, 0.49f);
    const auto dispatch = [&]()
    {
        ctx->clearUAV(counter->getUAV().get(), uint4(0));
        for (uint32_t offset = 0u; offset < parents;)
        {
            const uint32_t count = std::min(kDispatchItems, parents - offset);
            cb["gSparseOffset"] = offset;
            mpBuildGrowthCandidatesPass->execute(ctx, uint3(count, 1, 1));
            offset += count;
        }
        ctx->uavBarrier(counter.get());
        ctx->submit(true);
        uint32_t count = 0;
        counter->getBlob(&count, 0, sizeof(count));
        return count;
    };
    barrierSparseVoxels(ctx);
    barrierTopologyEvidence(ctx);
    for (const auto& page : mGridResources.indexPages) ctx->uavBarrier(page.get());
    var["EvidenceCB"]["gWriteCandidates"] = false;
    var["EvidenceCB"]["gCandidateCount"] = 0u;
    mGrowthCandidateCount = dispatch();
    mGrowthEvidenceBoundary = boundary;
    mGrowthEvidenceResolution = mVoxelResolution;
    if (!mGrowthCandidateCount) return;
    mpGrowthCandidates = mpDevice->createStructuredBuffer(kCandidateStride, mGrowthCandidateCount, flags);
    var["gGrowthCandidates"] = mpGrowthCandidates;
    var["EvidenceCB"]["gCandidateCount"] = mGrowthCandidateCount;
    var["EvidenceCB"]["gWriteCandidates"] = true;
    if (dispatch() != mGrowthCandidateCount) throw RuntimeError("Growth candidate count changed during initialization.");
    ctx->uavBarrier(mpGrowthCandidates.get());
    mTopologySettings.growthStatus = fmt::format("Collecting camera evidence for {} neighboring proposals", mGrowthCandidateCount);
}

void VoxelReconstructionNoLightTransport::evaluateGrowthCandidates(RenderContext* ctx, uint32_t viewID,
    const float4x4& viewProjection, const ref<Texture>& reference, const ref<Texture>& rendered)
{
    if (!mpGrowthCandidates || !mGrowthCandidateCount) return;
    if (!reference || !rendered || reference->getWidth() != rendered->getWidth() || reference->getHeight() != rendered->getHeight())
        throw RuntimeError("Growth evidence requires matching reference and rendered image sizes.");
    if (viewID >= TOPOLOGY_EVIDENCE_MAX_VIEWS) throw RuntimeError("Growth evidence camera index exceeds the supported view count.");
    if (mGrowthEvidenceSeenViews.size() <= viewID) mGrowthEvidenceSeenViews.resize(viewID + 1u, false);
    if (mGrowthEvidenceSeenViews[viewID]) return;
    auto var = mpEvaluateGrowthCandidatesPass->getRootVar();
    var["gGridDataParamBlock"] = mpGridBlock;
    var["gGrowthCandidates"] = mpGrowthCandidates;
    var["gGrowthReference"] = reference;
    var["gGrowthRendered"] = rendered;
    auto cb = var["EvidenceCB"];
    cb["gCandidateCount"] = mGrowthCandidateCount;
    cb["gEvidenceVP"] = viewProjection;
    cb["gEvidenceResolution"] = uint2(reference->getWidth(), reference->getHeight());
    cb["gMinAlphaDeficit"] = std::clamp(mTopologySettings.growthMinAlphaDeficit, 0.001f, 1.0f);
    cb["gForegroundAlphaMin"] = mTopologySettings.foregroundAlphaMin;
    cb["gBackgroundAlphaMax"] = mTopologySettings.backgroundAlphaMax;
    ctx->uavBarrier(rendered.get());
    for (uint32_t offset = 0u; offset < mGrowthCandidateCount;)
    {
        const uint32_t count = std::min(kDispatchItems, mGrowthCandidateCount - offset);
        var["CB"]["gSparseOffset"] = offset;
        mpEvaluateGrowthCandidatesPass->execute(ctx, uint3(count, 1, 1));
        offset += count;
    }
    ctx->uavBarrier(mpGrowthCandidates.get());
    mGrowthEvidenceSeenViews[viewID] = true;
    ++mGrowthEvidenceViews;
}

void VoxelReconstructionNoLightTransport::collectGrowthEvidence(RenderContext* ctx, const RenderData& renderData)
{
    const bool enabled = isCoarseToFine() ? mCoarseToFine.enableCoarseGrowth : mTopologySettings.enableGrowth;
    if (!enabled || !mTopologySettings.useGrowthEvidence)
    {
        resetGrowthEvidence();
        return;
    }
    const uint32_t start = getDeletionEvidenceStartIteration();
    const uint32_t boundary = mOptimizerParams.currentIteration + 1u;
    const uint32_t interval = std::max(1u, isCoarseToFine() ? mCoarseToFine.coarseGrowthInterval : mTopologySettings.growthInterval);
    // Existing scheduling skips growth on refinement/final-save boundaries.
    if (boundary < start || (boundary - start) % interval != 0u ||
        (isCoarseToFine() && boundary - mCoarseToFine.levelStartIteration >= mCoarseToFine.levelIterationBudget) ||
        (!isCoarseToFine() && boundary >= mOptimizerParams.maxIteration)) return;
    try
    {
        if (mOptimizerParams.currentView == 0u) beginGrowthEvidence(ctx, boundary);
        if (mGrowthEvidenceBoundary != boundary || mGrowthEvidenceResolution != mVoxelResolution) return;
        const uint32_t view = getTrainingViewIndex();
        if (view >= mReferenceImages.size() || view >= mReferenceCameras.size() || !mReferenceCameras[view])
            throw RuntimeError("Missing training camera or reference image for growth evidence.");
        evaluateGrowthCandidates(ctx, view, mReferenceCameras[view]->getViewProjMatrixNoJitter(),
            mReferenceImages[view], renderData.getTexture(kAccumulateOutputColor));
    }
    catch (const std::exception& error)
    {
        resetGrowthEvidence();
        if (isCoarseToFine()) mCoarseToFine.enableCoarseGrowth = false;
        else mTopologySettings.enableGrowth = false;
        mTopologySettings.growthStatus = "Growth paused: " + std::string(error.what());
        logError("{}", mTopologySettings.growthStatus);
    }
}

void VoxelReconstructionNoLightTransport::renderUIGrowthEvidence(Gui::Widgets& widget)
{
    if (widget.checkbox("Use Multi-view Growth Evidence", mTopologySettings.useGrowthEvidence)) resetGrowthEvidence();
    if (!mTopologySettings.useGrowthEvidence) return;
    widget.var("Growth Min Foreground Views", mTopologySettings.growthMinForegroundViews, 2u, uint32_t(TOPOLOGY_EVIDENCE_MAX_VIEWS), 1u);
    widget.var("Growth Background Veto Views", mTopologySettings.growthBackgroundVetoViews, 1u, uint32_t(TOPOLOGY_EVIDENCE_MAX_VIEWS), 1u);
    widget.var("Growth Min Alpha Deficit", mTopologySettings.growthMinAlphaDeficit, 0.001f, 1.0f, 0.01f);
    widget.text("Shared-face candidates; foreground alpha deficit supports growth, reliable background vetoes it.");
    widget.text("Existing foreground coverage gives no vote. Legacy penetration thresholds are ignored.");
    widget.text(fmt::format("Current evidence: {} proposals, {} distinct views", mGrowthCandidateCount, mGrowthEvidenceViews));
}
