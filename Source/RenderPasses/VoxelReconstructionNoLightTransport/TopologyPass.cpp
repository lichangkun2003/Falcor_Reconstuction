#include "VoxelReconstructionNoLightTransport.h"

uint32_t VoxelReconstructionNoLightTransport::getDeletionEvidenceStartIteration() const
{
    return mUpdatePass.mOpacityWarmupIterations + mUpdatePass.mOpacityRampIterations;
}

bool VoxelReconstructionNoLightTransport::shouldCollectDeletionEvidence() const
{
    const uint32_t start = getDeletionEvidenceStartIteration();
    // Use one jittered evidence sample per view and iteration, independent of
    // the training SPP. Repeated iterations improve 1-SPP coverage, while
    // per-view stamps still limit each camera to one foreground and one
    // background vote per evidence window.
    const bool finalSppSample = mRayMarchingPass.mSpp > 0u &&
        mRayMarchingPass.mSampleIndex == mRayMarchingPass.mSpp;
    return mTopologySettings.collectDeletionEvidence &&
        mOptimizerParams.currentIteration >= start &&
        finalSppSample &&
        mGridResources.gridData.activeVoxelCount > 0;
}

void VoxelReconstructionNoLightTransport::createTopologyPassResource(RenderContext* pRenderContext)
{
    ProgramDesc desc;
    desc.addShaderLibrary(TopologyPassShaderFilePath).csEntry("main");
    mpTopologyPass = ComputePass::create(mpDevice, desc, getReconstructionDefines(), true);
    mpTopologySummary = mpDevice->createStructuredBuffer(
        sizeof(uint32_t),
        4u,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    pRenderContext->clearUAV(mpTopologySummary->getUAV().get(), uint4(0));
}

void VoxelReconstructionNoLightTransport::resetDeletionEvidence(RenderContext* pRenderContext)
{
    for (const auto& page : mGridResources.topologyEvidencePages)
        pRenderContext->clearUAV(page->getUAV().get(), uint4(0));
    if (mpTopologySummary)
        pRenderContext->clearUAV(mpTopologySummary->getUAV().get(), uint4(0));

    mTopologySettings.candidateCount = 0;
    mTopologySettings.oneWindowCount = 0;
    mTopologySettings.protectedCount = 0;
    mTopologySettings.weakConflictCount = 0;
    mTopologySettings.completedWindows = 0;
}

void VoxelReconstructionNoLightTransport::evaluateDeletionEvidence(RenderContext* pRenderContext)
{
    if (!mTopologySettings.collectDeletionEvidence || !mpTopologyPass || !mpTopologySummary)
        return;

    const uint32_t start = getDeletionEvidenceStartIteration();
    const uint32_t interval = std::max(1u, mTopologySettings.evidenceInterval);
    if (mOptimizerParams.currentIteration <= start ||
        (mOptimizerParams.currentIteration - start) % interval != 0u)
        return;

    pRenderContext->clearUAV(mpTopologySummary->getUAV().get(), uint4(0));
    for (const auto& page : mGridResources.topologyEvidencePages)
        pRenderContext->uavBarrier(page.get());

    auto var = mpTopologyPass->getRootVar();
    var["gGridDataParamBlock"] = mpGridBlock;
    var["gTopologySummary"] = mpTopologySummary;
    auto cb = var["CB"];
    cb["gMinConflictViews"] = mTopologySettings.minDeletionConflictViews;
    cb["gMaxSupportViews"] = mTopologySettings.maxDeletionSupportViews;

    constexpr uint32_t batchSize = 65535u * 256u;
    for (uint32_t offset = 0; offset < mGridResources.gridData.activeVoxelCount; )
    {
        const uint32_t count = std::min(batchSize, mGridResources.gridData.activeVoxelCount - offset);
        cb["gSparseOffset"] = offset;
        mpTopologyPass->execute(pRenderContext, uint3(count, 1, 1));
        offset += count;
    }

    for (const auto& page : mGridResources.topologyEvidencePages)
        pRenderContext->uavBarrier(page.get());
    pRenderContext->uavBarrier(mpTopologySummary.get());
    pRenderContext->submit(true);

    uint32_t summary[4] = {};
    mpTopologySummary->getBlob(summary, 0, sizeof(summary));
    mTopologySettings.candidateCount = summary[0];
    mTopologySettings.oneWindowCount = summary[1];
    mTopologySettings.protectedCount = summary[2];
    mTopologySettings.weakConflictCount = summary[3];
    ++mTopologySettings.completedWindows;

    logInfo(
        "Deletion evidence window {} at iteration {}: {} candidates, {} first-window warnings, "
        "{} foreground-protected, {} weak background conflicts.",
        mTopologySettings.completedWindows,
        mOptimizerParams.currentIteration,
        summary[0], summary[1], summary[2], summary[3]
    );
}
