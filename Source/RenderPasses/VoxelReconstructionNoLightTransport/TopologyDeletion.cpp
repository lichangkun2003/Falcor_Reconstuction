#include "VoxelReconstructionNoLightTransport.h"

namespace
{
uint32_t pageCountForEntries(uint32_t count)
{
    return std::max(1u, (count + SPARSE_POOL_PAGE_SIZE - 1u) / SPARSE_POOL_PAGE_SIZE);
}

constexpr uint32_t kThreadsPerGroup = 256u;
constexpr uint32_t kMaximumDispatchItems = 65535u * kThreadsPerGroup;
}

void VoxelReconstructionNoLightTransport::createDeletionPassResources()
{
    const auto createPass = [&](const char* entryPoint)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(DeleteCompactPassShaderFilePath).csEntry(entryPoint);
        return ComputePass::create(mpDevice, desc, getReconstructionDefines(), true);
    };

    mpBuildDeletionListsPass = createPass("buildDeletionLists");
    mpClearDeletionIndicesPass = createPass("clearDeletionIndices");
    mpMoveDeletionSurvivorsPass = createPass("moveDeletionSurvivors");
    mpClearDeletionTailPass = createPass("clearDeletionTail");
}

void VoxelReconstructionNoLightTransport::deleteAndCompactCandidates(RenderContext* pRenderContext)
{
    const uint32_t oldActiveCount = mGridResources.gridData.activeVoxelCount;
    const uint32_t expectedCandidates = mTopologySettings.candidateCount;
    if (oldActiveCount == 0u || expectedCandidates == 0u)
    {
        mTopologySettings.deletionStatus = "Compaction skipped: no deletion candidates";
        return;
    }
    if (expectedCandidates > oldActiveCount)
    {
        mTopologySettings.deletionStatus = "Compaction aborted: candidate count exceeds the active pool";
        logError("{}", mTopologySettings.deletionStatus);
        return;
    }

    const uint32_t newActiveCount = oldActiveCount - expectedCandidates;
    const uint32_t listCapacity = std::max(1u, expectedCandidates);
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto holeIDs = mpDevice->createStructuredBuffer(sizeof(uint32_t), listCapacity, flags);
    auto donorIDs = mpDevice->createStructuredBuffer(sizeof(uint32_t), listCapacity, flags);
    auto counters = mpDevice->createStructuredBuffer(sizeof(uint32_t), 3u, flags);
    pRenderContext->clearUAV(counters->getUAV().get(), uint4(0));

    auto buildVar = mpBuildDeletionListsPass->getRootVar();
    buildVar["gGridDataParamBlock"] = mpGridBlock;
    buildVar["gHoleIDs"] = holeIDs;
    buildVar["gDonorIDs"] = donorIDs;
    buildVar["gDeletionCounters"] = counters;
    auto buildCB = buildVar["CB"];
    buildCB["gOldActiveCount"] = oldActiveCount;
    buildCB["gNewActiveCount"] = newActiveCount;
    buildCB["gListCapacity"] = listCapacity;

    barrierTopologyEvidence(pRenderContext);
    for (uint32_t offset = 0; offset < oldActiveCount; )
    {
        const uint32_t count = std::min(kMaximumDispatchItems, oldActiveCount - offset);
        buildCB["gSparseOffset"] = offset;
        mpBuildDeletionListsPass->execute(pRenderContext, uint3(count, 1, 1));
        offset += count;
    }
    pRenderContext->uavBarrier(holeIDs.get());
    pRenderContext->uavBarrier(donorIDs.get());
    pRenderContext->uavBarrier(counters.get());
    pRenderContext->submit(true);

    uint32_t listCounts[3] = {};
    counters->getBlob(listCounts, 0, sizeof(listCounts));
    if (listCounts[0] != expectedCandidates || listCounts[1] != listCounts[2] ||
        listCounts[1] > listCapacity)
    {
        mTopologySettings.deletionStatus = fmt::format(
            "Compaction aborted: evidence changed (expected {}, found {}, holes {}, donors {})",
            expectedCandidates, listCounts[0], listCounts[1], listCounts[2]);
        logError("{}", mTopologySettings.deletionStatus);
        return;
    }
    const uint32_t moveCount = listCounts[1];

    auto clearIndexVar = mpClearDeletionIndicesPass->getRootVar();
    clearIndexVar["gGridDataParamBlock"] = mpGridBlock;
    auto clearIndexCB = clearIndexVar["CB"];
    clearIndexCB["gOldActiveCount"] = oldActiveCount;
    clearIndexCB["gCooldownUntilIteration"] = uint32_t(std::min<uint64_t>(kMaximumTopologyIteration,
        uint64_t(mOptimizerParams.currentIteration) + mTopologySettings.deletionCooldownIterations));
    for (uint32_t offset = 0; offset < oldActiveCount; )
    {
        const uint32_t count = std::min(kMaximumDispatchItems, oldActiveCount - offset);
        clearIndexCB["gSparseOffset"] = offset;
        mpClearDeletionIndicesPass->execute(pRenderContext, uint3(count, 1, 1));
        offset += count;
    }
    for (const auto& page : mGridResources.indexPages) pRenderContext->uavBarrier(page.get());
    mGrowthCooldownPresent = true;

    if (moveCount > 0u)
    {
        auto moveVar = mpMoveDeletionSurvivorsPass->getRootVar();
        moveVar["gGridDataParamBlock"] = mpGridBlock;
        moveVar["gHoleIDs"] = holeIDs;
        moveVar["gDonorIDs"] = donorIDs;
        for (uint32_t page = 0; page < mGridResources.adamPages.size(); ++page)
            moveVar["gGeometryAdamPages"][page] = mGridResources.adamPages[page];
        auto moveCB = moveVar["CB"];
        moveCB["gPairCount"] = moveCount;
        for (uint32_t offset = 0; offset < moveCount; )
        {
            const uint32_t count = std::min(kMaximumDispatchItems, moveCount - offset);
            moveCB["gPairOffset"] = offset;
            mpMoveDeletionSurvivorsPass->execute(pRenderContext, uint3(count, 1, 1));
            offset += count;
        }
    }

    for (const auto& page : mGridResources.voxelPages) pRenderContext->uavBarrier(page.get());
    for (const auto& page : mGridResources.adamPages) pRenderContext->uavBarrier(page.get());
    for (const auto& page : mGridResources.radianceAdamIndexPages) pRenderContext->uavBarrier(page.get());
    for (const auto& page : mGridResources.cellIndexPages) pRenderContext->uavBarrier(page.get());
    for (const auto& page : mGridResources.indexPages) pRenderContext->uavBarrier(page.get());

    auto clearTailVar = mpClearDeletionTailPass->getRootVar();
    clearTailVar["gGridDataParamBlock"] = mpGridBlock;
    auto clearTailCB = clearTailVar["CB"];
    clearTailCB["gOldActiveCount"] = oldActiveCount;
    clearTailCB["gNewActiveCount"] = newActiveCount;
    for (uint32_t offset = newActiveCount; offset < oldActiveCount; )
    {
        const uint32_t count = std::min(kMaximumDispatchItems, oldActiveCount - offset);
        clearTailCB["gSparseOffset"] = offset;
        mpClearDeletionTailPass->execute(pRenderContext, uint3(count, 1, 1));
        offset += count;
    }
    pRenderContext->submit(true);

    const uint32_t oldPoolPages = uint32_t(mGridResources.voxelPages.size());
    const uint32_t oldRadiancePages = uint32_t(mGridResources.radianceAdamPages.size());
    const uint32_t requiredPoolPages = pageCountForEntries(newActiveCount);
    const uint32_t keepPoolPages = std::min(
        oldPoolPages,
        requiredPoolPages + (requiredPoolPages < oldPoolPages ? 1u : 0u));
    const uint32_t desiredRadiancePages = pageCountForEntries(std::max(1u, newActiveCount / 8u));
    const uint32_t keepRadiancePages = std::max(1u, std::min(oldRadiancePages, desiredRadiancePages));

    GridResources compacted = mGridResources;
    compacted.gridData.activeVoxelCount = newActiveCount;
    compacted.gridData.solidVoxelCount = newActiveCount;
    compacted.gridData.voxelCapacity = keepPoolPages * SPARSE_POOL_PAGE_SIZE;
    compacted.voxelPages.resize(keepPoolPages);
    compacted.gradPages.resize(keepPoolPages);
    compacted.topologyEvidencePages.resize(keepPoolPages);
    compacted.adamPages.resize(keepPoolPages);
    compacted.radianceAdamIndexPages.resize(keepPoolPages);
    compacted.cellIndexPages.resize(keepPoolPages);
    compacted.radianceAdamPages.resize(keepRadiancePages);

    auto compactedBlock = createSparseGridBlock(compacted);
    commitSparseGrid(std::move(compacted), compactedBlock, mVoxelResolution);
    // The compaction passes still reference the old parameter block, removed
    // pages, and temporary ID lists through their ProgramVars. Recreate these
    // rare-use passes so those references cannot keep released pages alive.
    createDeletionPassResources();

    // The independent radiance pool cannot be compacted safely in place without
    // a full reverse map. Keep the optimized SH coefficients, but restart their
    // Adam moments and mappings so deleted slots are genuinely reclaimed.
    for (const auto& page : mGridResources.radianceAdamIndexPages)
        pRenderContext->clearUAV(page->getUAV().get(), uint4(0xffffffffu));
    for (const auto& page : mGridResources.radianceAdamPages)
        pRenderContext->clearUAV(page->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mGridResources.radianceAdamCounter->getUAV().get(), uint4(0));
    clearSparseGradients(pRenderContext);
    resetDeletionEvidence(pRenderContext, true);
    if (mpPathRecordBuffer) pRenderContext->clearUAV(mpPathRecordBuffer->getUAV().get(), uint4(0));
    pRenderContext->submit(true);

    mTopologySettings.lastDeletedCount = expectedCandidates;
    mTopologySettings.lastReleasedPoolPages = oldPoolPages - keepPoolPages;
    mTopologySettings.lastReleasedRadiancePages = oldRadiancePages - keepRadiancePages;
    mTopologySettings.deletionStatus = fmt::format(
        "Deleted {} voxels; active {} -> {}; released {} pool pages and {} radiance-Adam pages",
        expectedCandidates, oldActiveCount, newActiveCount,
        mTopologySettings.lastReleasedPoolPages, mTopologySettings.lastReleasedRadiancePages);
    mPointCloud.status = mTopologySettings.deletionStatus;
    mPointCloud.clearAccumulation = true;
    mRayMarchingPass.mDrawMode = uint32_t(ABSDFDrawMode::Default);
    mRayMarchingPass.mOptionsChanged = true;
    logInfo("{}", mTopologySettings.deletionStatus);
}
