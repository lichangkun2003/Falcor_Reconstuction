#include "VoxelReconstructionNoLightTransport.h"

namespace
{
constexpr uint32_t kGrowthDispatchItems = 65535u * 256u;
}

void VoxelReconstructionNoLightTransport::createGrowthPassResources()
{
    const auto create = [&](const char* entry)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(GrowthPassShaderFilePath).csEntry(entry);
        return ComputePass::create(mpDevice, desc, getReconstructionDefines(), true);
    };
    mpProposeGrowthPass = create("proposeGrowth");
    mpInitializeGrowthPass = create("initializeGrowth");
    mpCommitGrowthPass = create("commitGrowth");
    mpClearGrowthClaimsPass = create("clearGrowthClaims");
    mpRollbackGrowthPass = create("rollbackGrowth");
    mpClearGrowthCooldownPass = create("clearGrowthCooldown");
}

void VoxelReconstructionNoLightTransport::resetGrowthCooldown(RenderContext* pRenderContext)
{
    if (!mGrowthCooldownPresent)
        return;
    if (!mpClearGrowthCooldownPass)
        createGrowthPassResources();
    auto var = mpClearGrowthCooldownPass->getRootVar();
    var["gGridDataParamBlock"] = mpGridBlock;
    const uint3 size = mGridResources.gridData.voxelCount;
    const uint32_t total = size.x * size.y * size.z;
    for (uint32_t offset = 0; offset < total;)
    {
        const uint32_t count = std::min(kGrowthDispatchItems, total - offset);
        var["CB"]["gSparseOffset"] = offset;
        mpClearGrowthCooldownPass->execute(pRenderContext, uint3(count, 1, 1));
        offset += count;
    }
    for (const auto& page : mGridResources.indexPages)
        pRenderContext->uavBarrier(page.get());
    mGrowthCooldownPresent = false;
}

void VoxelReconstructionNoLightTransport::growNeighborVoxels(RenderContext* pRenderContext)
{
    const uint32_t start = getDeletionEvidenceStartIteration();
    const uint32_t interval = std::max(1u, mTopologySettings.growthInterval);
    if (!mTopologySettings.enableGrowth || mOptimizerParams.currentIteration < start)
        return;
    // Anchor the cadence to the end of opacity warm-up + ramp. Skipped rounds
    // keep the last growth result visible and never claim/allocate new cells.
    if ((mOptimizerParams.currentIteration - start) % interval != 0u)
        return;
    mTopologySettings.lastGrowthCount = 0u;
    mTopologySettings.lastGrowthPages = 0u;
    const uint32_t parents = mGridResources.gridData.activeVoxelCount;
    if (parents == 0u)
        return;
    const uint32_t oldPages = uint32_t(mGridResources.voxelPages.size());
    uint32_t added = 0u;
    bool claimed = false;
    bool initializing = false;

    const auto indexBarrier = [&]()
    {
        for (const auto& page : mGridResources.indexPages)
            pRenderContext->uavBarrier(page.get());
    };
    const auto dispatch = [&](const ref<ComputePass>& pass, uint32_t count)
    {
        auto var = pass->getRootVar();
        var["gGridDataParamBlock"] = mpGridBlock;
        auto cb = var["CB"];
        cb["gParentCount"] = parents;
        cb["gNewCount"] = added;
        for (uint32_t offset = 0; offset < count;)
        {
            const uint32_t batch = std::min(kGrowthDispatchItems, count - offset);
            cb["gSparseOffset"] = offset;
            pass->execute(pRenderContext, uint3(batch, 1, 1));
            offset += batch;
        }
    };

    try
    {
        if (!mpProposeGrowthPass)
            createGrowthPassResources();
        // Compile/reflection-check all transaction stages before marking cells.
        for (const auto& pass :
             {mpProposeGrowthPass, mpInitializeGrowthPass, mpCommitGrowthPass, mpClearGrowthClaimsPass, mpRollbackGrowthPass})
            pass->getRootVar();
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        auto counter = mpDevice->createStructuredBuffer(sizeof(uint32_t), 1u, flags);
        pRenderContext->clearUAV(counter->getUAV().get(), uint4(0));
        for (const auto& pass : {mpProposeGrowthPass, mpInitializeGrowthPass})
        {
            auto var = pass->getRootVar();
            var["gGrowthCounter"] = counter;
            auto cb = var["CB"];
            cb["gCurrentIteration"] = std::min(mOptimizerParams.currentIteration, kMaximumTopologyIteration);
            cb["gGrowthWaitIterations"] = mTopologySettings.growthWaitIterations;
            cb["gEligibleIteration"] = uint32_t(std::min<uint64_t>(
                kMaximumTopologyIteration, uint64_t(mOptimizerParams.currentIteration) + mTopologySettings.growthProtectionIterations
            ));
            cb["gFacePenetration"] = std::clamp(mTopologySettings.growthFacePenetration, 0.0f, float(GROWTH_MAX_FACE_PENETRATION_VOXELS));
            cb["gShrink"] = std::clamp(mTopologySettings.growthShrink, 0.01f, 0.99f);
            cb["gContactOffset"] = std::clamp(mTopologySettings.growthContactOffset, 0.01f, 0.49f);
            cb["gInitialOpacity"] = std::clamp(mTopologySettings.growthInitialOpacity, 0.01f, 0.49f);
        }
        barrierSparseVoxels(pRenderContext);
        barrierTopologyEvidence(pRenderContext);
        indexBarrier();
        claimed = true;
        dispatch(mpProposeGrowthPass, parents);
        indexBarrier();
        pRenderContext->uavBarrier(counter.get());
        pRenderContext->submit(true);
        counter->getBlob(&added, 0, sizeof(added));
        if (added == 0u)
        {
            mTopologySettings.growthStatus = fmt::format("Iteration {}: no eligible empty neighbors (including newborn wait)",
                mOptimizerParams.currentIteration);
            return;
        }
        const uint64_t required = uint64_t(parents) + added;
        const uint64_t maximum = uint64_t(SPARSE_POOL_PAGE_SIZE) * SPARSE_POOL_MAX_PAGES;
        if (required > maximum)
            throw RuntimeError("Neighbor growth exceeds the sparse-pool page limit.");
        reserveSparseVoxelCapacity(pRenderContext, uint32_t(required));
        auto var = mpInitializeGrowthPass->getRootVar();
        for (uint32_t page = 0; page < mGridResources.adamPages.size(); ++page)
        {
            var["gGeometryAdamPages"][page] = mGridResources.adamPages[page];
            var["gRadianceAdamIndexPages"][page] = mGridResources.radianceAdamIndexPages[page];
        }
        pRenderContext->clearUAV(counter->getUAV().get(), uint4(0));
        initializing = true;
        // Parent count is frozen: newborns cannot become parents in this layer.
        dispatch(mpInitializeGrowthPass, parents);
        barrierSparseVoxels(pRenderContext);
        barrierSparseGradients(pRenderContext);
        barrierTopologyEvidence(pRenderContext);
        for (const auto& page : mGridResources.cellIndexPages)
            pRenderContext->uavBarrier(page.get());
        for (const auto& page : mGridResources.adamPages)
            pRenderContext->uavBarrier(page.get());
        for (const auto& page : mGridResources.radianceAdamIndexPages)
            pRenderContext->uavBarrier(page.get());
        pRenderContext->uavBarrier(counter.get());
        pRenderContext->submit(true);
        uint32_t initialized = 0u;
        counter->getBlob(&initialized, 0, sizeof(initialized));
        if (initialized != added)
            throw RuntimeError("Growth claim/initialization count mismatch.");
        dispatch(mpCommitGrowthPass, added);
        indexBarrier();
        pRenderContext->submit(true);
        // Only now expose the initialized, indexed children to rendering.
        mGridResources.gridData.activeVoxelCount = uint32_t(required);
        mGridResources.gridData.solidVoxelCount = uint32_t(required);
        mpGridBlock->getRootVar()["activeVoxelCount"] = uint32_t(required);
        mpGridBlock->getRootVar()["solidVoxelCount"] = uint32_t(required);
        mTopologySettings.lastGrowthCount = added;
        mTopologySettings.lastGrowthPages = uint32_t(mGridResources.voxelPages.size()) - oldPages;
        mTopologySettings.growthStatus = fmt::format(
            "Iteration {}: grew {} voxels ({} -> {}), added {} pool pages",
            mOptimizerParams.currentIteration,
            added,
            parents,
            required,
            mTopologySettings.lastGrowthPages
        );
        mPointCloud.clearAccumulation = true;
        logInfo("{}", mTopologySettings.growthStatus);
    }
    catch (const std::exception& error)
    {
        if (claimed)
        {
            try
            {
                mGridResources.gridData.activeVoxelCount = parents;
                mGridResources.gridData.solidVoxelCount = parents;
                mpGridBlock->getRootVar()["activeVoxelCount"] = parents;
                mpGridBlock->getRootVar()["solidVoxelCount"] = parents;
                if (initializing)
                    dispatch(mpRollbackGrowthPass, added);
                indexBarrier();
                dispatch(mpClearGrowthClaimsPass, parents);
                indexBarrier();
                barrierSparseVoxels(pRenderContext);
                pRenderContext->submit(true);
            }
            catch (const std::exception& rollbackError)
            {
                stopReconstruction();
                logError("Growth rollback failed; training stopped: {}", rollbackError.what());
            }
        }
        // Capacity/memory exhaustion must not silently drop part of the layer.
        // Existing reconstruction can continue if the rollback succeeded.
        mTopologySettings.enableGrowth = false;
        mTopologySettings.growthStatus = "Growth paused: " + std::string(error.what());
        logError("{}", mTopologySettings.growthStatus);
    }
}
