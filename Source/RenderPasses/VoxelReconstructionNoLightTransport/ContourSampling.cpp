#include "VoxelReconstructionNoLightTransport.h"
#include <cstring>

namespace
{
const char* kContourShader = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ContourSampling.cs.slang";
}

void VoxelReconstructionNoLightTransport::resetContourSampling()
{
    mContourViews.clear();
    mContourBatchActive = false;
    mContourResolution = uint2(0);
    mContourScratchCapacity = 0;
    mpContourLookup = nullptr;
    mpContourPixels = nullptr;
    mpContourCounter = nullptr;
    mpContourObservations = nullptr;
    // Release bindings as well as owners, including reference image handles.
    mpFindContours = nullptr;
    mpInitializeContourHistory = nullptr;
    mpPrepareContourBatch = nullptr;
    mpFinishContourBatch = nullptr;
}

void VoxelReconstructionNoLightTransport::beginContourSampling(
    RenderContext* ctx, uint32_t viewID, const float4x4& viewProjection)
{
    mContourBatchActive = false;
    if (!mContourSampling.enabled || !mEnableReconstruction || !mOptimizerParams.isRunning) return;
    const auto resolution = mRayMarchingPass.mOutputResolution;
    if (viewID >= mReferenceImages.size()) throw RuntimeError("Contour sampling camera is outside the reference set.");
    auto reference = mReferenceImages[viewID];
    if (reference->getWidth() != resolution.x || reference->getHeight() != resolution.y)
        throw RuntimeError("Contour sampling requires reference and render resolutions to match.");
    if (any(mContourResolution != resolution) || mContourViews.size() != mReferenceImages.size())
    {
        resetContourSampling();
        mContourResolution = resolution;
        mContourViews.resize(mReferenceImages.size());
    }
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    if (!mpContourLookup)
    {
        mpContourLookup = mpDevice->createTexture2D(resolution.x, resolution.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
        mpContourPixels = mpDevice->createStructuredBuffer(sizeof(uint32_t), resolution.x * resolution.y, flags);
        mpContourCounter = mpDevice->createStructuredBuffer(sizeof(uint32_t), 1u, flags);
        const auto makePass = [&](const char* entry)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kContourShader).csEntry(entry);
            return ComputePass::create(mpDevice, desc, DefineList(), true);
        };
        mpFindContours = makePass("findContours");
        mpInitializeContourHistory = makePass("initializeHistory");
        mpPrepareContourBatch = makePass("prepareBatch");
        mpFinishContourBatch = makePass("finishBatch");
    }
    auto& view = mContourViews[viewID];
    if (view.reference != reference || std::memcmp(&view.viewProjection, &viewProjection, sizeof(viewProjection)) != 0)
        view = {};
    if (!view.initialized)
    {
        ctx->clearUAV(mpContourCounter->getUAV().get(), uint4(0));
        auto var = mpFindContours->getRootVar();
        var["gReference"] = reference;
        var["gCounter"] = mpContourCounter;
        var["gPixels"] = mpContourPixels;
        var["CB"]["gResolution"] = resolution;
        mpFindContours->execute(ctx, uint3(resolution.x, resolution.y, 1u));
        ctx->uavBarrier(mpContourPixels.get());
        ctx->uavBarrier(mpContourCounter.get());
        ctx->submit(true);
        mpContourCounter->getBlob(&view.count, 0, sizeof(view.count));
        if (view.count > 0u)
        {
            view.history = mpDevice->createStructuredBuffer(sizeof(ContourHistory), view.count, flags);
            var = mpInitializeContourHistory->getRootVar();
            var["gReference"] = reference;
            var["gPixels"] = mpContourPixels;
            var["gHistory"] = view.history;
            var["CB"]["gResolution"] = resolution;
            var["CB"]["gCount"] = view.count;
            mpInitializeContourHistory->execute(ctx, uint3(view.count, 1u, 1u));
            ctx->uavBarrier(view.history.get());
        }
        view.reference = reference;
        view.viewProjection = viewProjection;
        view.initialized = true;
    }
    if (view.count == 0u) return;
    if (view.count > mContourScratchCapacity)
    {
        mpContourObservations = mpDevice->createStructuredBuffer(sizeof(ContourObservation), view.count * 16u, flags);
        mContourScratchCapacity = view.count;
    }
    ctx->clearUAV(mpContourLookup->getUAV().get(), uint4(0));
    auto var = mpPrepareContourBatch->getRootVar();
    var["gHistory"] = view.history;
    var["gObservations"] = mpContourObservations;
    var["gLookup"] = mpContourLookup;
    var["CB"]["gResolution"] = resolution;
    var["CB"]["gCount"] = view.count;
    mpPrepareContourBatch->execute(ctx, uint3(view.count, 1u, 1u));
    ctx->uavBarrier(mpContourLookup.get());
    ctx->uavBarrier(mpContourObservations.get());
    mContourBatchView = viewID;
    mContourBatchUniformFraction = mContourSampling.uniformFraction;
    mContourBatchHistoryRate = mContourSampling.historyRate;
    mContourBatchActive = true;
}

void VoxelReconstructionNoLightTransport::finishContourSampling(RenderContext* ctx)
{
    if (!mContourBatchActive) return;
    const auto& view = mContourViews[mContourBatchView];
    ctx->uavBarrier(mpContourObservations.get());
    auto var = mpFinishContourBatch->getRootVar();
    var["gHistory"] = view.history;
    var["gObservations"] = mpContourObservations;
    var["CB"]["gCount"] = view.count;
    var["CB"]["gHistoryRate"] = mContourBatchHistoryRate;
    mpFinishContourBatch->execute(ctx, uint3(view.count, 1u, 1u));
    ctx->uavBarrier(view.history.get());
    mContourBatchActive = false;
}

void VoxelReconstructionNoLightTransport::renderUIContourSampling(Gui::Widgets& widget)
{
    bool changed = widget.checkbox("Contour Importance Sampling", mContourSampling.enabled);
    widget.tooltip("Guide only reference pixels with strictly 0 < alpha < 1. Freeze the 4x4 proposal for all SPP of a view.");
    if (mContourSampling.enabled)
    {
        changed |= widget.var("Contour Uniform Fraction", mContourSampling.uniformFraction, 0.05f, 1.0f, 0.05f);
        changed |= widget.var("Contour History Rate", mContourSampling.historyRate, 0.01f, 1.0f, 0.05f);
        changed |= widget.button("Reset Contour Sampling History");
        uint64_t pixels = 0u;
        for (const auto& view : mContourViews) pixels += view.count;
        widget.text(fmt::format("Contour history: {} pixels, {:.2f} MiB", pixels, double(pixels * sizeof(ContourHistory)) / (1024.0 * 1024.0)));
    }
    if (changed)
    {
        resetContourSampling();
        mRayMarchingPass.mSampleIndex = 0u;
        mRayMarchingPass.mOptionsChanged = true;
    }
}
