/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "VoxelReconstructionNoLightTransport.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VoxelReconstructionNoLightTransport>();
}

VoxelReconstructionNoLightTransport::VoxelReconstructionNoLightTransport(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    mpDevice = pDevice;

    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);

    // Initial Data
    {
        // Lego
#if RECON_MODE == RECON_MODE_POINT_CLOUD
        mGridResources.gridData.solidVoxelCount = 0;
#else
        mGridResources.gridData.solidVoxelCount = 17650;        // 64
#endif

    }

    // Create Grid pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(ReflectTypesShaderFilePath).csEntry("main");
        DefineList defines = getReconstructionDefines();
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, true);
    }

    createInitializationPassResource();

    // RayMarchingPass
    {
        mRayMarchingPass.init();
    }

    // LossPass
    {
        mLossPass.init();
    }

    // PathRecord
    {
        mpPathRecordBuffer = mpDevice->createStructuredBuffer(
            sizeof(PathRecord),
            mRayMarchingPass.mOutputResolution.x * mRayMarchingPass.mOutputResolution.y,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
    }

}

Properties VoxelReconstructionNoLightTransport::getProperties() const
{
    return {};
}

RenderPassReflection VoxelReconstructionNoLightTransport::reflect(const CompileData& compileData)
{

    RenderPassReflection reflector;

    // Output
    reflector.addOutput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mRayMarchingPass.mOutputResolution.x, mRayMarchingPass.mOutputResolution.y, 1, 1);
    reflector.addOutput(kOutputColor, "Color")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mRayMarchingPass.mOutputResolution.x, mRayMarchingPass.mOutputResolution.y, 1, 1);
    reflector.addOutput(kAccumulateOutputColor, "AccuColor")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mRayMarchingPass.mOutputResolution.x, mRayMarchingPass.mOutputResolution.y, 1, 1);


    return reflector;
}

void VoxelReconstructionNoLightTransport::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData.getTexture("src");
    if (!mpScene)
        return;
    mFrameDim = renderData.getDefaultTextureDims();
    mInvFrameDim = 1.0f / float2(mFrameDim);
    beginFrame(pRenderContext, false);

    // Viewing an existing result must work even without the training images or PLY.
    if (mLoadReconstructionRequested)
    {
        if (!mReconstructionFilePaths.empty() && mSelectedReconstructionFile < mReconstructionFilePaths.size())
        {
            loadReconstruction(pRenderContext, mReconstructionFilePaths[mSelectedReconstructionFile]);
        }
        else
        {
            mReconstructionIOStatus = "Load failed: no file selected.";
            logWarning("Load reconstruction failed: no file selected.");
        }

        mLoadReconstructionRequested = false;
    }

    bool needsTrainingData = mEnableReconstruction || mOptimizerParams.isRunning || mInitVoxelData;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    needsTrainingData |= mPointCloud.startRequested;
#elif RECON_MODE == RECON_MODE_COARSE_TO_FINE
    needsTrainingData |= mCoarseToFine.startRequested || mCoarseToFine.advanceRequested || mRestoreCoarseCheckpointRequested;
#endif
    if ((needsTrainingData || mUseReferenceCamera || !mLoadedReconstructionForViewing) && mReferenceDataError.empty())
    {
        try { loadReferenceImages(); }
        catch (const std::exception& e)
        {
            mReferenceDataError = e.what();
            // Parsing can fail after a prefix of cameras has been appended. Retry the whole dataset.
            mReferenceCameras.clear();
            mReferenceImagePaths.clear();
            mReferenceImages.clear();
            logWarning("Reference data unavailable: {}. Saved reconstructions can still be loaded for viewing.", e.what());
        }
    }
    const bool referencesReady = !mReferenceCameras.empty() && mReferenceImages.size() == mReferenceCameras.size() &&
        mOptimizerParams.viewsPerIteration == mReferenceCameras.size();
    if (needsTrainingData && !referencesReady)
    {
        if (mReferenceDataError.empty())
        {
            endFrame(pRenderContext);
            return;
        }
        // A failed training/restore request must not prevent viewing an already loaded grid.
        mEnableReconstruction = false;
        mOptimizerParams.isRunning = false;
        mInitVoxelData = false;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
        mPointCloud.startRequested = false;
#elif RECON_MODE == RECON_MODE_COARSE_TO_FINE
        mRestoreCoarseCheckpointRequested = false;
        mCoarseToFine.startRequested = false;
        mCoarseToFine.advanceRequested = false;
        mCoarseToFine.pauseRequested = false;
        mCoarseToFine.paused = true;
#endif
        mReconstructionIOStatus = "Training/restore request cancelled: reference data is unavailable. The current grid remains available for viewing.";
    }

#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    if (mRestoreCoarseCheckpointRequested)
    {
        mRestoreCoarseCheckpointRequested = false;
        if (mSelectedReconstructionFile < mReconstructionFilePaths.size())
            restoreCoarseCheckpoint(pRenderContext, mReconstructionFilePaths[mSelectedReconstructionFile]);
        else mReconstructionIOStatus = "Restore failed: no file selected.";
    }
#endif
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    bool pointCloudInitFailed = false;
#endif
    if (mInitVoxelData)
    {
#if RECON_MODE == RECON_MODE_POINT_CLOUD
        const bool startAfterInit = mPointCloud.startRequested;
        pointCloudInitFailed = !initializePointCloudVoxelData(pRenderContext);
        mPointCloud.startRequested = startAfterInit && !pointCloudInitFailed;
#else
        initializeVoxelData(pRenderContext);
#if RECON_MODE == RECON_MODE_ORIGINAL
        mLoadedReconstructionForViewing = false;
#endif
#endif
        mInitVoxelData = false;
    }

#if RECON_MODE == RECON_MODE_POINT_CLOUD
    if (mPointCloud.startRequested && !pointCloudInitFailed)
    {
        const bool ready = mPointCloud.initialized || initializePointCloudVoxelData(pRenderContext);
        mPointCloud.startRequested = false;
        if (ready)
        {
            resetPointCloudOptimization(pRenderContext);
            mEnableReconstruction = true;
            mOptimizerParams.isRunning = true;
            mLoadedReconstructionForViewing = false;
        }
    }
    if (mPointCloud.clearAccumulation)
    {
        pRenderContext->clearUAV(renderData.getTexture(kAccumulateOutputColor)->getUAV().get(), float4(0));
        mPointCloud.clearAccumulation = false;
    }
#endif
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    if (mCoarseToFine.startRequested)
    {
        const bool ready = mCoarseToFine.initialized || initializeCoarseVoxelData(pRenderContext);
        mCoarseToFine.startRequested = false;
        if (ready)
        {
            mCoarseToFine.paused = false;
            mCoarseToFine.finished = false;
            mEnableReconstruction = true;
            mOptimizerParams.isRunning = true;
            resetCoarseSampling(pRenderContext);
        }
    }
    if (mCoarseToFine.advanceRequested)
    {
        mCoarseToFine.advanceRequested = false;
        advanceCoarseStage(pRenderContext);
    }
    if (mSaveReconstructionRequested && !mOptimizerParams.isRunning && mCoarseToFine.initialized)
    {
        saveCoarseStage(pRenderContext, renderData);
        mSaveReconstructionRequested = false;
    }
    if (mCoarseToFine.clearAccumulation)
    {
        if (mRayMarchingPass.accuColor)
            pRenderContext->clearUAV(mRayMarchingPass.accuColor->getUAV().get(), float4(0));
        mCoarseToFine.clearAccumulation = false;
    }
#else
    if (mSaveReconstructionRequested)
    {
        saveReconstruction(pRenderContext);
        mSaveReconstructionRequested = false;

        // 保存后刷新列表，立刻能在 dropdown 看到新文件
        mReconstructionFileListDirty = true;
    }
#endif

    // test input
    {
        ref<Texture> pDummy = renderData.getTexture("dummy");
        ref<Texture> pRef = testIndex < mReferenceImages.size() ? mReferenceImages[testIndex] : nullptr;

        if (pDummy && pRef)
        {
            pRenderContext->blit(pRef->getSRV(), pDummy->getRTV());
        }
        else if (pDummy) pRenderContext->clearRtv(pDummy->getRTV().get(), float4(0));
    }

    bool isLastSample = mRayMarchingPass.mSpp > 0 && mRayMarchingPass.mSampleIndex == mRayMarchingPass.mSpp - 1u;

    rayMarchingPass(pRenderContext, renderData);


    if (mEnableReconstruction && mOptimizerParams.isRunning && isLastSample)
    {
        mLossPass.mView = mOptimizerParams.currentView;
        runLossPass(pRenderContext, renderData);
        runGradientPass(pRenderContext, renderData);
        runUpdatePass(pRenderContext, renderData);
        runReducePass(pRenderContext, renderData);


        mRayMarchingPass.mSampleIndex = 0;
        mOptimizerParams.currentView++;
        if (mOptimizerParams.currentView >= mOptimizerParams.viewsPerIteration)
        {
            mOptimizerParams.currentView = 0;
            mOptimizerParams.currentIteration++;
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
            completeCoarseIteration(pRenderContext, renderData);
#else
            if (mOptimizerParams.currentIteration >= mOptimizerParams.maxIteration)
            {
                stopReconstruction();
                mSaveReconstructionRequested = true;
            }
#endif
        }
    }


#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    // Publish only after loss, gradients, updates and checkpoint evaluation have finished.
    pRenderContext->copyResource(renderData.getTexture(kAccumulateOutputColor).get(), mRayMarchingPass.accuColor.get());
    renderCoarsePreview(pRenderContext, renderData);
#endif
    endFrame(pRenderContext);
}

void VoxelReconstructionNoLightTransport::renderUI(Gui::Widgets& widget) {
    if (widget.checkbox("Check Primitive", mRayMarchingPass.mCheckPrimitive))
        mRayMarchingPass.mOptionsChanged = true;
    if (widget.dropdown("Draw Mode", reinterpret_cast<ABSDFDrawMode&>(mRayMarchingPass.mDrawMode)))
        mRayMarchingPass.mOptionsChanged = true;
    if (widget.checkbox("Render Background", mRayMarchingPass.mRenderBackGround))
        mRayMarchingPass.mOptionsChanged = true;

#if RECON_MODE != RECON_MODE_POINT_CLOUD
    if (widget.var("Solid Voxel Count", mGridResources.gridData.solidVoxelCount))
    {
        requestRecompile();
    }
#endif


    widget.var("Geometry Tau", mGradientPass.geometryTau, 0.0f, 0.2f, 1e-4f);
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    widget.var("Geometry Grad Clamp", mGradientPass.geometryGradClamp, 0.0f, 10.0f, 1e-4f);
#else
    widget.var("Geometry Grad Clamp", mGradientPass.geometryGradClamp, 0.0f, 10.0f, 1e-4f);
#endif
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    if (!mOptimizerParams.isRunning)
        widget.var("Spp", mRayMarchingPass.mSpp, 1u, 100u, 1u);
    else
        widget.text("Training Spp: " + std::to_string(mRayMarchingPass.mSpp));
#else
    widget.var("Spp", mRayMarchingPass.mSpp, 1u, 100u,1u);
#endif

    if (widget.checkbox("Use ReferenceCamera", mUseReferenceCamera))
    {
        mReferenceDataError.clear();
        mRayMarchingPass.mOptionsChanged = true;
    }
    const uint32_t maxCameraIndex = mReferenceCameras.empty() ? 0u : uint32_t(mReferenceCameras.size() - 1);
    testIndex = std::min(testIndex, maxCameraIndex);
    if (widget.slider("Camera Index", testIndex, 0u, maxCameraIndex)) mRayMarchingPass.mOptionsChanged = true;
    if (!mReferenceDataError.empty())
    {
        widget.text("Reference data: " + mReferenceDataError);
        if (widget.button("Retry loading reference data")) mReferenceDataError.clear();
    }
    else if (!mReferenceCameras.empty() && mReferenceImages.size() < mReferenceCameras.size())
        widget.text(fmt::format("Loading reference images: {} / {}", mReferenceImages.size(), mReferenceCameras.size()));
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    renderCoarseUI(widget);
#else
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    widget.text("Mode 1: point-cloud initialization");
    widget.text("PLY: " + (resolveReconstructionPath(ReferenceImageDir) / "init_points.ply").string());
    widget.text(mPointCloud.status);
    if (widget.button("Init / Reset from PLY")) mInitVoxelData = true;
#else
    widget.checkbox("Init Voxel Data", mInitVoxelData);
#endif
    widget.var("Max Iteration", mOptimizerParams.maxIteration);
    if (widget.checkbox("Enable Reconstruction", mEnableReconstruction))
    {
        if (mEnableReconstruction)
            startReconstruction();
        else
            stopReconstruction();
    }
#endif

    renderUIUpdatePass(widget);

    widget.text("Voxel Size: " + ToString(mGridResources.gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)mGridResources.gridData.voxelCount));
    widget.text("Grid Min: " + ToString(mGridResources.gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(mGridResources.gridData.solidVoxelCount));
    widget.text(
        "Solid Rate: " + std::to_string(mGridResources.gridData.solidVoxelCount / (float)mGridResources.gridData.totalVoxelCount())
    );
    widget.text("Ray Sample Index: " + std::to_string(mRayMarchingPass.mSampleIndex));

    if (auto group = widget.group("Reconstruction"))
    {
        widget.text("Current iteration: " + std::to_string(mOptimizerParams.currentIteration));
        widget.text("Current view: " + std::to_string(mOptimizerParams.currentView));
        widget.text("Is running: " + std::string(mOptimizerParams.isRunning ? "true" : "false"));
        widget.text(fmt::format("Mean loss: {:.8f}", mReduceLossPass.meanLoss));
    }

    if (auto group = widget.group("Reconstruction IO"))
    {
        widget.text("Directory: " + getReconstructionModeDirectory().string());
        if (widget.button("Refresh Files")) mReconstructionFileListDirty = true;
        if (mReconstructionFileListDirty)
        {
            refreshReconstructionFileList();
            mReconstructionFileListDirty = false;
        }

        Gui::DropdownList fileList;

        for (uint32_t i = 0; i < mReconstructionFilePaths.size(); i++)
        {
            fileList.push_back({i, mReconstructionFilePaths[i].lexically_relative(getReconstructionModeDirectory()).string()});
        }

        if (!fileList.empty())
        {
            widget.dropdown("Reconstruction File", fileList, mSelectedReconstructionFile);

            widget.text("Selected: " + mReconstructionFilePaths[mSelectedReconstructionFile].string());

            if (widget.button("Load Selected Reconstruction"))
            {
                mLoadReconstructionRequested = true;
            }
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
            if (widget.button("Restore Training Checkpoint"))
            {
                mReferenceDataError.clear();
                mRestoreCoarseCheckpointRequested = true;
            }
#endif
        }
        else
        {
            widget.text("No reconstruction .bin files found.");
        }

        if (!mReconstructionIOStatus.empty()) widget.text(mReconstructionIOStatus);
        widget.textbox("Name Tag", mReconstructionNameTag);
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
        if (!mLoadedReconstructionForViewing)
#endif
        if (widget.button("Save Reconstruction"))
        {
            mSaveReconstructionRequested = true;
        }
    }

    if (auto group = widget.group("Debugging"))
    {
        mpPixelDebug->renderUI(group);
    }

    // Show Loss Graph
    {
        Gui::Window lossWindow(widget, "Loss Curve", uint2(600, 300), uint2(20, 20), Gui::WindowFlags::Default);

        if (lossWindow.gui())
        {
            lossWindow.text(fmt::format("Iteration: {}", mOptimizerParams.currentIteration));

            lossWindow.text(fmt::format("Current view mean loss: {:.8f}", mReduceLossPass.meanLoss));

            if (!mReduceLossPass.iterationLossHistory.empty())
            {
                const auto& history = mReduceLossPass.iterationLossHistory;

                // 最多只显示最近 100 个 iteration
                const uint32_t maxDisplayCount = 100;

                uint32_t sampleCount = std::min(uint32_t(history.size()), maxDisplayCount);

                uint32_t startIndex = uint32_t(history.size()) - sampleCount;

                // callback 需要同时知道 history 和起始位置
                struct LossGraphData
                {
                    const std::vector<float>* history;
                    uint32_t startIndex;
                };

                LossGraphData graphData{&history, startIndex};

                auto lossCallback = [](void* pUserData, int32_t index) -> float
                {
                    auto* pData = static_cast<LossGraphData*>(pUserData);

                    uint32_t actualIndex = pData->startIndex + uint32_t(index);

                    float loss = (*pData->history)[actualIndex];

                    // 防止 log10(0)
                    loss = std::max(loss, 1e-12f);

                    return std::log10(loss);
                };

                // 显示当前 graph 对应的真实 iteration 范围
                lossWindow.text(fmt::format("Showing iterations: {} - {}", startIndex + 1, history.size()));

                lossWindow.graph("Log10 Average Loss (Last 100)", lossCallback, &graphData, sampleCount, 0, FLT_MAX, FLT_MAX, 0, 220);
            }

            lossWindow.release();
        }
    }

}


void VoxelReconstructionNoLightTransport::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mLoadedReconstructionForViewing = false;
    mReconstructionIOStatus.clear();
    mReferenceDataError.clear();
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    mPointCloud = {};
    mGridResources.gridData.solidVoxelCount = 0;
    mEnableReconstruction = false;
    mOptimizerParams.reset();
#endif
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    configureCoarseStages();
#endif
    UpdateVoxelGrid(mVoxelResolution);
    setupGridResouce(pRenderContext, true);



    // Auto Diff
    //mVoxelSHGradDim = mGridResources.gridData.totalVoxelCount() * SH_COUNT * 3;
    //std::vector<SceneGradients::GradConfig> gradConfigs;
    //gradConfigs.push_back(SceneGradients::GradConfig(
    //    GradientType::VoxelSH,
    //    mVoxelSHGradDim,
    //    1 // hashSize，第一版建议先用 1
    //));
    //mpSceneGradients = make_ref<SceneGradients>(mpDevice, gradConfigs, GradientAggregateMode::Direct);

    // RayMarching
    createRayMarchingPassResource(pRenderContext);

    // Loss Pass
    createLossPassResource(pRenderContext);

    // Gradient Pass
    createGradientPassResource(pRenderContext);

    //// Update Pass
    createUpdatePassResource(pRenderContext);

    // Reduce Pass
    createReducePassResource(pRenderContext);


}


void VoxelReconstructionNoLightTransport::beginFrame(RenderContext* pRenderContext, bool forceReset)
{
    mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
    setupGridResouce(pRenderContext, forceReset);
}

void VoxelReconstructionNoLightTransport::endFrame(RenderContext* pRenderContext)
{
    mpPixelDebug->endFrame(pRenderContext);
    mFrameCount++;
    mRayMarchingPass.mFrameIndex = mFrameCount;
}

//void VoxelReconstructionNoLightTransport::UpdateVoxelGrid(uint voxelResolution)
//{
//    float3 diag;
//    float length;
//    float3 center;
//    if (scene)
//    {
//        AABB aabb = scene->getSceneBounds();
//        diag = aabb.maxPoint - aabb.minPoint;
//        length = std::max(diag.z, std::max(diag.x, diag.y));
//        center = aabb.center();
//        diag *= 1.02f;
//        length *= 1.02f;
//    }
//    else
//    {
//        diag = float3(1);
//        length = 1.f;
//        center = float3(0);
//    }
//
//    mGridResources.gridData.voxelSize = float3(length / voxelResolution);
//    float3 temp = diag / mGridResources.gridData.voxelSize;
//
//    mGridResources.gridData.voxelCount = uint3(
//        (uint)math::ceil(temp.x / MinFactor.x) * MinFactor.x,
//        (uint)math::ceil(temp.y / MinFactor.y) * MinFactor.y,
//        (uint)math::ceil(temp.z / MinFactor.z) * MinFactor.z
//    );
//    mGridResources.gridData.gridMin = center - 0.5f * mGridResources.gridData.voxelSize * float3(mGridResources.gridData.voxelCount);
//    // mGridResources.gridData.solidVoxelCount = 0;
//}

void VoxelReconstructionNoLightTransport::UpdateVoxelGrid(uint voxelResolution)
{
    //
    // NeRF synthetic reconstruction domain.
    //
    // 手动指定 AABB。
    //
    const float3 aabbMin = float3(-1.3f);
    const float3 aabbMax = float3(1.3f);

    float3 diag = aabbMax - aabbMin;
    float3 center = 0.5f * (aabbMin + aabbMax);

    //
    // 最长边划分为 voxelResolution 个 voxel，
    // 其他方向根据实际 AABB 尺寸决定 voxel 数量。
    //
    float length = std::max(diag.x, std::max(diag.y, diag.z));

    //
    // 给边界留一点余量。
    //
    constexpr float padding = 1.02f;

    diag *= padding;
    length *= padding;

    //
    // Voxel size.
    //
    mGridResources.gridData.voxelSize = float3(length / static_cast<float>(voxelResolution));

    //
    // Calculate voxel count in XYZ.
    //
    float3 temp = diag / mGridResources.gridData.voxelSize;

    mGridResources.gridData.voxelCount = uint3(
        static_cast<uint>(math::ceil(temp.x / MinFactor.x)) * MinFactor.x,

        static_cast<uint>(math::ceil(temp.y / MinFactor.y)) * MinFactor.y,

        static_cast<uint>(math::ceil(temp.z / MinFactor.z)) * MinFactor.z
    );

    //
    // Actual voxel grid min position.
    //
    // 注意这里不是直接使用 aabbMin，
    // 因为 voxelCount 经过 ceil / MinFactor 对齐以后，
    // 实际 grid 大小可能略大于指定的 AABB。
    //
    mGridResources.gridData.gridMin = center - 0.5f * mGridResources.gridData.voxelSize * float3(mGridResources.gridData.voxelCount);

    logInfo(
        "Reconstruction voxel grid initialized:"
        " center=({}, {}, {}),"
        " voxelSize=({}, {}, {}),"
        " voxelCount=({}, {}, {}),"
        " gridMin=({}, {}, {})",
        center.x,
        center.y,
        center.z,
        mGridResources.gridData.voxelSize.x,
        mGridResources.gridData.voxelSize.y,
        mGridResources.gridData.voxelSize.z,
        mGridResources.gridData.voxelCount.x,
        mGridResources.gridData.voxelCount.y,
        mGridResources.gridData.voxelCount.z,
        mGridResources.gridData.gridMin.x,
        mGridResources.gridData.gridMin.y,
        mGridResources.gridData.gridMin.z
    );
}


void VoxelReconstructionNoLightTransport::setupGridResouce(RenderContext* pRenderContext, bool forceReset)
{
    if (!mpGridBlock)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock");

        if (!reflector)
            std::cout << "ComputerPass : ReflectTypes Error !!!!\n ";

        mpGridBlock = ParameterBlock::create(mpDevice, reflector);
    }
    ShaderVar gridBlock = mpGridBlock->getRootVar();

    // we only fully initialize resource once
    const bool initializeResource = !mGridResources.gridDataBuffer;

    // -----------------------------------------------------------------------------
    // Resource setup
    // -----------------------------------------------------------------------------
    if (initializeResource || forceReset)
    {
        mGridResources.gridDataBuffer = mpDevice->createStructuredBuffer(
            sizeof(VoxelData),
            mGridResources.gridData.totalVoxelCount(),
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        //mGridResources.gridDataBuffer = mpDevice->createStructuredBuffer(
        //    sizeof(VoxelData),
        //    mGridResources.gridData.solidVoxelCount,
        //    ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        //);
        mGridResources.vBuffer = mpDevice->createTexture3D(
            mGridResources.gridData.voxelCount.x, mGridResources.gridData.voxelCount.y, mGridResources.gridData.voxelCount.z,
            ResourceFormat::R32Int,
            1u,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        pRenderContext->clearUAV(mGridResources.gridDataBuffer->getUAV().get(), uint4(0));
    }


    gridBlock["gridDataBuffer"] = mGridResources.gridDataBuffer;
    gridBlock["vBuffer"] = mGridResources.vBuffer;
    gridBlock["voxelCount"] = mGridResources.gridData.voxelCount;
    gridBlock["voxelSize"] = mGridResources.gridData.voxelSize;
    gridBlock["gridMin"] = mGridResources.gridData.gridMin;
    gridBlock["solidVoxelCount"] = mGridResources.gridData.solidVoxelCount;

}



void VoxelReconstructionNoLightTransport::startReconstruction()
{
    mReferenceDataError.clear();
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    // GPU work and first-use initialization are performed at the next frame boundary.
    mPointCloud.startRequested = true;
#elif RECON_MODE == RECON_MODE_COARSE_TO_FINE
    if (mLoadedReconstructionForViewing)
    {
        mReconstructionIOStatus = "Viewing a saved result. Restore Training Checkpoint to continue its optimization.";
        return;
    }
    mCoarseToFine.startRequested = true;
#else
    mLoadedReconstructionForViewing = false;
    mEnableReconstruction = true;

    mOptimizerParams.isRunning = true;
    mOptimizerParams.currentIteration = 0;
    mOptimizerParams.currentView = 0;
    mRayMarchingPass.mSampleIndex = 0;

    mReduceLossPass.meanLoss = 0.0f;
    mReduceLossPass.iterationLossSum = 0.0f;
    mReduceLossPass.iterationLossCount = 0;
    mReduceLossPass.iterationLossHistory.clear();
#endif

}

void VoxelReconstructionNoLightTransport::stopReconstruction()
{
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    mPointCloud.startRequested = false;
    mEnableReconstruction = false;
    mOptimizerParams.isRunning = false;
    mOptimizerParams.currentView = 0;
    mRayMarchingPass.mSampleIndex = 0;
    mPointCloud.clearAccumulation = true;
#elif RECON_MODE == RECON_MODE_COARSE_TO_FINE
    mCoarseToFine.pauseRequested = true;
#else
    mEnableReconstruction = false;

    mOptimizerParams.isRunning = false;
    mOptimizerParams.currentIteration = 0;
    mOptimizerParams.currentView = 0;
#endif
}

bool VoxelReconstructionNoLightTransport::onMouseEvent(const MouseEvent& mouseEvent)
{
    bool ret = mpPixelDebug->onMouseEvent(mouseEvent);

    return ret;
}
