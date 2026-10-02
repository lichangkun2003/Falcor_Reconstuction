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
        mGridResources.gridData.solidVoxelCount = 0;

    }

    // Create Grid pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(ReflectTypesShaderFilePath).csEntry("main");
        DefineList defines = getReconstructionDefines();
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, true);
    }

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
    mFrameDim = mRayMarchingPass.mOutputResolution;
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
    needsTrainingData |= mPointCloud.startRequested;
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
        mPointCloud.startRequested = false;
        mReconstructionIOStatus = "Training/restore request cancelled: reference data is unavailable. The current grid remains available for viewing.";
    }

    bool pointCloudInitFailed = false;
    if (mInitVoxelData)
    {
        const bool startAfterInit = mPointCloud.startRequested;
        pointCloudInitFailed = !initializePointCloudVoxelData(pRenderContext);
        mPointCloud.startRequested = startAfterInit && !pointCloudInitFailed;
        mInitVoxelData = false;
    }

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
    if (mSaveReconstructionRequested)
    {
        saveReconstruction(pRenderContext);
        mSaveReconstructionRequested = false;

        // 保存后刷新列表，立刻能在 dropdown 看到新文件
        mReconstructionFileListDirty = true;
    }

    // test input
    if(!mEnableReconstruction)
    {
        ref<Texture> pDummy = renderData.getTexture("dummy");
        ref<Texture> pRef = testIndex < mReferenceImages.size() ? mReferenceImages[testIndex] : nullptr;

        if (pDummy && pRef)
        {
            pRenderContext->blit(pRef->getSRV(), pDummy->getRTV());
        }
        else if (pDummy) pRenderContext->clearRtv(pDummy->getRTV().get(), float4(0));
    }

    bool isFirstSample = mRayMarchingPass.mSampleIndex == 0u;
    bool isLastSample = mRayMarchingPass.mSpp > 0 && mRayMarchingPass.mSampleIndex == mRayMarchingPass.mSpp - 1u;
    // 前缀平均至少要有一个"别人"的采样：第 0 帧前缀为空，它的渲染结果只当后续采样的基线，不出梯度.
    // 只有 Spp == 1 时第 0 帧同时是唯一一帧，照旧出梯度（loss shader 会回退到累积图本身）.
    bool hasResidualBaseline = mRayMarchingPass.mSpp <= 1u || !isFirstSample;

    rayMarchingPass(pRenderContext, renderData);


    if (mEnableReconstruction && mOptimizerParams.isRunning)
    {
        // 批开始时清掉上一批留下的梯度，整批之内只做原子累加.
        // gradBuffer 的元素与 appearanceValid/geometryValid 计数一起累加.
        // update 时按计数求平均，得到的就是本批所有无偏梯度的平均.
        if (isFirstSample)
        {
            clearSparseGradients(pRenderContext);
        }

        if (hasResidualBaseline)
        {
            // 每一帧都反传自己的 PathRecord 并原子累加到 gradBuffer.
            // 第 0 帧没有前缀可用（见 hasResidualBaseline），只当基线.
            mLossPass.mView = mOptimizerParams.currentView;
            runLossPass(pRenderContext, renderData);
            runGradientPass(pRenderContext, renderData);
        }

        if (isLastSample)
        {
            runUpdatePass(pRenderContext, renderData);
            runReducePass(pRenderContext, renderData);


            mRayMarchingPass.mSampleIndex = 0;
            mOptimizerParams.currentView++;
            if (mOptimizerParams.currentView >= mOptimizerParams.viewsPerIteration)
            {
                mOptimizerParams.currentView = 0;
                mOptimizerParams.currentIteration++;
                if (mGridResources.radianceAdamCounter)
                {
                    uint32_t counter[2] = {};
                    mGridResources.radianceAdamCounter->getBlob(counter, 0, sizeof(counter));
                    const uint32_t capacity = uint32_t(mGridResources.radianceAdamPages.size()) * SPARSE_POOL_PAGE_SIZE;
                    if (counter[1])
                    {
                        if (capacity < mGridResources.gridData.activeVoxelCount)
                        {
                            const uint32_t target = uint32_t(std::min<uint64_t>(
                                mGridResources.gridData.activeVoxelCount,
                                std::max<uint64_t>(uint64_t(capacity) + SPARSE_POOL_PAGE_SIZE,
                                    uint64_t(capacity) * 6u / 5u)));
                            reserveRadianceAdamCapacity(pRenderContext, target);
                        }
                        // Concurrent overflow attempts did not acquire a slot.
                        // Reuse the first free slot on the next view.
                        const uint32_t reset[2] = {std::min(counter[0], capacity), 0u};
                        mGridResources.radianceAdamCounter->setBlob(reset, 0, sizeof(reset));
                    }
                }
                evaluateDeletionEvidence(pRenderContext);
                const bool reachedMaximumIteration =
                    mOptimizerParams.currentIteration >= mOptimizerParams.maxIteration;
                const uint32_t evidenceInterval = std::max(1u, mTopologySettings.evidenceInterval);
                const uint32_t deletionInterval = std::max(1u, mTopologySettings.deletionInterval);
                const uint32_t firstDeletionIteration =
                    getDeletionEvidenceStartIteration() + 2u * evidenceInterval;
                const bool periodicDeletionBoundary =
                    mOptimizerParams.currentIteration >= firstDeletionIteration &&
                    (mOptimizerParams.currentIteration - firstDeletionIteration) % deletionInterval == 0u;

                const bool hadCandidatesAtBoundary = mTopologySettings.candidateCount > 0u;
                if (hadCandidatesAtBoundary &&
                    (periodicDeletionBoundary || reachedMaximumIteration))
                {
                    try
                    {
                        deleteAndCompactCandidates(pRenderContext);
                    }
                    catch (const std::exception& e)
                    {
                        mTopologySettings.deletionStatus = std::string("Automatic compaction failed: ") + e.what();
                        logError("{}", mTopologySettings.deletionStatus);
                    }
                }

                // Do not save a final layer that has never been optimized.
                if (!reachedMaximumIteration && mOptimizerParams.isRunning)
                    growNeighborVoxels(pRenderContext);

                if (reachedMaximumIteration)
                {
                    stopReconstruction();
                    if (!hadCandidatesAtBoundary)
                        mTopologySettings.deletionStatus = "Maximum iteration reached; no additional confirmed candidates";
                    // Saving is deferred to the next frame, after compaction has
                    // produced a continuous active prefix and released pages.
                    mSaveReconstructionRequested = true;
                }
            }
        }
    }


    endFrame(pRenderContext);
}

void VoxelReconstructionNoLightTransport::renderUI(Gui::Widgets& widget) {
    if (widget.checkbox("Check Primitive", mRayMarchingPass.mCheckPrimitive))
        mRayMarchingPass.mOptionsChanged = true;
    if (widget.dropdown("Draw Mode", reinterpret_cast<ABSDFDrawMode&>(mRayMarchingPass.mDrawMode)))
        mRayMarchingPass.mOptionsChanged = true;
    if (widget.checkbox("Render Background", mRayMarchingPass.mRenderBackGround))
        mRayMarchingPass.mOptionsChanged = true;



    if (widget.var("Alpha Loss Weight", mLossPass.alphaLossWeight, 0.0f, 10.0f, 0.01f))
    {
        // Restart the sample batch so accumulated gradients use a single loss weight.
        mRayMarchingPass.mSampleIndex = 0;
    }
    widget.var("Geometry Tau (voxels)", mGradientPass.geometryTauVoxelFraction, 0.001f, 1.0f, 0.005f, false, "%.4f");
    widget.text(fmt::format("Geometry Tau (world): {:.6f}", getGeometryTauWorld()));
    widget.var("Geometry Grad Clamp", mGradientPass.geometryGradClamp, 0.0f, 10.0f, 1e-4f);
    widget.var("Alpha Geometry Weight", mGradientPass.alphaGeometryWeight, 0.0f, 1.0f, 0.01f);
    // 改 Spp 后立刻重开一批：否则若 mSampleIndex 已经 >= 新的 Spp.
    // isLastSample 就会永远为假，训练卡在"index 一直涨、loss 和 update 再也不跑"的状态.
    if (widget.var("Spp", mRayMarchingPass.mSpp, 1u, 100u, 1u))
    {
        mRayMarchingPass.mSampleIndex = 0;
    }

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
    widget.text("Point-cloud initialization");
    widget.text("PLY: " + (resolveReconstructionPath(ReferenceImageDir) / InitializationPointCloudFile).string());
    widget.text("Gaussian Opacity Threshold applies only to PLY files with Gaussian attributes.");
    // 高斯占位阈值，改动在下次 Init / Reset from PLY 时生效.
    widget.var("Gaussian Opacity Threshold", mPointCloud.opacityThreshold, 0.001f, 1.0f, 0.005f);
    widget.text(mPointCloud.status);
    if (widget.button("Init / Reset from PLY")) mInitVoxelData = true;
    widget.var("Max Iteration", mOptimizerParams.maxIteration);
    if (widget.checkbox("Enable Reconstruction", mEnableReconstruction))
    {
        if (mEnableReconstruction)
            startReconstruction();
        else
            stopReconstruction();
    }
    if (!mEnableReconstruction)
    {
        static const Gui::DropdownList kViewingResolutions = {
            {0, "800 x 800"},
            {1, "1920 x 1080"},
        };
        widget.dropdown("Viewing Resolution", kViewingResolutions, mViewingResolution);
        updateOutputResolution();
    }

    renderUIUpdatePass(widget);
    renderUITopology(widget);

    widget.text("Voxel Size: " + ToString(mGridResources.gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)mGridResources.gridData.voxelCount));
    widget.text("Grid Min: " + ToString(mGridResources.gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(mGridResources.gridData.solidVoxelCount));
    widget.text(fmt::format("Compact Pool: {} active / {} capacity ({} pages)",
        mGridResources.gridData.activeVoxelCount,
        mGridResources.gridData.voxelCapacity,
        mGridResources.voxelPages.size()));
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    const double poolGiB = double(mGridResources.gridData.voxelCapacity) *
        double(sizeof(VoxelData) + sizeof(GradRecord) + sizeof(GeometryAdamState) +
            sizeof(TopologyEvidence) + 2u * sizeof(uint32_t)) / bytesPerGiB;
    const uint32_t radianceCapacity = uint32_t(mGridResources.radianceAdamPages.size()) * SPARSE_POOL_PAGE_SIZE;
    const double radianceGiB = double(radianceCapacity) * sizeof(RadianceAdamState) / bytesPerGiB;
    const auto& count = mGridResources.gridData.voxelCount;
    const double indexGiB = double(count.x) * double(count.y) * double(count.z) * sizeof(int32_t) / bytesPerGiB;
    widget.text(fmt::format("Grid GPU storage: {:.2f} GiB pool + {:.2f} GiB radiance Adam + {:.2f} GiB index",
        poolGiB, radianceGiB, indexGiB));
    widget.text(fmt::format("Radiance Adam capacity: {} slots", radianceCapacity));
    widget.text(fmt::format("Spatial Index: {} page(s), {}^3 cells/page max",
        mGridResources.indexPages.size(), SPARSE_INDEX_PAGE_EDGE));
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
        }
        else
        {
            widget.text("No reconstruction .bin files found.");
        }

        if (!mReconstructionIOStatus.empty()) widget.text(mReconstructionIOStatus);
        widget.textbox("Name Tag", mReconstructionNameTag);
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

void VoxelReconstructionNoLightTransport::renderUITopology(Gui::Widgets& widget)
{
    auto group = widget.group("Topology", true);
    if (!group) return;

    group.text("Confirmed candidates are deleted and compacted periodically at complete iteration boundaries.");
    group.text("TopologyDebug is a viewing mode; training always renders with Default.");

    if (group.button("Show TopologyDebug"))
    {
        mRayMarchingPass.mDrawMode = uint32_t(ABSDFDrawMode::TopologyDebug);
        mRayMarchingPass.mOptionsChanged = true;
    }
    if (group.button("Show Default"))
    {
        mRayMarchingPass.mDrawMode = uint32_t(ABSDFDrawMode::Default);
        mRayMarchingPass.mOptionsChanged = true;
    }
    if (group.button("Show Deletion Candidates"))
    {
        mTopologySettings.debugLayer = uint32_t(TopologyDebugLayer::Deletion);
        mRayMarchingPass.mDrawMode = uint32_t(ABSDFDrawMode::TopologyDebug);
        mRayMarchingPass.mOptionsChanged = true;
    }
    if (group.button(mEnableReconstruction ? "Stop and Show Grown Voxels" : "Show Grown Voxels"))
    {
        if (mEnableReconstruction) stopReconstruction();
        mTopologySettings.debugLayer = uint32_t(TopologyDebugLayer::Growth);
        mRayMarchingPass.mDrawMode = uint32_t(ABSDFDrawMode::TopologyDebug);
        mRayMarchingPass.mOptionsChanged = true;
    }

    if (group.dropdown("Debug Layer", reinterpret_cast<TopologyDebugLayer&>(mTopologySettings.debugLayer)))
        mRayMarchingPass.mOptionsChanged = true;
    group.text("Occupied: blue; grown voxels: green; deletion candidates: red; optional context: gray.");
    if (mTopologySettings.debugLayer == uint32_t(TopologyDebugLayer::Growth))
        group.text("Shows surviving grown cells across all growth rounds, independent of opacity. Markers are not saved in v3 bins.");
    if (mTopologySettings.debugLayer != uint32_t(TopologyDebugLayer::Occupied))
    {
        if (group.checkbox("Show Occupied Context", mTopologySettings.showOccupiedContext))
            mRayMarchingPass.mOptionsChanged = true;
    }

    group.text("Deletion evidence");
    group.checkbox("Collect Deletion Evidence", mTopologySettings.collectDeletionEvidence);
    group.text(fmt::format(
        "Starts at iteration {} (opacity warm-up + ramp); window {} iterations; one evidence sample per view/iteration.",
        getDeletionEvidenceStartIteration(),
        std::max(1u, mTopologySettings.evidenceInterval)
    ));
    group.var("Evidence Interval (iterations)", mTopologySettings.evidenceInterval, 1u, 100u, 1u);
    group.var("Deletion Interval (iterations)", mTopologySettings.deletionInterval, 1u, 100u, 1u);
    group.var("Foreground Alpha Min", mTopologySettings.foregroundAlphaMin, 0.0f, 1.0f, 0.01f);
    group.var("Background Alpha Max", mTopologySettings.backgroundAlphaMax, 0.0f, 0.1f, 1e-4f, false, "%.4f");
    group.var("Min Removal Loss Delta", mTopologySettings.minRemovalLossDelta, 0.0f, 0.1f, 1e-5f, false, "%.6f");
    group.var("Min Evidence Transmittance", mTopologySettings.minEvidenceTransmittance, 0.0f, 1.0f, 0.01f);
    group.var("Min Background Conflict Views", mTopologySettings.minDeletionConflictViews, 1u, 1000u, 1u);
    group.var("Max Foreground Support Views", mTopologySettings.maxDeletionSupportViews, 0u, 1000u, 1u);
    group.text(fmt::format("Completed evidence windows: {}", mTopologySettings.completedWindows));
    group.text(fmt::format("Deletion candidates (two windows): {}", mTopologySettings.candidateCount));
    group.text(fmt::format("Pending background strikes: {}", mTopologySettings.oneWindowCount));
    group.text(fmt::format("Foreground-protected: {}", mTopologySettings.protectedCount));
    group.text(fmt::format("Weak background conflict: {}", mTopologySettings.weakConflictCount));

    group.text(fmt::format(
        "Automatic deletion starts at iteration {} and then runs every {} iterations when candidates exist.",
        getDeletionEvidenceStartIteration() + 2u * std::max(1u, mTopologySettings.evidenceInterval),
        std::max(1u, mTopologySettings.deletionInterval)));
    group.text(mTopologySettings.deletionStatus);
    group.text(fmt::format(
        "Last compaction: {} voxels, {} pool pages, {} radiance-Adam pages released",
        mTopologySettings.lastDeletedCount,
        mTopologySettings.lastReleasedPoolPages,
        mTopologySettings.lastReleasedRadiancePages));

    group.text("Ellipsoid neighbor growth (parent retained)");
    group.checkbox("Enable Growth", mTopologySettings.enableGrowth);
    group.text(fmt::format("Starts at iteration {}; one layer per full iteration, after deletion; no count budget.",
        getDeletionEvidenceStartIteration()));
    group.var("Growth Face Penetration (voxel widths)", mTopologySettings.growthFacePenetration, 0.0f,
        float(GROWTH_MAX_FACE_PENETRATION_VOXELS), 0.01f);
    group.text("Grow only when the ellipsoid extends beyond this depth inside the empty face neighbor; lower is more aggressive.");
    group.var("Child Scale Multiplier", mTopologySettings.growthShrink, 0.01f, 0.99f, 0.01f);
    group.var("Child Contact Offset (voxels)", mTopologySettings.growthContactOffset, 0.01f, 0.49f, 0.01f);
    group.var("Child Max Initial Opacity", mTopologySettings.growthInitialOpacity, 0.01f, 0.49f, 0.01f);
    group.var("Newborn Growth Wait (iterations)", mTopologySettings.growthWaitIterations, 0u, 100u, 1u);
    group.text("Newborns keep optimizing but cannot grow until this many full rounds finish; independent of deletion protection.");
    group.var("Newborn Protection (iterations)", mTopologySettings.growthProtectionIterations, 1u, 100u, 1u);
    group.var("Deleted Cell Cooldown (iterations)", mTopologySettings.deletionCooldownIterations, 0u, 100u, 1u);
    group.text(mTopologySettings.growthStatus);
    group.text(fmt::format("Last growth: {} voxels; {} pool pages added",
        mTopologySettings.lastGrowthCount, mTopologySettings.lastGrowthPages));
}

void VoxelReconstructionNoLightTransport::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mLoadedReconstructionForViewing = false;
    mReconstructionIOStatus.clear();
    mReferenceDataError.clear();
    mPointCloud = {};
    mGridResources.gridData.solidVoxelCount = 0;
    mEnableReconstruction = false;
    mOptimizerParams.reset();
    UpdateVoxelGrid(mVoxelResolution);
    setupGridResouce(pRenderContext, true);



    // RayMarching
    createRayMarchingPassResource(pRenderContext);
    updateOutputResolution();

    // Loss Pass
    createLossPassResource(pRenderContext);

    // Gradient Pass
    createGradientPassResource(pRenderContext);

    //// Update Pass
    createUpdatePassResource(pRenderContext);

    createTopologyPassResource(pRenderContext);
    createDeletionPassResources();

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
    if (!mpGridBlock || forceReset)
    {
        GridData grid = mGridResources.gridData;
        grid.solidVoxelCount = 0;
        grid.activeVoxelCount = 0;
        auto resources = allocateSparseGrid(pRenderContext, grid, 1u);
        auto block = createSparseGridBlock(resources);
        commitSparseGrid(std::move(resources), block, mVoxelResolution);
    }
    return;
}



void VoxelReconstructionNoLightTransport::updateOutputResolution()
{
    bool training = mEnableReconstruction || mOptimizerParams.isRunning;
    training |= mPointCloud.startRequested;
    const uint2 resolution = training ? uint2(800, 800) :
        (mViewingResolution == 0 ? uint2(800, 800) : uint2(1920, 1080));
    if (any(mRayMarchingPass.mOutputResolution != resolution))
    {
        mRayMarchingPass.mOutputResolution = resolution;
        mRayMarchingPass.mSampleIndex = 0;
        mRayMarchingPass.mOptionsChanged = true;
        requestRecompile();
    }
}

void VoxelReconstructionNoLightTransport::startReconstruction()
{
    mReferenceDataError.clear();
    updateOutputResolution();
    // GPU work and first-use initialization are performed at the next frame boundary.
    mPointCloud.startRequested = true;

}

void VoxelReconstructionNoLightTransport::stopReconstruction()
{
    mPointCloud.startRequested = false;
    mEnableReconstruction = false;
    mOptimizerParams.isRunning = false;
    mOptimizerParams.currentView = 0;
    mRayMarchingPass.mSampleIndex = 0;
    mPointCloud.clearAccumulation = true;
    updateOutputResolution();
}

bool VoxelReconstructionNoLightTransport::onMouseEvent(const MouseEvent& mouseEvent)
{
    bool ret = mpPixelDebug->onMouseEvent(mouseEvent);

    return ret;
}
