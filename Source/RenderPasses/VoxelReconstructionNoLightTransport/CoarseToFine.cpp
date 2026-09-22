#include "VoxelReconstructionNoLightTransport.h"

#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
namespace
{
const char* kRefineShader = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/RefineVoxelGrid.cs.slang";
const char* kEvaluationShader = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/EvaluationImages.cs.slang";
const auto kGridBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
}

uint32_t VoxelReconstructionNoLightTransport::getCoarseStageBudget(uint32_t stageIndex) const
{
    const uint32_t stageCount = uint32_t(mCoarseToFine.resolutions.size());
    if (stageCount == 0 || stageIndex >= stageCount || CTF_TOTAL_ITERATIONS < stageCount)
        throw RuntimeError("The total iteration budget must allow at least one iteration per stage.");
    // Reserve enough for every level, even when GRID_RESOLUTION adds more stages.
    uint32_t coarseBudget = std::min(uint32_t(CTF_REFINE_INTERVAL), uint32_t(CTF_TOTAL_ITERATIONS) / stageCount);
    if (coarseBudget >= CTF_PRUNE_INTERVAL)
        coarseBudget -= coarseBudget % CTF_PRUNE_INTERVAL;
    return stageIndex + 1 == stageCount ? CTF_TOTAL_ITERATIONS - coarseBudget * (stageCount - 1) : coarseBudget;
}

void VoxelReconstructionNoLightTransport::configureCoarseStages()
{
    const bool pauseAfterStage = mCoarseToFine.pauseAfterStage;
    mCoarseToFine = {};
    mCoarseToFine.pauseAfterStage = pauseAfterStage;
    for (uint32_t resolution = CTF_START_RESOLUTION;; resolution *= 2)
    {
        mCoarseToFine.resolutions.push_back(resolution);
        if (resolution == GRID_RESOLUTION) break;
    }
    mVoxelResolution = mCoarseToFine.resolutions.front();
    mCoarseToFine.stageBudget = getCoarseStageBudget(0);
    mOptimizerParams.reset();
    mEnableReconstruction = false;
    mSaveReconstructionRequested = false;
    mReduceLossPass.meanLoss = 0.f;
    mReduceLossPass.iterationLossSum = 0.f;
    mReduceLossPass.iterationLossCount = 0;
    mReduceLossPass.iterationLossHistory.clear();
}

void VoxelReconstructionNoLightTransport::replaceCoarseGrid(RenderContext* pRenderContext, const GridData& grid, uint32_t resolution)
{
    // Allocate everything before replacing the active grid. Optimizer settings and shader programs survive a resize.
    GridResources next;
    next.gridData = grid;
    next.gridDataBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), next.gridData.totalVoxelCount(), kGridBindFlags);
    next.vBuffer = mpDevice->createTexture3D(grid.voxelCount.x, grid.voxelCount.y, grid.voxelCount.z,
        ResourceFormat::R32Int, 1u, nullptr, kGridBindFlags);
    auto gradient = mpDevice->createStructuredBuffer(sizeof(GradRecord), next.gridData.totalVoxelCount(), kGridBindFlags);
    auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock");
    auto block = ParameterBlock::create(mpDevice, reflector);
    auto var = block->getRootVar();
    var["gridDataBuffer"] = next.gridDataBuffer;
    var["vBuffer"] = next.vBuffer;
    var["voxelCount"] = grid.voxelCount;
    var["voxelSize"] = grid.voxelSize;
    var["gridMin"] = grid.gridMin;
    var["solidVoxelCount"] = grid.solidVoxelCount;
    pRenderContext->clearUAV(next.gridDataBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(next.vBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(gradient->getUAV().get(), uint4(0));
    // Idle programs can otherwise retain a large previous grid after loading a smaller checkpoint.
    // Bind the replacement explicitly: this GFX backend cannot safely bind a null ParameterBlock.
    const auto rebindGrid = [&block](const auto& pass)
    {
        if (pass && pass->getVars()) pass->getRootVar()["gGridDataParamBlock"].setParameterBlock(block);
    };
    rebindGrid(mpInitializeDataPass);
    rebindGrid(mRayMarchingPass.mpFullScreenPass);
    rebindGrid(mGradientPass.mpComputePass);
    rebindGrid(mUpdatePass.mpComputePass);
    if (mGradientPass.mpComputePass && mGradientPass.mpComputePass->getVars())
        mGradientPass.mpComputePass->getRootVar()["gGradBuffer"].setBuffer(gradient);
    if (mUpdatePass.mpComputePass && mUpdatePass.mpComputePass->getVars())
        mUpdatePass.mpComputePass->getRootVar()["gGradBuffer"].setBuffer(gradient);
    mGridResources = std::move(next);
    mGradientPass.gradBuffer = gradient;
    mpGridBlock = block;
    mVoxelResolution = resolution;
    resetCoarseSampling(pRenderContext);
}

void VoxelReconstructionNoLightTransport::resetCoarseSampling(RenderContext* pRenderContext)
{
    mOptimizerParams.currentView = 0;
    mRayMarchingPass.mSampleIndex = 0;
    mRayMarchingPass.mOptionsChanged = true;
    mCoarseToFine.clearAccumulation = true;
    mReduceLossPass.iterationLossSum = 0.f;
    mReduceLossPass.iterationLossCount = 0;
    if (mpPathRecordBuffer) pRenderContext->clearUAV(mpPathRecordBuffer->getUAV().get(), uint4(0));
    if (mGradientPass.gradBuffer) pRenderContext->clearUAV(mGradientPass.gradBuffer->getUAV().get(), uint4(0));
}

bool VoxelReconstructionNoLightTransport::initializeCoarseVoxelData(RenderContext* pRenderContext)
{
    const auto previousState = mCoarseToFine;
    const auto previousGrid = mGridResources;
    const auto previousGradient = mGradientPass.gradBuffer;
    const auto previousBlock = mpGridBlock;
    const auto previousOptimizer = mOptimizerParams;
    const auto previousLoss = mReduceLossPass;
    const auto previousRay = mRayMarchingPass;
    const uint32_t previousResolution = mVoxelResolution, previousFrame = mFrameCount;
    try
    {
        configureCoarseStages();
        UpdateVoxelGrid(mVoxelResolution);
        mGridResources.gridData.solidVoxelCount = mGridResources.gridData.totalVoxelCount();
        replaceCoarseGrid(pRenderContext, mGridResources.gridData, mVoxelResolution);
        initializeOriginalVoxelData(pRenderContext);
        pRenderContext->uavBarrier(mGridResources.gridDataBuffer.get());
        pRenderContext->submit(true);
        mCoarseToFine.initialized = true;
        mCoarseToFine.paused = true;
        mFrameCount = 0;
        mRayMarchingPass.mFrameIndex = 0;
        ensureCoarseRunDirectory();
        logInfo("Coarse-to-fine initialized: {} -> {}, {} stages.", mVoxelResolution, GRID_RESOLUTION, mCoarseToFine.resolutions.size());
        return true;
    }
    catch (const std::exception& e)
    {
        mCoarseToFine = previousState;
        mGridResources = previousGrid;
        mGradientPass.gradBuffer = previousGradient;
        mpGridBlock = previousBlock;
        mOptimizerParams = previousOptimizer;
        mReduceLossPass = previousLoss;
        mRayMarchingPass = previousRay;
        mVoxelResolution = previousResolution;
        mFrameCount = previousFrame;
        mCoarseToFine.paused = true;
        mCoarseToFine.startRequested = false;
        mOptimizerParams.isRunning = false;
        mEnableReconstruction = false;
        logError("Coarse initialization failed; previous experiment retained: {}", e.what());
        return false;
    }
}

void VoxelReconstructionNoLightTransport::advanceCoarseStage(RenderContext* pRenderContext)
{
    auto& state = mCoarseToFine;
    if (!state.initialized || mOptimizerParams.isRunning || !state.lastSaveSucceeded ||
        state.stageIteration < state.stageBudget || state.stageIndex + 1 >= state.resolutions.size()) return;

    const auto previousGrid = mGridResources;
    const auto previousGradient = mGradientPass.gradBuffer;
    const auto previousBlock = mpGridBlock;
    const uint32_t previousResolution = mVoxelResolution;
    try
    {
        if (!mpRefineVoxelGridPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kRefineShader).csEntry("main");
            mpRefineVoxelGridPass = ComputePass::create(mpDevice, desc, getReconstructionDefines());
        }
        auto var = mpRefineVoxelGridPass->getRootVar();
        GridData nextGrid = previousGrid.gridData;
        nextGrid.voxelCount *= 2u;
        nextGrid.voxelSize *= 0.5f;
        nextGrid.solidVoxelCount = 0;
        // gridMin and voxelCount * voxelSize stay fixed in world space.
        replaceCoarseGrid(pRenderContext, nextGrid, state.resolutions[state.stageIndex + 1]);
        var["gSourceVoxels"] = previousGrid.gridDataBuffer;
        var["gDestinationVoxels"] = mGridResources.gridDataBuffer;
        var["CB"]["gSourceVoxelCount"] = previousGrid.gridData.voxelCount;
        var["CB"]["gDestinationVoxelCount"] = nextGrid.voxelCount;
        pRenderContext->uavBarrier(previousGrid.gridDataBuffer.get());
        mpRefineVoxelGridPass->execute(pRenderContext, nextGrid.voxelCount);
        pRenderContext->uavBarrier(mGridResources.gridDataBuffer.get());
        pRenderContext->submit(true);
        // Program variables also own references; release the old stage after GPU migration completes.
        var["gSourceVoxels"].setBuffer(nullptr);
        var["gDestinationVoxels"].setBuffer(nullptr);
        ++state.stageIndex;
        state.stageIteration = 0;
        state.stageBudget = getCoarseStageBudget(state.stageIndex);
        state.lastSaveSucceeded = false;
        state.lastCheckpointPath.clear();
        state.paused = false;
        state.finished = false;
        state.pauseRequested = false;
        mEnableReconstruction = true;
        mOptimizerParams.isRunning = true;
        logInfo("Refined to stage {}: resolution {}, budget {} complete iterations.", state.stageIndex + 1, mVoxelResolution, state.stageBudget);
    }
    catch (const std::exception& e)
    {
        mGridResources = previousGrid;
        mGradientPass.gradBuffer = previousGradient;
        mpGridBlock = previousBlock;
        mVoxelResolution = previousResolution;
        state.paused = true;
        mEnableReconstruction = false;
        mOptimizerParams.isRunning = false;
        logError("Refinement failed; the saved coarse stage is retained: {}", e.what());
    }
}

void VoxelReconstructionNoLightTransport::completeCoarseIteration(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& state = mCoarseToFine;
    ++state.stageIteration;
    state.lastSaveSucceeded = false;
    const float loss = mReduceLossPass.iterationLossHistory.empty() ? mReduceLossPass.meanLoss : mReduceLossPass.iterationLossHistory.back();
    state.lossHistory.push_back({mOptimizerParams.currentIteration, state.stageIndex, mVoxelResolution, state.stageIteration, loss});
    const bool stageComplete = state.stageIteration >= state.stageBudget;
    const bool pauseRequested = state.pauseRequested;
    if (!stageComplete && !pauseRequested && !mSaveReconstructionRequested) return;

    // Every checkpoint represents an entire sweep through the reference views.
    if (stageComplete || pauseRequested)
    {
        mEnableReconstruction = false;
        mOptimizerParams.isRunning = false;
        state.paused = true;
        state.finished = stageComplete && state.stageIndex + 1 == state.resolutions.size();
    }
    state.pauseRequested = false;
    mSaveReconstructionRequested = false;
    saveCoarseStage(pRenderContext, renderData);
    if (!state.lastSaveSucceeded)
    {
        mEnableReconstruction = false;
        mOptimizerParams.isRunning = false;
        state.paused = true;
    }
    else if (stageComplete && !state.finished && !state.pauseAfterStage && !pauseRequested)
        state.advanceRequested = true;
}

void VoxelReconstructionNoLightTransport::saveCoarseStage(RenderContext* pRenderContext, const RenderData& renderData)
{
    saveReconstruction(pRenderContext);
    if (!mCoarseToFine.lastSaveSucceeded) return;
    try
    {
        exportCoarseEvaluation(pRenderContext, renderData);
    }
    catch (const std::exception& e)
    {
        mCoarseToFine.lastSaveSucceeded = false;
        logError("Checkpoint saved, but evaluation export failed; retry Save Reconstruction before refinement: {}", e.what());
    }
    mReconstructionFileListDirty = true;
    if (mCoarseToFine.lastSaveSucceeded) recordCoarseStageResult();
}

void VoxelReconstructionNoLightTransport::renderCoarseUI(Gui::Widgets& widget)
{
    widget.text(fmt::format("Mode 2: {} / {} (target {})", mCoarseToFine.stageIndex + 1, mCoarseToFine.resolutions.size(), GRID_RESOLUTION));
    widget.text(fmt::format("Resolution: {}   Stage iterations: {} / {}", mVoxelResolution, mCoarseToFine.stageIteration, mCoarseToFine.stageBudget));
    widget.text(fmt::format("Default total budget: {} iterations (additional iterations extend it).", CTF_TOTAL_ITERATIONS));
    widget.text("One iteration visits every reference camera once.");
    widget.checkbox("Pause after each stage", mCoarseToFine.pauseAfterStage);
    renderCoarsePreviewUI(widget);
    if (mOptimizerParams.isRunning)
    {
        if (widget.button("Pause and save after this iteration")) stopReconstruction();
        if (mCoarseToFine.pauseRequested || mSaveReconstructionRequested)
            widget.text("Checkpoint pending: finishing the current iteration.");
        return;
    }
    if (widget.button("Initialize / reset coarse experiment")) mInitVoxelData = true;
    if (!mCoarseToFine.initialized || mCoarseToFine.stageIteration < mCoarseToFine.stageBudget)
    {
        if (widget.button(mCoarseToFine.initialized ? "Resume current stage" : "Start coarse-to-fine")) startReconstruction();
    }
    if (!mCoarseToFine.initialized) return;
    widget.var("Additional iterations", mCoarseToFine.extraIterations, 1u, 10000u, 1u);
    if (widget.button("Add iterations and continue this stage"))
    {
        mCoarseToFine.stageBudget = std::max(mCoarseToFine.stageBudget, mCoarseToFine.stageIteration) + mCoarseToFine.extraIterations;
        startReconstruction();
    }
    if (mCoarseToFine.stageIteration >= mCoarseToFine.stageBudget && mCoarseToFine.stageIndex + 1 < mCoarseToFine.resolutions.size())
    {
        if (mCoarseToFine.lastSaveSucceeded)
        {
            if (widget.button("Refine and start next stage")) mCoarseToFine.advanceRequested = true;
        }
        else widget.text("Save Reconstruction successfully before refining.");
    }
    if (mCoarseToFine.finished) widget.text("Final stage complete. You can add more iterations or load an earlier checkpoint.");
}

void VoxelReconstructionNoLightTransport::renderCoarsePreviewUI(Gui::Widgets& widget)
{
    auto& preview = mCoarseToFine.preview;
    Gui::DropdownList levels{{0u, "Current optimization level (live)"}};
    for (const auto& stage : preview.stages)
    {
        levels.push_back({stage.stageIndex + 1,
            fmt::format("Stage {}: resolution {} (saved at iteration {})", stage.stageIndex + 1, stage.resolution, stage.stageIteration)});
    }
    if (widget.dropdown("Display level", levels, preview.selectedStage))
    {
        preview.reloadRequested = true;
        preview.status.clear();
        mRayMarchingPass.mOptionsChanged = true;
    }
    widget.text("Completed stages appear after saving. Display selection does not change training.");
    if (preview.selectedStage != 0)
        widget.text("Viewing a saved stage. Use ReferenceCamera / Camera Index or move the scene camera.");
    if (!preview.status.empty()) widget.text(preview.status);
}

void VoxelReconstructionNoLightTransport::renderCoarsePreview(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& preview = mCoarseToFine.preview;
    if (preview.selectedStage == 0)
    {
        // Retain only the checkpoint list when returning to the live grid.
        preview.pass = nullptr;
        preview.gridBlock = nullptr;
        preview.grid = {};
        preview.accumulation = nullptr;
        preview.loadedPath.clear();
        preview.reloadRequested = false;
        return;
    }

    try
    {
        const auto stage = std::find_if(preview.stages.begin(), preview.stages.end(), [&](const CoarseStageResult& result)
            { return result.stageIndex + 1 == preview.selectedStage; });
        if (stage == preview.stages.end()) throw RuntimeError("The selected stage has no saved result.");
        if (preview.reloadRequested || preview.loadedPath != stage->path || !preview.gridBlock)
        {
            if (!loadCoarsePreview(pRenderContext, *stage))
            {
                preview.selectedStage = 0;
                mRayMarchingPass.mOptionsChanged = true;
                return;
            }
            preview.reloadRequested = false;
            auto& dict = renderData.getDictionary();
            dict[kRenderPassRefreshFlags] = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None) |
                RenderPassRefreshFlags::RenderOptionsChanged;
        }

        if (!preview.pass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(RayMarchingShaderFilePath).psEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            desc.addTypeConformances(mpScene->getTypeConformances());
            auto defines = mpScene->getSceneDefines();
            defines.add(getReconstructionDefines());
            defines.add("PREVIEW_ONLY", "1");
            preview.pass = FullScreenPass::create(mpDevice, desc, defines);
        }
        const auto& settings = mRayMarchingPass;
        const uint2 size = settings.mOutputResolution;
        if (!preview.accumulation || preview.accumulation->getWidth() != size.x || preview.accumulation->getHeight() != size.y)
            preview.accumulation = mpDevice->createTexture2D(size.x, size.y, ResourceFormat::RGBA32Float,
                1u, 1u, nullptr, kGridBindFlags);
        preview.pass->addDefine("CHECK_PRIMITIVE", settings.mCheckPrimitive ? "1" : "0");
        preview.pass->addDefine("USE_ENV_MAP", mpScene->getEnvMap() ? "1" : "0");
        preview.pass->addDefine("DIFF_MODE", "1");

        const auto camera = mUseReferenceCamera && testIndex < mReferenceCameras.size() ?
            mReferenceCameras[testIndex] : mpScene->getCamera();
        const auto& grid = preview.grid.gridData;
        auto var = preview.pass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        var["gGridDataParamBlock"] = preview.gridBlock;
        var["gAccuColor"] = preview.accumulation;
        var["GridData"]["gridMin"] = grid.gridMin;
        var["GridData"]["voxelSize"] = grid.voxelSize;
        var["GridData"]["voxelCount"] = grid.voxelCount;
        var["GridData"]["solidVoxelCount"] = grid.solidVoxelCount;
        auto cb = var["CB"];
        cb["pixelCount"] = size;
        cb["invVP"] = math::inverse(camera->getViewProjMatrixNoJitter());
        cb["shadowBias"] = settings.mShadowBias100 / 100.f / grid.voxelSize.x;
        cb["drawMode"] = settings.mDrawMode;
        cb["maxContributingVoxelCount"] = settings.mMaxContributingVoxelCount;
        cb["frameIndex"] = mFrameCount;
        cb["transmittanceThreshold"] = settings.mTransmittanceThreshold;
        cb["renderBackGround"] = settings.mRenderBackGround;
        cb["clearColor"] = float4(settings.mClearColor, 0);
        cb["enableReconstruction"] = false;
        cb["invSpp"] = 1.0f;

        // Private accumulation protects multi-sample training; no path buffer is bound for this program.
        const auto output = renderData.getTexture(kOutputColor);
        auto previewFbo = Fbo::create(mpDevice);
        // The shader writes both its RTV and accumulation UAV, so these must be distinct resources.
        previewFbo->attachColorTarget(output, 0);
        pRenderContext->clearUAV(preview.accumulation->getUAV().get(), float4(0));
        mpPixelDebug->prepareProgram(preview.pass->getProgram(), var);
        preview.pass->execute(pRenderContext, previewFbo);
        pRenderContext->copyResource(renderData.getTexture(kAccumulateOutputColor).get(), preview.accumulation.get());
    }
    catch (const std::exception& e)
    {
        preview.status = fmt::format("Preview unavailable: {}", e.what());
        preview.selectedStage = 0;
        mRayMarchingPass.mOptionsChanged = true;
        logWarning("{} Training state is unchanged.", preview.status);
    }
}

void VoxelReconstructionNoLightTransport::exportCoarseEvaluation(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Render the updated checkpoint with fixed cameras/seeds; training outputs precede the last parameter update.
    const auto imageDir = mCoarseToFine.runDirectory / "Images";
    std::filesystem::create_directories(imageDir);
    const auto stem = mCoarseToFine.lastCheckpointPath.stem().string();
    if (!mpEvaluationImagesPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kEvaluationShader).csEntry("main");
        mpEvaluationImagesPass = ComputePass::create(mpDevice, desc);
    }
    const uint2 size = mRayMarchingPass.mOutputResolution;
    auto rgb = mpDevice->createTexture2D(size.x, size.y, ResourceFormat::RGBA8Unorm, 1, 1, nullptr, kGridBindFlags);
    auto alpha = mpDevice->createTexture2D(size.x, size.y, ResourceFormat::RGBA8Unorm, 1, 1, nullptr, kGridBindFlags);
    std::ofstream csv(imageDir / (stem + "_evaluation.csv"));
    if (!csv) throw RuntimeError("Cannot create evaluation CSV.");
    csv << "view,spp,mean_loss\n" << std::setprecision(9);

    const bool enable = mEnableReconstruction;
    const uint32_t view = mOptimizerParams.currentView, lossView = mLossPass.mView;
    const uint32_t spp = mRayMarchingPass.mSpp, sample = mRayMarchingPass.mSampleIndex, frame = mRayMarchingPass.mFrameIndex;
    const auto restore = [&]()
    {
        mEnableReconstruction = enable;
        mOptimizerParams.currentView = view;
        mLossPass.mView = lossView;
        mRayMarchingPass.mSpp = spp;
        mRayMarchingPass.mSampleIndex = sample;
        mRayMarchingPass.mFrameIndex = frame;
        mCoarseToFine.clearAccumulation = true;
    };
    try
    {
        mEnableReconstruction = true;
        mRayMarchingPass.mSpp = CTF_EVALUATION_SPP;
        const uint32_t count = uint32_t(mReferenceCameras.size());
        std::vector<uint32_t> views{0u, count / 3u, 2u * count / 3u};
        views.erase(std::unique(views.begin(), views.end()), views.end());
        for (uint32_t evaluationView : views)
        {
            mOptimizerParams.currentView = evaluationView;
            mRayMarchingPass.mSampleIndex = 0;
            for (uint32_t i = 0; i < CTF_EVALUATION_SPP; ++i)
            {
                mRayMarchingPass.mFrameIndex = i;
                rayMarchingPass(pRenderContext, renderData);
            }
            mLossPass.mView = evaluationView;
            runLossPass(pRenderContext, renderData);
            auto var = mpEvaluationImagesPass->getRootVar();
            var["gColor"] = mRayMarchingPass.accuColor;
            var["gRgb"] = rgb;
            var["gAlpha"] = alpha;
            var["CB"]["gResolution"] = size;
            mpEvaluationImagesPass->execute(pRenderContext, size.x, size.y, 1);
            pRenderContext->submit(true);
            const std::string prefix = stem + "_view" + std::to_string(evaluationView);
            rgb->captureToFile(0, 0, imageDir / (prefix + "_rgb.png"), Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
            alpha->captureToFile(0, 0, imageDir / (prefix + "_alpha.png"), Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
            // Avoid runReducePass(): evaluation must not append points to training loss history.
            std::vector<float> losses(size_t(size.x) * size.y);
            mLossPass.lossBuffer->getSubresourceBlob(0, losses.data(), losses.size() * sizeof(float));
            double sum = 0.;
            for (float value : losses) sum += value;
            csv << evaluationView << ',' << CTF_EVALUATION_SPP << ',' << sum / double(losses.size()) << '\n';
        }
        csv.flush();
        if (!csv) throw RuntimeError("Failed to write evaluation CSV.");
    }
    catch (...)
    {
        restore();
        throw;
    }
    restore();
}
#endif
