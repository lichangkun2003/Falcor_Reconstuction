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
        mGridResources.gridData.solidVoxelCount = GRID_RESOLUTION == 128 ? 92562 : 545771;
    }

    // Create Grid pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(ReflectTypesShaderFilePath).csEntry("main");
        DefineList defines;
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, true);
    }

    // Create ProcessXuData pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(ProcessXuDataShaderFilePath).csEntry("main");
        DefineList defines;
        mpProcessXuDataPass = ComputePass::create(mpDevice, desc, defines, true);
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
    // Input
    reflector.addInput(kVBuffer, kVBuffer)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::R32Uint)
        .texture3D();

    reflector.addInput(kGBuffer, kGBuffer)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::Unknown)
        .rawBuffer(mGridResources.gridData.solidVoxelCount * sizeof(PrimitiveBSDF));

    reflector.addInput(kPBuffer, kPBuffer)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::Unknown)
        .rawBuffer(mGridResources.gridData.solidVoxelCount * sizeof(Ellipsoid));

    reflector.addInput(kBlockMap, kBlockMap)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Uint)
        .texture2D();

    // Output
    reflector.addOutput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mRayMarchingPass.mOutputResolution.x, mRayMarchingPass.mOutputResolution.y, 1, 1);
    reflector.addOutput(kOutputColor, "Color")
        .bindFlags(ResourceBindFlags::RenderTarget)
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

    if (mInitVoxelData)
    {
        proccessXuData(pRenderContext, renderData);
        mInitVoxelData = false;
    }

    if (mLoadReconstructionRequested)
    {
        if (!mReconstructionFilePaths.empty() && mSelectedReconstructionFile < mReconstructionFilePaths.size())
        {
            loadReconstruction(pRenderContext, mReconstructionFilePaths[mSelectedReconstructionFile]);
        }
        else
        {
            logWarning("Load reconstruction failed: no file selected.");
        }

        mLoadReconstructionRequested = false;
    }

    if (mSaveReconstructionRequested)
    {
        saveReconstruction(pRenderContext);
        mSaveReconstructionRequested = false;

        // 保存后刷新列表，立刻能在 dropdown 看到新文件
        mReconstructionFileListDirty = true;
    }

    // test input
    {
        ref<Texture> pDummy = renderData.getTexture("dummy");
        ref<Texture> pRef = mReferenceImages[testIndex];

        if (pDummy && pRef)
        {
            pRenderContext->blit(pRef->getSRV(), pDummy->getRTV());
        }
    }

    rayMarchingPass(pRenderContext, renderData);

    if (mEnableReconstruction && mOptimizerParams.isRunning)
    {
        mLossPass.mView = mOptimizerParams.currentView;
        runLossPass(pRenderContext, renderData);
        runGradientPass(pRenderContext, renderData);
        runUpdatePass(pRenderContext, renderData);
        runReducePass(pRenderContext, renderData);

        mOptimizerParams.currentView++;
        if (mOptimizerParams.currentView >= mOptimizerParams.viewsPerIteration)
        {
            mOptimizerParams.currentView = 0;
            mOptimizerParams.currentIteration++;
            if (mOptimizerParams.currentIteration >= mOptimizerParams.maxIteration)
            {
                stopReconstruction();
                mSaveReconstructionRequested = true;
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

    if (widget.var("Solid Voxel Count", mGridResources.gridData.solidVoxelCount))
    {
        requestRecompile();
    }


    widget.var("Geometry Tau", mGradientPass.geometryTau, 0.0f, 0.2f, 1e-4f);
    widget.var("Geometry Grad Clamp", mGradientPass.geometryTau, 0.0f, 10.0f, 1e-4f);

    widget.slider("Camera Index", testIndex, 0u, mOptimizerParams.viewsPerIteration - 1u);
    widget.checkbox("Init Voxel Data", mInitVoxelData);

    widget.var("Max Iteration", mOptimizerParams.maxIteration);
    if (widget.checkbox("Enable Reconstruction", mEnableReconstruction))
    {
        if (mEnableReconstruction)
            startReconstruction();
        else
            stopReconstruction();
    }

    renderUIUpdatePass(widget);

    widget.text("Voxel Size: " + ToString(mGridResources.gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)mGridResources.gridData.voxelCount));
    widget.text("Block Count: " + ToString((int3)mGridResources.gridData.blockCount3D()));
    widget.text("Grid Min: " + ToString(mGridResources.gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(mGridResources.gridData.solidVoxelCount));
    widget.text(
        "Solid Rate: " + std::to_string(mGridResources.gridData.solidVoxelCount / (float)mGridResources.gridData.totalVoxelCount())
    );

    if (auto group = widget.group("Reconstruction"))
    {
        widget.text("Current iteration: " + std::to_string(mOptimizerParams.currentIteration));
        widget.text("Current view: " + std::to_string(mOptimizerParams.currentView));
        widget.text("Is running: " + std::string(mOptimizerParams.isRunning ? "true" : "false"));
        widget.text(fmt::format("Mean loss: {:.8f}", mReduceLossPass.meanLoss));
    }

    if (auto group = widget.group("Reconstruction IO"))
    {
        if (mReconstructionFileListDirty)
        {
            refreshReconstructionFileList();
            mReconstructionFileListDirty = false;
        }

        Gui::DropdownList fileList;

        for (uint32_t i = 0; i < mReconstructionFilePaths.size(); i++)
        {
            fileList.push_back({i, mReconstructionFilePaths[i].filename().string()});
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

}


void VoxelReconstructionNoLightTransport::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    UpdateVoxelGrid(mpScene, mVoxelResolution);
    setupGridResouce(pRenderContext, true);

    //// RayMarching
    createRayMarchingPassResource(pRenderContext);

    // Loss Pass
    createLossPassResource(pRenderContext);

    // Gradient Pass
    createGradientPassResource(pRenderContext);

    //// Update Pass
    createUpdatePassResource(pRenderContext);

    // Reduce Pass
    createReducePassResource(pRenderContext);

    loadReferenceImages();
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

void VoxelReconstructionNoLightTransport::UpdateVoxelGrid(ref<Scene> scene, uint voxelResolution)
{
    float3 diag;
    float length;
    float3 center;
    if (scene)
    {
        AABB aabb = scene->getSceneBounds();
        diag = aabb.maxPoint - aabb.minPoint;
        length = std::max(diag.z, std::max(diag.x, diag.y));
        center = aabb.center();
        diag *= 1.02f;
        length *= 1.02f;
    }
    else
    {
        diag = float3(1);
        length = 1.f;
        center = float3(0);
    }

    mGridResources.gridData.voxelSize = float3(length / voxelResolution);
    float3 temp = diag / mGridResources.gridData.voxelSize;

    mGridResources.gridData.voxelCount = uint3(
        (uint)math::ceil(temp.x / MinFactor.x) * MinFactor.x,
        (uint)math::ceil(temp.y / MinFactor.y) * MinFactor.y,
        (uint)math::ceil(temp.z / MinFactor.z) * MinFactor.z
    );
    mGridResources.gridData.gridMin = center - 0.5f * mGridResources.gridData.voxelSize * float3(mGridResources.gridData.voxelCount);
    // mGridResources.gridData.solidVoxelCount = 0;
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
        mGridResources.blockOM = mpDevice->createTexture2D(
            mGridResources.gridData.blockGridSizeXY().x,
            mGridResources.gridData.blockGridSizeXY().y,
            ResourceFormat::RGBA32Uint,
            1u,
            1u,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        pRenderContext->clearUAV(mGridResources.gridDataBuffer->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(mGridResources.blockOM->getUAV().get(), uint4(0));
    }

    gridBlock["gridDataBuffer"] = mGridResources.gridDataBuffer;
    // gridBlock["blockOM"] = mGridResources.blockOM;
    gridBlock["voxelCount"] = mGridResources.gridData.voxelCount;
    gridBlock["voxelSize"] = mGridResources.gridData.voxelSize;
    gridBlock["gridMin"] = mGridResources.gridData.gridMin;
    gridBlock["solidVoxelCount"] = mGridResources.gridData.solidVoxelCount;
}



void VoxelReconstructionNoLightTransport::proccessXuData(RenderContext* pRenderContext, const RenderData& renderData)
{
    //pRenderContext->clearUAV(mGridResources.gridDataBuffer->getUAV().get(), uint4(0));
    //pRenderContext->clearUAV(mGridResources.blockOM->getUAV().get(), uint4(0));

    auto var = mpProcessXuDataPass->getRootVar();
    var[kVBuffer] = renderData.getTexture(kVBuffer);
    var[kGBuffer] = renderData.getResource(kGBuffer)->asBuffer();
    var[kPBuffer] = renderData.getResource(kPBuffer)->asBuffer();
    var[kBlockMap] = renderData.getTexture(kBlockMap);
    var["gGridDataParamBlock"] = mpGridBlock;

    auto cb = var["GridData"];
    cb["gLrCenter"] = mUpdatePass.mLrCenter;
    cb["gLrB"] = mUpdatePass.mLrB;

    ShaderVar gridBlock = mpGridBlock->getRootVar();
    gridBlock["blockOM"] = renderData.getTexture(kBlockMap);

    mpProcessXuDataPass->execute(pRenderContext, mGridResources.gridData.voxelCount);
}



void VoxelReconstructionNoLightTransport::startReconstruction()
{
    mEnableReconstruction = true;

    mOptimizerParams.isRunning = true;
    mOptimizerParams.currentIteration = 0;
    mOptimizerParams.currentView = 0;

    // 如果希望每次点击开始都重新初始化 voxel 数据
    // mInitVoxelData = true;
}

void VoxelReconstructionNoLightTransport::stopReconstruction()
{
    mEnableReconstruction = false;

    mOptimizerParams.isRunning = false;
    mOptimizerParams.currentIteration = 0;
    mOptimizerParams.currentView = 0;
}
