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
#include "VoxelReconstruction.h"


void VoxelReconstruction::createRayMarchingPassResource(RenderContext* pRenderContext)
{
    mRayMarchingPassResouce.init();
    mRayMarchingPassParams.init();

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mRayMarchingPassResouce.mpPointSampler = mpDevice->createSampler(samplerDesc);

    {
        ProgramDesc desc;
        desc.addShaderLibrary(VoxelPrime::RayMarchingShaderFilePath).psEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        desc.addTypeConformances(mpScene->getTypeConformances());
        mRayMarchingPassResouce.mpFullScreenPass = FullScreenPass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
}

void VoxelReconstruction::rayMarchingPass(RenderContext* pRenderContext, const RenderData& renderData)
{

    pRenderContext->clearUAV(mpPathRecordBuffer->getUAV().get(), uint4(0));

    RayMarchingPassResouce& resource = mRayMarchingPassResouce;
    RayMarchingPassParams& params = mRayMarchingPassParams;

    auto& dict = renderData.getDictionary();
    if (params.mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        params.mOptionsChanged = false;
    }

    ref<Camera> pCamera = mpScene->getCamera();
    ref<Texture> pOutputColor = renderData.getTexture(VoxelPrime::kOutputColor);
    pRenderContext->clearRtv(pOutputColor->getRTV().get(), float4(0));

    if (!params.mDisplayNDF)
    {
        resource.mpFullScreenPass->addDefine("CHECK_VISIBILITY", params.mCheckVisibility ? "1" : "0");
        resource.mpFullScreenPass->addDefine("CHECK_COVERAGE", params.mCheckCoverage ? "1" : "0");

        resource.mpFullScreenPass->addDefine("CHECK_PRIMITIVE", params.mCheckPrimitive ? "1" : "0");
        resource.mpFullScreenPass->addDefine("USE_MIP_MAP", params.mUseMipmap ? "1" : "0");

        ref<EnvMap> pEnvMap = mpScene->getEnvMap();
        resource.mpFullScreenPass->addDefine("USE_ENV_MAP", pEnvMap ? "1" : "0");
        if (pEnvMap)
        {
            if (!resource.mpEnvMapSampler || resource.mpEnvMapSampler->getEnvMap() != pEnvMap)
                resource.mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, pEnvMap);
        }

        // 必须在addDefine之后获取var
        auto var = resource.mpFullScreenPass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        if (pEnvMap)
            resource.mpEnvMapSampler->bindShaderData(var["gEnvMapSampler"]);

        var["gGridDataParamBlock"] = mpGridBlock;
        var["gPathRecordBuffer"] = mpPathRecordBuffer;

        auto cb_GridData = var["GridData"];
        cb_GridData["gridMin"] = mGridResources.gridData.gridMin;
        cb_GridData["voxelSize"] = mGridResources.gridData.voxelSize;
        cb_GridData["voxelCount"] = mGridResources.gridData.voxelCount;
        cb_GridData["solidVoxelCount"] = (uint)mGridResources.gridData.solidVoxelCount;

        auto cb = var["CB"];
        cb["pixelCount"] = params.mOutputResolution;
        cb["blockCount"] = mGridResources.gridData.blockCount3D();
        cb["invVP"] = math::inverse(pCamera->getViewProjMatrixNoJitter());
        cb["shadowBias"] = params.mShadowBias100 / 100 / mGridResources.gridData.voxelSize.x;
        cb["drawMode"] = params.mDrawMode;
        cb["maxBounce"] = params.mMaxBounce;
        cb["frameIndex"] = params.mFrameIndex;
        cb["minPdf"] = params.mMinPdf100 / 100;
        cb["trasmittanceThreshold"] = params.mTrasmittanceThreshold100 / 100;
        // cb["selectedPixel"] = mSelectedPixel;
        cb["renderBackGround"] = params.mRenderBackGround;
        cb["clearColor"] = float4(params.mClearColor, 0);
        cb["enableReconstruction"] = mEnableReconstruction;

        ref<Fbo> fbo = Fbo::create(mpDevice);
        fbo->attachColorTarget(pOutputColor, 0);
        resource.mpFullScreenPass->execute(pRenderContext, fbo);
    }
    else
    {
    }
}
