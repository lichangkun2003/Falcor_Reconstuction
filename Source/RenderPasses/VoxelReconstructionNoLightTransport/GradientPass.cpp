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

void VoxelReconstructionNoLightTransport::createGradientPassResource(RenderContext* pRenderContext)
{
    mGradientPass.init();

    {
        ProgramDesc desc;
        desc.addShaderLibrary(GradientPassShaderFilePath).csEntry("main");
        //desc.setShaderModel(Falcor::ShaderModel::SM6_7);
        DefineList defines;
        defines.add("DIFF_MODE", "1");
        mGradientPass.mpComputePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mGradientPass.gradBuffer = mpDevice->createStructuredBuffer(
        sizeof(GradRecord), mGridResources.gridData.totalVoxelCount(),
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );

    //mGradientPass.gradBuffer = mpDevice->createStructuredBuffer(
    //    sizeof(GradRecord), mGridResources.gridData.solidVoxelCount,
    //    ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    //);

}

void VoxelReconstructionNoLightTransport::runGradientPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->clearUAV(mGradientPass.gradBuffer->getUAV().get(), uint4(0));
    pRenderContext->uavBarrier(mpPathRecordBuffer.get());

    //mpSceneGradients->clearGrads(pRenderContext, GradientType::VoxelSH);

    //mGradientPass.mpComputePass->addDefine("CHECK_VISIBILITY", mRayMarchingPass.mCheckVisibility ? "1" : "0");
    //mGradientPass.mpComputePass->addDefine("CHECK_COVERAGE", mRayMarchingPass.mCheckCoverage ? "1" : "0");

    auto var = mGradientPass.mpComputePass->getRootVar();

    //mpSceneGradients->bindShaderData(var["gSceneGradients"]);

    var["gGridDataParamBlock"] = mpGridBlock;
    var["gDL_dColorBuffer"] = mLossPass.dL_dColor;
    var["gGradBuffer"] = mGradientPass.gradBuffer;
    var["gPathRecordBuffer"] = mpPathRecordBuffer;
    var["dummy"] = renderData.getTexture("dummy");
    var["gBackgroundMaskBuffer"] = mLossPass.mpBackGroundMask;


    auto cb = var["CB"];
    cb["gResolution"] = mRayMarchingPass.mOutputResolution;
    cb["gVoxelCount"] = mGridResources.gridData.voxelCount;
    cb["gGeometryTau"] = mGradientPass.geometryTau;
    cb["gGeometryGradClamp"] = mGradientPass.geometryGradClamp;
    cb["gBackgroundCarveWeight"] = 0.01f;

    mpPixelDebug->prepareProgram(mGradientPass.mpComputePass->getProgram(), mGradientPass.mpComputePass->getRootVar());

    mGradientPass.mpComputePass->execute(
        pRenderContext, uint3(mRayMarchingPass.mOutputResolution.x, mRayMarchingPass.mOutputResolution.y, 1)
    );

    pRenderContext->uavBarrier(mGradientPass.gradBuffer.get());

    mpSceneGradients->aggregateGrads(pRenderContext, GradientType::VoxelSH);

}
