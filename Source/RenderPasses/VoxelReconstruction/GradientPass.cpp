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

void VoxelReconstruction::createGradientPassResource(RenderContext* pRenderContext) {
    mGradientPass.init();

    {
        ProgramDesc desc;
        desc.addShaderLibrary(VoxelPrime::GradientPassShaderFilePath).csEntry("main");
        //desc.setShaderModel(Falcor::ShaderModel::SM6_7);
        DefineList defines;
        mGradientPass.mpComputePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mGradientPass.gradBuffer = mpDevice->createStructuredBuffer(
        sizeof(GradRecord), mGridResources.gridData.totalVoxelCount(),
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );

    //mGradientPass.mpVoxelGradAccumBuffer =
    //    mpDevice->createStructuredBuffer(
    //        sizeof(GradRecord), mGridResources.gridData.totalVoxelCount(),
    //        ResourceBindFlags::UnorderedAccess
    //    );

}
void VoxelReconstruction::runGradientPass(RenderContext* pRenderContext, const RenderData& renderData) {
    pRenderContext->clearUAV(mGradientPass.gradBuffer->getUAV().get(), uint4(0));

    mGradientPass.mpComputePass->addDefine("CHECK_VISIBILITY", mRayMarchingPassParams.mCheckVisibility ? "1" : "0");
    mGradientPass.mpComputePass->addDefine("CHECK_COVERAGE", mRayMarchingPassParams.mCheckCoverage ? "1" : "0");

    auto var = mGradientPass.mpComputePass->getRootVar();

    var["gDL_dColorBuffer"] = mLossPass.dL_dColor;
    var["gGradBuffer"] = mGradientPass.gradBuffer;
    var["gPathRecordBuffer"] = mpPathRecordBuffer;


    auto cb = var["CB"];
    cb["gResolution"] = mRayMarchingPassParams.mOutputResolution;
    cb["gVoxelCount"] = mGridResources.gridData.voxelCount;
    cb["trasmittanceThreshold"] = mRayMarchingPassParams.mTrasmittanceThreshold100 / 100;

    mpPixelDebug->prepareProgram(mGradientPass.mpComputePass->getProgram(), mGradientPass.mpComputePass->getRootVar());

    mGradientPass.mpComputePass->execute(
        pRenderContext, uint3(mRayMarchingPassParams.mOutputResolution.x, mRayMarchingPassParams.mOutputResolution.y, 1)
    );

}
