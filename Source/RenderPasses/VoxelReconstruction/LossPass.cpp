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

void VoxelReconstruction::createLossPassResource(RenderContext* pRenderContext)
{
    mLossPass.init();

    {
        ProgramDesc desc;
        desc.addShaderLibrary(VoxelPrime::LossPassShaderFilePath).csEntry("main");
        DefineList defines;
        mLossPass.mpComputePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mLossPass.lossBuffer = mpDevice->createTexture2D(
        mRayMarchingPassParams.mOutputResolution.x,
        mRayMarchingPassParams.mOutputResolution.y,
        ResourceFormat::R32Float,
        1u,
        1u,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mLossPass.dL_dColor = mpDevice->createTexture2D(
        mRayMarchingPassParams.mOutputResolution.x,
        mRayMarchingPassParams.mOutputResolution.y,
        ResourceFormat::RGBA32Float,
        1u,
        1u,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
}

void VoxelReconstruction::runLossPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->clearTexture(mLossPass.lossBuffer.get());
    pRenderContext->clearTexture(mLossPass.dL_dColor.get());

    auto var = mLossPass.mpComputePass->getRootVar();

    var["gRenderedColor"] = renderData.getTexture(VoxelPrime::kOutputColor);
    var["gReferenceImage"] = mReferenceImages[mLossPass.mView];
    var["gLossBuffer"] = mLossPass.lossBuffer;
    var["gDL_dColorBuffer"] = mLossPass.dL_dColor;

    auto cb = var["CB"];
    cb["gResolution"] = mRayMarchingPassParams.mOutputResolution;
    mLossPass.mpComputePass->execute(
        pRenderContext, uint3(mRayMarchingPassParams.mOutputResolution.x, mRayMarchingPassParams.mOutputResolution.y, 1)
    );

}

void VoxelReconstruction::loadReferenceImages()
{
    // TODO

    if (mReferenceImages.empty())
    {
        ref<Texture> image = mpDevice->createTexture2D(
            mRayMarchingPassParams.mOutputResolution.x,
            mRayMarchingPassParams.mOutputResolution.y,
            ResourceFormat::RGBA32Uint,
            1u,
            1u,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        mReferenceImages.push_back(image);
    }
}
