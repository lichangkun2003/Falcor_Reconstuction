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

void VoxelReconstructionNoLightTransport::createReducePassResource(RenderContext* pRenderContext)
{

    uint32_t width = mRayMarchingPass.mOutputResolution.x;
    uint32_t height = mRayMarchingPass.mOutputResolution.y;

    if (width == 0 || height == 0)
        return;

    uint32_t groupX = div_round_up(width, 16u);
    uint32_t groupY = div_round_up(height, 16u);
    uint32_t partialCount = std::max(groupX * groupY, 1u);

    // Reduce Texture Pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(ReduceTexturePassShaderFilePath).csEntry("main");
        DefineList defines;
        mReduceLossPass.mpReduceTexturePass = ComputePass::create(mpDevice, desc, defines);
    }

    // Reduce Buffer Pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(ReduceBufferPassShaderFilePath).csEntry("main");
        DefineList defines;
        mReduceLossPass.mpReduceBufferPass = ComputePass::create(mpDevice, desc, defines);
    }

    
    {
        mReduceLossPass.mpReduceBufferA = mpDevice->createStructuredBuffer(
            sizeof(float), partialCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );

        mReduceLossPass.mpReduceBufferB = mpDevice->createStructuredBuffer(
            sizeof(float), partialCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );

        mReduceLossPass.mpTotalLossReadback = mpDevice->createBuffer(sizeof(float), ResourceBindFlags::None, MemoryType::ReadBack);

    }



}
void VoxelReconstructionNoLightTransport::runReducePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    //pRenderContext->clearUAV(mReduceLossPass.mpReduceBufferA->getUAV().get(), float4(0.0f));
    //pRenderContext->clearUAV(mReduceLossPass.mpReduceBufferB->getUAV().get(), float4(0.0f));

    uint32_t width = mRayMarchingPass.mOutputResolution.x;
    uint32_t height = mRayMarchingPass.mOutputResolution.y;

    uint32_t groupX = div_round_up(width, 16u);
    uint32_t groupY = div_round_up(height, 16u);
    uint32_t partialCount = groupX * groupY;

    // 第一级：Texture2D<float> -> Buffer<float>
    auto var0 = mReduceLossPass.mpReduceTexturePass->getRootVar();
    var0["gLossBuffer"] = mLossPass.lossBuffer;
    var0["gPartialLoss"] = mReduceLossPass.mpReduceBufferA;

    auto cb0 = var0["CB"];
    cb0["gResolution"] = uint2(width, height);
    cb0["gNumGroups"] = uint2(groupX, groupY);

    mReduceLossPass.mpReduceTexturePass->execute(pRenderContext, width, height, 1);

    // 后续：Buffer<float> -> Buffer<float>
    uint32_t inputCount = partialCount;
    ref<Buffer> pInput = mReduceLossPass.mpReduceBufferA;
    ref<Buffer> pOutput = mReduceLossPass.mpReduceBufferB;

    while (inputCount > 1)
    {
        uint32_t outputCount = div_round_up(inputCount, 256u);

        auto var = mReduceLossPass.mpReduceBufferPass->getRootVar();
        var["gInput"] = pInput;
        var["gOutput"] = pOutput;
        auto cb = var["CB"];
        cb["gInputCount"] = inputCount;

        mReduceLossPass.mpReduceBufferPass->execute(pRenderContext, outputCount * 256u, 1, 1);

        std::swap(pInput, pOutput);
        inputCount = outputCount;

    }

    // 最后 pInput[0] 就是 total loss
    //mReduceLossPass.mpTotalLoss = pInput;

    // pInput[0] 就是最终 total loss
    pRenderContext->copyBufferRegion(mReduceLossPass.mpTotalLossReadback.get(), 0, pInput.get(), 0, sizeof(float));

    const float* pData = reinterpret_cast<const float*>(mReduceLossPass.mpTotalLossReadback->map());

    if (pData)
    {
        mReduceLossPass.meanLoss =
            pData[0] / std::max(float(mRayMarchingPass.mOutputResolution.x * mRayMarchingPass.mOutputResolution.y), 1.0f);
        mReduceLossPass.mpTotalLossReadback->unmap();
    }


}
