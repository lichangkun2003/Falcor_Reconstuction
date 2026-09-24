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

void VoxelReconstructionNoLightTransport::createUpdatePassResource(RenderContext* pRenderContext)
{
    mUpdatePass.init();

    {
        ProgramDesc desc;
        desc.addShaderLibrary(UpdatePassShaderFilePath).csEntry("main");
        DefineList defines = getReconstructionDefines();
        mUpdatePass.mpComputePass = ComputePass::create(mpDevice, desc, defines, true);
    }

}

void VoxelReconstructionNoLightTransport::runUpdatePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    //mUpdatePass.mpComputePass->addDefine("CHECK_VISIBILITY", mRayMarchingPass.mCheckVisibility ? "1" : "0");
    //mUpdatePass.mpComputePass->addDefine("CHECK_COVERAGE", mRayMarchingPass.mCheckCoverage ? "1" : "0");

    // 几何学习率由 scale 派生，放在这里而不是 renderUIUpdatePass 里.
    // 因为面板收起时那个函数会提前 return，mLrCenter/mLrB 就会停在 init() 的 0.
    // 于是"几何学不学"隐式地取决于 UI 面板有没有展开.
    mUpdatePass.mLrCenter = mLrCenterScale * 1e-3f;
    mUpdatePass.mLrB = mLrBScale * 1e-1f;

    // prune 默认关闭. pruneEllipsoid 直接写 occupied = 0.
    // 而全工程只有初始化阶段会写回 1，被删的体素本次运行里永远回不来.
    if ((mOptimizerParams.currentIteration) % 10 == 0)
    {
        //mUpdatePass.mEnableEllipsoidPruning = true;
    }


    auto var = mUpdatePass.mpComputePass->getRootVar();


    var["gGridDataParamBlock"] = mpGridBlock;
    var["gGradBuffer"] = mGradientPass.gradBuffer;

    //var["gVoxelSHGrads"] = mpSceneGradients->getGradsBuffer(GradientType::VoxelSH);

    auto cb = var["CB"];
    cb["gUseGradCountNormalize"] = mUpdatePass.mUseGradCountNormalize;
    cb["gGradScale"] = mUpdatePass.mGradScale;
    cb["gLrRadiance"] = mUpdatePass.mLrRadiance;
    cb["gLrCenter"] = mUpdatePass.mLrCenter;
    cb["gLrB"] = mUpdatePass.mLrB;
    cb["gLrOpacity"] = mUpdatePass.mLrOpacity;
    //cb["gVoxelSHGradDim"] = mVoxelSHGradDim;
    cb["gEllipsoidPruneThreshold"] = mUpdatePass.mEllipsoidPruneThreshold;
    cb["gEnableEllipsoidPruning"] = mUpdatePass.mEnableEllipsoidPruning;

    //mpPixelDebug->prepareProgram(mUpdatePass.mpComputePass->getProgram(), mUpdatePass.mpComputePass->getRootVar());

    mUpdatePass.mpComputePass->execute(pRenderContext, mGridResources.gridData.voxelCount);


    pRenderContext->uavBarrier(mGridResources.gridDataBuffer.get());

    mUpdatePass.mEnableEllipsoidPruning = false;
}

void VoxelReconstructionNoLightTransport::renderUIUpdatePass(Gui::Widgets& widget)
{
    auto group = widget.group("Update Pass", true);
    if (!group)
        return;

    group.checkbox("Normalize by grad count", mUpdatePass.mUseGradCountNormalize);

    group.var("Grad scale", mUpdatePass.mGradScale, 0.0f, 10.0f, 0.001f);

    group.text("Appearance Learning rates");

    group.var("LR radiance", mUpdatePass.mLrRadiance, 0.0f, 1.0f, 1e-4f);
    group.var("LR opacity", mUpdatePass.mLrOpacity, 0.0f, 100.0f, 1e-4f);

    group.text("Geometry learning rates");


    // 实际取值在 runUpdatePass 里由 scale 派生，这里只改 scale，并显示派生结果.
    group.var("LR center scale(1000x)", mLrCenterScale, 0.0f, 1.0f, 1e-6f);

    group.var("LR B scale(10x)", mLrBScale, 0.0f, 1.0f, 1e-7f);

    // 显示的是 runUpdatePass 实际用的值：lrCenter = scale * 1e-3, lrB = scale * 1e-1.
    group.text("  -> LR center = " + std::to_string(mUpdatePass.mLrCenter) + ",  LR B = " + std::to_string(mUpdatePass.mLrB));

    group.var("Ellipsoid Prune Threshold", mUpdatePass.mEllipsoidPruneThreshold, 0.0f, 1.0f, 1e-7f);

}
