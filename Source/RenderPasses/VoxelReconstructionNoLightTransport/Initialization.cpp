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

DefineList VoxelReconstructionNoLightTransport::getReconstructionDefines()
{
    DefineList defines;
    defines.add("RECON_MODE", std::to_string(RECON_MODE));
    defines.add("GRID_RESOLUTION", std::to_string(GRID_RESOLUTION));
    return defines;
}

void VoxelReconstructionNoLightTransport::createInitializationPassResource()
{
    ProgramDesc desc;
    desc.addShaderLibrary(InitializeDataShaderFilePath).csEntry("main");
    DefineList defines = getReconstructionDefines();
    mpInitializeDataPass = ComputePass::create(mpDevice, desc, defines, true);
}

void VoxelReconstructionNoLightTransport::initializeVoxelData(RenderContext* pRenderContext)
{
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    initializePointCloudVoxelData(pRenderContext);
#elif RECON_MODE == RECON_MODE_ORIGINAL
    initializeOriginalVoxelData(pRenderContext);
#elif RECON_MODE == RECON_MODE_COARSE_TO_FINE
    initializeCoarseVoxelData(pRenderContext);
#endif
}

void VoxelReconstructionNoLightTransport::initializeOriginalVoxelData(RenderContext* pRenderContext)
{
    // Preserve the original shader's learning-rate-dependent initialization.
    // In particular, do not clear gridDataBuffer: parameters may be retained when their learning rate is zero.
    auto var = mpInitializeDataPass->getRootVar();
    var["gGridDataParamBlock"] = mpGridBlock;

    auto cb = var["GridData"];
    cb["gLrCenter"] = mUpdatePass.mLrCenter;
    cb["gLrB"] = mUpdatePass.mLrB;
    cb["gLrRadiance"] = mUpdatePass.mLrRadiance;
    cb["gLrOpacity"] = mUpdatePass.mLrOpacity;

    mpInitializeDataPass->execute(pRenderContext, mGridResources.gridData.voxelCount);
}
