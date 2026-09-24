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

void VoxelReconstructionNoLightTransport::replaceReconstructionGrid(
    RenderContext* pRenderContext, const GridData& grid, uint32_t resolution, const void* voxelData, size_t byteSize)
{
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    GridResources next;
    next.gridData = grid;
    const uint32_t elementCount = next.gridData.totalVoxelCount();
    next.gridDataBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), elementCount, flags);
    next.vBuffer = mpDevice->createTexture3D(grid.voxelCount.x, grid.voxelCount.y, grid.voxelCount.z,
        ResourceFormat::R32Int, 1u, nullptr, flags);
    auto gradient = mpDevice->createStructuredBuffer(sizeof(GradRecord), elementCount, flags);
    auto block = ParameterBlock::create(mpDevice,
        mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock"));
    auto var = block->getRootVar();
    var["gridDataBuffer"] = next.gridDataBuffer;
    var["vBuffer"] = next.vBuffer;
    var["voxelCount"] = grid.voxelCount;
    var["voxelSize"] = grid.voxelSize;
    var["gridMin"] = grid.gridMin;
    var["solidVoxelCount"] = grid.solidVoxelCount;
    if (voxelData)
    {
        if (byteSize != size_t(elementCount) * sizeof(VoxelData))
            throw RuntimeError("Voxel payload size does not match the replacement grid.");
        pRenderContext->updateBuffer(next.gridDataBuffer.get(), voxelData, 0, byteSize);
    }
    else pRenderContext->clearUAV(next.gridDataBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(next.vBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(gradient->getUAV().get(), uint4(0));
    pRenderContext->uavBarrier(next.gridDataBuffer.get());
    if (voxelData) pRenderContext->submit(true);

    // Allocate/upload first. Keep all program bindings consistent if rebinding fails.
    const auto bind = [&](const ref<ParameterBlock>& gridBlock, const ref<Buffer>& gradBuffer)
    {
        const auto bindGrid = [&](const auto& pass)
        {
            if (pass && pass->getVars()) pass->getRootVar()["gGridDataParamBlock"].setParameterBlock(gridBlock);
        };
        bindGrid(mpInitializeDataPass);
        bindGrid(mRayMarchingPass.mpFullScreenPass);
        bindGrid(mGradientPass.mpComputePass);
        bindGrid(mUpdatePass.mpComputePass);
#if RECON_MODE == RECON_MODE_POINT_CLOUD
        bindGrid(mpInitializePointCloudPass);
#endif
        if (mGradientPass.mpComputePass && mGradientPass.mpComputePass->getVars())
            mGradientPass.mpComputePass->getRootVar()["gGradBuffer"].setBuffer(gradBuffer);
        if (mUpdatePass.mpComputePass && mUpdatePass.mpComputePass->getVars())
            mUpdatePass.mpComputePass->getRootVar()["gGradBuffer"].setBuffer(gradBuffer);
    };
    try { bind(block, gradient); }
    catch (...)
    {
        if (mpGridBlock) bind(mpGridBlock, mGradientPass.gradBuffer);
        throw;
    }
    mGridResources = std::move(next);
    mGradientPass.gradBuffer = gradient;
    mpGridBlock = block;
    mVoxelResolution = resolution;
}

void VoxelReconstructionNoLightTransport::resetLoadedReconstruction(RenderContext* pRenderContext)
{
    mEnableReconstruction = false;
    mOptimizerParams.reset();
    mInitVoxelData = false;
    mSaveReconstructionRequested = false;
    mLoadReconstructionRequested = false;
    mLoadedReconstructionForViewing = true;
    mFrameCount = 0;
    mRayMarchingPass.mFrameIndex = 0;
    mRayMarchingPass.mSampleIndex = 0;
    mRayMarchingPass.mOptionsChanged = true;
    mLossPass.mView = 0;
    mReduceLossPass.meanLoss = 0.f;
    mReduceLossPass.iterationLossSum = 0.f;
    mReduceLossPass.iterationLossCount = 0;
    mReduceLossPass.iterationLossHistory.clear();
    if (mpPathRecordBuffer) pRenderContext->clearUAV(mpPathRecordBuffer->getUAV().get(), uint4(0));
    if (mGradientPass.gradBuffer) pRenderContext->clearUAV(mGradientPass.gradBuffer->getUAV().get(), uint4(0));
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    mPointCloud.startRequested = false;
    mPointCloud.initialized = true;
    mPointCloud.clearAccumulation = true;
    mPointCloud.status = "Loaded voxel reconstruction; PLY initialization is not required.";
#endif
}

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
