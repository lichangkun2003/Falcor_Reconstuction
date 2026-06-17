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

void VoxelReconstructionNoLightTransport::createLossPassResource(RenderContext* pRenderContext)
{
    mLossPass.init();

    {
        ProgramDesc desc;
        desc.addShaderLibrary(LossPassShaderFilePath).csEntry("main");
        DefineList defines;
        mLossPass.mpComputePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mLossPass.lossBuffer = mpDevice->createTexture2D(
        mRayMarchingPass.mOutputResolution.x,
        mRayMarchingPass.mOutputResolution.y,
        ResourceFormat::R32Float,
        1u,
        1u,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mLossPass.dL_dColor = mpDevice->createTexture2D(
        mRayMarchingPass.mOutputResolution.x,
        mRayMarchingPass.mOutputResolution.y,
        ResourceFormat::RGBA32Float,
        1u,
        1u,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
}

void VoxelReconstructionNoLightTransport::runLossPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->clearTexture(mLossPass.lossBuffer.get());
    pRenderContext->clearTexture(mLossPass.dL_dColor.get());

    auto var = mLossPass.mpComputePass->getRootVar();

    var["gRenderedColor"] = renderData.getTexture(kOutputColor);
    var["gReferenceImage"] = mReferenceImages[mLossPass.mView];
    var["gLossBuffer"] = mLossPass.lossBuffer;
    var["gDL_dColorBuffer"] = mLossPass.dL_dColor;

    auto cb = var["CB"];
    cb["gResolution"] = mRayMarchingPass.mOutputResolution;


    mLossPass.mpComputePass->execute(pRenderContext, uint3(mRayMarchingPass.mOutputResolution.x, mRayMarchingPass.mOutputResolution.y, 1)
    );

    pRenderContext->uavBarrier(mLossPass.lossBuffer.get());
    pRenderContext->uavBarrier(mLossPass.dL_dColor.get());
}



bool VoxelReconstructionNoLightTransport::loadReferenceCamerasFromFile(const std::string& cameraFile)
{
    std::ifstream file(cameraFile);
    if (!file.is_open())
    {
        logError(std::string("Failed to open reference camera file: ") + cameraFile);
        return false;
    }

    mReferenceCameras.clear();

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        // 跳过注释行
        if (line[0] == '#')
            continue;

        // 只解析这种完整格式：
        // Index: 000 Pos: px py pz Tar: tx ty tz Up: ux uy uz
        if (line.find("Index:") != 0)
            continue;

        std::stringstream ss(line);

        std::string indexLabel;
        std::string posLabel;
        std::string tarLabel;
        std::string upLabel;

        uint32_t index = 0;
        float3 pos = float3(0.f);
        float3 target = float3(0.f);
        float3 up = float3(0.f, 1.f, 0.f);

        ss >> indexLabel >> index >> posLabel >> pos.x >> pos.y >> pos.z >> tarLabel >> target.x >> target.y >> target.z >> upLabel >>
            up.x >> up.y >> up.z;

        if (!ss)
        {
            logWarning(std::string("Failed to parse camera line: ") + line);
            continue;
        }

        if (indexLabel != "Index:" || posLabel != "Pos:" || tarLabel != "Tar:" || upLabel != "Up:")
        {
            logWarning(std::string("Invalid camera line format: ") + line);
            continue;
        }

        ref<Camera> cam = Camera::create();

        cam->setPosition(pos);
        cam->setTarget(target);
        cam->setUpVector(up);

        if (mpScene && mpScene->getCamera())
        {
            ref<Camera> sceneCam = mpScene->getCamera();

            cam->setAspectRatio(sceneCam->getAspectRatio());
            cam->setFocalLength(sceneCam->getFocalLength());
        }

        mReferenceCameras.push_back(cam);
        if (mReferenceCameras.size() >= mOptimizerParams.viewsPerIteration)
        {
            break;
        }
    }

    if (mReferenceCameras.empty())
    {
        logError(std::string("No valid cameras loaded from file: ") + cameraFile);
        return false;
    }

    {
        std::stringstream msg;
        msg << "Loaded " << mReferenceCameras.size() << " reference cameras from " << cameraFile;
        logInfo(msg.str());
    }

    return true;
}

void VoxelReconstructionNoLightTransport::loadReferenceImages()
{
    //if (!mReferenceImages.empty() && !mReferenceCameras.empty())
    //    return;

    if (!mReferenceImages.empty())
        return;

    mReferenceImages.clear();
    mReferenceCameras.clear();

    const std::filesystem::path imageDir = ReferenceImageDir;
    const std::filesystem::path cameraFile = ReferenceCameraFile;

    const uint32_t imageCount = mOptimizerParams.viewsPerIteration;

    if (!loadReferenceCamerasFromFile(cameraFile.string()))
    {
        throw RuntimeError("Failed to load reference cameras.");
    }

    if (mReferenceCameras.size() < imageCount)
    {
        std::stringstream err;
        err << "Reference camera count is smaller than image count. camera count = " << mReferenceCameras.size()
            << ", image count = " << imageCount;

        throw RuntimeError(err.str().c_str());
    }

    mReferenceImages.reserve(imageCount);

    for (uint32_t i = 0; i < imageCount; ++i)
    {
        std::stringstream ss;
        ss << "picture" << std::setw(3) << std::setfill('0') << i << ".exr";
        //ss << "white.exr";

        std::filesystem::path imagePath = imageDir / ss.str();
        imagePath = std::filesystem::weakly_canonical(imagePath);

        ref<Texture> image = Texture::createFromFile(mpDevice, imagePath, false, false);
         logInfo(
            "Reference texture created: size={}x{}, format={}", image->getWidth(), image->getHeight(), to_string(image->getFormat())
        );

        if (!image)
        {
            std::string msg = "Failed to load reference image: " + imagePath.string();
            throw RuntimeError(msg.c_str());
        }

        mReferenceImages.push_back(image);
    }

    if (mReferenceCameras.size() > imageCount)
    {
        mReferenceCameras.resize(imageCount);
    }

    logInfo("Reference dataset loaded.");
}


