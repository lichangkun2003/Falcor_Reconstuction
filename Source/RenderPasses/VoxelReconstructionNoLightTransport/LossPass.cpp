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

    mLossPass.mpBackGroundMask = mpDevice->createTexture2D(
        mRayMarchingPass.mOutputResolution.x,
        mRayMarchingPass.mOutputResolution.y,
        ResourceFormat::R32Uint,
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
    //pRenderContext->clearTexture(mLossPass.mpBackGroundMask.get());

    auto var = mLossPass.mpComputePass->getRootVar();

    var["gRenderedColor"] = renderData.getTexture(kAccumulateOutputColor);
    // 本帧未乘 invSpp 的渲染结果 C_k，loss shader 用它剥出前缀平均.
    // kOutputColor 就是 rayMarchingPass 里 attach 到 FBO 的那张，每帧被 clear 后重写.
    // 注意 mSampleIndex 已经在本帧的 rayMarchingPass 里自增过，所以此刻它等于 k + 1.
    var["gCurrentFrameColor"] = renderData.getTexture(kOutputColor);
    var["gReferenceImage"] = mReferenceImages[mLossPass.mView];
    var["gLossBuffer"] = mLossPass.lossBuffer;
    var["gDL_dColorBuffer"] = mLossPass.dL_dColor;
    var["gBackgroundMaskBuffer"] = mLossPass.mpBackGroundMask;

    auto cb = var["CB"];
    cb["gResolution"] = mRayMarchingPass.mOutputResolution;
    cb["gSpp"] = mRayMarchingPass.mSpp;
    // 前缀平均的除数是"已累积采样数"，不是整批的 gSpp.
    // (gSpp * accu - C_k) 还原的是累积和，除以 (gSampleCount - 1) == k 才是前 k 帧的平均.
    cb["gSampleCount"] = mRayMarchingPass.mSampleIndex + 1u;


    mLossPass.mpComputePass->execute(pRenderContext, uint3(mRayMarchingPass.mOutputResolution.x, mRayMarchingPass.mOutputResolution.y, 1)
    );

    pRenderContext->uavBarrier(mLossPass.lossBuffer.get());
    pRenderContext->uavBarrier(mLossPass.dL_dColor.get());
}



//bool VoxelReconstructionNoLightTransport::loadReferenceCamerasFromFile(const std::string& cameraFile)
//{
//    std::ifstream file(cameraFile);
//    if (!file.is_open())
//    {
//        logError(std::string("Failed to open reference camera file: ") + cameraFile);
//        return false;
//    }
//
//    mReferenceCameras.clear();
//
//    std::string line;
//
//    while (std::getline(file, line))
//    {
//        if (line.empty())
//            continue;
//
//        // 跳过注释行
//        if (line[0] == '#')
//            continue;
//
//        // 只解析这种完整格式：
//        // Index: 000 Pos: px py pz Tar: tx ty tz Up: ux uy uz
//        if (line.find("Index:") != 0)
//            continue;
//
//        std::stringstream ss(line);
//
//        std::string indexLabel;
//        std::string posLabel;
//        std::string tarLabel;
//        std::string upLabel;
//
//        uint32_t index = 0;
//        float3 pos = float3(0.f);
//        float3 target = float3(0.f);
//        float3 up = float3(0.f, 1.f, 0.f);
//
//        ss >> indexLabel >> index >> posLabel >> pos.x >> pos.y >> pos.z >> tarLabel >> target.x >> target.y >> target.z >> upLabel >>
//            up.x >> up.y >> up.z;
//
//        if (!ss)
//        {
//            logWarning(std::string("Failed to parse camera line: ") + line);
//            continue;
//        }
//
//        if (indexLabel != "Index:" || posLabel != "Pos:" || tarLabel != "Tar:" || upLabel != "Up:")
//        {
//            logWarning(std::string("Invalid camera line format: ") + line);
//            continue;
//        }
//
//        ref<Camera> cam = Camera::create();
//
//        cam->setPosition(pos);
//        cam->setTarget(target);
//        cam->setUpVector(up);
//
//        if (mpScene && mpScene->getCamera())
//        {
//            ref<Camera> sceneCam = mpScene->getCamera();
//
//            cam->setAspectRatio(sceneCam->getAspectRatio());
//            cam->setFocalLength(sceneCam->getFocalLength());
//        }
//
//        mReferenceCameras.push_back(cam);
//        if (mReferenceCameras.size() >= mOptimizerParams.viewsPerIteration)
//        {
//            break;
//        }
//    }
//
//    if (mReferenceCameras.empty())
//    {
//        logError(std::string("No valid cameras loaded from file: ") + cameraFile);
//        return false;
//    }
//
//    {
//        std::stringstream msg;
//        msg << "Loaded " << mReferenceCameras.size() << " reference cameras from " << cameraFile;
//        logInfo(msg.str());
//    }
//
//    return true;
//}

//void VoxelReconstructionNoLightTransport::loadReferenceImages()
//{
//    const uint32_t imageCount = mOptimizerParams.viewsPerIteration;
//
//    // 全部图片已经加载完成。
//    if (mReferenceImages.size() >= imageCount)
//        return;
//
//    // 第一帧：读取相机文件并预留图片空间。
//    if (mReferenceImages.empty())
//    {
//        const std::filesystem::path cameraFile = ReferenceCameraFile;
//
//        mReferenceCameras.clear();
//
//        if (!loadReferenceCamerasFromFile(cameraFile.string()))
//        {
//            throw RuntimeError("Failed to load reference cameras.");
//        }
//
//        if (mReferenceCameras.size() < imageCount)
//        {
//            std::stringstream err;
//            err << "Reference camera count is smaller than image count. camera count = " << mReferenceCameras.size()
//                << ", image count = " << imageCount;
//
//            throw RuntimeError(err.str());
//        }
//
//        if (mReferenceCameras.size() > imageCount)
//        {
//            mReferenceCameras.resize(imageCount);
//        }
//
//        mReferenceImages.reserve(imageCount);
//    }
//
//    // 本帧需要读取的图片序号。
//    const uint32_t imageIndex = static_cast<uint32_t>(mReferenceImages.size());
//
//    std::stringstream ss;
//    ss << "picture" << std::setw(3) << std::setfill('0') << imageIndex << ".exr";
//
//    std::filesystem::path imagePath = std::filesystem::path(ReferenceImageDir) / ss.str();
//
//    imagePath = std::filesystem::weakly_canonical(imagePath);
//
//    ref<Texture> image = Texture::createFromFile(mpDevice, imagePath, false, false);
//
//    if (!image)
//    {
//        throw RuntimeError("Failed to load reference image: " + imagePath.string());
//    }
//
//    logInfo(
//        "Reference image loaded: {}/{}, size={}x{}, format={}",
//        imageIndex + 1,
//        imageCount,
//        image->getWidth(),
//        image->getHeight(),
//        to_string(image->getFormat())
//    );
//
//    mReferenceImages.push_back(image);
//
//    if (mReferenceImages.size() == imageCount)
//    {
//        logInfo("Reference dataset loaded.");
//    }
//}


bool VoxelReconstructionNoLightTransport::loadReferenceCamerasFromFile(const std::string& cameraFile)
{
    const auto cameraPath = resolveReconstructionPath(cameraFile);
    std::ifstream file(cameraPath);
    if (!file.is_open())
    {
        logError("Failed to open NeRF transform file: " + cameraPath.string());
        return false;
    }

    nlohmann::json jsonData;

    try
    {
        file >> jsonData;
    }
    catch (const std::exception& e)
    {
        logError("Failed to parse NeRF transform file: " + std::string(e.what()));
        return false;
    }

    if (!jsonData.contains("camera_angle_x"))
    {
        logError("NeRF transform file does not contain camera_angle_x.");
        return false;
    }

    if (!jsonData.contains("frames") || !jsonData["frames"].is_array())
    {
        logError("NeRF transform file does not contain valid frames.");
        return false;
    }

    const float cameraAngleX = jsonData["camera_angle_x"].get<float>();

    mReferenceCameras.clear();
    mReferenceImagePaths.clear();

    const float aspectRatio =
        static_cast<float>(mRayMarchingPass.mOutputResolution.x) / static_cast<float>(mRayMarchingPass.mOutputResolution.y);

    //
    // Falcor 使用 frameHeight / focalLength 定义相机 FOV。
    // NeRF 给的是 horizontal FOV，所以需要换算成 Falcor focalLength。
    //
    const float frameHeight = Camera::kDefaultFrameHeight;
    const float frameWidth = frameHeight * aspectRatio;

    const float focalLength = 0.5f * frameWidth / std::tan(0.5f * cameraAngleX);

    const auto& frames = jsonData["frames"];

    for (size_t frameIndex = 0; frameIndex < frames.size(); ++frameIndex)
    {
        const auto& frame = frames[frameIndex];

        if (!frame.contains("file_path") || !frame.contains("transform_matrix"))
        {
            logWarning("Invalid NeRF frame at index " + std::to_string(frameIndex));
            continue;
        }

        //
        // ------------------------------------------------------------
        // 1. Image path
        // ------------------------------------------------------------
        //
        std::string relativeImagePath = frame["file_path"].get<std::string>();

        std::filesystem::path imagePath = resolveReconstructionPath(ReferenceImageDir) / relativeImagePath;

        // NeRF json 中通常写 "./train/r_0"，没有 .png
        if (!imagePath.has_extension())
        {
            imagePath += ".png";
        }

        imagePath = std::filesystem::weakly_canonical(imagePath);

        //
        // ------------------------------------------------------------
        // 2. Camera transform
        // ------------------------------------------------------------
        //
        const auto& matrix = frame["transform_matrix"];

        if (!matrix.is_array() || matrix.size() != 4)
        {
            logWarning("Invalid transform_matrix at frame " + std::to_string(frameIndex));
            continue;
        }

        bool validMatrix = true;

        for (uint32_t row = 0; row < 4; ++row)
        {
            if (!matrix[row].is_array() || matrix[row].size() != 4)
            {
                validMatrix = false;
                break;
            }
        }

        if (!validMatrix)
        {
            logWarning("Invalid 4x4 transform_matrix at frame " + std::to_string(frameIndex));
            continue;
        }

        //
        // NeRF synthetic / Blender:
        //
        // transform_matrix 是 Camera-to-World。
        //
        // column 0 : camera right
        // column 1 : camera up
        // column 2 : camera backward
        // column 3 : camera position
        //
        // 因此：
        //
        // position = column 3
        // up       = column 1
        // forward  = -column 2
        //

        float3 position = float3(matrix[0][3].get<float>(), matrix[1][3].get<float>(), matrix[2][3].get<float>());

        float3 up = float3(matrix[0][1].get<float>(), matrix[1][1].get<float>(), matrix[2][1].get<float>());

        float3 forward = float3(-matrix[0][2].get<float>(), -matrix[1][2].get<float>(), -matrix[2][2].get<float>());

        auto nerfToFalcor = [](const float3& v) { return float3(v.x, v.z, -v.y); };

        position = nerfToFalcor(position);
        up = math::normalize(nerfToFalcor(up));
        forward = math::normalize(nerfToFalcor(forward));

        float3 target = position + forward;

        //
        // ------------------------------------------------------------
        // 3. Create Falcor camera
        // ------------------------------------------------------------
        //
        ref<Camera> cam = Camera::create();

        cam->setPosition(position);
        cam->setTarget(target);
        cam->setUpVector(up);

        cam->setFrameHeight(frameHeight);
        cam->setAspectRatio(aspectRatio);
        cam->setFocalLength(focalLength);

        mReferenceCameras.push_back(cam);
        mReferenceImagePaths.push_back(imagePath);
    }

    if (mReferenceCameras.empty())
    {
        logError("No valid NeRF cameras loaded from: " + cameraFile);
        return false;
    }

    if (mReferenceCameras.size() != mReferenceImagePaths.size())
    {
        logError("NeRF camera/image count mismatch.");
        return false;
    }

    //
    // 直接使用整个 train 集。
    // NeRF synthetic train 正好是 100 张。
    //
    mOptimizerParams.viewsPerIteration = static_cast<uint32_t>(mReferenceCameras.size());

    logInfo("Loaded NeRF training dataset: {} views, FOVx={} rad, focalLength={}", mReferenceCameras.size(), cameraAngleX, focalLength);

    return true;
}


void VoxelReconstructionNoLightTransport::loadReferenceImages()
{
    //
    // 第一次进入时先读取 transforms_train.json。
    // 这一步会同时生成：
    //
    // mReferenceCameras
    // mReferenceImagePaths
    //
    if (mReferenceCameras.empty() || mReferenceImagePaths.empty())
    {
        mReferenceCameras.clear();
        mReferenceImagePaths.clear();
        mReferenceImages.clear();

        if (!loadReferenceCamerasFromFile(ReferenceCameraFile))
        {
            throw RuntimeError("Failed to load NeRF training dataset.");
        }

        if (mReferenceCameras.size() != mReferenceImagePaths.size())
        {
            throw RuntimeError("NeRF camera count does not match image path count.");
        }

        mOptimizerParams.viewsPerIteration = static_cast<uint32_t>(mReferenceCameras.size());

        mReferenceImages.reserve(mOptimizerParams.viewsPerIteration);
    }

    const uint32_t imageCount = mOptimizerParams.viewsPerIteration;

    //
    // 全部 train 图片已经加载完。
    //
    if (mReferenceImages.size() >= imageCount)
    {
        return;
    }

    //
    // 当前帧只加载一张图片。
    //
    const uint32_t imageIndex = static_cast<uint32_t>(mReferenceImages.size());

    if (imageIndex >= mReferenceImagePaths.size())
    {
        throw RuntimeError("Reference image index exceeds image path count.");
    }

    const std::filesystem::path& imagePath = mReferenceImagePaths[imageIndex];

    if (!std::filesystem::exists(imagePath))
    {
        throw RuntimeError("Reference image does not exist: " + imagePath.string());
    }

    ref<Texture> image = Texture::createFromFile(mpDevice, imagePath, false, true);

    if (!image)
    {
        throw RuntimeError("Failed to load reference image: " + imagePath.string());
    }

    logInfo(
        "Reference image loaded: {}/{}, path={}, size={}x{}, format={}",
        imageIndex + 1,
        imageCount,
        imagePath.string(),
        image->getWidth(),
        image->getHeight(),
        to_string(image->getFormat())
    );

    mReferenceImages.push_back(image);

    if (mReferenceImages.size() == imageCount)
    {
        logInfo("NeRF training dataset loaded. {} images.", imageCount);
    }
}


