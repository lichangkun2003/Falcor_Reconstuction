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

std::string VoxelReconstructionNoLightTransport::getOptimizedParamTag() const
{
    std::vector<std::string> tags;

    if (mUpdatePass.mLrRadiance > 0.0f)
        tags.push_back("radiance");

    if (mUpdatePass.mLrCenter > 0.0f)
        tags.push_back("specular");

    if (mUpdatePass.mLrB > 0.0f)
        tags.push_back("rough");


    if (tags.empty())
        return "none";

    std::string result = tags[0];

    for (size_t i = 1; i < tags.size(); ++i)
    {
        result += "_";
        result += tags[i];
    }

    return result;

}
std::filesystem::path VoxelReconstructionNoLightTransport::getDefaultReconstructionSavePath() const
{
    const std::filesystem::path outputDir = ReconstructionDataDir;

    uint3 voxelCount = mGridResources.gridData.voxelCount;

    std::string paramTag = getOptimizedParamTag();


    // 获取当天日期：month_day，例如 5_29
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &time);
#else
    localtime_r(&time, &localTime);
#endif

    int month = localTime.tm_mon + 1;
    int day = localTime.tm_mday;

    std::string dateTag = fmt::format("{}_{}", month, day);


    std::string nameTag = mReconstructionNameTag;

    std::string filename;

    if (nameTag.empty())
    {
        filename = fmt::format("recon{}_{}_{}.bin", dateTag, GRID_RESOLUTION, paramTag);
    }
    else
    {
        filename = fmt::format("recon{}_{}_{}_{}.bin", dateTag, nameTag, GRID_RESOLUTION, paramTag);
    }

    return outputDir / filename;
}



void VoxelReconstructionNoLightTransport::saveReconstruction(RenderContext* pRenderContext)
{
    if (!mGridResources.gridDataBuffer)
    {
        logWarning("Save reconstruction failed: gridDataBuffer is null.");
        return;
    }

    std::filesystem::path path = getDefaultReconstructionSavePath();

    const uint64_t elementCount = mGridResources.gridData.totalVoxelCount();
    const uint64_t byteSize = elementCount * sizeof(VoxelData);

    std::filesystem::create_directories(path.parent_path());

    // 确保 GPU update pass 已完成
    pRenderContext->submit(true);

    std::vector<uint8_t> data(byteSize);

    // 你的版本是 void getBlob(void* pData, size_t offset, size_t size) const
    mGridResources.gridDataBuffer->getBlob(data.data(), 0, size_t(byteSize));

    std::ofstream out(path, std::ios::binary);

    if (!out.is_open())
    {
        logError("Save reconstruction failed: cannot open file " + path.string());
        return;
    }

    uint32_t magic = 0x56525831; // "VRX1"
    uint32_t version = 1;
    uint3 voxelCount = mGridResources.gridData.voxelCount;
    uint32_t voxelDataSize = sizeof(VoxelData);

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&voxelCount), sizeof(voxelCount));
    out.write(reinterpret_cast<const char*>(&voxelDataSize), sizeof(voxelDataSize));

    out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(byteSize));

    out.close();

    logInfo(
        "Saved reconstruction to {}, voxelCount={}x{}x{}, params={}, bytes={}",
        path.string(),
        voxelCount.x,
        voxelCount.y,
        voxelCount.z,
        getOptimizedParamTag(),
        byteSize
    );
}


void VoxelReconstructionNoLightTransport::loadReconstruction(RenderContext* pRenderContext, const std::filesystem::path& path)
{
    if (!mGridResources.gridDataBuffer)
    {
        logWarning("Load reconstruction failed: gridDataBuffer is null.");
        return;
    }

    std::ifstream in(path, std::ios::binary);

    if (!in.is_open())
    {
        logError("Load reconstruction failed: cannot open file " + path.string());
        return;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint3 fileVoxelCount = uint3(0);
    uint32_t voxelDataSize = 0;

    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&fileVoxelCount), sizeof(fileVoxelCount));
    in.read(reinterpret_cast<char*>(&voxelDataSize), sizeof(voxelDataSize));

    if (magic != 0x56525831 || version != 1)
    {
        logError("Load reconstruction failed: invalid file header.");
        return;
    }

    if (any(fileVoxelCount != mGridResources.gridData.voxelCount))
    {
        logError(
            "Load reconstruction failed: voxel count mismatch. file={}x{}x{}, current={}x{}x{}",
            fileVoxelCount.x,
            fileVoxelCount.y,
            fileVoxelCount.z,
            mGridResources.gridData.voxelCount.x,
            mGridResources.gridData.voxelCount.y,
            mGridResources.gridData.voxelCount.z
        );
        return;
    }

    if (voxelDataSize != sizeof(VoxelData))
    {
        logError("Load reconstruction failed: VoxelData size mismatch. file={}, current={}", voxelDataSize, sizeof(VoxelData));
        return;
    }

    const uint64_t elementCount = mGridResources.gridData.totalVoxelCount();
    const uint64_t byteSize = elementCount * sizeof(VoxelData);

    std::vector<uint8_t> data(byteSize);

    in.read(reinterpret_cast<char*>(data.data()), std::streamsize(byteSize));

    if (!in)
    {
        logError("Load reconstruction failed: file is truncated.");
        return;
    }

    in.close();

    pRenderContext->updateBuffer(mGridResources.gridDataBuffer.get(), data.data(), 0, size_t(byteSize));

    logInfo("Loaded reconstruction from " + path.string());
}



void VoxelReconstructionNoLightTransport::refreshReconstructionFileList()
{
    mReconstructionFilePaths.clear();

    if (!std::filesystem::exists(ReconstructionDataDir))
    {
        std::filesystem::create_directories(ReconstructionDataDir);
        mSelectedReconstructionFile = 0;
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(ReconstructionDataDir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::filesystem::path& path = entry.path();

        if (path.extension() == ".bin")
        {
            mReconstructionFilePaths.push_back(path);
        }
    }

    std::sort(
        mReconstructionFilePaths.begin(),
        mReconstructionFilePaths.end(),
        [](const std::filesystem::path& a, const std::filesystem::path& b) { return a.filename().string() < b.filename().string(); }
    );

    if (mReconstructionFilePaths.empty())
    {
        mSelectedReconstructionFile = 0;
    }
    else
    {
        mSelectedReconstructionFile = std::min(mSelectedReconstructionFile, uint32_t(mReconstructionFilePaths.size() - 1));
    }
}
