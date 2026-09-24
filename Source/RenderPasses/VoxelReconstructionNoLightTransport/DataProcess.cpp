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
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <cstring>
#include <cwctype>
#include <slang-gfx.h>

namespace
{
constexpr uint32_t kReconstructionMagic = 0x56525831;

bool samePathComponent(const std::filesystem::path& a, const std::filesystem::path& b)
{
#if defined(_WIN32)
    auto lhs = a.native(), rhs = b.native();
    std::transform(lhs.begin(), lhs.end(), lhs.begin(), [](wchar_t value) { return wchar_t(std::towlower(value)); });
    std::transform(rhs.begin(), rhs.end(), rhs.begin(), [](wchar_t value) { return wchar_t(std::towlower(value)); });
    return lhs == rhs;
#else
    return a == b;
#endif
}

std::filesystem::path requireModeFile(const std::filesystem::path& path, const std::filesystem::path& modeDirectory)
{
    const auto file = std::filesystem::canonical(path);
    const auto directory = std::filesystem::canonical(modeDirectory);
    auto component = file.begin();
    for (auto rootComponent = directory.begin(); rootComponent != directory.end(); ++rootComponent, ++component)
    {
        if (component == file.end() || !samePathComponent(*component, *rootComponent))
            throw std::runtime_error("Select a reconstruction inside the current mode directory: " + modeDirectory.string());
    }
    if (component == file.end() || !std::filesystem::is_regular_file(file))
        throw std::runtime_error("The selected reconstruction is not a regular file.");
    return file;
}

uint32_t reconstructionDimensionLimit(const ref<Device>& device)
{
    const uint32_t limit = device->getGfxDevice()->getDeviceInfo().limits.maxTextureDimension3D;
    return limit > 0 ? limit : 2048u;
}

uint64_t checkedVoxelCount(uint3 count, uint32_t dimensionLimit)
{
    if (any(count == uint3(0)) || any(count > uint3(dimensionLimit)))
        throw std::runtime_error("Reconstruction voxel dimensions exceed the GPU's 3D texture limit.");
    const uint64_t maximum = uint64_t(std::numeric_limits<int32_t>::max());
    uint64_t elements = count.x;
    if (elements > maximum / count.y)
        throw std::runtime_error("Reconstruction exceeds the shader index range.");
    elements *= count.y;
    if (elements > maximum / count.z)
        throw std::runtime_error("Reconstruction exceeds the shader index range.");
    return elements * count.z;
}

// v1 文件头 = magic + version + voxelCount + voxelDataSize, 之后紧跟稠密的 payload.
// 调用方自己读 payload, 返回时 in 停在第一个数据字节.
// mode1/3 的 loadReconstruction 和 loadBakedReconstruction 共用它, 两边校验不会跑偏.
void readV1ReconstructionHeader(std::ifstream& in, const std::filesystem::path& path, uint32_t dimensionLimit,
    GridData& grid, uint32_t& resolution, uint64_t& byteSize)
{
    uint32_t magic = 0, version = 0, voxelDataSize = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&grid.voxelCount), sizeof(grid.voxelCount));
    in.read(reinterpret_cast<char*>(&voxelDataSize), sizeof(voxelDataSize));
    if (!in || magic != kReconstructionMagic || version != 1)
        throw std::runtime_error("This mode loads its current version 1 reconstruction files.");
    if (voxelDataSize != sizeof(VoxelData))
        throw std::runtime_error("Reconstruction VoxelData layout differs from the current build.");
    const uint64_t elementCount = checkedVoxelCount(grid.voxelCount, dimensionLimit);
    if (grid.voxelCount.x != grid.voxelCount.y || grid.voxelCount.x != grid.voxelCount.z)
        throw std::runtime_error("Only current cubic-grid mode 1/3 files are supported; older non-cubic/SVO files are not supported.");
    resolution = grid.voxelCount.x;
    byteSize = elementCount * sizeof(VoxelData);
    const uint64_t headerSize = sizeof(magic) + sizeof(version) + sizeof(grid.voxelCount) + sizeof(voxelDataSize);
    if (std::filesystem::file_size(path) != headerSize + byteSize)
        throw std::runtime_error("Reconstruction file size does not match its voxel dimensions and layout.");
    // v1 不存 AABB, 靠这个固定的 NeRF 重建域反推: extent = 2.6f * 1.02f.
    // Voxelization 那边必须按同一个 AABB 建格, 即勾选 "Match Reconstruction Grid";
    // 否则椭球会被放到错的位置.
    constexpr float extent = 2.6f * 1.02f;
    grid.voxelSize = float3(extent / float(resolution));
    grid.gridMin = -0.5f * grid.voxelSize * float3(grid.voxelCount);
}

} // namespace

std::filesystem::path VoxelReconstructionNoLightTransport::getReconstructionModeDirectory() const
{
    return resolveReconstructionPath(ReconstructionDataDir) / fmt::format("mode{}", RECON_MODE);
}


std::string VoxelReconstructionNoLightTransport::getOptimizedParamTag() const
{
    std::vector<std::string> tags;

    if (mUpdatePass.mLrRadiance > 0.0f)
        tags.push_back("radiance");
    if (mUpdatePass.mLrOpacity > 0.0f)
        tags.push_back("opacity");

    if (mUpdatePass.mLrCenter > 0.0f)
        tags.push_back("center");

    if (mUpdatePass.mLrB > 0.0f)
        tags.push_back("B");


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
    auto sceneDirectory = std::filesystem::path(ReferenceImageDir).lexically_normal();
    // A trailing separator gives an empty filename; use the last directory component.
    if (sceneDirectory.filename().empty()) sceneDirectory = sceneDirectory.parent_path();
    std::string sceneName = sceneDirectory.filename().string();
    if (sceneName.empty() || sceneName == "." || sceneName == "..") sceneName = "scene";

    const std::filesystem::path outputDir = getReconstructionModeDirectory();

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
        filename = fmt::format("{}_recon{}_{}_{}.bin", sceneName, dateTag, mVoxelResolution, paramTag);
    }
    else
    {
        filename = fmt::format("{}_recon{}_{}_{}_{}.bin", sceneName, dateTag, nameTag, mVoxelResolution, paramTag);
    }

    return outputDir / filename;
}



void VoxelReconstructionNoLightTransport::saveReconstruction(RenderContext* pRenderContext)
{
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    if (!mPointCloud.initialized)
    {
        logWarning("Save reconstruction skipped: initialize from PLY or load a reconstruction first.");
        return;
    }
#endif
    if (!mGridResources.gridDataBuffer)
    {
        logWarning("Save reconstruction failed: gridDataBuffer is null.");
        return;
    }

    std::filesystem::path path = getDefaultReconstructionSavePath();

    const uint64_t elementCount = mGridResources.gridData.totalVoxelCount();
    //const uint64_t elementCount = mGridResources.gridData.solidVoxelCount;
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

    saveLossHistory();

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



void VoxelReconstructionNoLightTransport::loadReconstruction(RenderContext* pRenderContext, const std::filesystem::path& selectedPath)
{
    try
    {
        const auto path = requireModeFile(selectedPath, getReconstructionModeDirectory());
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open reconstruction: " + path.string());
        const uint32_t dimensionLimit = reconstructionDimensionLimit(mpDevice);
        GridData grid{};
        uint32_t resolution = 0;
        uint64_t elementCount = 0, byteSize = 0;
        readV1ReconstructionHeader(in, path, dimensionLimit, grid, resolution, byteSize);
        elementCount = byteSize / sizeof(VoxelData);
        std::vector<uint8_t> data(static_cast<size_t>(byteSize));
        in.read(reinterpret_cast<char*>(data.data()), std::streamsize(byteSize));
        if (!in) throw std::runtime_error("Cannot read the complete reconstruction voxel payload.");
        in.close();
        grid.solidVoxelCount = 0;
        for (uint64_t i = 0; i < elementCount; ++i)
        {
            uint32_t occupied = 0;
            std::memcpy(&occupied, data.data() + i * sizeof(VoxelData) + offsetof(VoxelData, occupied), sizeof(occupied));
            if (occupied != 0) ++grid.solidVoxelCount;
        }
        auto status = fmt::format("Loaded for viewing: {} (mode {}, {}x{}x{}, {} occupied voxels)",
            path.filename().string(), RECON_MODE, grid.voxelCount.x, grid.voxelCount.y, grid.voxelCount.z, grid.solidVoxelCount);
        replaceReconstructionGrid(pRenderContext, grid, resolution, data.data(), data.size());
        resetLoadedReconstruction(pRenderContext);
        mReconstructionIOStatus = std::move(status);
        logInfo("{}", mReconstructionIOStatus);
    }
    catch (const std::exception& error)
    {
        mReconstructionIOStatus = std::string("Load failed: ") + error.what();
        logError("{}", mReconstructionIOStatus);
    }
}



void VoxelReconstructionNoLightTransport::refreshBakedFileList()
{
    mBakedFilePaths.clear();
    const auto directory = resolveReconstructionPath(BakeOutputDir);
    std::error_code error;
    if (!std::filesystem::exists(directory, error))
    {
        mSelectedBakedFile = 0;
        return;
    }

    try
    {
        for (const auto& entry : std::filesystem::directory_iterator(
                 directory, std::filesystem::directory_options::skip_permission_denied))
        {
            if (entry.is_regular_file() && samePathComponent(entry.path().extension(), ".bin"))
                mBakedFilePaths.push_back(entry.path());
        }
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        logWarning("Cannot list bake results in {}: {}", directory.string(), e.what());
    }

    std::sort(
        mBakedFilePaths.begin(),
        mBakedFilePaths.end(),
        [](const std::filesystem::path& a, const std::filesystem::path& b) { return a.generic_string() < b.generic_string(); }
    );
    mSelectedBakedFile = 0;
}

void VoxelReconstructionNoLightTransport::loadBakedReconstruction(
    RenderContext* pRenderContext, const std::filesystem::path& selectedPath
)
{
    try
    {
        // 关键: 这里刻意不调用 requireModeFile. 烘焙产物由 Voxelization 写在独立的资源目录里,
        // 不属于任何一个 mode 目录, 走那条校验会被直接拒绝.
        const std::filesystem::path directory = resolveReconstructionPath(BakeOutputDir);
        const std::filesystem::path path = selectedPath.is_absolute() ? selectedPath : directory / selectedPath;

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open bake result: " + path.string());
        const uint32_t dimensionLimit = reconstructionDimensionLimit(mpDevice);
        GridData grid{};
        uint32_t resolution = 0;
        uint64_t byteSize = 0;
        readV1ReconstructionHeader(in, path, dimensionLimit, grid, resolution, byteSize);

        std::vector<uint8_t> data(static_cast<size_t>(byteSize));
        in.read(reinterpret_cast<char*>(data.data()), std::streamsize(byteSize));
        if (!in) throw std::runtime_error("Cannot read the complete bake payload.");
        in.close();

        grid.solidVoxelCount = 0;
        for (uint64_t i = 0; i < byteSize / sizeof(VoxelData); ++i)
        {
            uint32_t occupied = 0;
            std::memcpy(&occupied, data.data() + i * sizeof(VoxelData) + offsetof(VoxelData, occupied), sizeof(occupied));
            if (occupied != 0) ++grid.solidVoxelCount;
        }

        // 这个数字必须和烘焙端日志里 "written occupied=N" 的 N 相等,
        // 不等就说明两边对 VoxelData 布局的理解不一致.
        auto status = fmt::format("Loaded bake result: {} ({}x{}x{}, {} occupied voxels)",
            path.filename().string(), grid.voxelCount.x, grid.voxelCount.y, grid.voxelCount.z, grid.solidVoxelCount);
        replaceReconstructionGrid(pRenderContext, grid, resolution, data.data(), data.size());
        resetLoadedReconstruction(pRenderContext);
        mReconstructionIOStatus = std::move(status);
        logInfo("{}", mReconstructionIOStatus);
    }
    catch (const std::exception& error)
    {
        mReconstructionIOStatus = std::string("Load failed: ") + error.what();
        logError("{}", mReconstructionIOStatus);
    }
}

void VoxelReconstructionNoLightTransport::refreshReconstructionFileList()
{
    const std::filesystem::path selectedPath = mSelectedReconstructionFile < mReconstructionFilePaths.size() ?
        mReconstructionFilePaths[mSelectedReconstructionFile] : std::filesystem::path{};
    mReconstructionFilePaths.clear();
    const auto directory = getReconstructionModeDirectory();
    std::error_code error;
    if (!std::filesystem::exists(directory, error))
    {
        mSelectedReconstructionFile = 0;
        return;
    }

    try
    {
        const auto entries = std::filesystem::recursive_directory_iterator(directory, std::filesystem::directory_options::skip_permission_denied);
        for (const auto& entry : entries)
        {
            if (entry.is_regular_file() && samePathComponent(entry.path().extension(), ".bin"))
            {
                try { mReconstructionFilePaths.push_back(requireModeFile(entry.path(), directory)); }
                catch (const std::exception& e) { logWarning("Skipped reconstruction file {}: {}", entry.path().string(), e.what()); }
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        logWarning("Cannot list reconstruction files in {}: {}", directory.string(), e.what());
    }

    std::sort(
        mReconstructionFilePaths.begin(),
        mReconstructionFilePaths.end(),
        [](const std::filesystem::path& a, const std::filesystem::path& b) { return a.generic_string() < b.generic_string(); }
    );
    mReconstructionFilePaths.erase(std::unique(mReconstructionFilePaths.begin(), mReconstructionFilePaths.end()), mReconstructionFilePaths.end());
    mSelectedReconstructionFile = 0;
    if (!selectedPath.empty())
    {
        const auto oldSelection = std::filesystem::weakly_canonical(selectedPath, error);
        const auto selected = std::find_if(mReconstructionFilePaths.begin(), mReconstructionFilePaths.end(),
            [&](const auto& path) { return samePathComponent(path, oldSelection); });
        if (selected != mReconstructionFilePaths.end())
            mSelectedReconstructionFile = uint32_t(std::distance(mReconstructionFilePaths.begin(), selected));
    }
}




void VoxelReconstructionNoLightTransport::saveLossHistory() const
{
    if (mReduceLossPass.iterationLossHistory.empty())
    {
        logWarning("Save loss history skipped: loss history is empty.");
        return;
    }

    const std::filesystem::path lossDir = getReconstructionModeDirectory() / "Loss";

    std::filesystem::create_directories(lossDir);

    // 获取当前日期：month_day
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

    // 文件名：日期_NameTag.csv
    std::string filename;

    if (mReconstructionNameTag.empty())
    {
        filename = fmt::format("{}.csv", dateTag);
    }
    else
    {
        filename = fmt::format("{}_{}.csv", dateTag, mReconstructionNameTag);
    }

    std::filesystem::path lossPath = lossDir / filename;

    std::ofstream out(lossPath);

    if (!out.is_open())
    {
        logError("Save loss history failed: cannot open file " + lossPath.string());
        return;
    }

    out << "iteration,mean_loss\n";

    out << std::setprecision(10);

    for (size_t i = 0; i < mReduceLossPass.iterationLossHistory.size(); ++i)
    {
        out << (i + 1) << "," << mReduceLossPass.iterationLossHistory[i] << "\n";
    }

    out.close();

    logInfo("Saved loss history to {}, iterations={}", lossPath.string(), mReduceLossPass.iterationLossHistory.size());
}
