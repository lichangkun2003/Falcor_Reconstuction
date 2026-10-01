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
constexpr uint32_t kReconstructionMagic = 0x56525832; // "VRX2"
constexpr uint32_t kSparseReconstructionVersion = 3;

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

} // namespace

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
    clearSparseGradients(pRenderContext);
    resetDeletionEvidence(pRenderContext);
    mPointCloud.startRequested = false;
    mPointCloud.initialized = true;
    mPointCloud.clearAccumulation = true;
    mPointCloud.status = "Loaded voxel reconstruction; PLY initialization is not required.";
}

DefineList VoxelReconstructionNoLightTransport::getReconstructionDefines()
{
    DefineList defines;
    defines.add("GRID_RESOLUTION", std::to_string(GRID_RESOLUTION));
    return defines;
}

std::filesystem::path VoxelReconstructionNoLightTransport::getReconstructionModeDirectory() const
{
    return resolveReconstructionPath(ReconstructionDataDir) / "mode1";
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

    if (mUpdatePass.mLrShape > 0.0f)
        tags.push_back("shape");

    if (mUpdatePass.mLrRotation > 0.0f)
        tags.push_back("rotation");
    if (mUpdatePass.mLrRadiance > 0.0f || mUpdatePass.mLrOpacity > 0.0f ||
        mUpdatePass.mLrCenter > 0.0f || mUpdatePass.mLrShape > 0.0f ||
        mUpdatePass.mLrRotation > 0.0f)
        tags.push_back("adam");


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



void VoxelReconstructionNoLightTransport::saveSparseReconstruction(
    RenderContext* pRenderContext, const std::filesystem::path& path)
{
    const uint32_t activeCount = mGridResources.gridData.activeVoxelCount;
    if (activeCount == 0 || mGridResources.voxelPages.empty())
    {
        logWarning("Save reconstruction skipped: the compact voxel pool is empty.");
        return;
    }

    std::filesystem::create_directories(path.parent_path());
    pRenderContext->submit(true);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw RuntimeError("Cannot open sparse reconstruction for writing: " + path.string());

    const uint32_t magic = kReconstructionMagic;
    const uint32_t version = kSparseReconstructionVersion;
    const uint3 voxelCount = mGridResources.gridData.voxelCount;
    const uint32_t voxelDataSize = sizeof(VoxelData);
    const uint32_t radianceCount = SH_COUNT;
    const uint32_t opacityCount = SH_OPACITY_COUNT;
    const float3 gridMin = mGridResources.gridData.gridMin;
    const float3 voxelSize = mGridResources.gridData.voxelSize;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&voxelCount), sizeof(voxelCount));
    out.write(reinterpret_cast<const char*>(&voxelDataSize), sizeof(voxelDataSize));
    out.write(reinterpret_cast<const char*>(&radianceCount), sizeof(radianceCount));
    out.write(reinterpret_cast<const char*>(&opacityCount), sizeof(opacityCount));
    out.write(reinterpret_cast<const char*>(&gridMin), sizeof(gridMin));
    out.write(reinterpret_cast<const char*>(&voxelSize), sizeof(voxelSize));
    out.write(reinterpret_cast<const char*>(&activeCount), sizeof(activeCount));

    uint32_t written = 0;
    while (written < activeCount)
    {
        const uint32_t page = written / SPARSE_POOL_PAGE_SIZE;
        const uint32_t count = std::min(activeCount - written, SPARSE_POOL_PAGE_SIZE);
        std::vector<uint32_t> cells(count);
        std::vector<VoxelData> voxels(count);
        mGridResources.cellIndexPages[page]->getBlob(cells.data(), 0, size_t(count) * sizeof(uint32_t));
        mGridResources.voxelPages[page]->getBlob(voxels.data(), 0, size_t(count) * sizeof(VoxelData));
        for (uint32_t i = 0; i < count; ++i)
        {
            out.write(reinterpret_cast<const char*>(&cells[i]), sizeof(cells[i]));
            out.write(reinterpret_cast<const char*>(&voxels[i]), sizeof(voxels[i]));
        }
        written += count;
    }
    if (!out) throw RuntimeError("Failed while writing sparse reconstruction payload.");
    out.close();
    saveLossHistory();
    const uint64_t payloadBytes = uint64_t(activeCount) * (sizeof(uint32_t) + sizeof(VoxelData));
    logInfo("Saved sparse reconstruction to {}, grid={}x{}x{}, active={}, payload={} bytes",
        path.string(), voxelCount.x, voxelCount.y, voxelCount.z, activeCount, payloadBytes);
}

void VoxelReconstructionNoLightTransport::loadSparseReconstruction(
    RenderContext* pRenderContext, const std::filesystem::path& selectedPath)
{
    const auto path = requireModeFile(selectedPath, getReconstructionModeDirectory());
    std::ifstream in(path, std::ios::binary);
    if (!in) throw RuntimeError("Cannot open sparse reconstruction: " + path.string());

    uint32_t magic = 0, version = 0, voxelDataSize = 0;
    uint32_t radianceCount = 0, opacityCount = 0, activeCount = 0;
    GridData grid{};
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&grid.voxelCount), sizeof(grid.voxelCount));
    in.read(reinterpret_cast<char*>(&voxelDataSize), sizeof(voxelDataSize));
    in.read(reinterpret_cast<char*>(&radianceCount), sizeof(radianceCount));
    in.read(reinterpret_cast<char*>(&opacityCount), sizeof(opacityCount));
    in.read(reinterpret_cast<char*>(&grid.gridMin), sizeof(grid.gridMin));
    in.read(reinterpret_cast<char*>(&grid.voxelSize), sizeof(grid.voxelSize));
    in.read(reinterpret_cast<char*>(&activeCount), sizeof(activeCount));
    if (!in || magic != kReconstructionMagic || version != kSparseReconstructionVersion)
        throw RuntimeError("Invalid sparse reconstruction header.");
    if (voxelDataSize != sizeof(VoxelData) || radianceCount != SH_COUNT || opacityCount != SH_OPACITY_COUNT)
        throw RuntimeError("Sparse reconstruction SH counts or VoxelData layout differ from the current build.");
    const uint64_t totalCells = checkedVoxelCount(grid.voxelCount, reconstructionDimensionLimit(mpDevice));
    if (activeCount == 0 || uint64_t(activeCount) > totalCells)
        throw RuntimeError("Sparse reconstruction has an invalid active voxel count.");
    const uint64_t headerBytes = sizeof(magic) + sizeof(version) + sizeof(grid.voxelCount) + sizeof(voxelDataSize) +
        sizeof(radianceCount) + sizeof(opacityCount) + sizeof(grid.gridMin) + sizeof(grid.voxelSize) + sizeof(activeCount);
    const uint64_t expectedBytes = headerBytes + uint64_t(activeCount) * (sizeof(uint32_t) + sizeof(VoxelData));
    if (std::filesystem::file_size(path) != expectedBytes)
        throw RuntimeError("Sparse reconstruction file size does not match its header.");

    grid.solidVoxelCount = activeCount;
    grid.activeVoxelCount = activeCount;
    const uint64_t maximumCapacity = uint64_t(SPARSE_POOL_PAGE_SIZE) * SPARSE_POOL_MAX_PAGES;
    if (uint64_t(activeCount) > maximumCapacity)
        throw RuntimeError("Sparse reconstruction exceeds the segmented parameter-pool limit.");
    auto resources = allocateSparseGrid(pRenderContext, grid, activeCount);
    auto block = createSparseGridBlock(resources);

    uint32_t loaded = 0;
    while (loaded < activeCount)
    {
        const uint32_t count = std::min(activeCount - loaded, SPARSE_POOL_PAGE_SIZE);
        std::vector<uint32_t> cells(count);
        std::vector<VoxelData> voxels(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            in.read(reinterpret_cast<char*>(&cells[i]), sizeof(cells[i]));
            in.read(reinterpret_cast<char*>(&voxels[i]), sizeof(voxels[i]));
            if (uint64_t(cells[i]) >= totalCells || voxels[i].occupied == 0)
                throw RuntimeError("Sparse reconstruction contains an invalid voxel entry.");
        }
        if (!in) throw RuntimeError("Cannot read the complete sparse reconstruction payload.");
        uploadSparseBatch(pRenderContext, block, loaded, cells.data(), count, voxels.data());
        loaded += count;
    }
    for (const auto& page : resources.voxelPages) pRenderContext->uavBarrier(page.get());
    for (const auto& page : resources.indexPages) pRenderContext->uavBarrier(page.get());
    pRenderContext->submit(true);
    const uint32_t resolution = std::max(grid.voxelCount.x, std::max(grid.voxelCount.y, grid.voxelCount.z));
    commitSparseGrid(std::move(resources), block, resolution);
    resetLoadedReconstruction(pRenderContext);
    mReconstructionIOStatus = fmt::format("Loaded sparse reconstruction: {} ({}x{}x{}, {} active voxels)",
        path.filename().string(), grid.voxelCount.x, grid.voxelCount.y, grid.voxelCount.z, activeCount);
    logInfo("{}", mReconstructionIOStatus);
}

void VoxelReconstructionNoLightTransport::saveReconstruction(RenderContext* pRenderContext)
{
    if (!mPointCloud.initialized)
    {
        logWarning("Save reconstruction skipped: initialize from PLY or load a reconstruction first.");
        return;
    }
    const auto path = getDefaultReconstructionSavePath();
    try
    {
        saveSparseReconstruction(pRenderContext, path);
        mReconstructionIOStatus = "Saved sparse reconstruction: " + path.filename().string();
    }
    catch (const std::exception& error)
    {
        mReconstructionIOStatus = std::string("Save failed: ") + error.what();
        logError("{}", mReconstructionIOStatus);
    }
}

void VoxelReconstructionNoLightTransport::loadReconstruction(
    RenderContext* pRenderContext, const std::filesystem::path& selectedPath)
{
    try
    {
        loadSparseReconstruction(pRenderContext, selectedPath);
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
