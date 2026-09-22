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

#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
constexpr uint64_t kMaxCheckpointMetadataBytes = 16ull * 1024 * 1024;

struct CoarseCheckpointHeader
{
    uint3 voxelCount = uint3(0);
    uint64_t elementCount = 0, byteSize = 0;
    nlohmann::json metadata;
};

// Leave the stream at the voxel payload so resume and read-only preview share the same size checks.
CoarseCheckpointHeader readCoarseCheckpointHeader(std::ifstream& in, const std::filesystem::path& path, uint32_t dimensionLimit)
{
    if (!in) throw std::runtime_error("Cannot open checkpoint: " + path.string());
    CoarseCheckpointHeader header;
    uint32_t magic = 0, version = 0, voxelDataSize = 0;
    uint64_t metadataSize = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&header.voxelCount), sizeof(header.voxelCount));
    in.read(reinterpret_cast<char*>(&voxelDataSize), sizeof(voxelDataSize));
    if (!in || magic != kReconstructionMagic || version != 2)
        throw std::runtime_error("Mode 2 requires a version 2 stage checkpoint; version 1 files lack stage and grid metadata.");
    if (voxelDataSize != sizeof(VoxelData))
        throw std::runtime_error("Checkpoint VoxelData layout differs from the current build.");
    in.read(reinterpret_cast<char*>(&metadataSize), sizeof(metadataSize));
    if (!in || metadataSize == 0 || metadataSize > kMaxCheckpointMetadataBytes)
        throw std::runtime_error("Invalid checkpoint metadata size.");
    header.elementCount = checkedVoxelCount(header.voxelCount, dimensionLimit);
    header.byteSize = header.elementCount * sizeof(VoxelData);
    const uint64_t headerSize = sizeof(magic) + sizeof(version) + sizeof(header.voxelCount) + sizeof(voxelDataSize) + sizeof(metadataSize);
    if (std::filesystem::file_size(path) != headerSize + metadataSize + header.byteSize)
        throw std::runtime_error("Checkpoint size does not match its grid and metadata (truncated or unexpected data).");
    std::string metadataText(static_cast<size_t>(metadataSize), '\0');
    in.read(metadataText.data(), std::streamsize(metadataSize));
    if (!in) throw std::runtime_error("Cannot read checkpoint metadata.");
    header.metadata = nlohmann::json::parse(metadataText);
    return header;
}

uint32_t checkpointUInt(const nlohmann::json& object, const char* key)
{
    const auto& value = object.at(key);
    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
        throw std::runtime_error(std::string("Invalid unsigned checkpoint field: ") + key);
    const uint64_t number = value.get<uint64_t>();
    if (number > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(std::string("Checkpoint field out of range: ") + key);
    return static_cast<uint32_t>(number);
}

float checkpointFloat(const nlohmann::json& object, const char* key, float minimum = 0.0f)
{
    const auto& value = object.at(key);
    if (!value.is_number())
        throw std::runtime_error(std::string("Invalid numeric checkpoint field: ") + key);
    const float number = value.get<float>();
    if (!std::isfinite(number) || number < minimum)
        throw std::runtime_error(std::string("Checkpoint field out of range: ") + key);
    return number;
}

float3 checkpointFloat3(const nlohmann::json& object, const char* key, bool positive)
{
    const auto& values = object.at(key);
    if (!values.is_array() || values.size() != 3)
        throw std::runtime_error(std::string("Invalid checkpoint vector: ") + key);
    float3 result;
    for (uint32_t i = 0; i < 3; ++i)
    {
        if (!values[i].is_number())
            throw std::runtime_error(std::string("Invalid checkpoint vector value: ") + key);
        result[i] = values[i].get<float>();
        if (!std::isfinite(result[i]) || (positive && result[i] <= 0.0f))
            throw std::runtime_error(std::string("Checkpoint vector out of range: ") + key);
    }
    return result;
}

GridData coarseResultGrid(const CoarseCheckpointHeader& header)
{
    const auto& metadata = header.metadata;
    if (checkpointUInt(metadata, "mode") != RECON_MODE_COARSE_TO_FINE ||
        checkpointUInt(metadata, "radianceShCount") != SH_COUNT || checkpointUInt(metadata, "opacityShCount") != SH_OPACITY_COUNT)
        throw std::runtime_error("The saved result has an incompatible reconstruction mode or SH layout.");
    GridData grid{};
    grid.voxelCount = header.voxelCount;
    const uint32_t resolution = checkpointUInt(metadata, "resolution");
    if (resolution != std::max(grid.voxelCount.x, std::max(grid.voxelCount.y, grid.voxelCount.z)))
        throw std::runtime_error("The saved result resolution does not match its grid dimensions.");
    grid.gridMin = checkpointFloat3(metadata, "gridMin", false);
    grid.voxelSize = checkpointFloat3(metadata, "voxelSize", true);
    grid.solidVoxelCount = checkpointUInt(metadata, "solidVoxelCount");
    if (grid.solidVoxelCount > header.elementCount)
        throw std::runtime_error("The saved result has an invalid solid voxel count.");
    for (uint32_t axis = 0; axis < 3; ++axis)
        if (!std::isfinite(grid.gridMin[axis] + grid.voxelSize[axis] * float(grid.voxelCount[axis])))
            throw std::runtime_error("The saved result grid extent is not finite.");
    return grid;
}

std::filesystem::path uniqueCheckpointPath(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
        return path;
    for (uint32_t suffix = 1; ; ++suffix)
    {
        auto candidate = path.parent_path() / fmt::format("{}_save{}{}", path.stem().string(), suffix, path.extension().string());
        if (!std::filesystem::exists(candidate))
            return candidate;
    }
}
#endif
} // namespace

std::filesystem::path VoxelReconstructionNoLightTransport::getReconstructionModeDirectory() const
{
    return std::filesystem::path(ReconstructionDataDir) / fmt::format("mode{}", RECON_MODE);
}

#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
void VoxelReconstructionNoLightTransport::ensureCoarseRunDirectory()
{
    if (!mCoarseToFine.runDirectory.empty())
        return;

    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &time);
#else
    localtime_r(&time, &localTime);
#endif
    std::ostringstream timestamp;
    timestamp << std::put_time(&localTime, "%Y%m%d_%H%M%S");
    const auto base = getReconstructionModeDirectory() / fmt::format("run_{}_target{}", timestamp.str(), GRID_RESOLUTION);
    std::filesystem::create_directories(base.parent_path());
    for (uint32_t suffix = 0; ; ++suffix)
    {
        const auto candidate = suffix == 0 ? base : std::filesystem::path(base.string() + "_" + std::to_string(suffix));
        if (std::filesystem::create_directory(candidate))
        {
            mCoarseToFine.runDirectory = candidate;
            return;
        }
    }
}

void VoxelReconstructionNoLightTransport::recordCoarseStageResult()
{
    auto& state = mCoarseToFine;
    if (!state.initialized || !state.lastSaveSucceeded || state.lastCheckpointPath.empty() ||
        state.stageBudget == 0 || state.stageIteration < state.stageBudget)
        return;

    auto& preview = state.preview;
    const CoarseStageResult result{state.stageIndex, mVoxelResolution, state.stageIteration, state.lastCheckpointPath};
    auto entry = std::lower_bound(preview.stages.begin(), preview.stages.end(), result.stageIndex,
        [](const CoarseStageResult& saved, uint32_t stageIndex) { return saved.stageIndex < stageIndex; });
    const bool exists = entry != preview.stages.end() && entry->stageIndex == result.stageIndex;
    const bool changed = !exists || entry->path != result.path || entry->resolution != result.resolution ||
        entry->stageIteration != result.stageIteration;
    if (exists)
        *entry = result;
    else
        preview.stages.insert(entry, result);
    if (changed && preview.selectedStage == result.stageIndex + 1)
        preview.reloadRequested = true;
}

bool VoxelReconstructionNoLightTransport::loadCoarsePreview(RenderContext* pRenderContext, const CoarseStageResult& result)
{
    auto& preview = mCoarseToFine.preview;
    try
    {
        const auto path = requireModeFile(result.path, getReconstructionModeDirectory());
        std::ifstream in(path, std::ios::binary);
        const auto header = readCoarseCheckpointHeader(in, path, reconstructionDimensionLimit(mpDevice));
        const auto& metadata = header.metadata;
        GridResources next;
        next.gridData = coarseResultGrid(header);
        const uint32_t resolution = checkpointUInt(metadata, "resolution");
        const uint32_t stageIndex = checkpointUInt(metadata, "stageIndex");
        const uint32_t stageIteration = checkpointUInt(metadata, "stageIteration");
        if (resolution != result.resolution || stageIndex != result.stageIndex || stageIteration != result.stageIteration)
            throw std::runtime_error("Saved preview does not match the selected stage result.");

        std::vector<uint8_t> data(static_cast<size_t>(header.byteSize));
        in.read(reinterpret_cast<char*>(data.data()), std::streamsize(header.byteSize));
        if (!in) throw std::runtime_error("Cannot read saved preview voxel payload.");

        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        next.gridDataBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), uint32_t(header.elementCount), flags);
        next.vBuffer = mpDevice->createTexture3D(header.voxelCount.x, header.voxelCount.y, header.voxelCount.z,
            ResourceFormat::R32Int, 1u, nullptr, flags);
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock");
        auto block = ParameterBlock::create(mpDevice, reflector);
        auto var = block->getRootVar();
        var["gridDataBuffer"] = next.gridDataBuffer;
        var["vBuffer"] = next.vBuffer;
        var["voxelCount"] = next.gridData.voxelCount;
        var["voxelSize"] = next.gridData.voxelSize;
        var["gridMin"] = next.gridData.gridMin;
        var["solidVoxelCount"] = next.gridData.solidVoxelCount;
        pRenderContext->clearUAV(next.vBuffer->getUAV().get(), uint4(0));
        pRenderContext->updateBuffer(next.gridDataBuffer.get(), data.data(), 0, data.size());
        pRenderContext->uavBarrier(next.gridDataBuffer.get());

        // Only publish preview resources after validation/allocation/upload; the active optimizer is untouched.
        auto loadedPath = path;
        auto status = fmt::format("Saved stage {}, resolution {}, iteration {}", stageIndex + 1, resolution, stageIteration);
        preview.grid = std::move(next);
        preview.gridBlock = std::move(block);
        preview.loadedPath = std::move(loadedPath);
        preview.status = std::move(status);
        preview.reloadRequested = false;
        return true;
    }
    catch (const std::exception& error)
    {
        preview.status = std::string("Saved stage preview failed: ") + error.what();
        logWarning("{}", preview.status);
        return false;
    }
}
#endif

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

#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    return mCoarseToFine.runDirectory / fmt::format(
        "{}_stage{}_res{}_iter{}.bin", sceneName, mCoarseToFine.stageIndex, mVoxelResolution, mCoarseToFine.stageIteration
    );
#else
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
#endif
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
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    mCoarseToFine.lastSaveSucceeded = false;
    try
    {
        if (mLoadedReconstructionForViewing)
            throw std::runtime_error("This result is open for viewing. Restore its training checkpoint before saving a new mode 2 checkpoint.");
        if (!mGridResources.gridDataBuffer || !mCoarseToFine.initialized)
            throw std::runtime_error("Initialize the coarse-to-fine grid before saving.");
        if (mOptimizerParams.currentView != 0)
            throw std::runtime_error("Coarse-to-fine checkpoints must be saved at an iteration boundary.");

        ensureCoarseRunDirectory();
        const auto path = uniqueCheckpointPath(getDefaultReconstructionSavePath());
        const auto& grid = mGridResources.gridData;
        const uint64_t elementCount = uint64_t(grid.voxelCount.x) * grid.voxelCount.y * grid.voxelCount.z;
        const uint64_t byteSize = elementCount * sizeof(VoxelData);

        nlohmann::json history = nlohmann::json::array();
        for (const auto& point : mCoarseToFine.lossHistory)
        {
            history.push_back({
                {"globalIteration", point.globalIteration}, {"stageIndex", point.stageIndex},
                {"resolution", point.resolution}, {"stageIteration", point.stageIteration}, {"loss", point.loss}
            });
        }
        nlohmann::json metadata = {
            {"mode", RECON_MODE}, {"targetResolution", GRID_RESOLUTION}, {"resolution", mVoxelResolution},
            {"resolutions", mCoarseToFine.resolutions},
            {"stageIndex", mCoarseToFine.stageIndex}, {"stageIteration", mCoarseToFine.stageIteration},
            {"stageBudget", mCoarseToFine.stageBudget}, {"currentIteration", mOptimizerParams.currentIteration},
            {"currentView", 0}, {"frameIndex", mFrameCount},
            {"finished", mCoarseToFine.finished}, {"pauseAfterStage", mCoarseToFine.pauseAfterStage},
            {"extraIterations", mCoarseToFine.extraIterations},
            {"gridMin", {grid.gridMin.x, grid.gridMin.y, grid.gridMin.z}},
            {"voxelSize", {grid.voxelSize.x, grid.voxelSize.y, grid.voxelSize.z}},
            {"solidVoxelCount", grid.solidVoxelCount},
            {"nameTag", mReconstructionNameTag}, {"referenceCameraFile", ReferenceCameraFile},
            {"referenceImageDir", ReferenceImageDir}, {"viewsPerIteration", mOptimizerParams.viewsPerIteration},
            {"imageWidth", mRayMarchingPass.mOutputResolution.x}, {"imageHeight", mRayMarchingPass.mOutputResolution.y},
            {"totalIterations", CTF_TOTAL_ITERATIONS}, {"refineInterval", CTF_REFINE_INTERVAL},
            {"finalIterations", getCoarseStageBudget(uint32_t(mCoarseToFine.resolutions.size() - 1))},
            {"maxPathHits", MAX_CONTRIBUTING_VOXELS_PER_RAY}, {"maxCandidates", MAX_CANDIDATES},
            {"radianceShCount", SH_COUNT}, {"opacityShCount", SH_OPACITY_COUNT},
            {"training", {
                {"lrRadiance", mUpdatePass.mLrRadiance}, {"lrOpacity", mUpdatePass.mLrOpacity},
                {"lrCenter", mUpdatePass.mLrCenter}, {"lrB", mUpdatePass.mLrB},
                {"normalizeByGradCount", mUpdatePass.mUseGradCountNormalize}, {"gradScale", mUpdatePass.mGradScale},
                {"geometryTau", mGradientPass.geometryTau}, {"geometryGradClamp", mGradientPass.geometryGradClamp},
                {"ellipsoidPruneThreshold", mUpdatePass.mEllipsoidPruneThreshold},
                {"spp", mRayMarchingPass.mSpp}, {"checkPrimitive", mRayMarchingPass.mCheckPrimitive},
                {"renderBackground", mRayMarchingPass.mRenderBackGround},
                {"clearColor", {mRayMarchingPass.mClearColor.x, mRayMarchingPass.mClearColor.y, mRayMarchingPass.mClearColor.z}},
                {"drawMode", mRayMarchingPass.mDrawMode}, {"shadowBias100", mRayMarchingPass.mShadowBias100},
                {"maxContributingVoxelCount", mRayMarchingPass.mMaxContributingVoxelCount},
                {"transmittanceThreshold", mRayMarchingPass.mTransmittanceThreshold},
                {"backgroundCarveWeight", 0.01f}, {"binaryOpacityWeight", 0.003f},
                // Keep the legacy field so earlier v2 readers can still inspect this checkpoint.
                {"pruneWarmupIterations", 0}, {"pruneAllStages", true}, {"pruneInterval", CTF_PRUNE_INTERVAL}
            }},
            {"lossHistory", history}
        };
        const std::string metadataText = metadata.dump();
        const uint64_t metadataSize = metadataText.size();
        if (metadataSize > kMaxCheckpointMetadataBytes)
            throw std::runtime_error("Checkpoint metadata exceeds the 16 MiB limit.");

        pRenderContext->submit(true);
        std::vector<uint8_t> data(static_cast<size_t>(byteSize));
        mGridResources.gridDataBuffer->getBlob(data.data(), 0, data.size());

        // Publish only complete files; the load list ignores the temporary .tmp extension.
        const std::filesystem::path temporaryPath = path.string() + ".tmp";
        std::ofstream out(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("Cannot open checkpoint for writing: " + temporaryPath.string());
        const uint32_t version = 2;
        const uint32_t voxelDataSize = sizeof(VoxelData);
        out.write(reinterpret_cast<const char*>(&kReconstructionMagic), sizeof(kReconstructionMagic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&grid.voxelCount), sizeof(grid.voxelCount));
        out.write(reinterpret_cast<const char*>(&voxelDataSize), sizeof(voxelDataSize));
        out.write(reinterpret_cast<const char*>(&metadataSize), sizeof(metadataSize));
        out.write(metadataText.data(), std::streamsize(metadataSize));
        out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(byteSize));
        out.close();
        if (!out)
            throw std::runtime_error("Checkpoint write failed: " + temporaryPath.string());
        std::filesystem::rename(temporaryPath, path);
        saveLossHistory();
        mCoarseToFine.lastCheckpointPath = path;
        mReconstructionFileListDirty = true;
        mCoarseToFine.lastSaveSucceeded = true;
        logInfo("Saved stage checkpoint to {}, resolution={}, stage iteration={}", path.string(), mVoxelResolution, mCoarseToFine.stageIteration);
    }
    catch (const std::exception& error)
    {
        logError("Save stage checkpoint failed: {}", error.what());
    }
#else
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
#endif
}


#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
void VoxelReconstructionNoLightTransport::restoreCoarseCheckpoint(RenderContext* pRenderContext, const std::filesystem::path& path)
{
    try
    {
        requireModeFile(path, getReconstructionModeDirectory());
        std::ifstream in(path, std::ios::binary);
        const auto header = readCoarseCheckpointHeader(in, path, reconstructionDimensionLimit(mpDevice));
        const auto fileVoxelCount = header.voxelCount;
        const auto elementCount = header.elementCount;
        const auto byteSize = header.byteSize;
        const auto& metadata = header.metadata;
        if (checkpointUInt(metadata, "mode") != RECON_MODE || checkpointUInt(metadata, "targetResolution") != GRID_RESOLUTION)
            throw std::runtime_error("Checkpoint mode or target resolution differs from RECON_MODE / GRID_RESOLUTION.");
        const uint32_t resolution = checkpointUInt(metadata, "resolution");
        if (resolution == 0 || any(fileVoxelCount > uint3(resolution)) ||
            std::max(fileVoxelCount.x, std::max(fileVoxelCount.y, fileVoxelCount.z)) != resolution)
            throw std::runtime_error("Checkpoint resolution does not match its voxel dimensions.");

        CoarseToFineState state;
        for (uint32_t value = CTF_START_RESOLUTION; ; value *= 2)
        {
            state.resolutions.push_back(value);
            if (value == GRID_RESOLUTION) break;
        }
        const auto& savedResolutions = metadata.at("resolutions");
        if (!savedResolutions.is_array() || savedResolutions.size() != state.resolutions.size())
            throw std::runtime_error("Checkpoint levels differ from the configured coarse-to-fine hierarchy.");
        for (size_t i = 0; i < state.resolutions.size(); ++i)
        {
            // Reuse strict integer validation instead of accepting JSON's lossy numeric conversions.
            if (checkpointUInt(nlohmann::json{{"value", savedResolutions[i]}}, "value") != state.resolutions[i])
                throw std::runtime_error("Checkpoint hierarchy differs from CTF_START_RESOLUTION / GRID_RESOLUTION.");
        }
        state.stageIndex = checkpointUInt(metadata, "stageIndex");
        state.stageIteration = checkpointUInt(metadata, "stageIteration");
        state.stageBudget = checkpointUInt(metadata, "stageBudget");
        state.extraIterations = checkpointUInt(metadata, "extraIterations");
        if (state.stageIndex >= state.resolutions.size() || state.resolutions[state.stageIndex] != resolution ||
            state.stageBudget == 0 || state.stageIteration > state.stageBudget || state.extraIterations == 0)
            throw std::runtime_error("Invalid checkpoint stage or iteration budget.");
        const uint32_t globalIteration = checkpointUInt(metadata, "currentIteration");
        const uint32_t frameIndex = checkpointUInt(metadata, "frameIndex");
        if (checkpointUInt(metadata, "currentView") != 0 || state.stageIteration > globalIteration)
            throw std::runtime_error("Checkpoint must describe a complete iteration boundary.");
        const uint32_t viewCount = checkpointUInt(metadata, "viewsPerIteration");
        if (viewCount == 0 || viewCount != mReferenceCameras.size() || viewCount != mReferenceImages.size())
            throw std::runtime_error("Load the same reference views before restoring this checkpoint.");
        if (checkpointUInt(metadata, "imageWidth") != mRayMarchingPass.mOutputResolution.x ||
            checkpointUInt(metadata, "imageHeight") != mRayMarchingPass.mOutputResolution.y ||
            checkpointUInt(metadata, "maxPathHits") != MAX_CONTRIBUTING_VOXELS_PER_RAY ||
            checkpointUInt(metadata, "maxCandidates") != MAX_CANDIDATES ||
            checkpointUInt(metadata, "radianceShCount") != SH_COUNT || checkpointUInt(metadata, "opacityShCount") != SH_OPACITY_COUNT)
            throw std::runtime_error("Checkpoint image dimensions or path capacity differs from the current configuration.");

        GridData grid{};
        grid.voxelCount = fileVoxelCount;
        grid.gridMin = checkpointFloat3(metadata, "gridMin", false);
        grid.voxelSize = checkpointFloat3(metadata, "voxelSize", true);
        grid.solidVoxelCount = checkpointUInt(metadata, "solidVoxelCount");
        if (grid.solidVoxelCount > elementCount)
            throw std::runtime_error("Invalid checkpoint solid voxel count.");
        for (uint32_t axis = 0; axis < 3; ++axis)
            if (!std::isfinite(grid.gridMin[axis] + grid.voxelSize[axis] * float(fileVoxelCount[axis])))
                throw std::runtime_error("Checkpoint grid extent is not finite.");

        const auto& training = metadata.at("training");
        auto update = mUpdatePass;
        update.mLrRadiance = checkpointFloat(training, "lrRadiance");
        update.mLrOpacity = checkpointFloat(training, "lrOpacity");
        update.mLrCenter = checkpointFloat(training, "lrCenter");
        update.mLrB = checkpointFloat(training, "lrB");
        update.mUseGradCountNormalize = training.at("normalizeByGradCount").get<bool>();
        update.mGradScale = checkpointFloat(training, "gradScale");
        update.mEllipsoidPruneThreshold = checkpointFloat(training, "ellipsoidPruneThreshold");
        update.mEnableEllipsoidPruning = false;
        const float geometryTau = checkpointFloat(training, "geometryTau");
        const float geometryGradClamp = checkpointFloat(training, "geometryGradClamp");
        const uint32_t spp = checkpointUInt(training, "spp");
        const uint32_t maxHits = checkpointUInt(training, "maxContributingVoxelCount");
        const float transmittanceThreshold = checkpointFloat(training, "transmittanceThreshold");
        const bool checkPrimitive = training.at("checkPrimitive").get<bool>();
        const bool renderBackground = training.at("renderBackground").get<bool>();
        const float3 clearColor = checkpointFloat3(training, "clearColor", false);
        const uint32_t drawMode = checkpointUInt(training, "drawMode");
        const float shadowBias100 = checkpointFloat(training, "shadowBias100");
        const uint32_t savedPruneWarmup = checkpointUInt(training, "pruneWarmupIterations");
        const bool savedPruneAllStages = training.value("pruneAllStages", false);
        const uint32_t savedPruneInterval = training.contains("pruneInterval") ? checkpointUInt(training, "pruneInterval") : 10u;
        if (checkpointFloat(training, "backgroundCarveWeight") != 0.01f || checkpointFloat(training, "binaryOpacityWeight") != 0.003f)
            throw std::runtime_error("Checkpoint gradient regularization differs from this build.");
        if (spp == 0 || maxHits == 0 || maxHits > MAX_CONTRIBUTING_VOXELS_PER_RAY || transmittanceThreshold > 1.0f ||
            !std::isfinite(update.mLrCenter * 1000.0f) || !std::isfinite(update.mLrB * 10.0f))
            throw std::runtime_error("Invalid checkpoint sampling or learning-rate settings.");

        state.finished = metadata.at("finished").get<bool>();
        state.pauseAfterStage = metadata.at("pauseAfterStage").get<bool>();
        if (state.finished && (state.stageIndex + 1 != state.resolutions.size() || state.stageIteration < state.stageBudget))
            throw std::runtime_error("Invalid final-stage completion flag.");
        const std::string nameTag = metadata.at("nameTag").get<std::string>();
        const std::string cameraFile = metadata.at("referenceCameraFile").get<std::string>();
        const std::string imageDirectory = metadata.at("referenceImageDir").get<std::string>();
        if (cameraFile != ReferenceCameraFile || imageDirectory != ReferenceImageDir)
            throw std::runtime_error("Checkpoint reference dataset differs from the currently configured dataset.");
        const uint32_t savedInterval = checkpointUInt(metadata, "refineInterval");
        const uint32_t savedFinalIterations = checkpointUInt(metadata, "finalIterations");
        const uint32_t savedTotalIterations = metadata.contains("totalIterations") ? checkpointUInt(metadata, "totalIterations") : 0u;
        const auto& history = metadata.at("lossHistory");
        if (!history.is_array() || history.size() != globalIteration)
            throw std::runtime_error("Checkpoint loss history does not match its global iteration count.");
        std::vector<float> iterationLossHistory;
        iterationLossHistory.reserve(history.size());
        uint32_t previousStage = 0, previousStageIteration = 0;
        for (size_t i = 0; i < history.size(); ++i)
        {
            const auto& point = history[i];
            StageLossRecord record{
                checkpointUInt(point, "globalIteration"), checkpointUInt(point, "stageIndex"),
                checkpointUInt(point, "resolution"), checkpointUInt(point, "stageIteration"), checkpointFloat(point, "loss")
            };
            if (record.globalIteration != i + 1 || record.stageIndex > state.stageIndex ||
                (i == 0 && record.stageIndex != 0) ||
                record.resolution != state.resolutions[record.stageIndex] ||
                (record.stageIndex == previousStage ? record.stageIteration != previousStageIteration + 1 :
                 record.stageIndex != previousStage + 1 || record.stageIteration != 1))
                throw std::runtime_error("Checkpoint loss history contains invalid stage progress.");
            previousStage = record.stageIndex;
            previousStageIteration = record.stageIteration;
            state.lossHistory.push_back(record);
            iterationLossHistory.push_back(record.loss);
        }
        if ((!history.empty() && (state.stageIteration > 0 ?
            previousStage != state.stageIndex || previousStageIteration != state.stageIteration :
            previousStage + 1 != state.stageIndex)) || (history.empty() && (state.stageIndex != 0 || state.stageIteration != 0)))
            throw std::runtime_error("Checkpoint stage position is inconsistent with its loss history.");

        std::vector<uint8_t> data(static_cast<size_t>(byteSize));
        in.read(reinterpret_cast<char*>(data.data()), std::streamsize(byteSize));
        if (!in) throw std::runtime_error("Cannot read checkpoint voxel payload.");
        in.close();

        // Validation and payload reads complete before the shared allocation/upload/binding transaction.
        replaceReconstructionGrid(pRenderContext, grid, resolution, data.data(), data.size());
        state.initialized = true;
        state.paused = true;
        state.lastSaveSucceeded = true;
        state.lastCheckpointPath = path;
        // A loaded checkpoint is also available as an immutable preview, including a manually saved partial stage.
        state.preview.stages.push_back({state.stageIndex, resolution, state.stageIteration, path});
        // Resume into a new run on first save so the inspected experiment stays intact.
        state.runDirectory.clear();
        mCoarseToFine = std::move(state);
        mLoadedReconstructionForViewing = false;
        mOptimizerParams.currentIteration = globalIteration;
        mOptimizerParams.currentView = 0;
        mOptimizerParams.viewsPerIteration = viewCount;
        mOptimizerParams.isRunning = false;
        mEnableReconstruction = false;
        mInitVoxelData = false;
        mSaveReconstructionRequested = false;
        mFrameCount = frameIndex;
        mRayMarchingPass.mFrameIndex = frameIndex;
        mRayMarchingPass.mSpp = spp;
        mRayMarchingPass.mMaxContributingVoxelCount = maxHits;
        mRayMarchingPass.mTransmittanceThreshold = transmittanceThreshold;
        mRayMarchingPass.mCheckPrimitive = checkPrimitive;
        mRayMarchingPass.mRenderBackGround = renderBackground;
        mRayMarchingPass.mClearColor = clearColor;
        mRayMarchingPass.mDrawMode = drawMode;
        mRayMarchingPass.mShadowBias100 = shadowBias100;
        mUpdatePass = update;
        mLrCenterScale = update.mLrCenter * 1000.0f;
        mLrBScale = update.mLrB * 10.0f;
        mGradientPass.geometryTau = geometryTau;
        mGradientPass.geometryGradClamp = geometryGradClamp;
        mReconstructionNameTag = nameTag;
        mReduceLossPass.iterationLossHistory = std::move(iterationLossHistory);
        mReduceLossPass.meanLoss = mReduceLossPass.iterationLossHistory.empty() ? 0.f : mReduceLossPass.iterationLossHistory.back();
        resetCoarseSampling(pRenderContext);
        if (savedInterval != CTF_REFINE_INTERVAL || savedTotalIterations != CTF_TOTAL_ITERATIONS ||
            savedFinalIterations != getCoarseStageBudget(uint32_t(mCoarseToFine.resolutions.size() - 1)))
            logWarning("Checkpoint stage budget restored. Future stages use the current CTF_TOTAL_ITERATIONS / CTF_REFINE_INTERVAL schedule; reset to apply the full new budget.");
        if (savedPruneWarmup != 0 || !savedPruneAllStages || savedPruneInterval != CTF_PRUNE_INTERVAL)
            logWarning("Checkpoint restored with volume pruning on every stage, once per {} complete stage iterations.", CTF_PRUNE_INTERVAL);
        logInfo("Loaded {} at resolution {}, stage iteration {}/{}; paused for inspection.", path.string(), resolution,
            mCoarseToFine.stageIteration, mCoarseToFine.stageBudget);
        mReconstructionIOStatus = fmt::format("Restored training checkpoint: {} (resolution {}, stage iteration {}/{}, paused)",
            path.filename().string(), resolution, mCoarseToFine.stageIteration, mCoarseToFine.stageBudget);
    }
    catch (const std::exception& error)
    {
        mReconstructionIOStatus = std::string("Restore checkpoint failed: ") + error.what();
        logError("{}", mReconstructionIOStatus);
    }
}
#endif

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
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
        const auto header = readCoarseCheckpointHeader(in, path, dimensionLimit);
        grid = coarseResultGrid(header);
        resolution = checkpointUInt(header.metadata, "resolution");
        elementCount = header.elementCount;
        byteSize = header.byteSize;
        CoarseToFineState viewState;
        viewState.initialized = true;
        viewState.paused = true;
        viewState.lastCheckpointPath = path;
        // Stage labels describe the file. They do not restore optimizer iteration counts or budgets.
        const auto stageLabel = [&](const CoarseCheckpointHeader& saved, const std::filesystem::path& file)
        {
            const auto savedGrid = coarseResultGrid(saved);
            const uint32_t stageIndex = checkpointUInt(saved.metadata, "stageIndex");
            if (stageIndex >= 32) throw std::runtime_error("Saved stage index is out of range.");
            return CoarseStageResult{stageIndex,
                std::max(savedGrid.voxelCount.x, std::max(savedGrid.voxelCount.y, savedGrid.voxelCount.z)),
                checkpointUInt(saved.metadata, "stageIteration"), file};
        };
        const auto selectedStage = stageLabel(header, path);
        viewState.stageIndex = selectedStage.stageIndex;
        viewState.resolutions.assign(selectedStage.stageIndex + 1, resolution);
        // A result remains viewable when its training hierarchy differs from the current experiment.
        const auto levels = header.metadata.find("resolutions");
        if (levels != header.metadata.end() && levels->is_array() && levels->size() <= 32)
        {
            std::vector<uint32_t> savedLevels;
            try
            {
                for (const auto& level : *levels)
                {
                    const uint32_t value = checkpointUInt(nlohmann::json{{"value", level}}, "value");
                    if (value == 0 || value > dimensionLimit) throw std::runtime_error("Invalid display level.");
                    savedLevels.push_back(value);
                }
                if (selectedStage.stageIndex < savedLevels.size() && savedLevels[selectedStage.stageIndex] == resolution)
                    viewState.resolutions = std::move(savedLevels);
            }
            catch (const std::exception&) { /* The optional training hierarchy is not required to display voxel data. */ }
        }
        viewState.preview.stages.push_back(selectedStage);
#else
        uint32_t magic = 0, version = 0, voxelDataSize = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&grid.voxelCount), sizeof(grid.voxelCount));
        in.read(reinterpret_cast<char*>(&voxelDataSize), sizeof(voxelDataSize));
        if (!in || magic != kReconstructionMagic || version != 1)
            throw std::runtime_error("This mode loads its current version 1 reconstruction files.");
        if (voxelDataSize != sizeof(VoxelData))
            throw std::runtime_error("Reconstruction VoxelData layout differs from the current build.");
        elementCount = checkedVoxelCount(grid.voxelCount, dimensionLimit);
        if (grid.voxelCount.x != grid.voxelCount.y || grid.voxelCount.x != grid.voxelCount.z)
            throw std::runtime_error("Only current cubic-grid mode 1/3 files are supported; older non-cubic/SVO files are not supported.");
        resolution = grid.voxelCount.x;
        byteSize = elementCount * sizeof(VoxelData);
        const uint64_t headerSize = sizeof(magic) + sizeof(version) + sizeof(grid.voxelCount) + sizeof(voxelDataSize);
        if (std::filesystem::file_size(path) != headerSize + byteSize)
            throw std::runtime_error("Reconstruction file size does not match its voxel dimensions and layout.");
        // Current mode 1/3 v1 files use this fixed NeRF reconstruction domain; v1 does not store its AABB.
        constexpr float extent = 2.6f * 1.02f;
        grid.voxelSize = float3(extent / float(resolution));
        grid.gridMin = -0.5f * grid.voxelSize * float3(grid.voxelCount);
#endif
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
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
        mCoarseToFine = std::move(viewState);
        // Discover saved stages from this run only; a bad neighbour must not invalidate the loaded result.
        try
        {
            std::map<uint32_t, std::pair<CoarseStageResult, std::filesystem::file_time_type>> stages;
            for (const auto& entry : std::filesystem::directory_iterator(path.parent_path()))
            {
                if (!entry.is_regular_file() || !samePathComponent(entry.path().extension(), ".bin")) continue;
                try
                {
                    const auto candidatePath = requireModeFile(entry.path(), getReconstructionModeDirectory());
                    std::ifstream candidate(candidatePath, std::ios::binary);
                    const auto candidateHeader = readCoarseCheckpointHeader(candidate, candidatePath, dimensionLimit);
                    auto result = stageLabel(candidateHeader, candidatePath);
                    const auto modified = std::filesystem::last_write_time(candidatePath);
                    auto old = stages.find(result.stageIndex);
                    if (old == stages.end() || result.stageIteration > old->second.first.stageIteration ||
                        (result.stageIteration == old->second.first.stageIteration && modified > old->second.second))
                        stages[result.stageIndex] = {std::move(result), modified};
                }
                catch (const std::exception& error)
                {
                    logWarning("Skipped saved-stage result {}: {}", entry.path().string(), error.what());
                }
            }
            // An explicitly selected older checkpoint wins over a newer file for the same stage.
            stages[selectedStage.stageIndex] = {selectedStage, std::filesystem::file_time_type::min()};
            std::vector<CoarseStageResult> results;
            for (auto& stage : stages) results.push_back(std::move(stage.second.first));
            mCoarseToFine.preview.stages = std::move(results);
        }
        catch (const std::exception& error)
        {
            logWarning("Loaded reconstruction, but could not list all saved stages: {}", error.what());
        }
#endif
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
#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
    if (mCoarseToFine.runDirectory.empty())
        throw std::runtime_error("Create a coarse-to-fine run directory before saving loss history.");
    const auto lossDir = mCoarseToFine.runDirectory / "Loss";
    std::filesystem::create_directories(lossDir);
    const auto path = lossDir / "training.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open loss history: " + path.string());
    out << "globalIteration,stage,resolution,stageIteration,loss\n" << std::setprecision(10);
    for (const auto& point : mCoarseToFine.lossHistory)
    {
        out << point.globalIteration << ',' << point.stageIndex << ',' << point.resolution << ','
            << point.stageIteration << ',' << point.loss << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("Failed to write loss history: " + path.string());
#else
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
#endif
}
