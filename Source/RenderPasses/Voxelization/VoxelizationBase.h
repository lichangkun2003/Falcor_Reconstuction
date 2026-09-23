#pragma once
#include "Voxel/VoxelGrid.slang"
#include "Voxel/ABSDF.slang"
#include "Math/Ellipsoid.slang"
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Math/VoxelizationUtility.h"
#include "Math/Random.h"
#include "VoxelizationShared.slang"
#include <random>
using namespace Falcor;

inline std::string ToString(float3 v)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}
inline std::string ToString(int2 v)
{
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ")";
    return oss.str();
}
inline std::string ToString(int3 v)
{
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

struct BufferDesc
{
    std::string name;
    std::string texname; // 如果不直接对应着色器资源，设为空字符串
    std::string desc;
    bool serialized;
    bool isInputOrOutut;
    size_t bytesPerElement;
};
using BufferlList = std::vector<BufferDesc>;

inline std::string kGBuffer = "gBuffer";
inline std::string kVBuffer = "vBuffer";
inline std::string kPBuffer = "pBuffer";
inline std::string kPolygonBuffer = "polygonBuffer";
inline std::string kPolygonRangeBuffer = "polygonRangeBuffer";
inline std::string kBlockMap = "blockMap";

class VoxelizationBase
{
public:
    static const int NDFLobeCount = 8;
    static GridData GlobalGridData;
    static uint3 MinFactor; // 网格的分辨率必须是此值的整数倍
    static bool FileUpdated;
    static std::string ResourceFolder;
    static bool LightChanged;

    // 手动网格。用于把 Voxelization 的网格对齐到 VoxelReconstructionNoLightTransport 的固定重建网格。
    // 开启后忽略场景 AABB，改用 ManualAABBMin / ManualAABBMax。
    // 两侧的建格算法逐行相同（见下面 UpdateVoxelGrid 的手动分支），因此只要 AABB 与 voxelResolution
    // 一致，得到的 gridMin / voxelSize / voxelCount 就是逐位相同的，椭球可以直接按体素坐标搬运。
    static bool UseManualAABB;
    static float3 ManualAABBMin;
    static float3 ManualAABBMax;

    static void UpdateVoxelGrid(ref<Scene> scene, uint voxelResolution)
    {
        float3 diag;
        float length;
        float3 center;
        if (UseManualAABB)
        {
            // 与 VoxelReconstructionNoLightTransport::UpdateVoxelGrid 完全相同的算法，
            // 只把 AABB 换成手动范围。改动这里时必须同步改那边，否则两个网格会对不上。
            center = 0.5f * (ManualAABBMin + ManualAABBMax);
            diag = ManualAABBMax - ManualAABBMin;
            length = std::max(diag.z, std::max(diag.x, diag.y));
            diag *= 1.02f;
            length *= 1.02f;
        }
        else if (scene)
        {
            AABB aabb = scene->getSceneBounds();
            diag = aabb.maxPoint - aabb.minPoint;
            length = std::max(diag.z, std::max(diag.x, diag.y));
            center = aabb.center();
            diag *= 1.02f;
            length *= 1.02f;
        }
        else
        {
            diag = float3(1);
            length = 1.f;
            center = float3(0);
        }

        GlobalGridData.voxelSize = float3(length / voxelResolution);
        float3 temp = diag / GlobalGridData.voxelSize;

        GlobalGridData.voxelCount = uint3(
            (uint)math::ceil(temp.x / MinFactor.x) * MinFactor.x,
            (uint)math::ceil(temp.y / MinFactor.y) * MinFactor.y,
            (uint)math::ceil(temp.z / MinFactor.z) * MinFactor.z
        );
        GlobalGridData.gridMin = center - 0.5f * GlobalGridData.voxelSize * float3(GlobalGridData.voxelCount);
        GlobalGridData.solidVoxelCount = 0;

        // 打印出来直接和重建 pass 启动时那行 "Reconstruction voxel grid initialized" 对照。
        logInfo(
            "Voxelization voxel grid initialized (manual={}):"
            " voxelSize=({}, {}, {}),"
            " voxelCount=({}, {}, {}),"
            " gridMin=({}, {}, {})",
            UseManualAABB,
            GlobalGridData.voxelSize.x,
            GlobalGridData.voxelSize.y,
            GlobalGridData.voxelSize.z,
            GlobalGridData.voxelCount.x,
            GlobalGridData.voxelCount.y,
            GlobalGridData.voxelCount.z,
            GlobalGridData.gridMin.x,
            GlobalGridData.gridMin.y,
            GlobalGridData.gridMin.z
        );
    }
};

struct SceneHeader
{
    uint meshCount;
    uint vertexCount;
    uint triangleCount;
};

struct MeshHeader
{
    uint meshID;
    uint materialID;
    uint vertexCount;
    uint triangleCount;
    uint triangleOffset;
};

inline ref<Buffer> copyToCpu(ref<Device> pDevice, RenderContext* pRenderContext, ref<Buffer> gpuBuffer)
{
    ref<Buffer> cpuBuffer = pDevice->createBuffer(gpuBuffer->getSize(), ResourceBindFlags::None, MemoryType::ReadBack);
    pRenderContext->copyResource(cpuBuffer.get(), gpuBuffer.get());
    return cpuBuffer;
}

class PolygonBufferGroup
{
private:
    GridData& gridData;
    ref<Device> mpDevice;
    std::vector<ref<Buffer>> mBuffers;
    std::vector<uint> voxelCount;     // 各组中的体素个数
    std::vector<uint> gBufferOffsets; // 每一组内的第一个体素在gBuffer中的偏移量
    std::vector<uint> polygonCount;   // 各组中的多边形个数

    std::vector<Polygon> currentPolygons; // 正在处理的Polygon
    uint currentVoxelCount = 0;
    uint currentPolygonCount = 0;

    void flushCurrent()
    {
        if (currentPolygonCount == 0)
            return;

        if (size() == 0)
            gBufferOffsets.push_back(0);
        else
            gBufferOffsets.push_back(voxelCount.back() + gBufferOffsets.back());

        ref<Buffer> buffer = mpDevice->createStructuredBuffer(
            sizeof(Polygon), currentPolygonCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        if (currentPolygons.size() > 0)
            buffer->setBlob(currentPolygons.data(), 0, size_t(currentPolygonCount) * sizeof(Polygon));

        mBuffers.push_back(buffer);
        voxelCount.push_back(currentVoxelCount);
        polygonCount.push_back(currentPolygonCount);

        currentPolygons.clear();
        currentVoxelCount = 0;
        currentPolygonCount = 0;
    }

public:
    uint maxPolygonCount = 256000;
    PolygonBufferGroup(ref<Device> device, GridData& gridData) : gridData(gridData), mpDevice(device) {}

    uint getVoxelOffset(uint index) const
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return gBufferOffsets[index];
    }

    uint getVoxelCount(uint index) const
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return voxelCount[index];
    }

    uint getPolygonCount(uint index) const
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return polygonCount[index];
    }

    uint size() const { return mBuffers.size(); }

    ref<Buffer> get(uint index)
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return mBuffers[index];
    }

    void reset()
    {
        mBuffers.clear();
        voxelCount.clear();
        polygonCount.clear();
        gBufferOffsets.clear();
        currentVoxelCount = 0;
        currentPolygonCount = 0;
        currentPolygons.reserve(maxPolygonCount);
        gridData.maxPolygonCount = 0;
        gridData.totalPolygonCount = 0;
    }

    // 用于CPU上已经裁剪完成的情况
    void setBlob(const std::vector<std::vector<Polygon>>& polygonArrays, std::vector<PolygonRange>& polygonRangeBuffer)
    {
        FALCOR_ASSERT(polygonRangeBuffer.size() == polygonArrays.size());
        reset();

        for (size_t v = 0; v < polygonArrays.size(); ++v)
        {
            const std::vector<Polygon>& polys = polygonArrays[v];
            const uint n = (uint)polys.size();

            FALCOR_ASSERT(n > 0 && n <= maxPolygonCount);

            if (currentPolygonCount + n > maxPolygonCount)
            {
                flushCurrent();
            }

            polygonRangeBuffer[v].count = n;
            polygonRangeBuffer[v].localHead = currentPolygonCount;

            gridData.maxPolygonCount = max(gridData.maxPolygonCount, n);
            gridData.totalPolygonCount += n;

            currentPolygons.insert(currentPolygons.end(), polys.begin(), polys.end());
            currentPolygonCount += n;
            currentVoxelCount++;
        }

        flushCurrent();
    }

    // 预分配空间，用于GPU上裁剪之前
    void reserve(std::vector<uint>& polygonCountBuffer, std::vector<PolygonRange>& polygonRangeBuffer)
    {
        FALCOR_ASSERT(polygonRangeBuffer.size() == polygonCountBuffer.size());
        reset();
        for (size_t v = 0; v < polygonRangeBuffer.size(); ++v)
        {
            const uint n = polygonCountBuffer[v]; // GPU上第一遍仅统计个数

            FALCOR_ASSERT(n > 0 && n <= maxPolygonCount);

            if (currentPolygonCount + n > maxPolygonCount)
            {
                flushCurrent();
            }

            polygonRangeBuffer[v].count = n;
            polygonRangeBuffer[v].localHead = currentPolygonCount;
            gridData.maxPolygonCount = max(gridData.maxPolygonCount, n);
            gridData.totalPolygonCount += n;

            currentPolygonCount += n;
            currentVoxelCount++;
        }

        flushCurrent();
    }
};
