#pragma once
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"
import Utils.Math.MathHelpers;

// 从 RayMarching.ps.slang 原样搬出来的公共部分：光追所需的资源声明 + DDA / Mipmap 两套 rayMarching。
// 烘焙（BakeReconstruction.cs.slang）复用同一份，避免两份 DDA 各自漂移。
// 改这里会同时影响前向渲染和烘焙，两边都要重新验证。
// 下面几段是从 RayMarching.ps.slang 逐字节搬过来的，未做任何改动。

// ---- 资源声明 ----
Texture3D<int> vBuffer;
StructuredBuffer<PrimitiveBSDF> gBuffer;
StructuredBuffer<Ellipsoid> pBuffer;
Texture2D<uint4> blockMap;

cbuffer GridData
{
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
    uint solidVoxelCount;
}

// ---- HitResult ----
struct HitResult
{
    bool hit;
    int3 cellInt;
    float3 cell;
};

// ---- 光追主体 ----
bool isSolidVoxel(int3 cellInt, out int offset)
{
    offset = vBuffer[cellInt];
    if (offset < 0 || offset > solidVoxelCount)
        return false;
    return gBuffer[offset].area > 0;
}

bool isSolidBlock(int3 blockInt)
{
    int component = blockInt.z / 32;
    uint value = 1 << blockInt.z % 32;
    uint4 zMap = blockMap[blockInt.xy];
    switch (component)
    {
    case 0:
        value &= zMap.x;
        break;
    case 1:
        value &= zMap.y;
        break;
    case 2:
        value &= zMap.z;
        break;
    case 3:
        value &= zMap.w;
        break;
    }
    return value != 0;
}

// 先后用体素和椭球裁剪线段
HitResult checkPrimitive(Ellipsoid ellipsoid, float3 cell, float3 direction, int3 cellInt)
{
    float3 from = cell - cellInt;
    int3 directions = int3(sign(direction));
    float3 border = (directions + 1) / 2;
    float3 t = (border - from) / direction;
    float tMin = 2.0f;
    if (direction.x != 0 && t.x < tMin)
        tMin = t.x;
    if (direction.y != 0 && t.y < tMin)
        tMin = t.y;
    if (direction.z != 0 && t.z < tMin)
        tMin = t.z;
    float3 to = from + tMin * direction;
    float2 inOut = ellipsoid.clip(from, to);
    HitResult result = { inOut.x >= 0, cellInt, cell + inOut.x * (to - from) };
    return result;
}

// 统一用网格坐标表示
HitResult rayMarching_DDA(float3 from, float3 direction, bool ignoreFirst)
{
    float3 cell = from;
    int3 cellInt = int3(floor(from));

    float3 reciprocal_direction = 1.0f / direction;
    // 可选的前进方向
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
            break;
        int offset;
        if (isSolidVoxel(cellInt, offset)) // 理论上不会出现面积为0的情况
        {
#if !CHECK_ELLIPSOID
            HitResult hit = { true, cellInt, cellInt + 0.5f };
            return hit;
#else
            Ellipsoid ellipsoid = pBuffer[offset];
            HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);
            if (hit.hit)
                return hit;
#endif
        }
        leave(cell, cellInt, directions, direction, reciprocal_direction);
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f };
    return hit;
}
// 统一用网格坐标表示
HitResult rayMarching_DDA(float3 from, float3 direction, float maxDistance, bool ignoreFirst)
{
    float3 cell = from;
    int3 cellInt = int3(floor(from));
    float distance = 0;

    float3 reciprocal_direction = 1.0f / direction;
    // 可选的前进方向
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
            break;
        if (maxDistance > 0 && distance > maxDistance) // 非正数表示不限制距离
            break;
        int offset;
        if (isSolidVoxel(cellInt, offset)) // 理论上不会出现面积为0的情况
        {
#if !CHECK_ELLIPSOID
            HitResult hit = { true, cellInt, cellInt + 0.5f };
            return hit;
#else
            Ellipsoid ellipsoid = pBuffer[offset];
            HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);
            if (hit.hit)
                return hit;
#endif
        }
        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f };
    return hit;
}

HitResult rayMarching_Mipmap(float3 from, float3 direction, bool ignoreFirst)
{
    float3 cell = from;
    int3 cellInt = int3(floor(from));

    float3 reciprocal_direction = 1.0f / direction;
    // 可选的前进方向
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
            break;

        int3 blockInt = cellInt / BLOCK_TO_VOXEL;
        if (isSolidBlock(blockInt))
        {
            int3 current = blockInt;
            while (all(current == blockInt))
            {
                if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
                    break;
                int offset;
                if (isSolidVoxel(cellInt, offset)) // 理论上不会出现面积为0的情况
                {
#if !CHECK_ELLIPSOID
                    HitResult hit = { true, cellInt, cellInt + 0.5f };
                    return hit;
#else
                    Ellipsoid ellipsoid = pBuffer[offset];
                    HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);
                    if (hit.hit)
                        return hit;
#endif
                }
                leave(cell, cellInt, directions, direction, reciprocal_direction);
                current = cellInt / BLOCK_TO_VOXEL;
            }
        }
        else
        {
            leaveBlock(cell, cellInt, directions, direction, reciprocal_direction);
        }
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f };
    return hit;
}

HitResult rayMarching_Mipmap(float3 from, float3 direction, float maxDistance, bool ignoreFirst)
{
    float3 cell = from;
    int3 cellInt = int3(floor(from));
    float distance = 0;

    float3 reciprocal_direction = 1.0f / direction;
    // 可选的前进方向
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
            break;
        if (distance > maxDistance) // 非正数表示不限制距离
            break;

        int3 blockInt = cellInt / BLOCK_TO_VOXEL;
        if (isSolidBlock(blockInt))
        {
            int3 current = blockInt;
            while (all(current == blockInt))
            {
                if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
                    break;
                if (distance > maxDistance) // 非正数表示不限制距离
                    break;
                int offset;
                if (isSolidVoxel(cellInt, offset)) // 理论上不会出现面积为0的情况
                {
#if !CHECK_ELLIPSOID
                    HitResult hit = { true, cellInt, cellInt + 0.5f };
                    return hit;
#else
                    Ellipsoid ellipsoid = pBuffer[offset];
                    HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);
                    if (hit.hit)
                        return hit;
#endif
                }
                distance += leave(cell, cellInt, directions, direction, reciprocal_direction);
                current = cellInt / BLOCK_TO_VOXEL;
            }
        }
        else
        {
            distance += leaveBlock(cell, cellInt, directions, direction, reciprocal_direction);
        }
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f };
    return hit;
}

HitResult rayMarching(float3 from, float3 direction, float maxDistance, bool ignoreFirst)
{
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
    {
        HitResult hit = { false, int3(from), from };
        return hit;
    }
#if USE_MIP_MAP
    if (maxDistance > 0)
        return rayMarching_Mipmap(from, direction, maxDistance, ignoreFirst);
    return rayMarching_Mipmap(from, direction, ignoreFirst);
#else
    if (maxDistance > 0)
        return rayMarching_DDA(from, direction, maxDistance, ignoreFirst);
    return rayMarching_DDA(from, direction, maxDistance, ignoreFirst);
#endif
}

// calcTrasmissionVisibility / calcLightVisibility 没有放在这里：
// 它们读 CB 里的 shadowBias，属于光传输专用的东西，烘焙用不到。
// 留在 RayMarching.ps.slang 里，公共头只保留两边都需要的光追部分。
