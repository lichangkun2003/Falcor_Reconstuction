#pragma once
#include "../HostDeviceShared.slang"
#include "Utils/Math/MathConstants.slangh"

#include "VoxelData.slang"

BEGIN_NAMESPACE_FALCOR

struct GridData{
    uint3 voxelCount;
    float3 gridMin;
    float3 voxelSize;
    uint solidVoxelCount;

    uint totalVoxelCount()
    {
        return voxelCount.x * voxelCount.y * voxelCount.z;
    }

    uint2 blockGridSizeXY()
    {
        return uint2(
            (voxelCount.x + BLOCK_TO_VOXEL - 1) / BLOCK_TO_VOXEL,
                     (voxelCount.y + BLOCK_TO_VOXEL - 1) / BLOCK_TO_VOXEL
        );
    }
    uint3 blockCount3D()
    {
        return uint3((voxelCount.x + BLOCK_TO_VOXEL - 1) / BLOCK_TO_VOXEL, (voxelCount.y + BLOCK_TO_VOXEL - 1) / BLOCK_TO_VOXEL, (voxelCount.z + BLOCK_TO_VOXEL - 1) / BLOCK_TO_VOXEL);
    }

#if !defined(HOST_CODE)
    RWStructuredBuffer<VoxelData> gridDataBuffer;
    RWTexture2D<uint4> blockOM;
#endif
};

#if !defined(HOST_CODE)
ParameterBlock<GridData> gGridDataParamBlock;

inline uint CellToIndex(uint3 cell, uint3 size)
{
    return uint(cell.x) + uint(cell.y) * size.x + uint(cell.z) * size.x * size.y;
}

inline int3 IndexToCell(int index, uint3 size)
{
    int z = index / (size.x * size.y);
    int y = (index % (size.x * size.y)) / size.x;
    int x = index % size.x;
    return int3(x, y, z);
}


bool isSolidVoxel(int3 cellInt)
{
    if (any(cellInt < 0) || any(cellInt >= int3(gGridDataParamBlock.voxelCount)) )
        return false;
    uint offset = CellToIndex(uint3(cellInt), gGridDataParamBlock.voxelCount);
    return gGridDataParamBlock.gridDataBuffer[offset].occupied != 0;
}


bool isSolidBlock(int3 blockInt)
{
    //uint2 blockSize = gGridDataParamBlock.blockGridSizeXY();

    //if (blockInt.x < 0 || blockInt.y < 0 || blockInt.z < 0)
    //    return false;
    //if (blockInt.x >= blockSize.x || blockInt.y >= blockSize.y || blockInt.z >= 128)
    //    return false;

    uint4 zBits = gGridDataParamBlock.blockOM[blockInt.xy];
    uint bit = 1u << (blockInt.z % 32);

    switch (blockInt.z / 32)
    {
    case 0: return (zBits.x & bit) != 0;
    case 1: return (zBits.y & bit) != 0;
    case 2: return (zBits.z & bit) != 0;
    case 3: return (zBits.w & bit) != 0;
    default: return false;
    }
}

bool setSolidBlock(int3 blockInt)
{
    uint2 blockSize = gGridDataParamBlock.blockGridSizeXY();

    if (blockInt.x < 0 || blockInt.y < 0 || blockInt.z < 0)
        return false;
    if (blockInt.x >= blockSize.x || blockInt.y >= blockSize.y || blockInt.z >= 128)
        return false;

    uint4 zBits = gGridDataParamBlock.blockOM[blockInt.xy];
    uint bit = 1u << (blockInt.z % 32);

    switch (blockInt.z / 32)
    {
    case 0: gGridDataParamBlock.blockOM[blockInt.xy].x |= bit; break;
    case 1: gGridDataParamBlock.blockOM[blockInt.xy].y |= bit; break;
    case 2: gGridDataParamBlock.blockOM[blockInt.xy].z |= bit; break;
    case 3: gGridDataParamBlock.blockOM[blockInt.xy].w |= bit; break;
    default: return false;
    }
    //InterlockedOr(gGridDataParamBlock.blockOM[blockInt.xy].x, bit);

    return true;
}

bool setSolidVoxel(int3 cellInt)
{
    if (any(cellInt < 0) || any(cellInt >= int3(gGridDataParamBlock.voxelCount)))
        return false;
    uint offset = CellToIndex(uint3(cellInt), gGridDataParamBlock.voxelCount);
    gGridDataParamBlock.gridDataBuffer[offset].occupied = 1;
    return true;
}

// DDA
float leave(inout float3 cell, inout int3 cellInt, int3 directions, float3 direction, float3 reciprocal_direction)
{
    float3 border = cellInt + (directions + 1) / 2;
    float3 t = (border - cell) * reciprocal_direction;
    float tMin = 2;
    int3 selectedDirection = 0;
    if (direction.x != 0 && t.x < tMin)
    {
        tMin = t.x;
        selectedDirection = int3(directions.x, 0, 0);
    }
    if (direction.y != 0 && t.y < tMin)
    {
        tMin = t.y;
        selectedDirection = int3(0, directions.y, 0);
    }
    if (direction.z != 0 && t.z < tMin)
    {
        tMin = t.z;
        selectedDirection = int3(0, 0, directions.z);
    }

    cell += tMin * direction;
    cellInt += selectedDirection;
    return tMin;
}

float leaveBlockFace(float c, int cInt, float reciprocal)
{
    int temp = cInt % BLOCK_TO_VOXEL;
    float t = -1;
    if (reciprocal < 0)
        t = (cInt - temp - c) * reciprocal;
    if (reciprocal > 0)
        t = (cInt - temp + BLOCK_TO_VOXEL - c) * reciprocal;
    return t > 0 ? t : 1e30;
}

float leaveBlock(inout float3 cell, inout int3 cellInt, int3 directions, float3 direction, float3 reciprocal_direction)
{
    float tMin = 2 * BLOCK_TO_VOXEL;
    float t = leaveBlockFace(cell.x, cellInt.x, reciprocal_direction.x);
    tMin = min(t, tMin);
    t = leaveBlockFace(cell.y, cellInt.y, reciprocal_direction.y);
    tMin = min(t, tMin);
    t = leaveBlockFace(cell.z, cellInt.z, reciprocal_direction.z);
    tMin = min(t, tMin);
    cell += tMin * direction;
    cellInt = int3(cell + direction * 1e-3);
    return tMin;
}

#endif

END_NAMESPACE_FALCOR
