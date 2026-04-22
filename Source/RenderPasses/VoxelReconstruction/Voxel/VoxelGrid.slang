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
    uint2 blockSize = gGridDataParamBlock.blockGridSizeXY();

    if (blockInt.x < 0 || blockInt.y < 0 || blockInt.z < 0)
        return false;
    if (blockInt.x >= blockSize.x || blockInt.y >= blockSize.y || blockInt.z >= 128)
        return false;

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



#endif

END_NAMESPACE_FALCOR
