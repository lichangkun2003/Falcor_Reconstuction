#pragma once
#include "../HostDeviceShared.slang"
#include "Utils/Math/MathConstants.slangh"

#include "VoxelData.slang"

BEGIN_NAMESPACE_FALCOR

struct GridData{
    uint3 gridSize;


    uint totalVoxelCount()
    {
        return gridSize.x * gridSize.y * gridSize.z;
    }

    uint2 blockGridSizeXY()
    {
        return uint2(
            (gridSize.x + BLOCK_TO_VOXEL - 1) / BLOCK_TO_VOXEL,
            (gridSize.y + BLOCK_TO_VOXEL - 1) / BLOCK_TO_VOXEL
        );
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


bool isSolidVoxel(int3 cellInt)
{
    if ( any(cellInt < 0) || any(cellInt >= int3(gGridDataParamBlock.gridSize)) )
        return false;
    uint offset = CellToIndex( uint3(cellInt),gGridDataParamBlock.gridSize);
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
#endif

END_NAMESPACE_FALCOR
