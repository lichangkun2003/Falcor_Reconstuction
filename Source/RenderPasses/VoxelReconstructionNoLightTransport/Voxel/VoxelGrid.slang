#pragma once
#include "../HostDeviceShared.slang"
#include "Utils/Math/MathConstants.slangh"

#include "VoxelData.slang"

BEGIN_NAMESPACE_FALCOR

struct GridData{
    uint3 voxelCount;
    float3 gridMin;
    float3 voxelSize;
    uint solidVoxelCount ;

    uint totalVoxelCount()
    {
        return voxelCount.x * voxelCount.y * voxelCount.z;
    }

#if !defined(HOST_CODE)
    RWTexture3D<int> vBuffer;
    RWStructuredBuffer<VoxelData> gridDataBuffer;
#endif
};

#if !defined(HOST_CODE)
ParameterBlock<GridData> gGridDataParamBlock;

inline int CellToIndex(uint3 cell, uint3 size)
{
    // return 0;
    //return gGridDataParamBlock.vBuffer[cell];
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
    int offset = CellToIndex(uint3(cellInt), gGridDataParamBlock.voxelCount);
    //return offset >= 0;
    return gGridDataParamBlock.gridDataBuffer[offset].occupied != 0;
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

#endif

END_NAMESPACE_FALCOR
