#pragma once
#include "../HostDeviceShared.slang"
#include "Utils/Math/MathConstants.slangh"

#include "VoxelData.slang"
#include "../GradRecord.slang"

#if RECON_MODE == RECON_MODE_POINT_CLOUD
#define VOXEL_DATA(ID) gGridDataParamBlock.voxelPages[NonUniformResourceIndex(uint(ID) / SPARSE_POOL_PAGE_SIZE)][uint(ID) % SPARSE_POOL_PAGE_SIZE]
#define GRAD_DATA(ID) gGridDataParamBlock.gradPages[NonUniformResourceIndex(uint(ID) / SPARSE_POOL_PAGE_SIZE)][uint(ID) % SPARSE_POOL_PAGE_SIZE]
#else
#define VOXEL_DATA(ID) gGridDataParamBlock.gridDataBuffer[ID]
#define GRAD_DATA(ID) gGradBuffer[ID]
#endif

BEGIN_NAMESPACE_FALCOR

struct GridData{
    uint3 voxelCount;
    float3 gridMin;
    float3 voxelSize;
    uint solidVoxelCount ;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    uint activeVoxelCount;
    uint voxelCapacity;
    uint3 indexPageCount;
#endif

    uint totalVoxelCount()
    {
        return voxelCount.x * voxelCount.y * voxelCount.z;
    }

#if !defined(HOST_CODE)
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    RWTexture3D<int> indexPages[SPARSE_INDEX_MAX_PAGES];
    RWStructuredBuffer<VoxelData> voxelPages[SPARSE_POOL_MAX_PAGES];
    RWStructuredBuffer<GradRecord> gradPages[SPARSE_POOL_MAX_PAGES];
    RWStructuredBuffer<uint> cellIndexPages[SPARSE_POOL_MAX_PAGES];
#else
    RWTexture3D<int> vBuffer;
    RWStructuredBuffer<VoxelData> gridDataBuffer;
#endif
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

// Spatial cell indices and compact voxel IDs are deliberately different quantities.
int lookupVoxelID(int3 cell)
{
    if (any(cell < 0) || any(cell >= int3(gGridDataParamBlock.voxelCount))) return -1;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    uint3 page = uint3(cell) / SPARSE_INDEX_PAGE_EDGE;
    uint pageID = page.x + page.y * gGridDataParamBlock.indexPageCount.x +
        page.z * gGridDataParamBlock.indexPageCount.x * gGridDataParamBlock.indexPageCount.y;
    int id = gGridDataParamBlock.indexPages[NonUniformResourceIndex(pageID)][uint3(cell) % SPARSE_INDEX_PAGE_EDGE];
    return id >= 0 && uint(id) < gGridDataParamBlock.activeVoxelCount ? id : -1;
#else
    return CellToIndex(uint3(cell), gGridDataParamBlock.voxelCount);
#endif
}

#if RECON_MODE == RECON_MODE_POINT_CLOUD
void setVoxelID(uint3 cell, uint id)
{
    uint3 page = cell / SPARSE_INDEX_PAGE_EDGE;
    uint pageID = page.x + page.y * gGridDataParamBlock.indexPageCount.x +
        page.z * gGridDataParamBlock.indexPageCount.x * gGridDataParamBlock.indexPageCount.y;
    gGridDataParamBlock.indexPages[NonUniformResourceIndex(pageID)][cell % SPARSE_INDEX_PAGE_EDGE] = int(id);
}

int3 getVoxelCell(uint voxelID)
{
    uint cellIndex = gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(voxelID / SPARSE_POOL_PAGE_SIZE)]
        [voxelID % SPARSE_POOL_PAGE_SIZE];
    return IndexToCell(int(cellIndex), gGridDataParamBlock.voxelCount);
}
#endif


bool isSolidVoxel(int3 cellInt)
{
    if (any(cellInt < 0) || any(cellInt >= int3(gGridDataParamBlock.voxelCount)) )
        return false;
    int offset = lookupVoxelID(cellInt);
    if (offset < 0) return false;
    //return offset >= 0;
    return VOXEL_DATA(offset).occupied != 0;
}


bool setSolidVoxel(int3 cellInt)
{
    if (any(cellInt < 0) || any(cellInt >= int3(gGridDataParamBlock.voxelCount)))
        return false;
    int offset = lookupVoxelID(cellInt);
    if (offset < 0) return false;
    VOXEL_DATA(offset).occupied = 1;
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
