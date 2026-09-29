#include "../Voxel/VoxelGrid.slang"

cbuffer CB
{
    uint gVoxelOffset;
    uint gVoxelCount;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gVoxelOffset + dispatchThreadId.x;
    if (voxelID >= gVoxelCount || voxelID >= gGridDataParamBlock.activeVoxelCount) return;

    uint cellIndex = gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(voxelID / SPARSE_POOL_PAGE_SIZE)]
        [voxelID % SPARSE_POOL_PAGE_SIZE];
    setVoxelID(uint3(IndexToCell(int(cellIndex), gGridDataParamBlock.voxelCount)), voxelID);
}
