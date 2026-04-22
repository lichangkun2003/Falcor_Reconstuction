//import RenderPasses.VoxelReconstruction.Voxel.VoxelGrid;

#include "../Voxel/VoxelGrid.slang"

Texture3D<int> vBuffer;
Texture2D<uint4> blockMap;

cbuffer GridData {
    uint solidVoxelCount;
}

bool isSolidVoxelXu(int3 cellInt, out int offset)
{
    offset = vBuffer[cellInt];
    //if (offset < 0 || offset > solidVoxelCount)
    if (offset < 0 )
        return false;
    return true;
    //return gBuffer[offset].area > 0;
}

bool isSolidBlockXu(int3 blockInt)
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

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint index = CellToIndex(dispatchThreadId, gGridDataParamBlock.voxelCount);
    int3 cellInt = int3(dispatchThreadId);
    int3 blockInt = cellInt / BLOCK_TO_VOXEL;
    if (!isSolidBlockXu(blockInt))
        return;
    int offset;
    if (isSolidVoxelXu(cellInt, offset)) {
        setSolidBlock(blockInt);
        setSolidVoxel(cellInt);
    }
}


