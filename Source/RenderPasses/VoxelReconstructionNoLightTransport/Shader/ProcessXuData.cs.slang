//import RenderPasses.VoxelReconstruction.Voxel.VoxelGrid;

#include "../Voxel/VoxelGrid.slang"

Texture3D<int> vBuffer;
Texture2D<uint4> blockMap;
StructuredBuffer<PrimitiveBSDF> gBuffer;
StructuredBuffer<Ellipsoid> pBuffer;

cbuffer GridData {
    uint solidVoxelCount;

    float gLrCenter;
    float gLrB;
    float gLrRadiance;
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

float3 getWorldUpperHemisphereLobeCenter(uint lobeId, uint lobeCount)
{
    float n = max(1.0f, float(lobeCount));

    // z/y 轴选择：
    // 这里我用 y 作为世界上方向，即 y >= 0 的上半球。
    // 如果你的世界坐标是 z-up，就把 y 换成 z。
    float minY = 0.15f;

    float u = (float(lobeId) + 0.5f) / n;

    float y = lerp(1.0f, minY, u);
    float r = sqrt(max(0.0f, 1.0f - y * y));

    float phi = 2.39996323f * float(lobeId); // golden angle

    return safeNormalize(float3(
        r * cos(phi),
        y,
        r * sin(phi)
    ));
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
        //setSolidBlock(blockInt);
        setSolidVoxel(cellInt);

        VoxelData data = gGridDataParamBlock.gridDataBuffer[index];
        //data.init();

        data.ellipsoid = pBuffer[offset];

        if (gLrRadiance > 0.0f) {
            data.radiance.init();
        }
       
        // -------------------------
        // geometry init
        // -------------------------
        if (gLrCenter > 0.0f)
        {
            data.ellipsoid.center = float3(0.5f, 0.5f, 0.5f);
        }

        // phi = (x-c)^T B (x-c) - 1
        if (gLrB > 0.0f)
        {
            float r = 0.35f;
            float invR2 = 1.0f / max(r * r, 1e-6f);

            data.ellipsoid.B = float3x3(
                invR2, 0.0f, 0.0f,
                0.0f, invR2, 0.0f,
                0.0f, 0.0f, invR2
            );
        }

        gGridDataParamBlock.gridDataBuffer[index] = data;

        //gGridDataParamBlock.gridDataBuffer[index].init();
    }
}


