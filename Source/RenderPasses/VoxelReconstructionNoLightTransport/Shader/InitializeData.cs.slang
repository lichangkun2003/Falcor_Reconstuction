//import RenderPasses.VoxelReconstruction.Voxel.VoxelGrid;

#include "../Voxel/VoxelGrid.slang"

Texture3D<int> vBuffer;
//StructuredBuffer<PrimitiveBSDF> gBuffer;
//StructuredBuffer<Ellipsoid> pBuffer;

cbuffer GridData {
    uint solidVoxelCount;

    float gLrCenter;
    float gLrShape;
    float gLrRadiance;
    float gLrOpacity;
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




[numthreads(4, 4, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    //uint index = CellToIndex(dispatchThreadId, gGridDataParamBlock.voxelCount);
    int3 cellInt = int3(dispatchThreadId);

    //gGridDataParamBlock.vBuffer[cellInt] = -1;

    int offset;
    //----------------------------------------------------
    //if (isSolidVoxelXu(cellInt, offset))
    {

        int index = CellToIndex(dispatchThreadId, gGridDataParamBlock.voxelCount);

        setSolidVoxel(cellInt);

        VoxelData data = gGridDataParamBlock.gridDataBuffer[index];
        data.opacity.init();
        data.opacity.coefficients[0] = 16.29f;

        //----------------------------------------------------
        //data.ellipsoid = pBuffer[offset];

        if (gLrRadiance > 0.0f) {
            data.radiance.init();
        }

        if( gLrOpacity > 0.0f) {
            data.opacity.init();
        }
       
        // -------------------------
        // geometry init
        // -------------------------
        if (gLrCenter > 0.0f)
        {
            data.ellipsoid.center = float3(0.5f, 0.5f, 0.5f);
        }

        if (gLrShape > 0.0f)
        {
            float3 initialScale = max(0.6f * gGridDataParamBlock.voxelSize, float3(1e-8f));
            data.ellipsoid.logScale = log(initialScale);
            data.ellipsoid.rotation = float4(1.0f, 0.0f, 0.0f, 0.0f);
        }

        gGridDataParamBlock.gridDataBuffer[index] = data;

        //gGridDataParamBlock.gridDataBuffer[index].init();
    }
}


