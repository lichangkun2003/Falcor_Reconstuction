//import RenderPasses.VoxelReconstruction.Voxel.VoxelGrid;

#include "../Voxel/VoxelGrid.slang"
#include "../Voxel/Shading.slang"

Texture3D<int> vBuffer;
Texture2D<uint4> blockMap;
StructuredBuffer<PrimitiveBSDF> gBuffer;
StructuredBuffer<Ellipsoid> pBuffer;

cbuffer GridData {
    uint solidVoxelCount;

    float gLrDiffuse;
    float gLrSpecular;
    float gLrRough;
    float gLrWeight;
    float gLrNormal;
    float gLrCoverageFunc;
    float gLrVisibilityFunc;

    float gLrCenter;
    float gLrB;
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

        data.bsdf.surface = gBuffer[offset].surface;
        data.bsdf.delta = gBuffer[offset].delta;
        //data.bsdf.surface.lobeCount = LOBE_COUNT;
        data.ellipsoid = pBuffer[offset];

        // 然后根据是否优化，覆盖初始值
        for (uint lobeId = 0; lobeId < LOBE_COUNT; lobeId++)
        {
            if (lobeId >= data.bsdf.surface.lobeCount)
                break;

            LobeBRDF lobe = data.bsdf.surface.lobes[lobeId];


            if (gLrDiffuse > 0.0f)
            {
                lobe.frostBite.diffuse = float3(1.0f);
            }

            if (gLrSpecular > 0.0f)
            {
                lobe.cookTorrence.specular = float3(0.2f);
            }

            if (gLrRough > 0.0f)
            {
                float rough = clamp(0.5f, MinRough, 1.0f);
                lobe.frostBite.rough = rough;
                lobe.cookTorrence.alpha = rough * rough;
            }

            if (gLrNormal > 0.0f)
            {
                float3 n = getWorldUpperHemisphereLobeCenter(lobeId, LOBE_COUNT);
                lobe.update(n);
            }

            data.bsdf.surface.lobes[lobeId] = lobe;

            if (gLrWeight > 0.0f)
            {
                data.bsdf.surface.weights[lobeId] =
                    1.0f / float(max(1u, data.bsdf.surface.lobeCount));
            }
        }

        if (gLrWeight > 0.0f)
        {
            // 如果你已经有 normalizeWeights 函数，也可以调用
            float weightSum = 0.0f;
            for (uint i = 0; i < LOBE_COUNT; i++)
            {
                if (i >= data.bsdf.surface.lobeCount)
                    break;
                weightSum += data.bsdf.surface.weights[i];
            }

            if (weightSum > 1e-6f)
            {
                float invSum = 1.0f / weightSum;
                for (uint i = 0; i < LOBE_COUNT; i++)
                {
                    if (i >= data.bsdf.surface.lobeCount)
                        break;
                    data.bsdf.surface.weights[i] *= invSum;
                }
            }
        }

        if (gLrCoverageFunc > 0.0f)
        {
            // 这里可以给 coverageFunc 一个可优化初始值
            // 比如初始为常数 1 或 0.8
            data.bsdf.coverageFunc.init();

            // calcEvenSH(0, p) = 1 / (2 * sqrt(pi))
            // 若想函数常数为 c，则 coeff[0] = c / calcEvenSH(0,p)
            float c = 1.0f;
            data.bsdf.coverageFunc.coefficients[0] =
                c / calcEvenSH(0, float3(0.0f, 0.0f, 1.0f));
        }


        if (gLrVisibilityFunc > 0.0f)
        {
            data.bsdf.visibilityFunc.init();

            float c = 1.0f;
            data.bsdf.visibilityFunc.coefficients[0] =
                c / calcEvenSH(0, float3(0.0f, 0.0f, 1.0f));
        }

        // -------------------------
        // geometry init
        // -------------------------
        if (gLrCenter > 0.0f)
        {
            data.ellipsoid.center = float3(0.5f, 0.5f, 0.5f);
        }

        // 如果开启 B 优化，可以把 B 初始化成一个稳定的球。
        // phi = (x-c)^T B (x-c) - 1
        // 若半径 r = 0.35，则 B = I / r^2。
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


