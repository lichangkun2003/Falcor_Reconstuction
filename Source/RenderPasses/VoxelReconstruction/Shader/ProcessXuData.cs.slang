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
        //setSolidBlock(blockInt);
        setSolidVoxel(cellInt);

        VoxelData data = gGridDataParamBlock.gridDataBuffer[index];

        data.bsdf.surface = gBuffer[offset].surface;
        data.bsdf.delta = gBuffer[offset].delta;
        // data.bsdf.surface.lobeCount = LOBE_COUNT;
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
                // 保留原始区域中心也可以；如果你要统一初始化，可以在这里设置
                float3 n = safeNormalize(lobe.n);
                if (dot(n, n) > 0.0f)
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


        gGridDataParamBlock.gridDataBuffer[index] = data;

        //gGridDataParamBlock.gridDataBuffer[index].init();
    }
}


