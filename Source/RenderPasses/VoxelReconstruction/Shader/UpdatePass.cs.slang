#include "../Voxel/VoxelGrid.slang"
#include "../GradRecord.slang"

// Input
StructuredBuffer<GradRecord> gGradBuffer;

cbuffer CB {
    // 是否按当前 voxel 命中的 pixel 数做平均
    bool gUseGradCountNormalize;

    // 全局梯度缩放，第一版可以设为 1.0
    float gGradScale;

    float gLrDiffuse;
    float gLrSpecular;
    float gLrRough;
    float gLrWeight;
    float gLrNormal;
    float gLrCoverageFunc;
    float gLrVisibilityFunc;
}

void updateLobeNormal(inout LobeBRDF lobe, float3 gradNormal, float lr)
{
    if (lr <= 0.0f)
        return;

    float3 n = safeNormalize(lobe.n);

    // normal 是单位向量，只保留切平面梯度
    gradNormal = gradNormal - n * dot(gradNormal, n);

    n = safeNormalize(n - lr * gradNormal);

    // 更新 normal 后必须重建 TBN
    lobe.update(n);
}

void normalizeWeights(inout SurfaceBRDF surface)
{
    float weightSum = 0.0f;

    for (uint i = 0; i < LOBE_COUNT; i++)
    {
        if (i >= surface.lobeCount)
            break;

        surface.weights[i] = max(surface.weights[i], 0.0f);
        weightSum += surface.weights[i];
    }

    if (weightSum > 1e-6f)
    {
        float invSum = 1.0f / weightSum;

        for (uint i = 0; i < LOBE_COUNT; i++)
        {
            if (i >= surface.lobeCount)
                break;

            surface.weights[i] *= invSum;
        }
    }
    else
    {
        float uniformWeight = 1.0f / max(1.0f, float(surface.lobeCount));

        for (uint i = 0; i < LOBE_COUNT; i++)
        {
            if (i >= surface.lobeCount)
                break;

            surface.weights[i] = uniformWeight;
        }
    }
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint3 voxel = dispatchThreadId;
    if (voxel.x >= gGridDataParamBlock.voxelCount.x ||
    voxel.y >= gGridDataParamBlock.voxelCount.y ||
        voxel.z >= gGridDataParamBlock.voxelCount.z)
    {
        return;
    }

    uint index = CellToIndex(voxel, gGridDataParamBlock.voxelCount);

    GradRecord grad = gGradBuffer[index];

    // 这里 valid 实际上表示当前 voxel 累计了多少个 pixel contribution
    if (grad.valid == 0)
        return;

    float gradScale = gGradScale;

    if (gUseGradCountNormalize)
    {
        gradScale *= 1.0f / max(1.0f, float(grad.valid));
    }

    // Voexl Data : gGridDataParamBlock.gridDataBuffer[index]....
    VoxelData voxelData = gGridDataParamBlock.gridDataBuffer[index];

    PrimitiveBSDF_New bsdf = voxelData.bsdf;

    for (uint lobeId = 0; lobeId < LOBE_COUNT; lobeId++)
    {
        LobeBRDF lobe = bsdf.surface.lobes[lobeId];

        if (lobeId >= bsdf.surface.lobeCount)
            break;


        float3 gradDiffuse =
            grad.diffuse[lobeId] * gradScale;

        float3 gradSpecular =
            grad.specular[lobeId] * gradScale;

        float gradRough =
            grad.rough[lobeId] * gradScale;

        float gradWeight =
            grad.weight[lobeId] * gradScale;

        float3 gradNormal =
            grad.normal[lobeId] * gradScale;

        // diffuse
        if (gLrDiffuse > 0.0f)
        {
            gradDiffuse /= max(1.0f, float(grad.valid));
            if (any(isnan(gradDiffuse)) || any(isinf(gradDiffuse)))
            {
                gradDiffuse = float3(0.0f);
            }
            float3 step = gLrDiffuse * gradDiffuse;

            step = clamp(step, float3(-0.003f), float3(0.003f));

            lobe.frostBite.diffuse -= step;
            lobe.frostBite.diffuse = clamp(lobe.frostBite.diffuse, float3(0.0f), float3(2.0f));
        }

        // specular
        if (gLrSpecular > 0.0f)
        {
            lobe.cookTorrence.specular -= gLrSpecular * gradSpecular;
            lobe.cookTorrence.specular = saturate(lobe.cookTorrence.specular);
            lobe.specularWeight = lobe.evalSpecularWeight();
        }

        // rough
        if (gLrRough > 0.0f)
        {
            float rough = lobe.frostBite.rough;
            rough -= gLrRough * gradRough;
            rough = clamp(rough, MinRough, 1.0f);

            lobe.frostBite.rough = rough;

            // alpha 不单独优化，由 rough 派生
            lobe.cookTorrence.alpha = rough * rough;
        }

        // weight
        if (gLrWeight > 0.0f)
        {
            bsdf.surface.weights[lobeId] -= gLrWeight * gradWeight;
            bsdf.surface.weights[lobeId] = max(bsdf.surface.weights[lobeId], 0.0f);
        }

        // normal
        updateLobeNormal(lobe, gradNormal, gLrNormal);

        bsdf.surface.lobes[lobeId] = lobe;
    }

    // weight 更新后统一归一化
    if (gLrWeight > 0.0f)
    {
        normalizeWeights(bsdf.surface);
    }

    // -------------------------
    // coverageFunc update
    // -------------------------
#if CHECK_COVERAGE
    if (gLrCoverageFunc > 0.0f)
    {
        for (uint i = 0; i < EVEN_SH_COUNT; i++)
        {
            bsdf.coverageFunc.coefficients[i] -=
                gLrCoverageFunc *
                grad.coverageFunc.coefficients[i] *
                gradScale;
        }
    }
#endif

    // -------------------------
    // visibilityFunc update
    // -------------------------
#if CHECK_VISIBILITY
    if (gLrVisibilityFunc > 0.0f)
    {
        for (uint i = 0; i < EVEN_SH_COUNT; i++)
        {
            bsdf.visibilityFunc.coefficients[i] -=
                gLrVisibilityFunc *
                grad.visibilityFunc.coefficients[i] *
                gradScale;
        }
    }
#endif

    voxelData.bsdf = bsdf;

    gGridDataParamBlock.gridDataBuffer[index] = voxelData;
}
