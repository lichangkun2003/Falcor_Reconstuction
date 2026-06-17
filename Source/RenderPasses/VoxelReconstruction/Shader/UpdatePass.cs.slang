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

    float gLrCenter;
    float gLrB;
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
bool isBadFloat(float x)
{
    return isnan(x) || isinf(x);
}

bool isBadFloat3(float3 x)
{
    return any(isnan(x)) || any(isinf(x));
}

float sanitizeFloat(float x)
{
    if (isBadFloat(x))
        return 0.0f;
    return x;
}

float3 sanitizeFloat3(float3 x)
{
    if (isBadFloat3(x))
        return float3(0.0f);
    return x;
}

void updateEllipsoidCenter(inout Ellipsoid ellipsoid, float3 gradCenter, float lr)
{
    if (lr <= 0.0f)
        return;

    gradCenter = sanitizeFloat3(gradCenter);

    float3 step = lr * gradCenter;

    // 防止一次跨太远。center 是 voxel-local 坐标，建议单步不要太大。
    step = clamp(step, float3(-0.02f), float3(0.02f));

    ellipsoid.center -= step;

    // center 保持在 voxel local 范围内。
    // 可以稍微留边，避免跑出 voxel 后数值很怪。
    ellipsoid.center = clamp(
        ellipsoid.center,
        float3(0.01f),
        float3(0.99f)
    );
}

void updateEllipsoidB(inout Ellipsoid ellipsoid, GradRecord grad, float lr, float gradScale)
{
    if (lr <= 0.0f)
        return;

    float gB00 = sanitizeFloat(grad.B00 * gradScale);
    float gB11 = sanitizeFloat(grad.B11 * gradScale);
    float gB22 = sanitizeFloat(grad.B22 * gradScale);
    float gB01 = sanitizeFloat(grad.B01 * gradScale);
    float gB02 = sanitizeFloat(grad.B02 * gradScale);
    float gB12 = sanitizeFloat(grad.B12 * gradScale);

    float sB00 = lr * gB00;
    float sB11 = lr * gB11;
    float sB22 = lr * gB22;
    float sB01 = lr * gB01;
    float sB02 = lr * gB02;
    float sB12 = lr * gB12;

    // B 的梯度更容易爆，单步限制小一点
    sB00 = clamp(sB00, -0.01f, 0.01f);
    sB11 = clamp(sB11, -0.01f, 0.01f);
    sB22 = clamp(sB22, -0.01f, 0.01f);
    sB01 = clamp(sB01, -0.005f, 0.005f);
    sB02 = clamp(sB02, -0.005f, 0.005f);
    sB12 = clamp(sB12, -0.005f, 0.005f);

    ellipsoid.B[0][0] -= sB00;
    ellipsoid.B[1][1] -= sB11;
    ellipsoid.B[2][2] -= sB22;

    // 保持 B 对称
    ellipsoid.B[0][1] -= sB01;
    ellipsoid.B[1][0] = ellipsoid.B[0][1];

    ellipsoid.B[0][2] -= sB02;
    ellipsoid.B[2][0] = ellipsoid.B[0][2];

    ellipsoid.B[1][2] -= sB12;
    ellipsoid.B[2][1] = ellipsoid.B[1][2];

    
    ellipsoid.B[0][0] = max(ellipsoid.B[0][0], 1e-4f);
    ellipsoid.B[1][1] = max(ellipsoid.B[1][1], 1e-4f);
    ellipsoid.B[2][2] = max(ellipsoid.B[2][2], 1e-4f);

    // 如果 B 变得非正定，需要增加 SPD projection。
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
    if (grad.valid == 0 && grad.geometryValid == 0)
        return;

    float materialGradScale = gGradScale;
    float geometryGradScale = gGradScale;

    if (gUseGradCountNormalize)
    {
        materialGradScale *= 1.0f / max(1.0f, float(grad.valid));
        geometryGradScale *= 1.0f / max(1.0f, float(grad.geometryValid));
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
            grad.diffuse[lobeId] * materialGradScale;

        float3 gradSpecular =
            grad.specular[lobeId] * materialGradScale;

        float gradRough =
            grad.rough[lobeId] * materialGradScale;

        float gradWeight =
            grad.weight[lobeId] * materialGradScale;

        float3 gradNormal =
            grad.normal[lobeId] * materialGradScale;

        // diffuse
        if (gLrDiffuse > 0.0f)
        {
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
                materialGradScale;
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
                materialGradScale;
        }
    }
#endif

    voxelData.bsdf = bsdf;

    // -------------------------
    // geometry update
    // -------------------------
    if (grad.geometryValid > 0)
    {
        Ellipsoid ellipsoid = voxelData.ellipsoid;

        if (gLrCenter > 0.0f)
        {
            float3 gradCenter = grad.center * geometryGradScale;
            updateEllipsoidCenter(ellipsoid, gradCenter, gLrCenter);
        }

        if (gLrB > 0.0f)
        {
            updateEllipsoidB(ellipsoid, grad, gLrB, geometryGradScale);
        }

        voxelData.ellipsoid = ellipsoid;
    }

    gGridDataParamBlock.gridDataBuffer[index] = voxelData;
}
