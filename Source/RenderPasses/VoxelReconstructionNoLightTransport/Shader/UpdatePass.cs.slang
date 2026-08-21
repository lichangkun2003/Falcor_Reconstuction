#include "../Voxel/VoxelGrid.slang"
#include "../GradRecord.slang"

// Input
StructuredBuffer<GradRecord> gGradBuffer;

StructuredBuffer<float> gVoxelSHGrads;

cbuffer CB {
    // 是否按当前 voxel 命中的 pixel 数做平均
    bool gUseGradCountNormalize;

    // 全局梯度缩放，第一版可以设为 1.0
    float gGradScale;

    float gLrRadiance;
    float gLrCenter;
    float gLrB;
    float gLrOpacity;

    float gEllipsoidPruneThreshold;
    bool gEnableEllipsoidPruning;


    uint gVoxelSHGradDim;
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

float detSym3x3(float a, float b, float c, float d, float e, float f)
{
    return a * (b * c - f * f)
         - d * (d * c - e * f)
         + e * (d * f - b * e);
}

bool isSPD(float3x3 B)
{
    float a = B[0][0];
    float b = B[1][1];
    float c = B[2][2];

    float d = B[0][1];
    float e = B[0][2];
    float f = B[1][2];

    if (a <= 1e-5f)
        return false;

    if (a * b - d * d <= 1e-5f)
        return false;

    if (detSym3x3(a, b, c, d, e, f) <= 1e-5f)
        return false;

    return true;
}

float3x3 makeTiltedEllipsoidB_Debug()
{
    float rx = 0.20f;
    float ry = 0.45f;
    float rz = 0.30f;

    float dx = 1.0f / (rx * rx);
    float dy = 1.0f / (ry * ry);
    float dz = 1.0f / (rz * rz);

    float angle = 0.78539816339f; // 45 degrees
    float c = cos(angle);
    float s = sin(angle);

    float3x3 B;

    // B = R * D * R^T, rotation around Z
    B[0][0] = c * c * dx + s * s * dy;
    B[1][1] = s * s * dx + c * c * dy;
    B[0][1] = c * s * (dx - dy);
    B[1][0] = B[0][1];

    B[0][2] = 0.0f;
    B[2][0] = 0.0f;

    B[1][2] = 0.0f;
    B[2][1] = 0.0f;

    B[2][2] = dz;

    return B;
}

void updateEllipsoidB(inout Ellipsoid ellipsoid, GradRecord grad, float lr, float gradScale)
{
    if (lr <= 0.0f)
        return;

    // --------------------------------
    // 1. 读取并 sanitize gradB
    // --------------------------------
    float gB00 = sanitizeFloat(grad.B00 * gradScale);
    float gB11 = sanitizeFloat(grad.B11 * gradScale);
    float gB22 = sanitizeFloat(grad.B22 * gradScale);
    float gB01 = sanitizeFloat(grad.B01 * gradScale);
    float gB02 = sanitizeFloat(grad.B02 * gradScale);
    float gB12 = sanitizeFloat(grad.B12 * gradScale);

    // 梯度保护
    float gradClamp = 10.0f;

    gB00 = clamp(gB00, -gradClamp, gradClamp);
    gB11 = clamp(gB11, -gradClamp, gradClamp);
    gB22 = clamp(gB22, -gradClamp, gradClamp);
    gB01 = clamp(gB01, -gradClamp, gradClamp);
    gB02 = clamp(gB02, -gradClamp, gradClamp);
    gB12 = clamp(gB12, -gradClamp, gradClamp);

    float3x3 oldB = ellipsoid.B;

    // --------------------------------
    // 2. 当前 B -> Cholesky L
    // --------------------------------
    float3x3 B = ellipsoid.B;

    // 强制对称
    B[1][0] = B[0][1];
    B[2][0] = B[0][2];
    B[2][1] = B[1][2];

    float3x3 L;
    bool ok = cholesky3x3SPD(B, L);

    if (!ok)
    {
        // 当前 B 已经不是 SPD 时，退回轴对齐合法 B
        B[0][0] = max(abs(B[0][0]), 1e-4f);
        B[1][1] = max(abs(B[1][1]), 1e-4f);
        B[2][2] = max(abs(B[2][2]), 1e-4f);

        B[0][1] = 0.0f; B[1][0] = 0.0f;
        B[0][2] = 0.0f; B[2][0] = 0.0f;
        B[1][2] = 0.0f; B[2][1] = 0.0f;

        ok = cholesky3x3SPD(B, L);

        if (!ok)
            return;
    }

    // --------------------------------
    // 3. 把 GradRecord 还原成 full matrix G = dL/dB
    //
    // 注意：
    // grad.B01 是对“对称参数 B01=B10”的梯度。
    // 如果放进 full matrix，off-diagonal 要乘 0.5。
    // 否则会 double count。
    // --------------------------------
    float3x3 G;

    G[0][0] = gB00;
    G[1][1] = gB11;
    G[2][2] = gB22;

    G[0][1] = 0.5f * gB01;
    G[1][0] = 0.5f * gB01;

    G[0][2] = 0.5f * gB02;
    G[2][0] = 0.5f * gB02;

    G[1][2] = 0.5f * gB12;
    G[2][1] = 0.5f * gB12;

    // --------------------------------
    // 4. B = L L^T，所以 dLoss/dL = 2 G L
    // 手写 lower-triangular 部分，避免 mul 行列约定问题
    // --------------------------------
    float gL00 =
        2.0f * (G[0][0] * L[0][0] + G[0][1] * L[1][0] + G[0][2] * L[2][0]);

    float gL10 =
        2.0f * (G[1][0] * L[0][0] + G[1][1] * L[1][0] + G[1][2] * L[2][0]);

    float gL11 =
        2.0f * (G[1][1] * L[1][1] + G[1][2] * L[2][1]);

    float gL20 =
        2.0f * (G[2][0] * L[0][0] + G[2][1] * L[1][0] + G[2][2] * L[2][0]);

    float gL21 =
        2.0f * (G[2][1] * L[1][1] + G[2][2] * L[2][1]);

    float gL22 =
        2.0f * (G[2][2] * L[2][2]);

    gL00 = sanitizeFloat(gL00);
    gL10 = sanitizeFloat(gL10);
    gL11 = sanitizeFloat(gL11);
    gL20 = sanitizeFloat(gL20);
    gL21 = sanitizeFloat(gL21);
    gL22 = sanitizeFloat(gL22);

    // --------------------------------
    // 5. 更新 L
    // 对角项用 log-space，保证为正
    // 非对角项直接更新
    // --------------------------------
    float lrDiag = lr;
    float lrOff = lr * 0.25f;

    // 非对角项单步限制
    float stepOff10 = clamp(lrOff * gL10, -0.01f, 0.01f);
    float stepOff20 = clamp(lrOff * gL20, -0.01f, 0.01f);
    float stepOff21 = clamp(lrOff * gL21, -0.01f, 0.01f);

    L[1][0] -= stepOff10;
    L[2][0] -= stepOff20;
    L[2][1] -= stepOff21;

    // 对角项 log-space
    float s00 = log(max(L[0][0], 1e-6f));
    float s11 = log(max(L[1][1], 1e-6f));
    float s22 = log(max(L[2][2], 1e-6f));

    float gS00 = sanitizeFloat(gL00 * L[0][0]);
    float gS11 = sanitizeFloat(gL11 * L[1][1]);
    float gS22 = sanitizeFloat(gL22 * L[2][2]);

    float stepS00 = clamp(lrDiag * gS00, -0.02f, 0.02f);
    float stepS11 = clamp(lrDiag * gS11, -0.02f, 0.02f);
    float stepS22 = clamp(lrDiag * gS22, -0.02f, 0.02f);

    s00 -= stepS00;
    s11 -= stepS11;
    s22 -= stepS22;

    L[0][0] = exp(s00);
    L[1][1] = exp(s11);
    L[2][2] = exp(s22);

    // 防止 L 极端
    L[0][0] = clamp(L[0][0], 0.02f, 50.0f);
    L[1][1] = clamp(L[1][1], 0.02f, 50.0f);
    L[2][2] = clamp(L[2][2], 0.02f, 50.0f);

    L[1][0] = clamp(L[1][0], -50.0f, 50.0f);
    L[2][0] = clamp(L[2][0], -50.0f, 50.0f);
    L[2][1] = clamp(L[2][1], -50.0f, 50.0f);

    // --------------------------------
    // 6. 重新生成 B
    // --------------------------------
    float3x3 newB = buildBFromCholesky(L);

    newB[0][0] = sanitizeFloat(newB[0][0]);
    newB[0][1] = sanitizeFloat(newB[0][1]);
    newB[0][2] = sanitizeFloat(newB[0][2]);

    newB[1][0] = newB[0][1];
    newB[1][1] = sanitizeFloat(newB[1][1]);
    newB[1][2] = sanitizeFloat(newB[1][2]);

    newB[2][0] = newB[0][2];
    newB[2][1] = newB[1][2];
    newB[2][2] = sanitizeFloat(newB[2][2]);

    // 理论上 L L^T 一定 SPD，这里只是最后保险
    if (!isSPD(newB))
    {
        ellipsoid.B = oldB;
        return;
    }

    //newB = makeTiltedEllipsoidB_Debug();
    ellipsoid.B = newB;
    
}

void updateOpacitySH(inout SphericalHarmonicsOpacity opacity,GradRecord grad,float lr,float gradScale)
{
    if (lr <= 0.0f)
        return;

    SH_MAX_ITERS(SH_OPACITY_COUNT)
    for (uint i = 0; i < SH_OPACITY_COUNT; ++i)
    {
        float g =
            grad.opacityGrad.coefficients[i] *
            gradScale;

        g = sanitizeFloat(g);

        float step = clamp(
            lr * g,
            -0.05f,
            0.05f
        );

        // 梯度下降
        opacity.coefficients[i] -= step;

        //opacity.coefficients[i] = clamp(
        //    sanitizeFloat(opacity.coefficients[i]),
        //    -10.0f,
        //    10.0f
        //);
    }
}

void updateRadianceSH(inout SphericalHarmonicsRGB radiance, GradRecord grad, float lr, float gradScale)
{
    if (lr <= 0.0f)
        return;

    for (uint i = 0; i < SH_COUNT; i++)
    {
        float3 g = grad.radianceGrad.coefficients[i] * gradScale;

        g = sanitizeFloat3(g);

        float3 step = lr * g;

        // 防止 SH 更新一步过大，第一版建议限制一下
        step = clamp(step, float3(-0.05f), float3(0.05f));

        radiance.coefficients[i] -= step;

        // 防止参数本身变成 NaN/Inf
        radiance.coefficients[i] = sanitizeFloat3(radiance.coefficients[i]);
    }
}


float calcEllipsoidVoxelVolumeRatio(Ellipsoid ellipsoid)
{
    float detB = determinant(ellipsoid.B);

    if (isBadFloat(detB) || detB <= 1e-12f)
        return -1.0f;

    // 局部体素为 [0,1]^3，体积为 1
    float ratio = (4.0f * PI / 3.0f) / sqrt(detB);

    if (isBadFloat(ratio))
        return -1.0f;

    return ratio;
}

void pruneEllipsoid(inout VoxelData voxel)
{
    if (voxel.occupied == 0)
        return;

    // 删除截断椭球体积占比较小的voxel
    float ratio = calcEllipsoidVoxelVolumeRatio(voxel.ellipsoid);
    // B异常时暂不删除
    if (ratio < 0.0f)
        return;

    if (ratio < gEllipsoidPruneThreshold)
        voxel.occupied = 0;
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

    int index = CellToIndex(voxel, gGridDataParamBlock.voxelCount);

    if (index < 0)
        return;

    GradRecord grad = gGradBuffer[(uint)index];

    // 这里 valid 实际上表示当前 voxel 累计了多少个 pixel contribution
    //if (grad.radianceValid == 0 && grad.geometryValid == 0)
    //    return;

    float appearanceGradScale = gGradScale;
    float geometryGradScale = gGradScale;

    if (gUseGradCountNormalize)
    {
        appearanceGradScale *= 1.0f / max(1.0f, float(grad.appearanceValid));
        geometryGradScale *= 1.0f / max(1.0f, float(grad.geometryValid));
    }

    VoxelData voxelData = gGridDataParamBlock.gridDataBuffer[index];
    if (voxelData.occupied == 0)
        return;

    // -------------------------
    // appearance  SH update
    // -------------------------
    if (grad.appearanceValid > 0)
    {
        if (gLrRadiance > 0.0f)
        {
            updateRadianceSH(voxelData.radiance,grad,gLrRadiance,appearanceGradScale);
        }

        if (gLrOpacity > 0.0f)
        {
            updateOpacitySH(voxelData.opacity,grad,gLrOpacity,appearanceGradScale);
        }
    }

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

    if (gEnableEllipsoidPruning)
    {
        pruneEllipsoid(voxelData);
    }


    gGridDataParamBlock.gridDataBuffer[index] = voxelData;
}
