#pragma once
#include "RayMarchingCommon.slang" // 资源声明 + HitResult + rayMarching，与前向渲染共用同一份
#include "Shading.slang"           // PrimitiveBSDF / SurfaceBRDF / LobeBRDF
#include "Math/LebedevQuadrature.slang"

// ============================================================================
// 把 Voxelization 的 ABSDF 表达烘焙成 VoxelReconstructionNoLightTransport 的表达。
//
// 假设：纯白天光（skymap 恒为 1）。只算直接光——射线逃出网格就拿到白色，
// 命中别的体素就记 0。要考虑反弹时，把 V[k] 换成读命中体素在该方向上的 radiance 即可。
//
// 对应关系（推导见工程笔记）：
//   alpha(v) = coverage(v)                     —— 原模型的 mixture 概率，逐项对应到确定性合成
//   L(v)     = ∫ surface.eval(l,v,h) · internalVis(l) · V(l) dl
//
// 关键：必须用 surface.eval，不能用 PrimitiveBSDF::eval。
// 后者会再乘一次 coverage，而 coverage 已经作为 alpha 单独输出了，乘两次就是重复计数。
// ============================================================================

// ---------------------------------------------------------------------------
// 输出布局：必须与 VoxelReconstructionNoLightTransport 的 VoxelData 逐字节一致。
// 那边是 { uint occupied; Ellipsoid ellipsoid; SphericalHarmonicsRGB radiance; SphericalHarmonicsOpacity opacity; }
// 两个插件的 VoxelData 同名，无法同时 include，所以这里手写等价布局。
//
// 偏移（StructuredBuffer 的紧凑排布，float3 按 4 字节对齐）：
//     0  uint       occupied
//     4  float3     ellipsoid.center
//    16  float3x3   ellipsoid.B
//    52  float3[9]  radiance.coefficients
//   160  float[9]   opacity.coefficients
//   196  总大小
// 任一字段改动都必须同步改那边的 VoxelData.slang / SphericalHarmonics.slang，
// 并且两边 sizeof 都必须是 196（重建端加载时会校验 voxelDataSize，对不上会直接报错）。
// ---------------------------------------------------------------------------
struct ReconEllipsoid
{
    float3 center;
    float3x3 B;
};

struct ReconVoxelData
{
    uint occupied;
    ReconEllipsoid ellipsoid;
    float3 radiance[9];
    float opacity[9];
};

RWStructuredBuffer<ReconVoxelData> bakeOutput;

cbuffer BakeCB
{
    // logit 之前把 coverage 夹到的边界。coverage 可能取到精确的 0 或 1，
    // 直接取 logit 会得到 ±inf，会把整条 SH 污染成 NaN。
    float coverageEps;
    float bakePad0;
    float bakePad1;
    float bakePad2;
}

// Lebedev 权重之和为 1，乘 4pi 才是一整套球面积分权重。
// 这里不引用已有的 PI 宏，避免与 import 进来的符号重名。
#define BAKE_FOUR_PI 12.56637061435917295f

// 重建端的 SH 基与 Voxelization 的 calcSH 在 i = 0..8 上逐行相同
// （都来自 Math/SphericalHarmonics.slang，SQRTPI 也一致），所以直接复用 calcSH。
#define BAKE_SH_COUNT 9

float bakeLogit(float p)
{
    return log(p / (1.0f - p));
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int index = (int)tid.x;

    if (index >= (int)(voxelCount.x * voxelCount.y * voxelCount.z))
        return;

    int3 cellInt = IndexToCell(index, voxelCount);

    ReconVoxelData outData;
    outData.occupied = 0;
    outData.ellipsoid.center = float3(0);
    outData.ellipsoid.B = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < BAKE_SH_COUNT; i++)
    {
        outData.radiance[i] = float3(0);
        outData.opacity[i] = 0.0f;
    }

    // 空体素也要写回去——重建端是稠密网格，靠 occupied 区分
    int offset = vBuffer[cellInt];
    if (offset < 0 || offset >= (int)solidVoxelCount)
    {
        bakeOutput[index] = outData;
        return;
    }

    PrimitiveBSDF bsdf = gBuffer[offset];
    if (bsdf.area <= 0)
    {
        bakeOutput[index] = outData;
        return;
    }

    Ellipsoid e = pBuffer[offset];

    // 发射点取椭球中心（网格坐标）。网格是等大的（voxelSize 三个分量相同），
    // 所以网格空间和世界空间的方向一致，Lebedev 方向可以同时喂给 DDA 和 BRDF。
    // 若哪天 AABB 不是立方体，这个前提就没了，必须改成带轴缩放的转换。
    float3 origin = float3(cellInt) + e.center;

    // 入射方向的可见性。与出射方向 v 无关，所以每个方向只算一次。
    // V = 是否逃出网格 · 体素内部自遮挡
    float V[LEBEDEV_DIRECTION_COUNT];
    // coverage 是出射方向 v 的函数，这里按方向采样，最后投影成 opacity 的 SH
    float cov[LEBEDEV_DIRECTION_COUNT];

    for (uint k = 0; k < LEBEDEV_DIRECTION_COUNT; k++)
    {
        float3 l = LebedevSamples[k].direction;
        // ignoreFirst 跳过自己所在的体素，否则会立刻撞到自己的椭球
        HitResult hit = rayMarching(origin, l, -1, true);
        float vis = hit.hit ? 0.0f : 1.0f; // 直接光：只有逃出网格才拿到白色天光
        V[k] = vis * bsdf.calcInternalVisibility(l);
        cov[k] = bsdf.calcCoverage(l);
    }

    float3 radianceCoeff[BAKE_SH_COUNT];
    float opacityCoeff[BAKE_SH_COUNT];
    for (int i = 0; i < BAKE_SH_COUNT; i++)
    {
        radianceCoeff[i] = float3(0);
        opacityCoeff[i] = 0.0f;
    }

    for (uint j = 0; j < LEBEDEV_DIRECTION_COUNT; j++)
    {
        float3 v = LebedevSamples[j].direction;

        // L(v) = ∫ surface.eval(l,v,h) · internalVis(l) · V(l) dl
        float3 L = float3(0);
        for (uint k = 0; k < LEBEDEV_DIRECTION_COUNT; k++)
        {
            if (V[k] <= 0.0f)
                continue;
            float3 l = LebedevSamples[k].direction;
            // l 与 v 恰好反向时 normalize(0) 会出 NaN。这个方向的贡献本就是 0
            // （两个 lobe 的 l.z <= 0 早退都会返回 0），直接跳过。
            float3 lv = l + v;
            float lvLen2 = dot(lv, lv);
            if (lvLen2 < 1e-12f)
                continue;
            float3 h = lv * rsqrt(lvLen2); // 微表面半程向量，与渲染器 sample() 的约定一致
            L += LebedevSamples[k].weight * V[k] * bsdf.surface.eval(l, v, h);
        }
        L *= BAKE_FOUR_PI;

        float opacityValue = bakeLogit(clamp(cov[j], coverageEps, 1.0f - coverageEps));
        float weight = BAKE_FOUR_PI * LebedevSamples[j].weight;

        for (int i = 0; i < BAKE_SH_COUNT; i++)
        {
            float Y = calcSH(i, v);
            radianceCoeff[i] += weight * L * Y;
            opacityCoeff[i] += weight * opacityValue * Y;
        }
    }

    outData.occupied = 1;
    outData.ellipsoid.center = e.center;
    outData.ellipsoid.B = e.B;
    for (int i = 0; i < BAKE_SH_COUNT; i++)
    {
        outData.radiance[i] = radianceCoeff[i];
        outData.opacity[i] = opacityCoeff[i];
    }

    bakeOutput[index] = outData;
}
