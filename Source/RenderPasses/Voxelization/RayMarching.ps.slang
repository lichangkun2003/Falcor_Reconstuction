#include "Voxel/VoxelGrid.slang"
#include "Math/VoxelizationUtility.slang"
#include "VoxelizationShared.slang"
#include "MixLightSampler.slang"
import Rendering.Materials.StandardMaterial;
import Utils.Sampling.UniformSampleGenerator;
#include "RayMarchingCommon.slang" // 资源声明 + HitResult + DDA，与烘焙共用同一份

#define MAX_BOUNCE 4

RWStructuredBuffer<float4> selectedVoxel;

cbuffer CB
{
    uint2 pixelCount;
    uint3 blockCount;
    float4x4 invVP;
    float shadowBias;
    uint drawMode;
    uint sampleStrategy;
    uint maxBounce;
    uint frameIndex;
    float minPdf;
    float trasmittanceThreshold;
    bool renderBackGround;
    float4 clearColor;
    uint2 selectedPixel;
}

struct PsOut
{
    float4 color : SV_Target0;
};

// 这两个读 CB 里的 shadowBias，只有前向渲染用得到，所以不放进 RayMarchingCommon.slang。
// 烘焙只需要 DDA，用不着阴影射线。
float calcTrasmissionVisibility(HitResult hit, float3 direction)
{
    hit = rayMarching(hit.cell, direction, -1, true);
    return hit.hit ? 0 : 1;
}

float calcLightVisibility(HitResult hit, float3 direction, float3 mainNormal, float distance)
{
    float ignoreDistance = 2; // 忽略光源附近的体素
    if (distance >= 0 && distance <= ignoreDistance)
        return 1;
    float3 from = hit.cell + shadowBias * mainNormal;
    hit = rayMarching(from, direction, distance - ignoreDistance, true);
    return hit.hit ? 0 : 1;
}

float3 coverageTest(HitResult hit, uint offset, float3 direction)
{
    float coverage = 0;
    float times = maxBounce;
    while (times > 0)
    {
        PrimitiveBSDF bsdf = gBuffer[offset];
        coverage += (1 - coverage) * bsdf.calcCoverage(direction);
        if (coverage >= 1 - trasmittanceThreshold)
        {
            coverage = 1;
            break;
        }
        hit = rayMarching(hit.cell, direction, -1, true);
        if (!hit.hit)
            break;
        offset = vBuffer[hit.cellInt];
        times--;
    }
    return valueToColor(coverage);
}

float3 debugShading(HitResult hit, uint offset, float3 v)
{
    PrimitiveBSDF bsdf = gBuffer[offset];
    Ellipsoid e = pBuffer[offset];
    float result = 0;
    switch (drawMode)
    {
    case ABSDFDrawMode::Normal:
        return bsdf.surface.getMainNormal(v) * 0.5 + 0.5;
    case ABSDFDrawMode::Ellipsoid:
        float3 localPos = hit.cell - hit.cellInt;
        float3 d = localPos - e.center;
        float3 p = d - v * dot(d, v);
        float3 Bp = mul(e.B, p);
        float3 Bn = mul(e.B, v);
        float denom = max(dot(v, Bn), 1e-8f);
        float val = dot(p, Bp) - (dot(p, Bn) * dot(p, Bn)) / denom;
        float t = 1.0f - safeSqrt(val);
        return pdfToColor(t, 1.0f);
    case ABSDFDrawMode::PolygonsProjArea:
        result = bsdf.polygonsProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::TotalProjArea:
        result = bsdf.totalProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::PrimitiveProjArea:
        result = bsdf.primitiveProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::InternalVisibility:
        result = bsdf.calcInternalVisibility(v);
        break;
    case ABSDFDrawMode::TotalArea:
        result = bsdf.area;
        break;
    default:
        break;
    }
    return valueToColor(result);
}

float3 shading(uint offset, float3 l, float3 v, float3 h, float3 lightColor)
{
    PrimitiveBSDF BSDF = gBuffer[offset];
    BSDF.updateCoverage(v, trasmittanceThreshold);
    return lightColor * BSDF.eval(l, v, h);
}

// 随机采样方向，不确定光源是否可见
float3 shadingDirect(HitResult hit, uint offset, float3 v, inout UniformSampleGenerator sg)
{
    PrimitiveBSDF BSDF = gBuffer[offset];
    float3 posW = gridMin + hit.cell * voxelSize;
    LightSample ls = {};
    bool valid = sampleLight(posW, v, BSDF, sg, ls);

    if (!valid || ls.pdf <= 0 || all(ls.h == 0))
        return 0;

    ls.pdf = max(ls.pdf, minPdf);
    float3 mainNormal = BSDF.surface.getMainNormal(ls.l);
    float visibility = calcLightVisibility(hit, ls.l, mainNormal, ls.distance / voxelSize.x);

    if (visibility > 0)
        return visibility * shading(offset, ls.l, v, ls.h, ls.lightColor) / ls.pdf;
    return 0;
}

float3 shadingGlobal(HitResult hit, float3 v, uint bounce, inout UniformSampleGenerator sg)
{
    // 非递归版本，先计算所有命中点，再倒回来统计
    HitResult[MAX_BOUNCE + 1] hits;
    float3[MAX_BOUNCE + 1] ls; // 第i个采样方向即第i个体素指向第i+1个体素的方向
    float3[MAX_BOUNCE + 1] hs;
    float[MAX_BOUNCE + 1] pdfs; // 第i个pdf即对第i个体素的BSDF采样得到的权重
    bounce = min(bounce, MAX_BOUNCE);
    uint shadingPointCount = bounce + 1; // 要着色的点的个数，未中断时为primary hit + bounce
    hits[0] = hit;
    bool leaveGrid = false;
    for (int i = 0; i < bounce; i++)
    {
        uint offset = vBuffer[hits[i].cellInt];
        PrimitiveBSDF BSDF = gBuffer[offset];
        float3 currentV = i == 0 ? v : -ls[i - 1];
        BSDF.updateCoverage(currentV, trasmittanceThreshold);
        BSDF.sample(currentV, ls[i], hs[i], sg);

        pdfs[i] = BSDF.evalPdf(ls[i], currentV, hs[i]);
        if (pdfs[i] <= 0)
        {
            shadingPointCount = i + 1; // 0,1,...,i;第i+1个点到第i个点的间接光照无法计算，但仍要计算直接光照
            break;
        }
        pdfs[i] = max(pdfs[i], minPdf);
        hits[i + 1] = rayMarching(hits[i].cell, ls[i], -1, true);
        if (!hits[i + 1].hit)
        {
            leaveGrid = true;
            shadingPointCount = i + 1; // 0,1,...,i
            break;
        }
    }

    float3 reflectLights[MAX_BOUNCE + 1];
    for (int i = shadingPointCount - 1; i >= 0; i--)
    {
        reflectLights[i] = 0;
        HitResult currentHit = hits[i];
        uint currentOffset = vBuffer[currentHit.cellInt];
        float3 currentV = i == 0 ? v : -ls[i - 1];

        reflectLights[i] += shadingDirect(currentHit, currentOffset, currentV, sg); // 直接光照
        if (i == shadingPointCount - 1)                                             // 最后一个着色点不计算间接光照
        {
#if USE_ENV_MAP
            if (leaveGrid && all(hs[i] == 0))
            {
                float3 incidentLight = gEnvMapSampler.eval(ls[i]);
                reflectLights[i] += shading(currentOffset, ls[i], currentV, hs[i], incidentLight) / pdfs[i]; // 透射环境光
            }
#endif
        }
        else
        {
            if (pdfs[i] <= 0)
                continue;
            HitResult nextHit = hits[i + 1];
            if (nextHit.hit)
            {
                uint nextOffset = vBuffer[nextHit.cellInt];
                float3 incidentLight = reflectLights[i + 1];
                reflectLights[i] += shading(currentOffset, ls[i], currentV, hs[i], incidentLight) / pdfs[i]; // 间接光照
            }
        }
    }
    return reflectLights[0];
}

float3 screenCoordToCell(float2 screenCoord, float NDCDepth)
{
    float4 NDC = float4(2 * screenCoord.x - 1, 1 - 2 * screenCoord.y, NDCDepth, 1);
    float4 temp = mul(invVP, NDC);
    float4 posWorld = temp / temp.w;
    float3 cell = (posWorld.xyz - gridMin) / voxelSize;
    return cell;
}
float2 randomJitter()
{
    UniformSampleGenerator sg = UniformSampleGenerator(0, frameIndex);
    float2 jitter = Next2(sg) - 0.5f;
    jitter /= pixelCount;
    return jitter;
}
float4 renderBackground(float3 direction)
{
#if USE_ENV_MAP
    if (renderBackGround)
    {
        float3 color = gScene.envMap.eval(direction);
        return float4(color, 1);
    }
#endif
    return clearColor;
}

PsOut main(float2 texCoord: TEXCOORD, float4 pos: SV_POSITION)
{
    PsOut psOut = {};
    uint2 pixel = uint2(texCoord * pixelCount);
    float2 jitter = randomJitter();
    texCoord = saturate(texCoord + jitter);
    float3 from = screenCoordToCell(texCoord, 0);
    float3 to = screenCoordToCell(texCoord, 1);
    float3 delta = to - from;
    float3 direction = normalize(delta); // 网格/世界空间下的方向相同
    psOut.color = renderBackground(direction);

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0)
        return psOut;

    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;
    HitResult hit = rayMarching(from, direction, -1, false);
    if (all(pixel == selectedPixel))
    {
        if (hit.hit)
        {
            int offset = vBuffer[hit.cellInt];
            selectedVoxel[0] = float4(-direction, (float)offset);
        }
        else
        {
            selectedVoxel[0] = float4(-1);
        }
    }

    if (hit.hit)
    {
        float3 color = 0;
        uint offset = vBuffer[hit.cellInt];
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
#if USE_EMISSIVE_LIGHTS
            float3 posW = gridMin + hit.cell * voxelSize;
            if (checkHitEmissive(posW, voxelSize.x, color))
            {
                psOut.color = float4(color, 1);
                return psOut; // 命中自发光面片时中止后续计算
            }
#endif
            color = shadingGlobal(hit, -direction, maxBounce, sg);
            break;
        case ABSDFDrawMode::Coverage:
            color = coverageTest(hit, offset, direction);
            break;
        default:
            color = debugShading(hit, offset, -direction);
            break;
        }
        psOut.color = float4(color, 1);
    }

    return psOut;
}
