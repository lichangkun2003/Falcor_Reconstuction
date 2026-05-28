//import RenderPasses.VoxelReconstruction.Voxel.VoxelGrid;

import Rendering.Materials.StandardMaterial;
import Utils.Sampling.UniformSampleGenerator;
import Utils.Math.MathHelpers;
import Scene.SceneTypes;
import Scene.Scene;

#include "../Math/AABB.slang"
#include "../Math/Random.slang"
#include "../Voxel/Shading.slang"
#include "../Voxel/VoxelGrid.slang"
#include "../MixLightSampler.slang"
#include "../PathRecord.slang"

cbuffer GridData
{
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
    uint solidVoxelCount;
}

cbuffer CB
{
    uint2 pixelCount;
    uint3 blockCount;
    float4x4 invVP;
    float shadowBias;
    uint drawMode;
    //uint sampleStrategy;
    uint maxBounce;
    uint frameIndex;
    float minPdf;
    float trasmittanceThreshold;
    bool renderBackGround;
    float4 clearColor;
    // uint2 selectedPixel;
    bool enableReconstruction;
}

RWStructuredBuffer<PathRecord> gPathRecordBuffer;


struct PsOut
{
    float4 color : SV_Target0;
};

struct HitResult
{
    bool hit;
    int3 cellInt;
    float3 cell;
};

float3 screenCoordToCell(float2 screenCoord, float NDCDepth){
    float4 NDC = float4(2 * screenCoord.x - 1, 1 - 2 * screenCoord.y, NDCDepth, 1);
    float4 temp = mul(invVP, NDC);
    float4 posWorld = temp / temp.w;
    float3 cell = (posWorld.xyz - gridMin) / voxelSize;
    return cell;
}
float2 randomJitter(){
    UniformSampleGenerator sg = UniformSampleGenerator(0, frameIndex);
    float2 jitter = Next2(sg) - 0.5f;
    jitter /= pixelCount;
    return jitter;
}

float4 renderBackground(float3 direction){
    #if USE_ENV_MAP
    if (renderBackGround)
    {
        float3 color = gScene.envMap.eval(direction);
        return float4(color, 1);
    }
    #endif
    return clearColor;
}

// 与体素内部primitive求交，即判断from-to线段上有无高密度点
HitResult checkPrimitive(DensityPrimitive prim, float3 cell, float3 direction, int3 cellInt)
{
    // voxel-local entry point
    float3 from = cell - float3(cellInt);

    int3 directions = int3(sign(direction));
    float3 border = float3((directions + 1) / 2);

    float3 t = (border - from) / direction;

    float tMin = 2.0f;
    if (direction.x != 0 && t.x < tMin)
        tMin = t.x;
    if (direction.y != 0 && t.y < tMin)
        tMin = t.y;
    if (direction.z != 0 && t.z < tMin)
        tMin = t.z;

    float3 to = from + tMin * direction;

    float segLen = length(to - from);

    const int SAMPLE_COUNT = 8;
    // const float hitAlpha = 0.1f;
    const float hitAlpha = (randomJitter() * pixelCount).x + 0.5;

    float targetTau = -log(1.0f - hitAlpha);
    float tau = 0.0f;

    float3 localHit = to;
    bool hit = false;

    [unroll]
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        float u0 = float(i) / float(SAMPLE_COUNT);
        float u1 = float(i + 1) / float(SAMPLE_COUNT);
        float um = 0.5f * (u0 + u1);

        float3 p = lerp(from, to, um);

        float sigma = prim.evalDensity(p);
        float dt = segLen / float(SAMPLE_COUNT);

        tau += sigma * dt;

        if (!hit && tau >= targetTau)
        {
            localHit = p;
            hit = true;
            break;
        }
    }

    HitResult result = {
        hit,
        cellInt,
        float3(cellInt) + localHit
    };

    return result;
}

// 先后用体素和椭球裁剪线段
HitResult checkPrimitive(Ellipsoid ellipsoid, float3 cell, float3 direction, int3 cellInt)
{
    float3 from = cell - cellInt;
    int3 directions = int3(sign(direction));
    float3 border = (directions + 1) / 2;
    float3 t = (border - from) / direction;
    float tMin = 2.0f;
    if (direction.x != 0 && t.x < tMin)
        tMin = t.x;
    if (direction.y != 0 && t.y < tMin)
        tMin = t.y;
    if (direction.z != 0 && t.z < tMin)
        tMin = t.z;
    float3 to = from + tMin * direction;
    float2 inOut = ellipsoid.clip(from, to);
    HitResult result = { inOut.x >= 0, cellInt, cell + inOut.x * (to - from) };
    return result;
}

HitResult rayMarching_Mipmap(float3 from, float3 direction, bool ignoreFirst)
{
    float3 cell = from;
    int3 cellInt = int3(floor(from));

    float3 reciprocal_direction = 1.0f / direction;
    // 可选的前进方向
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
            break;

        int3 blockInt = cellInt / BLOCK_TO_VOXEL;
        if (isSolidBlock(blockInt))
        {
            int3 current = blockInt;
            while (all(current == blockInt))
            {
                if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
                    break;
                //int offset;
                if (isSolidVoxel(cellInt)) // 理论上不会出现面积为0的情况
                {
#if !CHECK_PRIMITIVE
                    HitResult hit = { true, cellInt, cellInt + 0.5f };
                    return hit;
#else
                    uint offset = CellToIndex(cellInt, gGridDataParamBlock.voxelCount);
                    Ellipsoid ellipsoid = gGridDataParamBlock.gridDataBuffer[offset].ellipsoid;
                    HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);
                    //HitResult hit = checkPrimitive(gGridDataParamBlock.gridDataBuffer[offset].densityPrimitive, cell, direction, cellInt);
                    if (hit.hit)
                        return hit;
#endif
                }
                leave(cell, cellInt, directions, direction, reciprocal_direction);
                current = cellInt / BLOCK_TO_VOXEL;
            }
        }
        else
        {
            leaveBlock(cell, cellInt, directions, direction, reciprocal_direction);
        }
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f };
    return hit;
}

HitResult rayMarching_Mipmap(float3 from, float3 direction, float maxDistance, bool ignoreFirst)
{
    float3 cell = from;
    int3 cellInt = int3(floor(from));
    float distance = 0;

    float3 reciprocal_direction = 1.0f / direction;
    // 可选的前进方向
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
            break;
        if (distance > maxDistance) // 非正数表示不限制距离
            break;

        int3 blockInt = cellInt / BLOCK_TO_VOXEL;
        if (isSolidBlock(blockInt))
        {
            int3 current = blockInt;
            while (all(current == blockInt))
            {
                if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
                    break;
                if (distance > maxDistance) // 非正数表示不限制距离
                    break;
                //int offset;
                if (isSolidVoxel(cellInt)) // 理论上不会出现面积为0的情况
                {
#if !CHECK_PRIMITIVE
                    HitResult hit = { true, cellInt, cellInt + 0.5f };
                    return hit;
#else
                    uint offset = CellToIndex(cellInt, gGridDataParamBlock.voxelCount);
                    Ellipsoid ellipsoid = gGridDataParamBlock.gridDataBuffer[offset].ellipsoid;
                    HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);
                    //HitResult hit = checkPrimitive(gGridDataParamBlock.gridDataBuffer[offset].densityPrimitive, cell, direction, cellInt);
                    if (hit.hit)
                        return hit;
#endif
                }
                distance += leave(cell, cellInt, directions, direction, reciprocal_direction);
                current = cellInt / BLOCK_TO_VOXEL;
            }
        }
        else
        {
            distance += leaveBlock(cell, cellInt, directions, direction, reciprocal_direction);
        }
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f };
    return hit;
}


HitResult rayMarching(float3 from, float3 direction, float maxDistance, bool ignoreFirst){
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
    {
        HitResult hit = { false, int3(from), from };
        return hit;
    }
#if USE_MIP_MAP
    if (maxDistance > 0)
        return rayMarching_Mipmap(from, direction, maxDistance, ignoreFirst);
    return rayMarching_Mipmap(from, direction, ignoreFirst);
#else
    // if (maxDistance > 0)
    //     return rayMarching_DDA(from, direction, maxDistance, ignoreFirst);
    // return rayMarching_DDA(from, direction, maxDistance, ignoreFirst);
    return rayMarching_Mipmap(from, direction, ignoreFirst);
#endif
}

float3 shading(uint offset, float3 l, float3 v, float3 h, float3 lightColor)
{
    PrimitiveBSDF_New BSDF = gGridDataParamBlock.gridDataBuffer[offset].bsdf;
    BSDF.updateCoverage(v, trasmittanceThreshold);
    return lightColor * BSDF.eval(l, v, h);
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


// 随机采样方向，不确定光源是否可见
float3 shadingDirect(HitResult hit, uint offset, float3 v, uint2 pixel,inout UniformSampleGenerator sg)
{
    PrimitiveBSDF_New BSDF = gGridDataParamBlock.gridDataBuffer[offset].bsdf;
    float3 posW = gridMin + hit.cell * voxelSize;
    LightSample ls = {};
    bool valid = sampleLight(posW, v, BSDF, sg, ls);

    if (!valid || ls.pdf <= 0 || all(ls.h == 0))
        return 0;

    ls.pdf = max(ls.pdf, minPdf);
    float3 mainNormal = BSDF.surface.getMainNormal(ls.l);
    float visibility = calcLightVisibility(hit, ls.l, mainNormal, ls.distance / voxelSize.x);

    if (enableReconstruction) {
        uint bufferIndex = PixelToIndex(pixel, pixelCount);
        gPathRecordBuffer[bufferIndex].lightDirPdf.xyz = ls.l;
        gPathRecordBuffer[bufferIndex].lightDirPdf.w = ls.pdf;
        gPathRecordBuffer[bufferIndex].lightColor.xyz = visibility * ls.lightColor;
        gPathRecordBuffer[bufferIndex].valid = 1;
        gPathRecordBuffer[bufferIndex].hitCell = float4(hit.cell,0.0f);
        gPathRecordBuffer[bufferIndex].hitCellInt = hit.cellInt;
        gPathRecordBuffer[bufferIndex].viewDir = float4(v,0.0f);
    }

    if (visibility > 0) 
        return visibility * shading(offset, ls.l, v, ls.h, ls.lightColor) / ls.pdf;
    return 0;
}

float3 shadingGlobal(HitResult hit, float3 v, uint bounce, uint2 pixel,inout UniformSampleGenerator sg)
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
        uint offset = CellToIndex(hits[i].cellInt, gGridDataParamBlock.voxelCount);
        PrimitiveBSDF_New BSDF = gGridDataParamBlock.gridDataBuffer[offset].bsdf;
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
        uint currentOffset = CellToIndex(currentHit.cellInt, gGridDataParamBlock.voxelCount);
        float3 currentV = i == 0 ? v : -ls[i - 1];

        // direct lighting
        reflectLights[i] += shadingDirect(currentHit, currentOffset, currentV,pixel,sg);

        //if (enableReconstruction) {
        //    uint bufferIndex = PixelToIndex(pixel, pixelCount);
        //    gPathRecordBuffer[bufferIndex].valid = 1;
        //    gPathRecordBuffer[bufferIndex].hitCell = hits[i].cell;
        //    gPathRecordBuffer[bufferIndex].hitCellInt = hits[i].cellInt;
        //    //gPathRecordBuffer[bufferIndex].lightDirPdf.xyz = pdfs[i];
        //    //gPathRecordBuffer[bufferIndex].lightDirPdf.w = pdfs[i];
        //    gPathRecordBuffer[bufferIndex].viewDir.xyz = currentV;
        //}

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
                uint nextOffset = CellToIndex(nextHit.cellInt, gGridDataParamBlock.voxelCount);
                float3 incidentLight = reflectLights[i + 1];
                reflectLights[i] += shading(currentOffset, ls[i], currentV, hs[i], incidentLight) / pdfs[i]; // 间接光照
            }
        }
    }
    return reflectLights[0];

    return float3(0);
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

    uint bufferIndex = PixelToIndex(pixel, pixelCount);
    gPathRecordBuffer[bufferIndex].init();

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0)
        return psOut;

    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;
    HitResult hit = rayMarching(from, direction, -1, false);


    if (hit.hit) {
        float3 color = 0;
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
            color = shadingGlobal(hit, -direction, maxBounce, pixel,sg);
            break;
        case ABSDFDrawMode::Occupancy:
            color = hit.cell;
            break;
        }
        //color = clamp(color, float3(0.0f), float3(1e4f)); 
        psOut.color = float4(color, 1);
    }

    return psOut;

}
