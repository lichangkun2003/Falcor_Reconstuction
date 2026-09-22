import Rendering.Materials.StandardMaterial;
import Utils.Sampling.UniformSampleGenerator;
import Utils.Math.MathHelpers;
import Scene.SceneTypes;
import Scene.Scene;
import Utils.Debug.PixelDebug;


#include "DiffShading.slang"

#include "../Math/AABB.slang"
#include "../Math/Random.slang"
//#include "../Voxel/Shading.slang"
#include "../Voxel/VoxelGrid.slang"
#include "../PathRecord.slang"

#ifndef PREVIEW_ONLY
#define PREVIEW_ONLY 0
#endif


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
    float4x4 invVP;
    float shadowBias;
    uint drawMode;
    //uint sampleStrategy;
    uint frameIndex;
    bool renderBackGround;
    float4 clearColor;
    //uint2 selectedPixel;
    bool enableReconstruction;
    float invSpp;
    uint maxContributingVoxelCount;
    float transmittanceThreshold;
}

#if !PREVIEW_ONLY
RWStructuredBuffer<PathRecord> gPathRecordBuffer;
#endif
RWTexture2D gAccuColor;


struct PsOut
{
    float4 color : SV_Target0;
};

struct HitResult
{
    bool hit;
    int3 cellInt;
    float3 cell;

    float gmin;
    float4 xStar;
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

void calcVoxelLocalSegment(float3 cell,float3 direction,int3 cellInt,out float3 localFrom,out float3 localTo)
{
    localFrom = cell - float3(cellInt);

    int3 directions = int3(sign(direction));
    float3 border = float3(directions + 1) * 0.5f;

    float3 t = float3(1e30f);

    if (direction.x != 0.0f)
        t.x = (border.x - localFrom.x) / direction.x;

    if (direction.y != 0.0f)
        t.y = (border.y - localFrom.y) / direction.y;

    if (direction.z != 0.0f)
        t.z = (border.z - localFrom.z) / direction.z;

    float tMin = min(t.x, min(t.y, t.z));

    // safety
    tMin = max(tMin, 0.0f);

    localTo = localFrom + tMin * direction;
}





float evalEllipsoidImplicit(Ellipsoid ellipsoid, float3 xLocal)
{
    float3 q = xLocal - ellipsoid.center;
    float3 Bq = mul(ellipsoid.B, q);
    return dot(q, Bq) - 1.0f;
}

// 计算 ray segment 上的最小值
void calcEllipsoidGMinOnSegment(
    Ellipsoid ellipsoid,
    float3 localFrom,
    float3 localTo,
    int3 cellInt,
    out float gmin,
    out float3 xStarGlobal)
{
    float3 seg = localTo - localFrom;

    float3 o = localFrom;
    float3 d = seg;

    float3 m = o - ellipsoid.center;

    float3 Bd = mul(ellipsoid.B, d);
    float3 Bm = mul(ellipsoid.B, m);

    float a = dot(d, Bd);
    float b = 2.0f * dot(d, Bm);
    float c = dot(m, Bm) - 1.0f;

    float tStar = 0.0f;

    if (abs(a) > 1e-8f)
        tStar = clamp(-b / (2.0f * a), 0.0f, 1.0f);
    else
        tStar = 0.0f;

    float3 xStarLocal = localFrom + tStar * seg;

    gmin = evalEllipsoidImplicit(ellipsoid, xStarLocal);

    xStarGlobal = float3(cellInt) + xStarLocal;
}

float calcCandidateSortScore(
    Ellipsoid ellipsoid,
    int3 cellInt,
    float gmin,
    float3 xStarCell,
    float3 rayOriginCell)
{
    // ---------------------------------------------------------
    // 1. gmin -> 近似世界空间 surface gap
    // ---------------------------------------------------------

    // xStarCell 是 global cell-space。
    float3 xLocal =
        xStarCell - float3(cellInt);

    float3 q =
        xLocal - ellipsoid.center;

    float3 Bq =
        mul(ellipsoid.B, q);

    float3 gradPhiWorld =
        (2.0f * Bq) /
        max(voxelSize, float3(1e-6f));

    float gradNormWorld =
        max(length(gradPhiWorld), 1e-6f);

    // 一阶 signed-distance approximation。
    // 对 near-miss candidate，gmin >= 0。
    float gapWorld =
        max(gmin, 0.0f) / gradNormWorld;

    // ---------------------------------------------------------
    // 2. gap normalization
    // ---------------------------------------------------------

    float voxelScale =
        max(voxelSize.x, 1e-6f);

    float gapNorm =
        gapWorld / voxelScale;

    // ---------------------------------------------------------
    // 3. 距离相机（当前先用 ray near-plane origin）
    // ---------------------------------------------------------

    float3 deltaWorld =
        (xStarCell - rayOriginCell) * voxelSize;

    float depthWorld =
        length(deltaWorld);

    float sceneDiagonal =
        length(float3(voxelCount) * voxelSize);

    float depthNorm =
        depthWorld / max(sceneDiagonal, 1e-6f);

    // ---------------------------------------------------------
    // 4. Weighted score
    // gap 主导，depth 只作为弱约束
    // ---------------------------------------------------------

    const float depthWeight = 0.05f;

    return gapNorm +
           depthWeight * depthNorm;
}


// 先后用体素和椭球裁剪线段
HitResult checkPrimitive(Ellipsoid ellipsoid, float3 cell, float3 direction, int3 cellInt)
{
    float3 localFrom;
    float3 localTo;

    calcVoxelLocalSegment(cell, direction, cellInt, localFrom, localTo);

    // 计算线段上的最小隐式函数值，以及对应位置
    float gmin;
    float3 xStarGlobal;

    calcEllipsoidGMinOnSegment(
        ellipsoid,
        localFrom,
        localTo,
        cellInt,
        gmin,
        xStarGlobal
    );

    float2 inOut = ellipsoid.clip(localFrom, localTo);

    bool hardHit =
        inOut.x >= 0.0f;

    HitResult result;
    result.hit = hardHit;
    result.cellInt = cellInt;
    result.gmin = gmin;
    result.xStar =
        float4(xStarGlobal, 1.0f);

    if (hardHit)
    {
        // inOut.x是相对于[localFrom, localTo]的线段参数
        result.cell =
            cell +
            inOut.x *
            (localTo - localFrom);
    }
    else
    {
        // 未命中时cell不会用于继续追踪
        result.cell = cell;
    }

    return result;
}

// Traverse occupied voxels directly with the voxel DDA.
HitResult rayMarchingDDA(
    float3 from,
    float3 direction,
    bool ignoreFirst,
    int3 initialCellInt,
    float3 rayOriginCell,
    uint bufferIndex,
    float transmittanceBefore,
    uint frontHitCount,
    inout uint candidateCount)
{
    float3 cell = from;
    // int3 cellInt = int3(floor(from));
    int3 cellInt = initialCellInt;

    float3 reciprocal_direction = 1.0f / direction;
    // 可选的前进方向
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true) 
    {
        if (any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount))
            break;

        if (isSolidVoxel(cellInt))
        {
#if !CHECK_PRIMITIVE
            HitResult hit = { true, cellInt, cellInt + 0.5f, -1.0f,
                float4(cellInt + 0.5f, 1.0f) };
            return hit;
#else
            uint offset = CellToIndex(cellInt, gGridDataParamBlock.voxelCount);
            Ellipsoid ellipsoid = gGridDataParamBlock.gridDataBuffer[offset].ellipsoid;
            HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);

            // hard-hit不放入candidate数组，直接返回。
            // 随后会被写入VoxelRecord。
            if (hit.hit)
                return hit;

#if !PREVIEW_ONLY
            // candidate数组只保存未命中的体素
            bool validMiss = !isnan(hit.gmin) && !isinf(hit.gmin) && hit.gmin >= 0.0f;
            if (enableReconstruction && validMiss)
            {
                GeometryCandidateRecord cand;
                cand.valid = 1;
                cand.cellInt = cellInt;
                cand.gmin = hit.gmin;
                cand.xStar = hit.xStar;
                cand.transmittanceBefore = transmittanceBefore;
                cand.frontHitCount = frontHitCount;

                cand.sortScore = calcCandidateSortScore(
                    ellipsoid,
                    cellInt,
                    hit.gmin,
                    hit.xStar.xyz,
                    rayOriginCell
                );

                // candidate数组还没有装满
                if (candidateCount < (uint)MAX_CANDIDATES)
                {
                    uint insertIndex = candidateCount;

                    // 数组按gmin从小到大排序
                    while (insertIndex > 0 && gPathRecordBuffer[bufferIndex].candidates[insertIndex - 1].sortScore > cand.sortScore)
                    {
                        gPathRecordBuffer[bufferIndex].candidates[insertIndex] =
                            gPathRecordBuffer[bufferIndex].candidates[insertIndex - 1];
                        insertIndex--;
                    }

                    gPathRecordBuffer[bufferIndex].candidates[insertIndex] = cand;
                    candidateCount++;
                }
                // 数组已经装满，只考虑替换当前最大的sortScore
                else if (cand.sortScore < gPathRecordBuffer[bufferIndex].candidates[MAX_CANDIDATES - 1].sortScore)
                {
                    uint insertIndex = MAX_CANDIDATES - 1;

                    while (insertIndex > 0 && gPathRecordBuffer[bufferIndex].candidates[insertIndex - 1].sortScore > cand.sortScore)
                    {
                        gPathRecordBuffer[bufferIndex].candidates[insertIndex] =
                            gPathRecordBuffer[bufferIndex].candidates[insertIndex - 1];
                        insertIndex--;
                    }

                    gPathRecordBuffer[bufferIndex].candidates[insertIndex] = cand;
                }
            }

#endif // !PREVIEW_ONLY
#endif // CHECK_PRIMITIVE
        }
        leave(cell, cellInt, directions, direction, reciprocal_direction);
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f,
                      1e30f,float4(cellInt + 0.5f, 1.0f) };
    return hit;
}

// HitResult rayMarching(float3 from, float3 direction, bool ignoreFirst, int3 initialCellInt)
HitResult rayMarching(
    float3 from,
    float3 direction,
    bool ignoreFirst,
    int3 initialCellInt,
    float3 rayOriginCell,
    uint bufferIndex,
    float transmittanceBefore,
    uint frontHitCount,
    inout uint candidateCount)
{
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
    {
        HitResult hit = { false, int3(from), from, 1e30f,
            float4(from, 1.0f) };
        return hit;
    }

    return rayMarchingDDA(
        from,
        direction,
        ignoreFirst,
        initialCellInt,
        rayOriginCell,

        bufferIndex,
        transmittanceBefore,
        frontHitCount,
        candidateCount
    );
}

float4 shadingGlobal(
    HitResult firstHit,
    float3 direction,
    uint contributingVoxelCount,
    uint2 pixel,
    float3 rayOriginCell,
    inout UniformSampleGenerator sg,
    inout uint candidateCount)
{
    uint bufferIndex = PixelToIndex(pixel, pixelCount);

    //HitResult hits[MAX_CONTRIBUTING_VOXELS_PER_RAY];
    contributingVoxelCount = min(contributingVoxelCount, (uint)MAX_CONTRIBUTING_VOXELS_PER_RAY);

    float3 accumulatedColor = float3(0.0f);
    float transmittance = 1.0f;

    HitResult currentHit = firstHit;
    uint actualHitCount = 0;
    for (uint i = 0; i < contributingVoxelCount; ++i)
    {
        if (!currentHit.hit)
            break;

        if (any(currentHit.cellInt < int3(0)) ||
            any(currentHit.cellInt >= int3(voxelCount)))
        {
            break;
        }

        // 保存当前有效命中，后续可用于梯度计算
        //hits[actualHitCount] = currentHit;
        actualHitCount++;

        uint voxelIndex = CellToIndex(
            uint3(currentHit.cellInt),
            gGridDataParamBlock.voxelCount
        );

        VoxelData data =
            gGridDataParamBlock.gridDataBuffer[voxelIndex];

        // direction 是相机指向场景；
        // -direction 是体素指向相机的观察方向
        float3 radiance = data.radiance.calc(-direction);
        radiance = max(radiance, 0.0f);

        float opacity = data.opacity.calcOpacity(-direction);
        opacity = saturate(opacity);

#if !PREVIEW_ONLY
        if (enableReconstruction) {
            VoxelRecord voxelRecord;
            voxelRecord.cellInt = currentHit.cellInt;
            voxelRecord.hitPosition = currentHit.cell;
            voxelRecord.radiance = radiance;
            voxelRecord.opacity = opacity;
            voxelRecord.transmittanceBefore = transmittance;

            voxelRecord.gmin =currentHit.gmin;
            voxelRecord.xStar =currentHit.xStar;

            gPathRecordBuffer[bufferIndex].voxels[i] = voxelRecord;
        }
#endif

        {
            //float weight = transmittance * opacity;

            //print("hit index : ", i);
            //print("cell : ", currentHit.cellInt);
            //print("opacity : ", opacity);
            //print("T_before : ", transmittance);
            //print("weight : ", weight);
        }

        accumulatedColor += transmittance * opacity * radiance;
        transmittance *= 1.0f - opacity;

        if (transmittance <= transmittanceThreshold)
            break;
        if (actualHitCount >= contributingVoxelCount)
            break;

        currentHit = rayMarching(
            currentHit.cell,
            direction,
            true,
            currentHit.cellInt,
            rayOriginCell,

            bufferIndex,
            transmittance,
            actualHitCount,
            candidateCount);

        //if (i == 0) {
        //    print("first radiance : ", radiance);
        //    print("first opacity : ", opacity);
        //}
            
    }

    accumulatedColor += transmittance * renderBackground(direction).xyz;
    float renderAlpha = 1.0f - transmittance;

#if !PREVIEW_ONLY
    if (enableReconstruction) {
        gPathRecordBuffer[bufferIndex].contributingVoxelCount = actualHitCount;
        gPathRecordBuffer[bufferIndex].viewDir = normalize(-direction);
        gPathRecordBuffer[bufferIndex].backgroundColor = renderBackground(direction).xyz;
        gPathRecordBuffer[bufferIndex].finalColor = accumulatedColor;
    }
#endif

    print("actualHitCount : ", actualHitCount);

    return float4(accumulatedColor, renderAlpha);
}




PsOut main(float2 texCoord: TEXCOORD, float4 pos: SV_POSITION)
{
    PsOut psOut = {};
    uint2 pixel = uint2(texCoord * pixelCount);
    printSetPixel(pixel);

    float2 jitter = randomJitter();
    texCoord = saturate(texCoord + jitter);
    float3 from = screenCoordToCell(texCoord, 0);
    float3 to = screenCoordToCell(texCoord, 1);
    float3 delta = to - from;
    float3 direction = normalize(delta); // 网格/世界空间下的方向相同
    //psOut.color = renderBackground(direction);
    float3 backgroundColor = renderBackground(direction).xyz;
    psOut.color = float4(backgroundColor, 0.0f);

    float3 rayOriginCell = from;

    uint bufferIndex = PixelToIndex(pixel, pixelCount);
#if !PREVIEW_ONLY
    gPathRecordBuffer[bufferIndex].init();
    gPathRecordBuffer[bufferIndex].backgroundColor = renderBackground(direction).xyz;
    gPathRecordBuffer[bufferIndex].viewDir = normalize(-direction);
#endif

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0) {
        gAccuColor[pixel].xyz += psOut.color.xyz * invSpp;
        gAccuColor[pixel].w += 0.0f;
        return psOut;
    }
        
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;


    HitResult hit;
    uint candidateCount = 0;
    hit = rayMarching(
        from,
        direction,
        false,
        int3(floor(from)),
        rayOriginCell,
        bufferIndex,
        1.0f,
        0,
        candidateCount
        );



    if (hit.hit)
    {
#if !PREVIEW_ONLY
        gPathRecordBuffer[bufferIndex].valid = 1;
#endif

        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
            psOut.color = shadingGlobal(hit, direction, maxContributingVoxelCount, pixel, rayOriginCell, sg,candidateCount);
            break;
        }
    }

#if !PREVIEW_ONLY
    if (enableReconstruction)
    {
        gPathRecordBuffer[bufferIndex].candidateCount = candidateCount;
    }
#endif

    //print("candidateCount : ", candidateCount);

    gAccuColor[pixel].xyz += psOut.color.xyz * invSpp;
    gAccuColor[pixel].w += psOut.color.w * invSpp;

    print("Alpha : ", gAccuColor[pixel].w);

    return psOut;

}
