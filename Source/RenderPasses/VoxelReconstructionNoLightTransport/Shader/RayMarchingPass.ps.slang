import Rendering.Materials.StandardMaterial;
import Utils.Sampling.UniformSampleGenerator;
import Utils.Math.MathHelpers;
import Scene.SceneTypes;
import Scene.Scene;


#include "DiffShading.slang"

#include "../Math/AABB.slang"
#include "../Math/Random.slang"
//#include "../Voxel/Shading.slang"
#include "../Voxel/VoxelGrid.slang"
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
    uint frameIndex;
    bool renderBackGround;
    float4 clearColor;
    //uint2 selectedPixel;
    bool enableReconstruction;
    float invSpp;
}

RWStructuredBuffer<PathRecord> gPathRecordBuffer;
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
};

struct RayMarchResult
{
    HitResult hit;

    uint candidateCount;
    GeometryCandidateRecord candidates[MAX_CANDIDATES];

    MUTATING
    void init()
    {
        hit.hit = false;
        hit.cellInt = int3(0);
        hit.cell = float3(0.0f);

        candidateCount = 0;

        for (uint i = 0; i < MAX_CANDIDATES; i++)
        {
            candidates[i].valid = 0;
            candidates[i].cellInt = int3(0);
            candidates[i].gmin = 1e30f;
            candidates[i].xStar = float4(0.0f);
            candidates[i].radiance = float4(0.0f);
            candidates[i].behindColor = float4(0.0f);
        }
    }
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

// 先后用体素和椭球裁剪线段
HitResult checkPrimitive(Ellipsoid ellipsoid, float3 cell, float3 direction, int3 cellInt)
{
    float3 from;
    float3 to ;

    calcVoxelLocalSegment(cell, direction, cellInt, from, to);

    float2 inOut = ellipsoid.clip(from, to);
    HitResult result = { inOut.x >= 0, cellInt, cell + inOut.x * (to - from) };
    return result;
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

// candidate 插入 top-K
void insertCandidateTopK(
    inout RayMarchResult result,
    GeometryCandidateRecord candidate)
{
    if (candidate.valid == 0)
        return;

    if (result.candidateCount < MAX_CANDIDATES)
    {
        result.candidates[result.candidateCount] = candidate;
        result.candidateCount++;
        return;
    }

    uint worstIndex = 0;
    float worstG = result.candidates[0].gmin;

    for (uint i = 1; i < MAX_CANDIDATES; i++)
    {
        if (result.candidates[i].gmin > worstG)
        {
            worstG = result.candidates[i].gmin;
            worstIndex = i;
        }
    }

    if (candidate.gmin < worstG)
    {
        result.candidates[worstIndex] = candidate;
    }
}

// 每遇到一个 solid voxel，都可以生成 candidate
GeometryCandidateRecord makeGeometryCandidate(
    int3 cellInt,
    float3 cell,
    float3 direction,
    float3 viewDir)
{
    GeometryCandidateRecord cand;

    cand.valid = 0;
    cand.cellInt = cellInt;
    cand.gmin = 1e30f;
    cand.xStar = float4(0.0f);
    cand.radiance = float4(0.0f);
    cand.behindColor = float4(0.0f);

    if (any(cellInt < int3(0)) || any(cellInt >= int3(gGridDataParamBlock.voxelCount)))
        return cand;

    int offset = CellToIndex(uint3(cellInt), gGridDataParamBlock.voxelCount);

    if ( offset < 0)
        return cand;

    VoxelData voxel = gGridDataParamBlock.gridDataBuffer[(uint)offset];

    //if (voxel.occupied == 0)
    //    return cand;

    float3 localFrom;
    float3 localTo;

    calcVoxelLocalSegment(cell, direction, cellInt, localFrom, localTo);

    float gmin;
    float3 xStarGlobal;

    calcEllipsoidGMinOnSegment(
        voxel.ellipsoid,
        localFrom,
        localTo,
        cellInt,
        gmin,
        xStarGlobal
    );

    float3 L = voxel.radiance.calc(viewDir);

    cand.valid = 1;
    cand.cellInt = cellInt;
    cand.gmin = gmin;
    cand.xStar = float4(xStarGlobal, 1.0f);
    cand.radiance = float4(L, 1.0f);
    cand.behindColor = float4(0.0f);

    return cand;
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

    if (maxDistance > 0)
        return rayMarching_Mipmap(from, direction, maxDistance, ignoreFirst);
    return rayMarching_Mipmap(from, direction, ignoreFirst);

}

RayMarchResult rayMarchingCollectCandidates_Mipmap(float3 from, float3 direction, bool ignoreFirst)
{
    RayMarchResult result;
    result.init();

    float3 cell = from;
    int3 cellInt = int3(floor(from));

    direction = normalizeSafe(direction);

    if (all(direction == float3(0.0f)))
        return result;

    float3 reciprocal_direction = 1.0f / direction;
    int3 directions = int3(sign(direction));

    float3 viewDir = -direction;

    if (ignoreFirst)
        leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (any(cellInt < int3(0)) || any(cellInt >= int3(voxelCount)))
            break;

        int3 blockInt = cellInt / BLOCK_TO_VOXEL;

        if (isSolidBlock(blockInt))
        {
            int3 current = blockInt;

            while (all(current == blockInt))
            {
                if (any(cellInt < int3(0)) || any(cellInt >= int3(voxelCount)))
                    break;

                if (isSolidVoxel(cellInt))
                {
                    // 1. collect geometry candidate first.
                    GeometryCandidateRecord cand = makeGeometryCandidate(
                        cellInt,
                        cell,
                        direction,
                        viewDir
                    );

                    insertCandidateTopK(result, cand);

    #if !CHECK_PRIMITIVE
                    HitResult hit = { true, cellInt, float3(cellInt) + 0.5f };
                    result.hit = hit;
                    return result;
    #else
                    uint offset = CellToIndex(uint3(cellInt), gGridDataParamBlock.voxelCount);

                    Ellipsoid ellipsoid = gGridDataParamBlock.gridDataBuffer[offset].ellipsoid;

                    HitResult hit = checkPrimitive(ellipsoid, cell, direction, cellInt);

                    if (hit.hit)
                    {
                        result.hit = hit;
                        return result;
                    }
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

    return result;
}

RayMarchResult rayMarchingCollectCandidates(float3 from, float3 direction, float maxDistance, bool ignoreFirst)
{
    // 第一版先不处理 maxDistance，和原始 -1 用法一致。
    return rayMarchingCollectCandidates_Mipmap(from, direction, ignoreFirst);
}

float3 shadeHitOnly(HitResult hit, float3 viewDir)
{
    if (!hit.hit)
        return float3(0.0f);

    if (any(hit.cellInt < int3(0)) || any(hit.cellInt >= int3(gGridDataParamBlock.voxelCount)))
        return float3(0.0f);

    if (!isSolidVoxel(hit.cellInt))
        return float3(0.0f);

    uint offset = CellToIndex(uint3(hit.cellInt), gGridDataParamBlock.voxelCount);

    VoxelData voxel = gGridDataParamBlock.gridDataBuffer[offset];

    // if (voxel.occupied == 0)
    //    return float3(0.0f);

    return voxel.radiance.calc(viewDir);
}



float3 computeNextColorAfterCurrentVoxel(HitResult hit, float3 direction)
{
    float3 cell = hit.cell;
    int3 cellInt = hit.cellInt;

    direction = normalizeSafe(direction);

    if (all(direction == float3(0.0f)))
        return float3(0.0f);

    float3 reciprocal_direction = 1.0f / direction;
    int3 directions = int3(sign(direction));

    // leave current voxel explicitly.
    leave(cell, cellInt, directions, direction, reciprocal_direction);

    cell += 1e-5f * direction;

    HitResult nextHit = rayMarching(cell, direction, -1.0f, false);

    if (!nextHit.hit)
        return renderBackground(direction).rgb;

    return shadeHitOnly(nextHit, -direction);
}

float3 shadingGlobal(HitResult hit, float3 v, uint bounce, uint2 pixel,inout UniformSampleGenerator sg)
{
    uint indexPathRecordBuffer = pixel.y * pixelCount.x + pixel.x;
    SceneGradientInfo gradInfo = SceneGradientInfo(
        SceneGradientFlag.make(GradientMode.None),
            pixel,
            indexPathRecordBuffer
        );

    SceneGradientInfo.Differential dGradInfo;
    dGradInfo.flag = SceneGradientFlag.make(GradientMode.None);

    DifferentialPair<SceneGradientInfo> dpGradInfo = diffPair(gradInfo, dGradInfo);
    return evalPixelColorFromPathAD(gradInfo, hit.cellInt, float4(v, 0.0f), gGridDataParamBlock.voxelCount);

    //return shadeHitOnly(hit, normalizeSafe(v));
}

void recordPath(
    uint2 pixel,
    RayMarchResult result,
    float3 forwardColor,
    float3 LHit,
    float3 CBehind,
    float3 viewDir,
    float3 rayDir)
{
    uint bufferIndex = PixelToIndex(pixel, pixelCount);

    gPathRecordBuffer[bufferIndex].valid = result.hit.hit ? 1 : 0;

    gPathRecordBuffer[bufferIndex].viewDir = float4(viewDir, 0.0f);
    gPathRecordBuffer[bufferIndex].rayDir = float4(rayDir, 0.0f);

    gPathRecordBuffer[bufferIndex].pathContribution = float4(forwardColor, 1.0f);
    gPathRecordBuffer[bufferIndex].nextContribution = float4(CBehind, 1.0f);

    if (result.hit.hit)
    {
        gPathRecordBuffer[bufferIndex].hitCellInt = result.hit.cellInt;
        gPathRecordBuffer[bufferIndex].hitCell = float4(result.hit.cell, 1.0f);
    }

    gPathRecordBuffer[bufferIndex].candidateCount = result.candidateCount;

    for (uint i = 0; i < MAX_CANDIDATES; i++)
    {
        if (i >= result.candidateCount)
        {
            gPathRecordBuffer[bufferIndex].candidates[i].valid = 0;
            continue;
        }

        GeometryCandidateRecord cand = result.candidates[i];

        if (result.hit.hit)
        {
            if (all(cand.cellInt == result.hit.cellInt))
                cand.behindColor = float4(CBehind, 1.0f);
            else
                cand.behindColor = float4(LHit, 1.0f);
        }
        else
        {
            cand.behindColor = float4(forwardColor, 1.0f);
        }

        gPathRecordBuffer[bufferIndex].candidates[i] = cand;
    }
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
    gPathRecordBuffer[bufferIndex].backGroundColor = renderBackground(direction);

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0) {
        gAccuColor[pixel].xyz += psOut.color.xyz * invSpp;
        gAccuColor[pixel].w = 1.0f;
        return psOut;
    }
        


    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;

    //HitResult hit;
    //hit = rayMarching(from, direction, -1, false);

    RayMarchResult marchResult = rayMarchingCollectCandidates(from, direction, -1.0f, false);
    HitResult hit = marchResult.hit;
    float3 forwardColor = psOut.color.rgb;
    float3 LHit = float3(0.0f);
    float3 CBehind = forwardColor;

    if (hit.hit)
    {
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
            LHit = shadingGlobal(hit, -direction, 0, pixel, sg);
            CBehind = computeNextColorAfterCurrentVoxel(hit, direction);
            forwardColor = LHit;
            break;

        case ABSDFDrawMode::Occupancy:
            forwardColor = (float3(hit.cellInt) + 0.5f) / float3(voxelCount);
            break;
        }
    }
    psOut.color = float4(forwardColor, 1.0f);

    if (enableReconstruction && drawMode == ABSDFDrawMode::Default)
    {
        recordPath(
        pixel,
        marchResult,
        forwardColor,
        LHit,
        CBehind,
        -direction,
        direction
    );
    }

    gAccuColor[pixel].xyz += psOut.color.xyz * invSpp;
    gAccuColor[pixel].w = 1.0f;

    return psOut;

}
