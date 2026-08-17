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
    uint maxContributingVoxelCount;
    float transmittanceThreshold;
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

HitResult rayMarching_Mipmap(float3 from, float3 direction, bool ignoreFirst, int3 initialCellInt)
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

HitResult rayMarching(float3 from, float3 direction, bool ignoreFirst, int3 initialCellInt){
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
    {
        HitResult hit = { false, int3(from), from };
        return hit;
    }

    return rayMarching_Mipmap(from, direction, ignoreFirst, initialCellInt);
}

float3 shadingGlobal(HitResult firstHit, float3 direction, uint contributingVoxelCount, uint2 pixel,inout UniformSampleGenerator sg)
{
    uint bufferIndex = PixelToIndex(pixel, pixelCount);

    HitResult hits[MAX_CONTRIBUTING_VOXELS_PER_RAY];
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
        hits[actualHitCount] = currentHit;
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
        float opacity = data.opacity.calcOpacity(-direction);
        opacity = saturate(opacity);

        if (enableReconstruction) {
            VoxelRecord voxelRecord;
            voxelRecord.cellInt = currentHit.cellInt;
            voxelRecord.hitPosition = currentHit.cell;
            voxelRecord.radiance = radiance;
            voxelRecord.opacity = opacity;
            voxelRecord.transmittanceBefore = transmittance;

            gPathRecordBuffer[bufferIndex].voxels[i] = voxelRecord;
        }


        accumulatedColor += transmittance * opacity * radiance;
        transmittance *= 1.0f - opacity;

        if (transmittance <= transmittanceThreshold)
            break;

        currentHit = rayMarching(
            currentHit.cell,
            direction,
            true,
            currentHit.cellInt
        );

        if (i == 0) {
            print("first opacitySH0 : ", data.opacity.coefficients[0]);
            print("first opacity : ", opacity);
        }
            

    }

    accumulatedColor += transmittance * renderBackground(direction).xyz;

    if (enableReconstruction) {
        gPathRecordBuffer[bufferIndex].contributingVoxelCount = actualHitCount;
        gPathRecordBuffer[bufferIndex].viewDir = normalize(-direction);
        gPathRecordBuffer[bufferIndex].backgroundColor = renderBackground(direction).xyz;
        gPathRecordBuffer[bufferIndex].finalColor = accumulatedColor;
    }

    print("actualHitCount : ", actualHitCount);


    return accumulatedColor;
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
    psOut.color = renderBackground(direction);

    uint bufferIndex = PixelToIndex(pixel, pixelCount);
    gPathRecordBuffer[bufferIndex].init();
    gPathRecordBuffer[bufferIndex].backgroundColor = renderBackground(direction).xyz;

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0) {
        gAccuColor[pixel].xyz += psOut.color.xyz * invSpp;
        gAccuColor[pixel].w = 1.0f;
        return psOut;
    }
        
    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;

    HitResult hit;
    hit = rayMarching(from, direction, false,int3(floor(from)));


    if (hit.hit)
    {
        gPathRecordBuffer[bufferIndex].valid = 1;

        float3 color = 0;
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
            color = shadingGlobal(hit, direction, maxContributingVoxelCount, pixel, sg);
            psOut.color = float4(color, 1.0f);
            break;
        }
    }

    gAccuColor[pixel].xyz += psOut.color.xyz * invSpp;
    gAccuColor[pixel].w = 1.0f;

    return psOut;

}
