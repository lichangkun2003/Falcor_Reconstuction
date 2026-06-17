import Rendering.Materials.StandardMaterial;
import Utils.Sampling.UniformSampleGenerator;
import Utils.Math.MathHelpers;
import Scene.SceneTypes;
import Scene.Scene;

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

    if (maxDistance > 0)
        return rayMarching_Mipmap(from, direction, maxDistance, ignoreFirst);
    return rayMarching_Mipmap(from, direction, ignoreFirst);

}



float3 shadingGlobal(HitResult hit, float3 v, uint bounce, uint2 pixel,inout UniformSampleGenerator sg)
{
    uint offset = CellToIndex(hit.cellInt, gGridDataParamBlock.voxelCount);

    VoxelData voxel = gGridDataParamBlock.gridDataBuffer[offset];

    float3 viewDir = normalizeSafe(v);

    float3 color = voxel.radiance.calc(viewDir);

    if (enableReconstruction) {
        uint bufferIndex = PixelToIndex(pixel, pixelCount);
        gPathRecordBuffer[bufferIndex].valid = 1;
        gPathRecordBuffer[bufferIndex].hitCellInt = hit.cellInt;
        gPathRecordBuffer[bufferIndex].hitCell = float4(hit.cell, 1.f);
        gPathRecordBuffer[bufferIndex].viewDir = float4(viewDir, 0.f);
    }


    return color;

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
    HitResult hit;


    hit = rayMarching(from, direction, -1, false);
    
    //HitResult hit = rayMarching(from, direction, -1, false);


    if (hit.hit) {
        float3 color = 0;
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
            color = shadingGlobal(hit, -direction, 0, pixel, sg);
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
