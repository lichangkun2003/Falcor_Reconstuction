#include "../GradRecord.slang"
#include "../PathRecord.slang"
//#include "../Voxel/Shading.slang"
#include "../Voxel/VoxelGrid.slang"
#include "DiffShading.slang"

import Utils.Debug.PixelDebug;
import DiffRendering.SceneGradientInfo;


// Input
Texture2D<float4> gDL_dColorBuffer;
StructuredBuffer<PathRecord> gPathRecordBuffer;
Texture2D<uint> gBackgroundMaskBuffer;


// Output
RWStructuredBuffer<GradRecord> gGradBuffer;
RWTexture2D<float4> dummy;

cbuffer CB {
    uint2 gResolution;
    uint3 gVoxelCount;
    float gGeometryTau;         // soft-hit sigmoid 温度，越小越接近 hard hit
    float gGeometryGradClamp;   // 防止几何梯度爆炸

}

#define INTERLOCKED_ADD_FLOAT(DEST, VALUE)                                                                             \
    {                                                                                                                  \
        float expected_ = DEST;                                                                                        \
        float original_ = 0.0f;                                                                                        \
        [allow_uav_condition]                                                                                          \
        for (uint iter_ = 0; iter_ < 256; iter_++)                                                                     \
        {                                                                                                              \
            float desired_ = expected_ + (VALUE);                                                                      \
            InterlockedCompareExchangeFloatBitwise(                     \
            DEST,                                                   \
            expected_,                                               \
            desired_,                                                \
            original_                                                \
        );                              \
            if (asuint(original_) == asuint(expected_))                                                                \
                break;                                                                                                 \
            expected_ = original_;                                                                                     \
        }                                                                                                              \
    }

void calcGeometryGradFromCandidate(
    GeometryCandidateRecord cand,
    float3 dL_dC,
    bool isReferenceBackground,
    out GradRecord geometryGradRecord)
{
    geometryGradRecord.init();

    if (cand.valid == 0)
        return;

    if (any(cand.cellInt < int3(0)) || any(cand.cellInt >= int3(gVoxelCount)))
        return;

    if (!isSolidVoxel(cand.cellInt))
        return;

    uint voxelIndex = CellToIndex(uint3(cand.cellInt), gVoxelCount);

    VoxelData voxel = gGridDataParamBlock.gridDataBuffer[voxelIndex];

    if (voxel.occupied == 0)
        return;

    float tau = max(gGeometryTau, 1e-5f);

    float gmin = cand.gmin;

    if (isBadFloat(gmin))
        return;

    float w = sigmoid(-gmin / tau);

    float3 LCandidate = cand.radiance.xyz;
    float3 CBehind = cand.behindColor.xyz;

    LCandidate = sanitizeFloat3(LCandidate);
    CBehind = sanitizeFloat3(CBehind);

    //// dLoss / dw
    // float dL_dW = dot(dL_dC, LCandidate - CBehind);
    //// dw / dgmin
    // float dW_dG = -w * (1.0f - w) / tau;
    // float dL_dG = dL_dW * dW_dG;
    float dL_dG = 0.0f;
    if (isReferenceBackground)
    {
        dL_dG = -0.01 * w / tau;
    }
    else
    {
        float dL_dW =
        dot(dL_dC, LCandidate - CBehind);

        float dW_dG =
        -w * (1.0f - w) / tau;

        dL_dG = dL_dW * dW_dG;
    }


    dL_dG = sanitizeFloat(dL_dG);

    // 防止几何梯度爆炸
    dL_dG = clamp(dL_dG, -gGeometryGradClamp, gGeometryGradClamp);

    Ellipsoid e = voxel.ellipsoid;

    // xStar 是 global cell-space，转 voxel local
    float3 xLocal = cand.xStar.xyz - float3(cand.cellInt);

    float3 q = xLocal - e.center;

    q = sanitizeFloat3(q);

    float3 Bq = mul(e.B, q);

    Bq = sanitizeFloat3(Bq);

    geometryGradRecord.geometryValid = 1;

    // dF / dCenter = -2 B q
    geometryGradRecord.center = dL_dG * (-2.0f * Bq);

    // dF / dB
    geometryGradRecord.B00 = dL_dG * q.x * q.x;
    geometryGradRecord.B11 = dL_dG * q.y * q.y;
    geometryGradRecord.B22 = dL_dG * q.z * q.z;

    // B is symmetric, off-diagonal appears twice
    geometryGradRecord.B01 = dL_dG * 2.0f * q.x * q.y;
    geometryGradRecord.B02 = dL_dG * 2.0f * q.x * q.z;
    geometryGradRecord.B12 = dL_dG * 2.0f * q.y * q.z;

    geometryGradRecord.center = sanitizeFloat3(geometryGradRecord.center);
    geometryGradRecord.B00 = sanitizeFloat(geometryGradRecord.B00);
    geometryGradRecord.B11 = sanitizeFloat(geometryGradRecord.B11);
    geometryGradRecord.B22 = sanitizeFloat(geometryGradRecord.B22);
    geometryGradRecord.B01 = sanitizeFloat(geometryGradRecord.B01);
    geometryGradRecord.B02 = sanitizeFloat(geometryGradRecord.B02);
    geometryGradRecord.B12 = sanitizeFloat(geometryGradRecord.B12);
}


void accumulateGeometryGradRecord(uint indexGradBuffer, GradRecord gradRecord)
{
    InterlockedAdd(gGradBuffer[indexGradBuffer].geometryValid, gradRecord.geometryValid);

    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].center.x,
        gradRecord.center.x
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].center.y,
        gradRecord.center.y
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].center.z,
        gradRecord.center.z
    );

    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B00,
        gradRecord.B00
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B11,
        gradRecord.B11
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B22,
        gradRecord.B22
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B01,
        gradRecord.B01
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B02,
        gradRecord.B02
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B12,
        gradRecord.B12
    );
}


[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    printSetPixel(dispatchThreadId.xy);
    uint2 pixel = dispatchThreadId.xy;

    if (pixel.x >= gResolution.x || pixel.y >= gResolution.y)
        return;

    uint indexPathRecordBuffer = pixel.y * gResolution.x + pixel.x;

    PathRecord rec = gPathRecordBuffer[indexPathRecordBuffer];

    if (rec.valid == 0 && rec.candidateCount == 0)
        return;

    float3 dL_dC = gDL_dColorBuffer.Load(int3(pixel, 0)).rgb;

    //if (all(dL_dC == float3(0.0f)))
    //    return;

    bool isReferenceBackground = gBackgroundMaskBuffer[pixel] == 1;

    // -------------------------
    // 1. Radiance gradient
    // only hard hit
    // -------------------------
    if (rec.valid != 0)
    {
        float3 gRadiance = dL_dC;


        if ( !isReferenceBackground ) {
            SceneGradientInfo gradInfo = SceneGradientInfo(
            SceneGradientFlag.make(GradientMode.Scene),
            pixel,
            indexPathRecordBuffer
        );

            SceneGradientInfo.Differential dGradInfo;
            dGradInfo.flag = SceneGradientFlag.make(GradientMode.Scene);

            DifferentialPair<SceneGradientInfo> dpGradInfo = diffPair(gradInfo, dGradInfo);

            bwd_diff(evalPixelColorFromPathAD)(dpGradInfo, rec.hitCellInt, rec.viewDir, gVoxelCount, dL_dC);
        }
            

    }

    // -------------------------
    // 2. Geometry gradient
    // all candidates
    // -------------------------
    //float3 gGeometry = dL_dC;
    //if (isReferenceBackground) {
    //    gGeometry = float3(0.2);
    //}

    for (uint i = 0; i < rec.candidateCount && i < MAX_CANDIDATES; i++)
    {
        GeometryCandidateRecord cand = rec.candidates[i];

        if (cand.valid == 0)
            continue;

        if (any(cand.cellInt < int3(0)) || any(cand.cellInt >= int3(gVoxelCount)))
            continue;

        GradRecord geometryGradRecord;
        calcGeometryGradFromCandidate(cand, dL_dC, isReferenceBackground, geometryGradRecord);

        if (geometryGradRecord.geometryValid == 0)
            continue;

        uint geometryIndexGradBuffer = CellToIndex(uint3(cand.cellInt), gVoxelCount);

        accumulateGeometryGradRecord(geometryIndexGradBuffer, geometryGradRecord);
    }

    //GeometryCandidateRecord cand = rec.candidates[0];
    //uint indexGradBuffer = CellToIndex(uint3(cand.cellInt), gVoxelCount);
    //print("grad B00 :", gGradBuffer[indexGradBuffer].B00);
    //print("grad B11 :", gGradBuffer[indexGradBuffer].B11);
    //print("grad B22 :", gGradBuffer[indexGradBuffer].B22);
    //print("grad B01 :", gGradBuffer[indexGradBuffer].B01);
    //print("grad B02 :", gGradBuffer[indexGradBuffer].B02);
    //print("grad B12 :", gGradBuffer[indexGradBuffer].B12);


    dummy[pixel] = float4(abs(dL_dC), 1.0f);
}


