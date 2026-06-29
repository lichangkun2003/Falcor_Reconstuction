#include "../GradRecord.slang"
#include "../PathRecord.slang"
//#include "../Voxel/Shading.slang"
#include "../Voxel/VoxelGrid.slang"

import Utils.Debug.PixelDebug;


// Input
Texture2D<float4> gDL_dColorBuffer;
StructuredBuffer<PathRecord> gPathRecordBuffer;


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

// theta Gradient
void calcRadianceGrad(PathRecord rec,float3 dL_dC,out GradRecord radianceGradRecord)
{
    radianceGradRecord.init();
    
    if (rec.valid == 0)
        return;

    float3 viewDir = normalizeSafe(rec.viewDir.xyz);

    if (all(viewDir == float3(0.0f)))
        return;

    // 计算 radiance SH 的梯度
    radianceGradRecord.radianceValid = 1;

    for (uint i = 0; i < SH_COUNT; i++)
    {
        float Y = calcSH(i, viewDir);

        // C = sum_i coeff_i * Y_i(viewDir)
        // dL / d coeff_i = dL / dC * Y_i(viewDir)
        radianceGradRecord.radianceGrad.coefficients[i] = dL_dC * Y;
    }

}

void accumulateRadianceGradRecord(uint indexGradBuffer, GradRecord gradRecord)
{
    InterlockedAdd(gGradBuffer[indexGradBuffer].radianceValid, gradRecord.radianceValid);

    for (uint i = 0; i < SH_COUNT; i++)
    {
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].radianceGrad.coefficients[i].x,
                              gradRecord.radianceGrad.coefficients[i].x
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].radianceGrad.coefficients[i].y,
                              gradRecord.radianceGrad.coefficients[i].y
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].radianceGrad.coefficients[i].z,
                              gradRecord.radianceGrad.coefficients[i].z
        );
    }
}

void calcGeometryGradFromCandidate(
    GeometryCandidateRecord cand,
    float3 dL_dC,
    out GradRecord geometryGradRecord)
{
    geometryGradRecord.init();

    if (cand.valid == 0)
        return;

    if (any(cand.cellInt < int3(0)) || any(cand.cellInt >= int3(gVoxelCount)))
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

    // dLoss / dw
    float dL_dW = dot(dL_dC, LCandidate - CBehind);

    // dw / dgmin
    float dW_dG = -w * (1.0f - w) / tau;

    float dL_dG = dL_dW * dW_dG;

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
    if (all(dL_dC == float3(0.0f)))
        return;

    // -------------------------
    // 1. Radiance gradient
    // only hard hit
    // -------------------------
    if (rec.valid != 0)
    {
        if (all(rec.hitCellInt >= int3(0)) && all(rec.hitCellInt < int3(gVoxelCount)))
        {
            GradRecord radianceGradRecord;
            calcRadianceGrad(rec, dL_dC, radianceGradRecord);

            if (radianceGradRecord.radianceValid != 0)
            {
                uint radianceIndexGradBuffer = CellToIndex(uint3(rec.hitCellInt), gVoxelCount);
                accumulateRadianceGradRecord(radianceIndexGradBuffer, radianceGradRecord);
            }
        }
    }

    // -------------------------
    // 2. Geometry gradient
    // all candidates
    // -------------------------
    for (uint i = 0; i < rec.candidateCount && i < MAX_CANDIDATES; i++)
    {
        GeometryCandidateRecord cand = rec.candidates[i];

        if (cand.valid == 0)
            continue;

        if (any(cand.cellInt < int3(0)) || any(cand.cellInt >= int3(gVoxelCount)))
            continue;

        GradRecord geometryGradRecord;
        calcGeometryGradFromCandidate(cand, dL_dC, geometryGradRecord);

        if (geometryGradRecord.geometryValid == 0)
            continue;

        uint geometryIndexGradBuffer = CellToIndex(uint3(cand.cellInt), gVoxelCount);

        accumulateGeometryGradRecord(geometryIndexGradBuffer, geometryGradRecord);
    }

    dummy[pixel] = float4(abs(dL_dC), 1.0f);
}


