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
    //float trasmittanceThreshold;
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
// theta Gradient
void calc_dL_dTheta(PathRecord rec,float3 dL_dC,out GradRecord radianceGradRecord,out GradRecord geometryGradRecord)
{
    radianceGradRecord.init();
    geometryGradRecord.init();

    if (rec.valid == 0)
        return;

    float3 viewDir = normalizeSafe(rec.viewDir.xyz);

    if (all(viewDir == float3(0.0f)))
        return;

    // 当前只计算 radiance SH 的梯度
    radianceGradRecord.radianceValid = 1;

    for (uint i = 0; i < SH_COUNT; i++)
    {
        float Y = calcSH(i, viewDir);

        // C = sum_i coeff_i * Y_i(viewDir)
        // dL / d coeff_i = dL / dC * Y_i(viewDir)
        radianceGradRecord.radianceGrad.coefficients[i] = dL_dC * Y;
    }

    // geometry 暂时不算
    geometryGradRecord.geometryValid = 0;
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

    if (rec.valid == 0)
        return;

    float3 dL_dC = gDL_dColorBuffer.Load(int3(pixel, 0)).rgb;

    GradRecord radianceGradRecord;
    GradRecord geometryGradRecord;
    calc_dL_dTheta(rec, dL_dC, radianceGradRecord, geometryGradRecord);
    uint radianceIndexGradBuffer = CellToIndex(rec.hitCellInt, gVoxelCount);
    accumulateRadianceGradRecord(radianceIndexGradBuffer, radianceGradRecord);

    // geormetry
    if (geometryGradRecord.geometryValid != 0 && all(rec.xCellInt >= int3(0))){
        uint xIndex = CellToIndex(rec.xCellInt, gVoxelCount);
        accumulateGeometryGradRecord(xIndex, geometryGradRecord);
    }
    

    //dummy[pixel] = float4(materialGradRecord.diffuse[0],1.0f);
}


