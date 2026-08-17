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

//void calcGeometryGradFromCandidate(
//    GeometryCandidateRecord cand,
//    float3 dL_dC,
//    bool isReferenceBackground,
//    out GradRecord geometryGradRecord)
//{
//    geometryGradRecord.init();

//    if (cand.valid == 0)
//        return;

//    if (any(cand.cellInt < int3(0)) || any(cand.cellInt >= int3(gVoxelCount)))
//        return;

//    if (!isSolidVoxel(cand.cellInt))
//        return;

//    uint voxelIndex = CellToIndex(uint3(cand.cellInt), gVoxelCount);

//    VoxelData voxel = gGridDataParamBlock.gridDataBuffer[voxelIndex];

//    if (voxel.occupied == 0)
//        return;

//    float tau = max(gGeometryTau, 1e-5f);

//    float gmin = cand.gmin;

//    if (isBadFloat(gmin))
//        return;

//    float w = sigmoid(-gmin / tau);

//    float3 LCandidate = cand.radiance.xyz;
//    float3 CBehind = cand.behindColor.xyz;

//    LCandidate = sanitizeFloat3(LCandidate);
//    CBehind = sanitizeFloat3(CBehind);

//    //// dLoss / dw
//    // float dL_dW = dot(dL_dC, LCandidate - CBehind);
//    //// dw / dgmin
//    // float dW_dG = -w * (1.0f - w) / tau;
//    // float dL_dG = dL_dW * dW_dG;
//    float dL_dG = 0.0f;
//    if (isReferenceBackground)
//    {
//        dL_dG = -0.01 * w / tau;
//    }
//    else
//    {
//        float dL_dW =
//        dot(dL_dC, LCandidate - CBehind);

//        float dW_dG =
//        -w * (1.0f - w) / tau;

//        dL_dG = dL_dW * dW_dG;
//    }


//    dL_dG = sanitizeFloat(dL_dG);

//    // 防止几何梯度爆炸
//    dL_dG = clamp(dL_dG, -gGeometryGradClamp, gGeometryGradClamp);

//    Ellipsoid e = voxel.ellipsoid;

//    // xStar 是 global cell-space，转 voxel local
//    float3 xLocal = cand.xStar.xyz - float3(cand.cellInt);

//    float3 q = xLocal - e.center;

//    q = sanitizeFloat3(q);

//    float3 Bq = mul(e.B, q);

//    Bq = sanitizeFloat3(Bq);

//    geometryGradRecord.geometryValid = 1;

//    // dF / dCenter = -2 B q
//    geometryGradRecord.center = dL_dG * (-2.0f * Bq);

//    // dF / dB
//    geometryGradRecord.B00 = dL_dG * q.x * q.x;
//    geometryGradRecord.B11 = dL_dG * q.y * q.y;
//    geometryGradRecord.B22 = dL_dG * q.z * q.z;

//    // B is symmetric, off-diagonal appears twice
//    geometryGradRecord.B01 = dL_dG * 2.0f * q.x * q.y;
//    geometryGradRecord.B02 = dL_dG * 2.0f * q.x * q.z;
//    geometryGradRecord.B12 = dL_dG * 2.0f * q.y * q.z;

//    geometryGradRecord.center = sanitizeFloat3(geometryGradRecord.center);
//    geometryGradRecord.B00 = sanitizeFloat(geometryGradRecord.B00);
//    geometryGradRecord.B11 = sanitizeFloat(geometryGradRecord.B11);
//    geometryGradRecord.B22 = sanitizeFloat(geometryGradRecord.B22);
//    geometryGradRecord.B01 = sanitizeFloat(geometryGradRecord.B01);
//    geometryGradRecord.B02 = sanitizeFloat(geometryGradRecord.B02);
//    geometryGradRecord.B12 = sanitizeFloat(geometryGradRecord.B12);
//}


//void accumulateGeometryGradRecord(uint indexGradBuffer, GradRecord gradRecord)
//{
//    InterlockedAdd(gGradBuffer[indexGradBuffer].geometryValid, gradRecord.geometryValid);

//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].center.x,
//        gradRecord.center.x
//    );
//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].center.y,
//        gradRecord.center.y
//    );
//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].center.z,
//        gradRecord.center.z
//    );

//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].B00,
//        gradRecord.B00
//    );
//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].B11,
//        gradRecord.B11
//    );
//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].B22,
//        gradRecord.B22
//    );
//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].B01,
//        gradRecord.B01
//    );
//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].B02,
//        gradRecord.B02
//    );
//    INTERLOCKED_ADD_FLOAT(
//        gGradBuffer[indexGradBuffer].B12,
//        gradRecord.B12
//    );
//}

void calcAppearanceGrad(
    float3 dL_dRadiance,
    float dL_dOpacity,
    float opacity,
    float3 viewDir,
    out SphericalHarmonicsRGBGrad radianceGrad,
    out SphericalHarmonicsOpacity opacityGrad)
{
    radianceGrad.init();
    opacityGrad.init();
    opacityGrad.coefficients[0] = 0.0f;


    // dLoss/dRadiance : radiance SH 系数梯度
    radianceGrad.addGrad(
        dL_dRadiance,
        viewDir
    );

    /*
    内部完成：
    opacity = sigmoid(logit)
    dLoss/dlogit = dLoss/dopacity * opacity * (1-opacity)
    然后再乘对应的 SH 基函数。
    */
    opacityGrad.addOpacityGrad(
        dL_dOpacity,
        opacity,
        viewDir
    );
}

void accumulateAppearanceGrad(
    uint voxelIndex,
    SphericalHarmonicsRGBGrad radianceGrad,
    SphericalHarmonicsOpacity opacityGrad)
{
    InterlockedAdd(
        gGradBuffer[voxelIndex].appearanceValid,
        1
    );

    SH_MAX_ITERS(SH_COUNT)
    for (uint i = 0; i < SH_COUNT; ++i)
    {
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[voxelIndex].radianceGrad.coefficients[i].x,
            radianceGrad.coefficients[i].x
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[voxelIndex].radianceGrad.coefficients[i].y,
            radianceGrad.coefficients[i].y
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[voxelIndex].radianceGrad.coefficients[i].z,
            radianceGrad.coefficients[i].z
        );
    }

    SH_MAX_ITERS(SH_OPACITY_COUNT)
    for (uint i = 0; i < SH_OPACITY_COUNT; ++i)
    {
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[voxelIndex].opacityGrad.coefficients[i],
            opacityGrad.coefficients[i]
        );
    }
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
    bool isReferenceBackground = gBackgroundMaskBuffer[pixel] == 1;

    // -------------------------
    // 1. Appearance gradient
    // -------------------------
    if (rec.valid != 0)
    {
        uint hitCount = min(rec.contributingVoxelCount, (uint)MAX_CONTRIBUTING_VOXELS_PER_RAY);
        float3 behindColor =
        sanitizeFloat3(rec.backgroundColor.xyz);
        for (int i = int(hitCount) - 1; i >= 0; --i) {
            VoxelRecord voxelRec = rec.voxels[i];
            if (any(voxelRec.cellInt < int3(0)) || any(voxelRec.cellInt >= int3(gVoxelCount))){
                continue;
            }

            float3 radiance = sanitizeFloat3(voxelRec.radiance);
            float opacity = clamp(sanitizeFloat(voxelRec.opacity),0.0f,1.0f);
            float transmittanceBefore = clamp(sanitizeFloat(voxelRec.transmittanceBefore),0.0f,1.0f);

            float3 dL_dRadiance = dL_dC * transmittanceBefore * opacity;
            float dL_dOpacity = dot(dL_dC,transmittanceBefore *(radiance - behindColor));

            dL_dRadiance = sanitizeFloat3(dL_dRadiance);
            dL_dOpacity = sanitizeFloat(dL_dOpacity);

            /*
            对参考图背景区域，不允许错误体素通过修改颜色
            拟合背景，只让 opacity 梯度推动错误体素透明。
            */
            if (isReferenceBackground)
            {
                dL_dRadiance = float3(0.0f);
            }

            SphericalHarmonicsRGBGrad radianceGrad;
            SphericalHarmonicsOpacity opacityGrad;

            calcAppearanceGrad(
                dL_dRadiance,
                dL_dOpacity,
                opacity,
                rec.viewDir,
                radianceGrad,
                opacityGrad
            );

            uint voxelIndex = CellToIndex(uint3(voxelRec.cellInt),gVoxelCount);

            //if(i == 0)
            //    print("local opacity coeff grad:",opacityGrad.coefficients[0]);

            accumulateAppearanceGrad(voxelIndex,radianceGrad,opacityGrad);

            /*
            更新当前体素及其后方组成的局部颜色：
            必须放在当前体素梯度计算之后。
            */
            behindColor = opacity * radiance +(1.0f - opacity) * behindColor;
            behindColor = sanitizeFloat3(behindColor);
        }
  
    }

    // -------------------------
    // 2. Geometry gradient
    // all candidates
    // -------------------------

    //for (uint i = 0; i < rec.candidateCount && i < MAX_CANDIDATES; i++)
    //{
    //    GeometryCandidateRecord cand = rec.candidates[i];

    //    if (cand.valid == 0)
    //        continue;

    //    if (any(cand.cellInt < int3(0)) || any(cand.cellInt >= int3(gVoxelCount)))
    //        continue;

    //    GradRecord geometryGradRecord;
    //    calcGeometryGradFromCandidate(cand, dL_dC, isReferenceBackground, geometryGradRecord);

    //    if (geometryGradRecord.geometryValid == 0)
    //        continue;

    //    uint geometryIndexGradBuffer = CellToIndex(uint3(cand.cellInt), gVoxelCount);

    //    accumulateGeometryGradRecord(geometryIndexGradBuffer, geometryGradRecord);
    //}

    //VoxelRecord voxelRec = rec.voxels[0];
    //uint indexGradBuffer = CellToIndex(uint3(voxelRec.cellInt), gVoxelCount);
    //print("grad opacity :   ", gGradBuffer[indexGradBuffer].opacityGrad.coefficients[0]);
    //print("appearanceValid :", gGradBuffer[indexGradBuffer].appearanceValid);
    //print("average :        ", gGradBuffer[indexGradBuffer].opacityGrad.coefficients[0] / gGradBuffer[indexGradBuffer].appearanceValid);
    //print("Loss:", dL_dC);


    dummy[pixel] = float4(abs(dL_dC), 1.0f);
}


