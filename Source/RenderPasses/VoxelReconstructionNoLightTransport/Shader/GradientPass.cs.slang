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
    float gGeometryTau;             // soft-hit sigmoid 温度，越小越接近 hard hit
    float gGeometryGradClamp;       // 防止几何梯度爆炸
    float gBackgroundCarveWeight;   // 背景几何清除项的权重

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

void calcHardHitGeometryGrad(
    VoxelRecord voxelRec,
    float3 behindColor,
    float3 dL_dC,
    bool isReferenceBackground,
    out GradRecord geometryGradRecord)
{
    geometryGradRecord.init();

    int3 cellInt = voxelRec.cellInt;

    if (any(cellInt < int3(0)) || any(cellInt >= int3(gVoxelCount)))
        return;
    if (!isSolidVoxel(cellInt))
        return;

    uint voxelIndex = CellToIndex( uint3(cellInt), gVoxelCount);
    VoxelData voxel = gGridDataParamBlock.gridDataBuffer[voxelIndex];

    if (voxel.occupied == 0)
        return;

    float gmin = voxelRec.gmin;

    if (isBadFloat(gmin))
        return;

    float tau = max(gGeometryTau, 1e-5f);
    float w = sigmoid(-gmin / tau);

    float dL_dG = 0.0f;

    if (isReferenceBackground)
    {
        // 当前函数只处理实际hard-hit，
        // 因此背景区域可以直接使用carve梯度。
        dL_dG = -gBackgroundCarveWeight * w / tau;
    }
    else
    {
        float3 radiance = sanitizeFloat3( voxelRec.radiance);

        float opacity =clamp(sanitizeFloat(voxelRec.opacity),0.0f,1.0f);

        float transmittanceBefore =
            clamp(
                sanitizeFloat(
                    voxelRec.transmittanceBefore
                ),
                0.0f,
                1.0f
            );

        behindColor = sanitizeFloat3(behindColor);

        /*
        dC/dw = T_before * opacity * (radiance - behindColor)
        */
        float3 dC_dW =transmittanceBefore * opacity * (radiance - behindColor);

        float dL_dW = dot(dL_dC, dC_dW);
        float dW_dG =-w *(1.0f - w) /tau;

        dL_dG = dL_dW * dW_dG;
    }

    dL_dG =
        clamp(
            sanitizeFloat(dL_dG),
            - gGeometryGradClamp,
            gGeometryGradClamp
        );

    Ellipsoid ellipsoid =
        voxel.ellipsoid;

    // xStar保存在global cell-space，
    // 这里转换回当前体素局部坐标。
    float3 xLocal =
        voxelRec.xStar.xyz -
        float3(cellInt);

    float3 q =sanitizeFloat3(xLocal -ellipsoid.center);

    float3 Bq =sanitizeFloat3(mul(ellipsoid.B, q));

    geometryGradRecord.geometryValid = 1;

    // dg/dcenter = -2Bq
    geometryGradRecord.center =dL_dG *(-2.0f * Bq);

    // dg/dB
    geometryGradRecord.B00 =dL_dG * q.x * q.x;
    geometryGradRecord.B11 =dL_dG * q.y * q.y;
    geometryGradRecord.B22 =dL_dG * q.z * q.z;

    // B是对称矩阵，非对角参数出现两次
    geometryGradRecord.B01 = dL_dG * 2.0f * q.x * q.y;
    geometryGradRecord.B02 = dL_dG * 2.0f * q.x * q.z;
    geometryGradRecord.B12 = dL_dG * 2.0f * q.y * q.z;

    geometryGradRecord.center = sanitizeFloat3( geometryGradRecord.center);

    geometryGradRecord.B00 = sanitizeFloat( geometryGradRecord.B00);
    geometryGradRecord.B11 = sanitizeFloat( geometryGradRecord.B11);
    geometryGradRecord.B22 = sanitizeFloat( geometryGradRecord.B22);
    geometryGradRecord.B01 = sanitizeFloat( geometryGradRecord.B01);
    geometryGradRecord.B02 = sanitizeFloat( geometryGradRecord.B02);
    geometryGradRecord.B12 = sanitizeFloat( geometryGradRecord.B12);
}

void calcMissCandidateGeometryGrad(
    GeometryCandidateRecord cand,
    float3 candidateBehindColor,
    float3 dL_dC,
    float3 viewDir,
    out GradRecord geometryGradRecord)
{
    geometryGradRecord.init();

    if (cand.valid == 0)
        return;

    int3 cellInt = cand.cellInt;

    if (any(cellInt < int3(0)) || any(cellInt >= int3(gVoxelCount)))
        return;
    if (!isSolidVoxel(cellInt))
        return;

    uint voxelIndex = CellToIndex(uint3(cellInt), gVoxelCount);

    VoxelData voxel = gGridDataParamBlock.gridDataBuffer[voxelIndex];

    if (voxel.occupied == 0)
        return;

    float gmin = cand.gmin;

    if (isBadFloat(gmin))
        return;

    float tau = max(gGeometryTau, 1e-5f);
    float w = sigmoid(-gmin / tau);

    float3 radiance = sanitizeFloat3(voxel.radiance.calc(viewDir));

    float opacity = clamp( sanitizeFloat( voxel.opacity.calcOpacity(viewDir) ),0.0f,1.0f );

    float transmittanceBefore = clamp( sanitizeFloat( cand.transmittanceBefore),0.0f,1.0f);

    candidateBehindColor = sanitizeFloat3(candidateBehindColor);

    /*
    假设 near-miss candidate 逐渐变为 hit：
    dC/dw = T_before * opacity * (radiance - candidateBehindColor)
    */
    float3 dC_dW = transmittanceBefore * opacity * (radiance - candidateBehindColor);
    float dL_dW = dot(dL_dC, dC_dW);
    float dW_dG = -w * (1.0f - w) / tau;
    float dL_dG = dL_dW * dW_dG;

    dL_dG =clamp( sanitizeFloat(dL_dG),- gGeometryGradClamp, gGeometryGradClamp);

    Ellipsoid ellipsoid = voxel.ellipsoid;

    // cand.xStar 是 global cell-space。
    float3 xLocal = cand.xStar.xyz - float3(cellInt);

    float3 q =sanitizeFloat3(xLocal - ellipsoid.center);
    float3 Bq =sanitizeFloat3(mul(ellipsoid.B, q));

    geometryGradRecord.geometryValid = 1;

    // dg / dcenter = -2Bq
    geometryGradRecord.center = dL_dG * (-2.0f * Bq);

    // dg / dB
    geometryGradRecord.B00 = dL_dG * q.x * q.x;
    geometryGradRecord.B11 = dL_dG * q.y * q.y;
    geometryGradRecord.B22 = dL_dG * q.z * q.z;
    geometryGradRecord.B01 = dL_dG * 2.0f * q.x * q.y;
    geometryGradRecord.B02 = dL_dG * 2.0f * q.x * q.z;
    geometryGradRecord.B12 = dL_dG * 2.0f * q.y * q.z;

    geometryGradRecord.center =sanitizeFloat3(geometryGradRecord.center);

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
    // 1. Appearance gradient & hit geometry gradient
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
            accumulateAppearanceGrad(voxelIndex, radianceGrad, opacityGrad);

            // -------------------------
            // Hard-hit geometry gradient
            // -------------------------
            GradRecord geometryGradRecord;
            calcHardHitGeometryGrad(
                voxelRec,
                behindColor,
                dL_dC,
                isReferenceBackground,
                geometryGradRecord
            );

            if (geometryGradRecord.geometryValid != 0)
            {
                accumulateGeometryGradRecord(
                    voxelIndex,
                    geometryGradRecord
                );
            }

            /*
            更新当前体素及其后方组成的局部颜色：
            必须放在当前体素梯度计算之后。
            */
            behindColor = opacity * radiance +(1.0f - opacity) * behindColor;
            behindColor = sanitizeFloat3(behindColor);
        }
  
    }

    // -------------------------
    // 2. Near-miss candidate geometry gradient
    // -------------------------
    /*
    candidate 本身没有实际命中。
    参考图是背景时，candidate 当前已经是 miss，不需要继续执行 carve。
    只有参考图是前景时，才通过 candidate 梯度 推动椭球表面向 ray 扩张。
    */
    if (!isReferenceBackground)
    {
        uint candidateCount = min( rec.candidateCount, (uint)MAX_CANDIDATES);
        uint hitCount = min( rec.contributingVoxelCount,(uint)MAX_CONTRIBUTING_VOXELS_PER_RAY);

        for (uint i = 0; i < candidateCount; ++i)
        {
            GeometryCandidateRecord cand = rec.candidates[i];

            if (cand.valid == 0)
                continue;
            if (any(cand.cellInt < int3(0)) || any(cand.cellInt >= int3(gVoxelCount)))
                continue;
            if (!isSolidVoxel(cand.cellInt))
                continue;

            /*
            cand.frontHitCount 表示 candidate 前面已经有多少个 hard-hit。
            因此 candidate 后方的 hard-hit 范围是：[cand.frontHitCount, hitCount)
            */
            uint behindStart = min( cand.frontHitCount, hitCount);
            float3 candidateBehindColor =sanitizeFloat3(rec.backgroundColor.xyz);

            /*
            从最后一个 hard-hit 向前合成，直到 candidate 后方的第一个 hard-hit。
            */
            for ( int j = int(hitCount) - 1; j >= int(behindStart); --j)
            {
                VoxelRecord behindRec = rec.voxels[j];

                float3 behindRadiance = sanitizeFloat3( behindRec.radiance);
                float behindOpacity =clamp(sanitizeFloat(behindRec.opacity),0.0f,1.0f);

                candidateBehindColor = behindOpacity * behindRadiance + (1.0f - behindOpacity) * candidateBehindColor;
                candidateBehindColor = sanitizeFloat3( candidateBehindColor);
            }

            GradRecord geometryGradRecord;
            calcMissCandidateGeometryGrad(
                cand,
                candidateBehindColor,
                dL_dC,
                rec.viewDir,
                geometryGradRecord
            );

            if (geometryGradRecord.geometryValid == 0)
                continue;

            uint voxelIndex = CellToIndex( uint3(cand.cellInt), gVoxelCount);
            accumulateGeometryGradRecord(
                voxelIndex,
                geometryGradRecord
            );
        }
    }

    // VoxelRecord voxelRec = rec.voxels[0];
    // uint indexGradBuffer = CellToIndex(uint3(voxelRec.cellInt), gVoxelCount);
    // print("grad opacity :   ", gGradBuffer[indexGradBuffer].opacityGrad.coefficients[0]);
    // print("appearanceValid :", gGradBuffer[indexGradBuffer].appearanceValid);
    // print("average :        ", gGradBuffer[indexGradBuffer].opacityGrad.coefficients[0] / gGradBuffer[indexGradBuffer].appearanceValid);
    //print("Loss:", dL_dC);
    print("rec.candidateCount : ", rec.candidateCount);
    if (rec.candidateCount > 0) {
        GeometryCandidateRecord cand = rec.candidates[0];
        uint indexGradBuffer = CellToIndex(uint3(cand.cellInt), gVoxelCount);
        GradRecord accumulatedGrad = gGradBuffer[indexGradBuffer];
        if (accumulatedGrad.geometryValid != 0)
        {
            float invCount =
                1.0f / float(accumulatedGrad.geometryValid);

            float3 averageCenterGrad =
                accumulatedGrad.center * invCount;

            float averageB00 =
                accumulatedGrad.B00 * invCount;
            float averageB11 =
                accumulatedGrad.B11 * invCount;
            float averageB22 =
                accumulatedGrad.B22 * invCount;

            float averageB01 =
                accumulatedGrad.B01 * invCount;
            float averageB02 =
                accumulatedGrad.B02 * invCount;
            float averageB12 =
                accumulatedGrad.B12 * invCount;

            print("candidate gmin:",
                cand.gmin);

            print("partial geometryValid:",
                accumulatedGrad.geometryValid);

            print("average center grad:",
                averageCenterGrad);

            print("average center grad norm:",
                length(averageCenterGrad));

            print("average B00 grad:", averageB00);
            print("average B11 grad:", averageB11);
            print("average B22 grad:", averageB22);

            print("average B01 grad:", averageB01);
            print("average B02 grad:", averageB02);
            print("average B12 grad:", averageB12);

            float BGradNorm =
                sqrt(
                    averageB00 * averageB00 +
                    averageB11 * averageB11 +
                    averageB22 * averageB22 +
                     2.0f * averageB01 * averageB01 +
                    2.0f * averageB02 * averageB02 +
                    2.0f * averageB12 * averageB12
                );

            print("average B grad norm:",
                BGradNorm);
        }
        
    }


    dummy[pixel] = float4(abs(dL_dC), 1.0f);
}


