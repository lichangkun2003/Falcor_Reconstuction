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
RWTexture2D<float4> dummy;

cbuffer CB {
    uint2 gResolution;
    uint3 gVoxelCount;
    float gGeometryTauWorld;        // Current voxel width * Geometry Tau (voxels); shared by hit/miss.
    float gGeometryGradClamp;       // Clamp dL/dg after converting the world-distance gradient.
    float gAlphaGeometryWeight;     // Geometry-only multiplier; dL/dAlpha already includes the alpha-loss weight.
    float gBackgroundCarveWeight;   // 背景几何清除项的权重
    float gBinaryOpacityWeight;
    bool gCollectDeletionEvidence;
    uint gEvidenceViewID;
    float gMinRemovalLossDelta;
    float gMinEvidenceTransmittance;
    float gAlphaLossWeight;

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

void fillGaussianEllipsoidGradient(
    GaussianEllipsoid ellipsoid,
    float3 xLocal,
    float dL_dG,
    out GradRecord geometryGradRecord)
{
    float3 voxelSize = max(gGridDataParamBlock.voxelSize, float3(1e-8f));
    float3x3 rotation = gaussianRotationMatrix(ellipsoid.rotation);
    float3 qWorld = (xLocal - ellipsoid.center) * voxelSize;
    float3 principal = mul(transpose(rotation), qWorld);
    float3 invScaleSquared = exp(clamp(-2.0f * ellipsoid.logScale, float3(-60.0f), float3(60.0f)));
    float3 weightedPrincipal = principal * invScaleSquared;

    geometryGradRecord.geometryValid = 1;
    geometryGradRecord.center = dL_dG * (-2.0f * mul(rotation, weightedPrincipal) * voxelSize);
    geometryGradRecord.logScale = dL_dG * (-2.0f * weightedPrincipal * principal);
    // The quaternion update applies a local/right-multiplied axis-angle step.
    geometryGradRecord.rotation = dL_dG * (2.0f * cross(weightedPrincipal, principal));

    geometryGradRecord.center = sanitizeFloat3(geometryGradRecord.center);
    geometryGradRecord.logScale = sanitizeFloat3(geometryGradRecord.logScale);
    geometryGradRecord.rotation = sanitizeFloat3(geometryGradRecord.rotation);
}

float calcAlphaGeometryDistanceGrad(
    float dL_dAlpha,
    float transmittanceBefore,
    float opacity,
    float transmittanceAfter,
    float w,
    float tau)
{
    if (gAlphaGeometryWeight <= 0.0f)
        return 0.0f;

    // Gate one hard hit or insert one virtual miss into the recorded hard path.
    // A = 1 - product(1 - opacity_i * gate_i), so dA/dgate_i = T_before * opacity_i * T_after.
    // The local gate surrogate has dw/dd = -w*(1-w)/tau; forward remains a hard intersection.
    float dA_dGate =
        saturate(sanitizeFloat(transmittanceBefore)) *
        saturate(sanitizeFloat(opacity)) *
        saturate(sanitizeFloat(transmittanceAfter));
    float dL_dGate = gAlphaGeometryWeight * sanitizeFloat(dL_dAlpha) * dA_dGate;
    return dL_dGate * (-w * (1.0f - w) / tau);
}

void calcHardHitGeometryGrad(
    VoxelRecord voxelRec,
    float3 behindColor,
    float3 dL_dC,
    float dL_dA,
    float transmittanceAfter,
    bool isReferenceBackground,
    out GradRecord geometryGradRecord)
{
    geometryGradRecord.init();

    int3 cellInt = getVoxelCell(voxelRec.voxelID);

    if (any(cellInt < int3(0)) || any(cellInt >= int3(gVoxelCount)))
        return;
    if (!isSolidVoxel(cellInt))
        return;

    uint voxelIndex = voxelRec.voxelID;
    VoxelData voxel = VOXEL_DATA(voxelIndex);

    if (voxel.occupied == 0)
        return;

    float gmin = voxelRec.gmin;

    if (isBadFloat(gmin))
        return;

    GaussianEllipsoid ellipsoid = voxel.ellipsoid;
    float3 xLocal = voxelRec.xStar - float3(cellInt);
    float distanceWorld;
    float distanceScale;
    if (!calcGaussianEllipsoidWorldDistance(
        ellipsoid, xLocal, gGridDataParamBlock.voxelSize, gmin, distanceWorld, distanceScale))
        return;

    float tau = max(gGeometryTauWorld, 1e-8f);
    float w = sigmoid(-distanceWorld / tau);

    float dL_dDistance = 0.0f;

    if (isReferenceBackground)
    {
        // 当前函数只处理实际hard-hit，
        // 因此背景区域可以直接使用carve梯度。
        dL_dDistance = -gBackgroundCarveWeight * w / tau;
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
        float dW_dDistance = -w * (1.0f - w) / tau;

        dL_dDistance = dL_dW * dW_dDistance;
    }

    // Add the alpha geometry proxy to the existing RGB/carve branch before the shared clamp.
    dL_dDistance += calcAlphaGeometryDistanceGrad(
        dL_dA, voxelRec.transmittanceBefore, voxelRec.opacity, transmittanceAfter, w, tau);

    // Surface normalization is a stop-gradient preconditioner for the existing dg/d(theta).
    float dL_dG =
        clamp(
            sanitizeFloat(dL_dDistance * distanceScale),
            - gGeometryGradClamp,
            gGeometryGradClamp
        );

    fillGaussianEllipsoidGradient(ellipsoid, xLocal, dL_dG, geometryGradRecord);
    if (isReferenceBackground && geometryGradRecord.geometryValid != 0)
        geometryGradRecord.backgroundCarveValid = 1;
}

void calcMissCandidateGeometryGrad(
    GeometryCandidateRecord cand,
    float3 candidateBehindColor,
    float3 dL_dC,
    float dL_dA,
    float transmittanceAfter,
    float3 viewDir,
    out GradRecord geometryGradRecord)
{
    geometryGradRecord.init();

    if (cand.valid == 0)
        return;

    int3 cellInt = getVoxelCell(cand.voxelID);

    if (any(cellInt < int3(0)) || any(cellInt >= int3(gVoxelCount)))
        return;
    if (!isSolidVoxel(cellInt))
        return;

    uint voxelIndex = cand.voxelID;

    VoxelData voxel = VOXEL_DATA(voxelIndex);

    if (voxel.occupied == 0)
        return;

    float gmin = cand.gmin;

    if (isBadFloat(gmin))
        return;

    GaussianEllipsoid ellipsoid = voxel.ellipsoid;
    float3 xLocal = cand.xStar - float3(cellInt);
    float distanceWorld;
    float distanceScale;
    if (!calcGaussianEllipsoidWorldDistance(
        ellipsoid, xLocal, gGridDataParamBlock.voxelSize, gmin, distanceWorld, distanceScale))
        return;

    float tau = max(gGeometryTauWorld, 1e-8f);
    float w = sigmoid(-distanceWorld / tau);

    float3 radiance = sanitizeFloat3(voxel.radiance.calc(viewDir));
    radiance = max(radiance, 0.0f);

    float opacity = clamp( sanitizeFloat( voxel.opacity.calcOpacity(viewDir) ),0.0f,1.0f );

    float transmittanceBefore = clamp( sanitizeFloat( cand.transmittanceBefore),0.0f,1.0f);

    candidateBehindColor = sanitizeFloat3(candidateBehindColor);

    /*
    假设 near-miss candidate 逐渐变为 hit：
    dC/dw = T_before * opacity * (radiance - candidateBehindColor)
    */
    float3 dC_dW = transmittanceBefore * opacity * (radiance - candidateBehindColor);
    float dL_dW = dot(dL_dC, dC_dW);
    float dW_dDistance = -w * (1.0f - w) / tau;
    float dL_dDistance = dL_dW * dW_dDistance;
    dL_dDistance += calcAlphaGeometryDistanceGrad(
        dL_dA, transmittanceBefore, opacity, transmittanceAfter, w, tau);

    float dL_dG = clamp(sanitizeFloat(dL_dDistance * distanceScale), -gGeometryGradClamp, gGeometryGradClamp);

    fillGaussianEllipsoidGradient(ellipsoid, xLocal, dL_dG, geometryGradRecord);
}


void accumulateGeometryGradRecord(uint indexGradBuffer, GradRecord gradRecord)
{
    InterlockedAdd(GRAD_DATA(indexGradBuffer).geometryValid, gradRecord.geometryValid);
    InterlockedAdd(GRAD_DATA(indexGradBuffer).backgroundCarveValid, gradRecord.backgroundCarveValid);

    INTERLOCKED_ADD_FLOAT(
        GRAD_DATA(indexGradBuffer).center.x,
        gradRecord.center.x
    );
    INTERLOCKED_ADD_FLOAT(
        GRAD_DATA(indexGradBuffer).center.y,
        gradRecord.center.y
    );
    INTERLOCKED_ADD_FLOAT(
        GRAD_DATA(indexGradBuffer).center.z,
        gradRecord.center.z
    );

    INTERLOCKED_ADD_FLOAT(GRAD_DATA(indexGradBuffer).logScale.x, gradRecord.logScale.x);
    INTERLOCKED_ADD_FLOAT(GRAD_DATA(indexGradBuffer).logScale.y, gradRecord.logScale.y);
    INTERLOCKED_ADD_FLOAT(GRAD_DATA(indexGradBuffer).logScale.z, gradRecord.logScale.z);
    INTERLOCKED_ADD_FLOAT(GRAD_DATA(indexGradBuffer).rotation.x, gradRecord.rotation.x);
    INTERLOCKED_ADD_FLOAT(GRAD_DATA(indexGradBuffer).rotation.y, gradRecord.rotation.y);
    INTERLOCKED_ADD_FLOAT(GRAD_DATA(indexGradBuffer).rotation.z, gradRecord.rotation.z);
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
        GRAD_DATA(voxelIndex).appearanceValid,
        1
    );

    SH_MAX_ITERS(SH_COUNT)
    for (uint i = 0; i < SH_COUNT; ++i)
    {
        INTERLOCKED_ADD_FLOAT(
            GRAD_DATA(voxelIndex).radianceGrad.coefficients[i].x,
            radianceGrad.coefficients[i].x
        );

        INTERLOCKED_ADD_FLOAT(
            GRAD_DATA(voxelIndex).radianceGrad.coefficients[i].y,
            radianceGrad.coefficients[i].y
        );

        INTERLOCKED_ADD_FLOAT(
            GRAD_DATA(voxelIndex).radianceGrad.coefficients[i].z,
            radianceGrad.coefficients[i].z
        );
    }

    SH_MAX_ITERS(SH_OPACITY_COUNT)
    for (uint i = 0; i < SH_OPACITY_COUNT; ++i)
    {
        INTERLOCKED_ADD_FLOAT(
            GRAD_DATA(voxelIndex).opacityGrad.coefficients[i],
            opacityGrad.coefficients[i]
        );
    }
}




void recordUniqueDeletionEvidence(uint voxelIndex, bool foreground)
{
    uint previousView = 0u;
    if (foreground)
    {
        // Most pixels/samples for a view arrive after its first vote. Avoid an
        // unnecessary atomic in that common case; the exchange still resolves
        // races between the first voting pixels.
        if (TOPOLOGY_DATA(voxelIndex).lastForegroundView == gEvidenceViewID)
            return;
        InterlockedExchange(
            TOPOLOGY_DATA(voxelIndex).lastForegroundView,
            gEvidenceViewID,
            previousView
        );
        if (previousView != gEvidenceViewID)
            InterlockedAdd(TOPOLOGY_DATA(voxelIndex).packedCountsAndFlags, 1u);
    }
    else
    {
        if (TOPOLOGY_DATA(voxelIndex).lastBackgroundView == gEvidenceViewID)
            return;
        InterlockedExchange(
            TOPOLOGY_DATA(voxelIndex).lastBackgroundView,
            gEvidenceViewID,
            previousView
        );
        if (previousView != gEvidenceViewID)
            InterlockedAdd(
                TOPOLOGY_DATA(voxelIndex).packedCountsAndFlags,
                1u << kTopologyEvidenceBackgroundShift
            );
    }
}

void collectDeletionEvidence(
    uint voxelIndex,
    uint referenceClass,
    float3 dL_dC,
    float dL_dA,
    float3 radiance,
    float3 behindColor,
    float opacity,
    float transmittanceBefore,
    float transmittanceAfter)
{
    if (!gCollectDeletionEvidence ||
        transmittanceBefore < gMinEvidenceTransmittance ||
        (referenceClass != 1u && referenceClass != 2u))
        return;

    // Exact loss change for locally removing this hard hit while holding all
    // other recorded hits fixed. Positive means removal hurts (keep it);
    // negative means removal improves the pixel (background conflict).
    float3 colorDelta = transmittanceBefore * opacity * (behindColor - radiance);
    float alphaDelta = -transmittanceBefore * opacity * transmittanceAfter;
    float removalLossDelta =
        dot(dL_dC, colorDelta) + dot(colorDelta, colorDelta) +
        dL_dA * alphaDelta + gAlphaLossWeight * alphaDelta * alphaDelta;
    removalLossDelta = sanitizeFloat(removalLossDelta);

    if (referenceClass == 2u && removalLossDelta > gMinRemovalLossDelta)
        recordUniqueDeletionEvidence(voxelIndex, true);
    else if (referenceClass == 1u && removalLossDelta < -gMinRemovalLossDelta)
        recordUniqueDeletionEvidence(voxelIndex, false);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;

    if (pixel.x >= gResolution.x || pixel.y >= gResolution.y)
        return;
    dummy[pixel] = float4(0.0f);
    uint indexPathRecordBuffer = pixel.y * gResolution.x + pixel.x;
    PathRecord rec = gPathRecordBuffer[indexPathRecordBuffer];

    if (rec.valid == 0 && rec.candidateCount == 0)
        return;

    float4 dL_dRGBA = gDL_dColorBuffer[pixel];
    float3 dL_dC = dL_dRGBA.xyz;
    float dL_dA = dL_dRGBA.w;
    uint referenceClass = gBackgroundMaskBuffer[pixel];
    bool isReferenceBackground = referenceClass == 1u;

    // -------------------------
    // 1. Appearance gradient & hit geometry gradient
    // -------------------------
    if (rec.valid != 0)
    {
        uint hitCount = min(rec.contributingVoxelCount, (uint)MAX_CONTRIBUTING_VOXELS_PER_RAY);
        float3 behindColor = sanitizeFloat3(rec.backgroundColor.xyz);
        float transmittanceAfter = 1.0f;
        for (int i = int(hitCount) - 1; i >= 0; --i) {
            VoxelRecord voxelRec = rec.voxels[i];
            int3 voxelCell = getVoxelCell(voxelRec.voxelID);
            if (any(voxelCell < int3(0)) || any(voxelCell >= int3(gVoxelCount))){
                continue;
            }

            float3 radiance = sanitizeFloat3(voxelRec.radiance);
            float opacity = clamp(sanitizeFloat(voxelRec.opacity),0.0f,1.0f);
            float transmittanceBefore = clamp(sanitizeFloat(voxelRec.transmittanceBefore),0.0f,1.0f);

            float3 dL_dRadiance = dL_dC * transmittanceBefore * opacity;
            float dL_dOpacityRGB = dot(dL_dC, transmittanceBefore * (radiance - behindColor));
            float dA_dOpacity = transmittanceBefore * transmittanceAfter;
            float dL_dOpacityAlpha = dL_dA * dA_dOpacity;
            //float dL_dOpacityBinary = gBinaryOpacityWeight * (1.0f - 2.0f * opacity);
            float dL_dOpacity = dL_dOpacityRGB + dL_dOpacityAlpha;

            dL_dRadiance = sanitizeFloat3(dL_dRadiance);
            dL_dOpacity = sanitizeFloat(dL_dOpacity);

            /*
            RayMarchingPass 里 radiance 被 max(radiance, 0) 夹过，所以 radiance 为 0 的
            通道正处在死区：渲染值恒为 0，当前视角下 loss 对该通道的偏导确实是 0，
            任何方向都不会降低 loss。但两半梯度的后果完全不同：

              dL_dRadiance < 0（渲染比目标暗，该更亮）
                  -> 更新会抬高 SH，是唯一能把它推出死区的方向，保留；

              dL_dRadiance > 0（渲染比目标亮，该更暗）
                  -> 更新会压低 SH，把系数推得更负，而渲染早已不变，没有恢复力。
                     这正是之前 radiance DC 跑到 -3.5、单项最大 +44 的机制。

            所以死区里只留"唤醒"的那一半。注意不能直接归零，那会变成 dead ReLU，
            系数一旦落到负区就永远回不来。
            已经比目标亮的体素冻在黑色本来就是局部最优：它该做的是变透明（opacity 梯度），
            不是变亮。
            */
            float3 atClampMask = float3(
                radiance.x <= 0.0f ? 1.0f : 0.0f,
                radiance.y <= 0.0f ? 1.0f : 0.0f,
                radiance.z <= 0.0f ? 1.0f : 0.0f
            );
            dL_dRadiance = lerp(dL_dRadiance, min(dL_dRadiance, float3(0.0f)), atClampMask);

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

            uint voxelIndex = voxelRec.voxelID;
            accumulateAppearanceGrad(voxelIndex, radianceGrad, opacityGrad);
            collectDeletionEvidence(
                voxelIndex,
                referenceClass,
                dL_dC,
                dL_dA,
                radiance,
                behindColor,
                opacity,
                transmittanceBefore,
                transmittanceAfter
            );

            // -------------------------
            // Hard-hit geometry gradient
            // -------------------------
            GradRecord geometryGradRecord;
            calcHardHitGeometryGrad(
                voxelRec,
                behindColor,
                dL_dC,
                dL_dA,
                transmittanceAfter,
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

            transmittanceAfter *= (1.0f - opacity);
            transmittanceAfter = clamp(sanitizeFloat(transmittanceAfter), 0.0f, 1.0f);
        }

    }

    // -------------------------
    // 2. Near-miss candidate geometry gradient
    // -------------------------
    /*
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
            int3 candidateCell = getVoxelCell(cand.voxelID);
            if (any(candidateCell < int3(0)) || any(candidateCell >= int3(gVoxelCount)))
                continue;
            if (!isSolidVoxel(candidateCell))
                continue;

            /*
            cand.frontHitCount 表示 candidate 前面已经有多少个 hard-hit。
            因此 candidate 后方的 hard-hit 范围是：[cand.frontHitCount, hitCount)
            */
            uint behindStart = min( cand.frontHitCount, hitCount);
            float3 candidateBehindColor =sanitizeFloat3(rec.backgroundColor.xyz);
            float candidateTransmittanceAfter = 1.0f;

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
                candidateTransmittanceAfter *= 1.0f - behindOpacity;
            }

            GradRecord geometryGradRecord;
            calcMissCandidateGeometryGrad(
                cand,
                candidateBehindColor,
                dL_dC,
                dL_dA,
                candidateTransmittanceAfter,
                rec.viewDir,
                geometryGradRecord
            );

            if (geometryGradRecord.geometryValid == 0)
                continue;

            uint voxelIndex = cand.voxelID;
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


    dummy[pixel] = float4(abs(dL_dC), 1.0f);
}


