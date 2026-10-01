#include "../Voxel/VoxelGrid.slang"
#include "../GradRecord.slang"
#include "../GeometryAdamState.slang"

// Input
#if RECON_MODE != RECON_MODE_POINT_CLOUD
StructuredBuffer<GradRecord> gGradBuffer;
#else
RWStructuredBuffer<GeometryAdamState> gGeometryAdamPages[SPARSE_POOL_MAX_PAGES];
RWStructuredBuffer<uint> gRadianceAdamIndexPages[SPARSE_POOL_MAX_PAGES];
RWStructuredBuffer<RadianceAdamState> gRadianceAdamPages[SPARSE_POOL_MAX_PAGES];
RWStructuredBuffer<uint> gRadianceAdamCounter;
#endif

//StructuredBuffer<float> gVoxelSHGrads;

cbuffer CB {
    // 是否按当前 voxel 命中的 pixel 数做平均
    bool gUseGradCountNormalize;

    // 全局梯度缩放，第一版可以设为 1.0
    float gGradScale;

    float gLrRadiance;
    float gLrCenter;
    float gLrShape;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    float gLrRotation;
#endif
    float gLrOpacity;

    // Applied after Adam normalization. A value above one accelerates geometry
    // updates in proportion to the fraction of samples coming from known-empty
    // reference background pixels.
    float gBackgroundCarveAdamMultiplier;

    float gEllipsoidPruneThreshold;
    bool gEnableEllipsoidPruning;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    uint gSparseUpdateOffset;
    uint gRadianceAdamCapacity;
#endif


    //uint gVoxelSHGradDim;
}




void updateEllipsoidCenter(inout GaussianEllipsoid ellipsoid, float3 gradCenter, float lr)
{
    if (lr <= 0.0f)
        return;

    gradCenter = sanitizeFloat3(gradCenter);

    float3 step = lr * gradCenter;

    // 防止一次跨太远。center 是 voxel-local 坐标，建议单步不要太大。
    step = clamp(step, float3(-0.02f), float3(0.02f));

    ellipsoid.center -= step;

    // center 保持在 voxel local 范围内。
    // 可以稍微留边，避免跑出 voxel 后数值很怪。
    ellipsoid.center = clamp(
        ellipsoid.center,
        float3(0.01f),
        float3(0.99f)
    );
}

void updateEllipsoidShape(
    inout GaussianEllipsoid ellipsoid,
    GradRecord grad,
    float lr,
    float gradScale,
    float3 voxelSize)
{
    if (lr <= 0.0f)
        return;

    float3 logScaleGrad = clamp(
        sanitizeFloat3(grad.logScale * gradScale),
        float3(-10.0f),
        float3(10.0f)
    );
    float3 logScaleStep = clamp(lr * logScaleGrad, float3(-0.02f), float3(0.02f));
    ellipsoid.logScale -= logScaleStep;

    float3 minScale = max(voxelSize * 1e-4f, float3(1e-8f));
    float3 maxScale = max(voxelSize * 4.0f, minScale);
    ellipsoid.logScale = clamp(
        sanitizeFloat3(ellipsoid.logScale),
        log(minScale),
        log(maxScale)
    );

    float3 rotationGrad = clamp(
        sanitizeFloat3(grad.rotation * gradScale),
        float3(-10.0f),
        float3(10.0f)
    );
    float3 deltaAngle = -lr * rotationGrad;
    float angle = length(deltaAngle);
    if (angle > 0.02f)
    {
        deltaAngle *= 0.02f / angle;
        angle = 0.02f;
    }

    if (angle > 1e-8f)
    {
        float halfAngle = 0.5f * angle;
        float3 axis = deltaAngle / angle;
        float4 deltaRotation = float4(cos(halfAngle), axis * sin(halfAngle));
        ellipsoid.rotation = normalizeGaussianRotation(
            multiplyGaussianRotations(ellipsoid.rotation, deltaRotation)
        );
    }
    else
    {
        ellipsoid.rotation = normalizeGaussianRotation(ellipsoid.rotation);
    }
}

#if RECON_MODE == RECON_MODE_POINT_CLOUD
// Moments are updated once per view, after that view's SPP gradients have been averaged.
// Each parameter group uses an independent step count so an LR of zero does not age it.
static const float kAdamBeta1 = 0.9f;
static const float kAdamBeta2 = 0.999f;
static const float kAdamEpsilon = 1e-8f;

float3 adamStepWithBias(float3 gradient, inout float3 mean, inout float3 squaredMean,
    float bias1, float bias2, float lr)
{
    gradient = clamp(sanitizeFloat3(gradient), float3(-10.0f), float3(10.0f));
    mean = kAdamBeta1 * mean + (1.0f - kAdamBeta1) * gradient;
    squaredMean = kAdamBeta2 * squaredMean + (1.0f - kAdamBeta2) * gradient * gradient;
    float3 correctedMean = mean / bias1;
    float3 correctedSquaredMean = max(squaredMean / bias2, float3(0.0f));
    return sanitizeFloat3(lr * correctedMean / (sqrt(correctedSquaredMean) + kAdamEpsilon));
}

float3 adamStepAtStep(float3 gradient, inout float3 mean, inout float3 squaredMean, uint steps, float lr)
{
    float bias1 = max(1.0f - pow(kAdamBeta1, float(steps)), 1e-8f);
    float bias2 = max(1.0f - pow(kAdamBeta2, float(steps)), 1e-8f);
    return adamStepWithBias(gradient, mean, squaredMean, bias1, bias2, lr);
}

float3 adamStep(float3 gradient, inout float3 mean, inout float3 squaredMean, inout uint steps, float lr)
{
    steps += 1;
    return adamStepAtStep(gradient, mean, squaredMean, steps, lr);
}

float adamStepWithBias(float gradient, inout float mean, inout float squaredMean,
    float bias1, float bias2, float lr)
{
    gradient = clamp(sanitizeFloat(gradient), -10.0f, 10.0f);
    mean = kAdamBeta1 * mean + (1.0f - kAdamBeta1) * gradient;
    squaredMean = kAdamBeta2 * squaredMean + (1.0f - kAdamBeta2) * gradient * gradient;
    float correctedMean = mean / bias1;
    float correctedSquaredMean = max(squaredMean / bias2, 0.0f);
    return sanitizeFloat(lr * correctedMean / (sqrt(correctedSquaredMean) + kAdamEpsilon));
}

void updateEllipsoidCenterAdam(
    inout GaussianEllipsoid ellipsoid, float3 gradient, float lr,
    uint adamPage, uint adamOffset)
{
    float3 mean = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].centerMean;
    float3 squaredMean = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].centerSquaredMean;
    uint steps = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].centerSteps;
    float3 step = adamStep(gradient, mean, squaredMean, steps, lr);
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].centerMean = mean;
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].centerSquaredMean = squaredMean;
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].centerSteps = steps;
    ellipsoid.center -= clamp(step, float3(-0.02f), float3(0.02f));
    ellipsoid.center = clamp(ellipsoid.center, float3(0.01f), float3(0.99f));
}

void updateEllipsoidLogScaleAdam(
    inout GaussianEllipsoid ellipsoid, float3 gradient, float lr,
    float3 voxelSize, uint adamPage, uint adamOffset)
{
    float3 mean = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].logScaleMean;
    float3 squaredMean = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].logScaleSquaredMean;
    uint steps = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].logScaleSteps;
    float3 step = adamStep(gradient, mean, squaredMean, steps, lr);
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].logScaleMean = mean;
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].logScaleSquaredMean = squaredMean;
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].logScaleSteps = steps;
    ellipsoid.logScale -= clamp(step, float3(-0.02f), float3(0.02f));
    float3 minScale = max(voxelSize * 1e-4f, float3(1e-8f));
    float3 maxScale = max(voxelSize * 4.0f, minScale);
    ellipsoid.logScale = clamp(sanitizeFloat3(ellipsoid.logScale), log(minScale), log(maxScale));
}

void updateEllipsoidRotationAdam(
    inout GaussianEllipsoid ellipsoid, float3 gradient, float lr,
    uint adamPage, uint adamOffset)
{
    // The gradient and Adam moments are local tangent-space axis-angle vectors.
    // Do not optimize the four quaternion components independently.
    float3 mean = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].rotationMean;
    float3 squaredMean = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].rotationSquaredMean;
    uint steps = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].rotationSteps;
    float3 deltaAngle = -adamStep(gradient, mean, squaredMean, steps, lr);
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].rotationMean = mean;
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].rotationSquaredMean = squaredMean;
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].rotationSteps = steps;
    float angle = length(deltaAngle);
    if (angle > 0.02f)
    {
        deltaAngle *= 0.02f / angle;
        angle = 0.02f;
    }
    if (angle > 1e-8f)
    {
        float halfAngle = 0.5f * angle;
        float3 axis = deltaAngle / angle;
        float4 deltaRotation = float4(cos(halfAngle), axis * sin(halfAngle));
        ellipsoid.rotation = normalizeGaussianRotation(
            multiplyGaussianRotations(ellipsoid.rotation, deltaRotation)
        );
    }
    else
    {
        ellipsoid.rotation = normalizeGaussianRotation(ellipsoid.rotation);
    }
}

void updateRadianceSHAdam(
    uint voxelIndex, float lr, float gradScale, uint adamPage, uint adamOffset)
{
    // Stream gradients, coefficients and moments directly from their buffers.
    // Loading complete VoxelData/GradRecord values here spills SH arrays into
    // thread-local storage at high resolutions.
    uint slot = gRadianceAdamIndexPages[NonUniformResourceIndex(adamPage)][adamOffset];
    if (slot == 0xffffffffu)
    {
        if (gRadianceAdamCounter[0] < gRadianceAdamCapacity)
        {
            InterlockedAdd(gRadianceAdamCounter[0], 1u, slot);
            if (slot < gRadianceAdamCapacity)
                gRadianceAdamIndexPages[NonUniformResourceIndex(adamPage)][adamOffset] = slot;
        }
        else slot = gRadianceAdamCapacity;
    }
    if (slot >= gRadianceAdamCapacity)
    {
        uint previousOverflow;
        InterlockedOr(gRadianceAdamCounter[1], 1u, previousOverflow);
        // Never address an unallocated page. Overflow is rare; the host grows
        // this pool at the next iteration boundary. Keep this view's radiance
        // update with the old SGD rule until moments can be allocated.
        SH_MAX_ITERS(SH_COUNT)
        for (uint i = 0; i < SH_COUNT; ++i)
        {
            float3 gradient = GRAD_DATA(voxelIndex).radianceGrad.coefficients[i] * gradScale;
            float3 coefficient = VOXEL_DATA(voxelIndex).radiance.coefficients[i];
            VOXEL_DATA(voxelIndex).radiance.coefficients[i] = sanitizeFloat3(
                coefficient - clamp(lr * sanitizeFloat3(gradient), float3(-0.05f), float3(0.05f)));
        }
        return;
    }
    uint radiancePage = slot / SPARSE_POOL_PAGE_SIZE;
    uint radianceOffset = slot % SPARSE_POOL_PAGE_SIZE;
    uint steps = gRadianceAdamPages[NonUniformResourceIndex(radiancePage)][radianceOffset].steps + 1u;
    gRadianceAdamPages[NonUniformResourceIndex(radiancePage)][radianceOffset].steps = steps;
    float bias1 = max(1.0f - pow(kAdamBeta1, float(steps)), 1e-8f);
    float bias2 = max(1.0f - pow(kAdamBeta2, float(steps)), 1e-8f);
    SH_MAX_ITERS(SH_COUNT)
    for (uint i = 0; i < SH_COUNT; ++i)
    {
        float3 gradient = GRAD_DATA(voxelIndex).radianceGrad.coefficients[i] * gradScale;
        float3 mean = gRadianceAdamPages[NonUniformResourceIndex(radiancePage)][radianceOffset].mean[i];
        float3 squaredMean =
            gRadianceAdamPages[NonUniformResourceIndex(radiancePage)][radianceOffset].squaredMean[i];
        float3 step = adamStepWithBias(gradient, mean, squaredMean, bias1, bias2, lr);
        gRadianceAdamPages[NonUniformResourceIndex(radiancePage)][radianceOffset].mean[i] = mean;
        gRadianceAdamPages[NonUniformResourceIndex(radiancePage)][radianceOffset].squaredMean[i] = squaredMean;
        float3 coefficient = VOXEL_DATA(voxelIndex).radiance.coefficients[i];
        VOXEL_DATA(voxelIndex).radiance.coefficients[i] =
            sanitizeFloat3(coefficient - clamp(step, float3(-0.05f), float3(0.05f)));
    }
}

void updateOpacitySHAdam(
    uint voxelIndex, float lr, float gradScale, uint adamPage, uint adamOffset)
{
    // Optimize the stored SH logits. The forward sigmoid and all geometry
    // proxy terms continue to use the real opacity without an artificial floor.
    uint steps = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].opacitySteps + 1u;
    gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].opacitySteps = steps;
    float bias1 = max(1.0f - pow(kAdamBeta1, float(steps)), 1e-8f);
    float bias2 = max(1.0f - pow(kAdamBeta2, float(steps)), 1e-8f);
    SH_MAX_ITERS(SH_OPACITY_COUNT)
    for (uint i = 0; i < SH_OPACITY_COUNT; ++i)
    {
        float gradient = GRAD_DATA(voxelIndex).opacityGrad.coefficients[i] * gradScale;
        float mean = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].opacityMean[i];
        float squaredMean =
            gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].opacitySquaredMean[i];
        float step = adamStepWithBias(gradient, mean, squaredMean, bias1, bias2, lr);
        gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].opacityMean[i] = mean;
        gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset].opacitySquaredMean[i] = squaredMean;
        float coefficient = VOXEL_DATA(voxelIndex).opacity.coefficients[i];
        VOXEL_DATA(voxelIndex).opacity.coefficients[i] =
            sanitizeFloat(coefficient - clamp(step, -0.01f, 0.01f));
    }
}
#endif


void updateOpacitySH(inout SphericalHarmonicsOpacity opacity,GradRecord grad,float lr,float gradScale)
{
    if (lr <= 0.0f)
        return;

    SH_MAX_ITERS(SH_OPACITY_COUNT)
    for (uint i = 0; i < SH_OPACITY_COUNT; ++i)
    {
        float g =
            grad.opacityGrad.coefficients[i] *
            gradScale;

        g = sanitizeFloat(g);

        float step = clamp(
            lr * g,
            -0.05f,
            0.05f
        );

        // 梯度下降
        opacity.coefficients[i] -= step;

        //opacity.coefficients[i] = clamp(
        //    sanitizeFloat(opacity.coefficients[i]),
        //    -10.0f,
        //    10.0f
        //);
    }
}

void updateRadianceSH(inout SphericalHarmonicsRGB radiance, GradRecord grad, float lr, float gradScale)
{
    if (lr <= 0.0f)
        return;

    for (uint i = 0; i < SH_COUNT; i++)
    {
        float3 g = grad.radianceGrad.coefficients[i] * gradScale;

        g = sanitizeFloat3(g);

        float3 step = lr * g;

        // 防止 SH 更新一步过大，第一版建议限制一下
        step = clamp(step, float3(-0.05f), float3(0.05f));

        radiance.coefficients[i] -= step;

        // 防止参数本身变成 NaN/Inf
        radiance.coefficients[i] = sanitizeFloat3(radiance.coefficients[i]);
    }
}


float calcEllipsoidVoxelVolumeRatio(GaussianEllipsoid ellipsoid)
{
    float3 scale = gaussianEllipsoidScale(ellipsoid);
    float voxelVolume = gGridDataParamBlock.voxelSize.x *
        gGridDataParamBlock.voxelSize.y * gGridDataParamBlock.voxelSize.z;

    if (isBadFloat3(scale) || voxelVolume <= 1e-30f)
        return -1.0f;

    // 局部体素为 [0,1]^3，体积为 1
    float ratio = (4.0f * PI / 3.0f) * scale.x * scale.y * scale.z / voxelVolume;

    if (isBadFloat(ratio))
        return -1.0f;

    return ratio;
}

void pruneEllipsoid(inout VoxelData voxel)
{
    if (voxel.occupied == 0)
        return;

    // 删除截断椭球体积占比较小的voxel
    float ratio = calcEllipsoidVoxelVolumeRatio(voxel.ellipsoid);
    // B异常时暂不删除
    if (ratio < 0.0f)
        return;

    if (ratio < gEllipsoidPruneThreshold)
        voxel.occupied = 0;
}



#if RECON_MODE == RECON_MODE_POINT_CLOUD
[numthreads(256, 1, 1)]
#else
[numthreads(8, 8, 1)]
#endif
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    uint index = gSparseUpdateOffset + dispatchThreadId.x;
    if (index >= gGridDataParamBlock.activeVoxelCount) return;
#else
    uint3 voxel = dispatchThreadId;
    if (voxel.x >= gGridDataParamBlock.voxelCount.x ||
    voxel.y >= gGridDataParamBlock.voxelCount.y ||
        voxel.z >= gGridDataParamBlock.voxelCount.z)
    {
        return;
    }

    int index = CellToIndex(voxel, gGridDataParamBlock.voxelCount);

    if (index < 0)
        return;
#endif

    // Check the compact validity counters before loading the full gradient,
    // voxel, or Adam records. Most high-resolution voxels are untouched by a
    // single view.
    uint appearanceValid = GRAD_DATA((uint)index).appearanceValid;
    uint geometryValid = GRAD_DATA((uint)index).geometryValid;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    bool updateAppearance = appearanceValid > 0u && (gLrRadiance > 0.0f || gLrOpacity > 0.0f);
    bool updateGeometry = geometryValid > 0u &&
        (gLrCenter > 0.0f || gLrShape > 0.0f || gLrRotation > 0.0f);
#else
    bool updateAppearance = appearanceValid > 0u && (gLrRadiance > 0.0f || gLrOpacity > 0.0f);
    bool updateGeometry = geometryValid > 0u && (gLrCenter > 0.0f || gLrShape > 0.0f);
#endif
    if (!updateAppearance && !updateGeometry && !gEnableEllipsoidPruning)
        return;

#if RECON_MODE != RECON_MODE_POINT_CLOUD
    GradRecord grad = GRAD_DATA((uint)index);
#endif

    // 这里 valid 实际上表示当前 voxel 累计了多少个 pixel contribution
    float appearanceGradScale = gGradScale;
    float geometryGradScale = gGradScale;

    if (gUseGradCountNormalize)
    {
        appearanceGradScale *= 1.0f / max(1.0f, float(appearanceValid));
        geometryGradScale *= 1.0f / max(1.0f, float(geometryValid));
    }

#if RECON_MODE == RECON_MODE_POINT_CLOUD
    if (VOXEL_DATA(index).occupied == 0)
#else
    VoxelData voxelData = VOXEL_DATA(index);
    if (voxelData.occupied == 0)
#endif
        return;

#if RECON_MODE == RECON_MODE_POINT_CLOUD
    uint adamPage = uint(index) / SPARSE_POOL_PAGE_SIZE;
    uint adamOffset = uint(index) % SPARSE_POOL_PAGE_SIZE;
#endif

    // -------------------------
    // appearance  SH update
    // -------------------------
    if (updateAppearance)
    {
        if (gLrRadiance > 0.0f)
        {
#if RECON_MODE == RECON_MODE_POINT_CLOUD
            updateRadianceSHAdam(
                uint(index), gLrRadiance, appearanceGradScale, adamPage, adamOffset);
#else
            updateRadianceSH(voxelData.radiance,grad,gLrRadiance,appearanceGradScale);
#endif
        }

        if (gLrOpacity > 0.0f)
        {
#if RECON_MODE == RECON_MODE_POINT_CLOUD
            updateOpacitySHAdam(
                uint(index), gLrOpacity, appearanceGradScale, adamPage, adamOffset);
#else
            updateOpacitySH(voxelData.opacity,grad,gLrOpacity,appearanceGradScale);
#endif
        }
    }

    // -------------------------
    // geometry update
    // -------------------------
    if (updateGeometry)
    {
#if RECON_MODE == RECON_MODE_POINT_CLOUD
        GaussianEllipsoid ellipsoid = VOXEL_DATA(index).ellipsoid;
#else
        GaussianEllipsoid ellipsoid = voxelData.ellipsoid;
#endif

        float geometryLrScale = 1.0f;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
        float backgroundCarveFraction = saturate(
            float(GRAD_DATA(index).backgroundCarveValid) / max(1.0f, float(geometryValid)));
        geometryLrScale = lerp(1.0f, max(gBackgroundCarveAdamMultiplier, 1.0f), backgroundCarveFraction);
#endif

#if RECON_MODE == RECON_MODE_POINT_CLOUD
        if (gLrCenter > 0.0f)
        {
            updateEllipsoidCenterAdam(
                ellipsoid, GRAD_DATA(index).center * geometryGradScale, gLrCenter * geometryLrScale, adamPage, adamOffset);
        }
        if (gLrShape > 0.0f)
        {
            updateEllipsoidLogScaleAdam(ellipsoid, GRAD_DATA(index).logScale * geometryGradScale,
                gLrShape * geometryLrScale, gGridDataParamBlock.voxelSize, adamPage, adamOffset);
        }
        if (gLrRotation > 0.0f)
        {
            updateEllipsoidRotationAdam(
                ellipsoid, GRAD_DATA(index).rotation * geometryGradScale, gLrRotation * geometryLrScale, adamPage, adamOffset);
        }
#else
        if (gLrCenter > 0.0f)
        {
            float3 gradCenter = grad.center * geometryGradScale;
            updateEllipsoidCenter(ellipsoid, gradCenter, gLrCenter);
        }

        if (gLrShape > 0.0f)
        {
            updateEllipsoidShape(
                ellipsoid,
                grad,
                gLrShape,
                geometryGradScale,
                gGridDataParamBlock.voxelSize
            );
        }
#endif

#if RECON_MODE == RECON_MODE_POINT_CLOUD
        VOXEL_DATA(index).ellipsoid = ellipsoid;
#else
        voxelData.ellipsoid = ellipsoid;
#endif
    }

    if (gEnableEllipsoidPruning)
    {
#if RECON_MODE == RECON_MODE_POINT_CLOUD
        if (calcEllipsoidVoxelVolumeRatio(VOXEL_DATA(index).ellipsoid) < gEllipsoidPruneThreshold &&
            calcEllipsoidVoxelVolumeRatio(VOXEL_DATA(index).ellipsoid) >= 0.0f)
            VOXEL_DATA(index).occupied = 0;
#else
        pruneEllipsoid(voxelData);
#endif
    }


#if RECON_MODE != RECON_MODE_POINT_CLOUD
    VOXEL_DATA(index) = voxelData;
#endif
}
