#include "../Voxel/VoxelGrid.slang"
#include "../GradRecord.slang"
#include "../GeometryAdamState.slang"

// Input
#if RECON_MODE != RECON_MODE_POINT_CLOUD
StructuredBuffer<GradRecord> gGradBuffer;
#else
RWStructuredBuffer<GeometryAdamState> gGeometryAdamPages[SPARSE_POOL_MAX_PAGES];
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

    float gEllipsoidPruneThreshold;
    bool gEnableEllipsoidPruning;
#if RECON_MODE == RECON_MODE_POINT_CLOUD
    uint gSparseUpdateOffset;
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

float3 adamStep(float3 gradient, inout float3 mean, inout float3 squaredMean, inout uint steps, float lr)
{
    gradient = clamp(sanitizeFloat3(gradient), float3(-10.0f), float3(10.0f));
    mean = kAdamBeta1 * mean + (1.0f - kAdamBeta1) * gradient;
    squaredMean = kAdamBeta2 * squaredMean + (1.0f - kAdamBeta2) * gradient * gradient;
    steps += 1;
    float bias1 = max(1.0f - pow(kAdamBeta1, float(steps)), 1e-8f);
    float bias2 = max(1.0f - pow(kAdamBeta2, float(steps)), 1e-8f);
    float3 correctedMean = mean / bias1;
    float3 correctedSquaredMean = max(squaredMean / bias2, float3(0.0f));
    return sanitizeFloat3(lr * correctedMean / (sqrt(correctedSquaredMean) + kAdamEpsilon));
}

void updateEllipsoidCenterAdam(inout GaussianEllipsoid ellipsoid, float3 gradient, float lr, inout GeometryAdamState state)
{
    float3 step = adamStep(gradient, state.centerMean, state.centerSquaredMean, state.centerSteps, lr);
    ellipsoid.center -= clamp(step, float3(-0.02f), float3(0.02f));
    ellipsoid.center = clamp(ellipsoid.center, float3(0.01f), float3(0.99f));
}

void updateEllipsoidLogScaleAdam(
    inout GaussianEllipsoid ellipsoid, float3 gradient, float lr,
    float3 voxelSize, inout GeometryAdamState state)
{
    float3 step = adamStep(gradient, state.logScaleMean, state.logScaleSquaredMean, state.logScaleSteps, lr);
    ellipsoid.logScale -= clamp(step, float3(-0.02f), float3(0.02f));
    float3 minScale = max(voxelSize * 1e-4f, float3(1e-8f));
    float3 maxScale = max(voxelSize * 4.0f, minScale);
    ellipsoid.logScale = clamp(sanitizeFloat3(ellipsoid.logScale), log(minScale), log(maxScale));
}

void updateEllipsoidRotationAdam(
    inout GaussianEllipsoid ellipsoid, float3 gradient, float lr,
    inout GeometryAdamState state)
{
    // The gradient and Adam moments are local tangent-space axis-angle vectors.
    // Do not optimize the four quaternion components independently.
    float3 deltaAngle = -adamStep(
        gradient,
        state.rotationMean,
        state.rotationSquaredMean,
        state.rotationSteps,
        lr
    );
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

    GradRecord grad = GRAD_DATA((uint)index);

    // 这里 valid 实际上表示当前 voxel 累计了多少个 pixel contribution
    //if (grad.radianceValid == 0 && grad.geometryValid == 0)
    //    return;

    float appearanceGradScale = gGradScale;
    float geometryGradScale = gGradScale;

    if (gUseGradCountNormalize)
    {
        appearanceGradScale *= 1.0f / max(1.0f, float(grad.appearanceValid));
        geometryGradScale *= 1.0f / max(1.0f, float(grad.geometryValid));
    }

    VoxelData voxelData = VOXEL_DATA(index);
    if (voxelData.occupied == 0)
        return;

    // -------------------------
    // appearance  SH update
    // -------------------------
    if (grad.appearanceValid > 0)
    {
        if (gLrRadiance > 0.0f)
        {
            updateRadianceSH(voxelData.radiance,grad,gLrRadiance,appearanceGradScale);
        }

        if (gLrOpacity > 0.0f)
        {
            updateOpacitySH(voxelData.opacity,grad,gLrOpacity,appearanceGradScale);
        }
    }

    // -------------------------
    // geometry update
    // -------------------------
    if (grad.geometryValid > 0)
    {
        GaussianEllipsoid ellipsoid = voxelData.ellipsoid;

#if RECON_MODE == RECON_MODE_POINT_CLOUD
        uint adamPage = uint(index) / SPARSE_POOL_PAGE_SIZE;
        uint adamOffset = uint(index) % SPARSE_POOL_PAGE_SIZE;
        GeometryAdamState state = gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset];

        if (gLrCenter > 0.0f)
            updateEllipsoidCenterAdam(ellipsoid, grad.center * geometryGradScale, gLrCenter, state);
        if (gLrShape > 0.0f)
            updateEllipsoidLogScaleAdam(ellipsoid, grad.logScale * geometryGradScale,
                gLrShape, gGridDataParamBlock.voxelSize, state);
        if (gLrRotation > 0.0f)
            updateEllipsoidRotationAdam(
                ellipsoid, grad.rotation * geometryGradScale, gLrRotation, state);

        if (gLrCenter > 0.0f || gLrShape > 0.0f || gLrRotation > 0.0f)
            gGeometryAdamPages[NonUniformResourceIndex(adamPage)][adamOffset] = state;
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

        voxelData.ellipsoid = ellipsoid;
    }

    if (gEnableEllipsoidPruning)
    {
        pruneEllipsoid(voxelData);
    }


    VOXEL_DATA(index) = voxelData;
}
