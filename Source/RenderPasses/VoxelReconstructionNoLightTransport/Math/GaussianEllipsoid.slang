#pragma once
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

// 3DGS-style ellipsoid parameterization.
// center is voxel-local. exp(logScale) stores the three world-space semi-axes.
// rotation is a unit quaternion in (w, x, y, z) order and maps principal axes
// to world axes.
struct GaussianEllipsoid
{
    float3 center;
    float3 logScale;
    float4 rotation;
};

#if !defined(HOST_CODE)

float4 normalizeGaussianRotation(float4 q)
{
    float lengthSquared = dot(q, q);
    if (lengthSquared <= 1e-20f || isnan(lengthSquared) || isinf(lengthSquared))
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    return q * rsqrt(lengthSquared);
}

float4 multiplyGaussianRotations(float4 lhs, float4 rhs)
{
    return float4(
        lhs.x * rhs.x - dot(lhs.yzw, rhs.yzw),
        lhs.x * rhs.y + rhs.x * lhs.y + lhs.z * rhs.w - lhs.w * rhs.z,
        lhs.x * rhs.z + rhs.x * lhs.z + lhs.w * rhs.y - lhs.y * rhs.w,
        lhs.x * rhs.w + rhs.x * lhs.w + lhs.y * rhs.z - lhs.z * rhs.y
    );
}

float3x3 gaussianRotationMatrix(float4 rotation)
{
    float4 q = normalizeGaussianRotation(rotation);
    float w = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    return float3x3(
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w),        2.0f * (x * z + y * w),
        2.0f * (x * y + z * w),        1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w),
        2.0f * (x * z - y * w),        2.0f * (y * z + x * w),        1.0f - 2.0f * (x * x + y * y)
    );
}

float3 gaussianEllipsoidScale(GaussianEllipsoid ellipsoid)
{
    return exp(clamp(ellipsoid.logScale, float3(-30.0f), float3(30.0f)));
}

float3 gaussianEllipsoidPrincipalOffset(
    GaussianEllipsoid ellipsoid,
    float3 xLocal,
    float3 voxelSize)
{
    float3 qWorld = (xLocal - ellipsoid.center) * voxelSize;
    return mul(transpose(gaussianRotationMatrix(ellipsoid.rotation)), qWorld);
}

float gaussianEllipsoidImplicit(
    GaussianEllipsoid ellipsoid,
    float3 xLocal,
    float3 voxelSize)
{
    float3 principal = gaussianEllipsoidPrincipalOffset(ellipsoid, xLocal, voxelSize);
    float3 invScaleSquared = exp(clamp(-2.0f * ellipsoid.logScale, float3(-60.0f), float3(60.0f)));
    return dot(principal * invScaleSquared, principal) - 1.0f;
}

float3 gaussianEllipsoidImplicitGradientWorld(
    GaussianEllipsoid ellipsoid,
    float3 xLocal,
    float3 voxelSize)
{
    float3x3 rotation = gaussianRotationMatrix(ellipsoid.rotation);
    float3 principal = gaussianEllipsoidPrincipalOffset(ellipsoid, xLocal, voxelSize);
    float3 invScaleSquared = exp(clamp(-2.0f * ellipsoid.logScale, float3(-60.0f), float3(60.0f)));
    return 2.0f * mul(rotation, principal * invScaleSquared);
}

// First-order signed distance at the implicit minimum on a ray segment.
// Normalize with the gradient at the radially projected surface point, where
// it cannot vanish. This is accurate near the hit boundary, not an exact SDF.
// Backward treats distanceScale as constant: d(distance)/d(theta) ~= scale * dg/d(theta).
bool calcGaussianEllipsoidWorldDistance(
    GaussianEllipsoid ellipsoid,
    float3 xLocal,
    float3 voxelSize,
    float gmin,
    out float distanceWorld,
    out float distanceScale)
{
    distanceWorld = 0.0f;
    distanceScale = 0.0f;

    float3 scale = gaussianEllipsoidScale(ellipsoid);
    float3 unitOffset = gaussianEllipsoidPrincipalOffset(ellipsoid, xLocal, voxelSize) / scale;
    float radiusSquared = dot(unitOffset, unitOffset);
    if (isnan(gmin) || isinf(gmin) || isnan(radiusSquared) || isinf(radiusSquared) || radiusSquared <= 1e-12f)
        return false; // Center hits have no unique radial direction; skip their geometry gradient.

    float3 surfaceDirection = unitOffset * rsqrt(radiusSquared);
    float surfaceGradNorm = 2.0f * length(surfaceDirection / scale);
    if (isnan(surfaceGradNorm) || isinf(surfaceGradNorm) || surfaceGradNorm <= 0.0f)
        return false;

    distanceScale = 1.0f / surfaceGradNorm;
    distanceWorld = gmin * distanceScale;
    return !isnan(distanceWorld) && !isinf(distanceWorld);
}

void calcGaussianEllipsoidMinimumOnSegment(
    GaussianEllipsoid ellipsoid,
    float3 localFrom,
    float3 localTo,
    float3 voxelSize,
    out float gmin,
    out float3 xStarLocal)
{
    float3 scale = gaussianEllipsoidScale(ellipsoid);
    float3x3 worldToPrincipal = transpose(gaussianRotationMatrix(ellipsoid.rotation));
    float3 fromWorld = (localFrom - ellipsoid.center) * voxelSize;
    float3 segmentWorld = (localTo - localFrom) * voxelSize;
    float3 fromUnit = mul(worldToPrincipal, fromWorld) / scale;
    float3 segmentUnit = mul(worldToPrincipal, segmentWorld) / scale;

    float denominator = dot(segmentUnit, segmentUnit);
    float tStar = denominator > 1e-20f
        ? clamp(-dot(fromUnit, segmentUnit) / denominator, 0.0f, 1.0f)
        : 0.0f;

    float3 closestUnit = fromUnit + tStar * segmentUnit;
    gmin = dot(closestUnit, closestUnit) - 1.0f;
    xStarLocal = localFrom + tStar * (localTo - localFrom);
}

float2 clipGaussianEllipsoid(
    GaussianEllipsoid ellipsoid,
    float3 localFrom,
    float3 localTo,
    float3 voxelSize)
{
    float3 scale = gaussianEllipsoidScale(ellipsoid);
    float3x3 worldToPrincipal = transpose(gaussianRotationMatrix(ellipsoid.rotation));
    float3 fromUnit = mul(worldToPrincipal, (localFrom - ellipsoid.center) * voxelSize) / scale;
    float3 segmentUnit = mul(worldToPrincipal, (localTo - localFrom) * voxelSize) / scale;

    float a = dot(segmentUnit, segmentUnit);
    float b = 2.0f * dot(fromUnit, segmentUnit);
    float c = dot(fromUnit, fromUnit) - 1.0f;

    if (a <= 1e-20f)
        return c <= 0.0f ? float2(0.0f, 1.0f) : float2(-1.0f, -1.0f);

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
        return float2(-1.0f, -1.0f);

    float root = sqrt(max(discriminant, 0.0f));
    float inverseDenominator = 0.5f / a;
    float entry = max(0.0f, (-b - root) * inverseDenominator);
    float exit = min(1.0f, (-b + root) * inverseDenominator);
    return entry <= exit ? float2(entry, exit) : float2(-1.0f, -1.0f);
}

#endif

END_NAMESPACE_FALCOR
