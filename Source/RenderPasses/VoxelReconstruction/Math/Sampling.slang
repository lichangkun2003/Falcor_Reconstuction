#pragma once
#include "Projection.slang"
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR
struct SamplingRay
{
    float3 origin;
    float3 direction;
};

inline float2 sampleDisk(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2 * PI * u.y;
    return float2(r * cos(phi), r * sin(phi));
}

inline float3 sampleSphere(float2 u)
{
    float phi = 2 * PI * u.y;
    float cosTheta = 1.0f - 2.0f * u.x;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// 沿给定方向取半个球面，从半球面内的随机点(对投影面积平均)沿反方向发出射线
inline SamplingRay rayToVoxel(float2 uDisk, float3 direction, Basis2 basis, float3 cellCenter)
{
    float2 p = sampleDisk(uDisk);
    float r = 0.8660254f;
    float u = p.x * r;
    float w = p.y * r;

    float h = sqrt(r * r - u * u - w * w);
    float3 point = h * direction + u * basis.u + w * basis.w;
    SamplingRay ray = { point + cellCenter , -direction };
    return ray;
}

END_NAMESPACE_FALCOR
