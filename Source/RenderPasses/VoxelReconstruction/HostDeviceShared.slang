#pragma once
#include "Utils/HostDeviceShared.slangh"

#include "Defines.h"

#if defined(HOST_CODE)
#include <vector>
using namespace Falcor::math;
#endif


#if !defined(HOST_CODE)
#define MUTATING [mutating]
#else
#define MUTATING
#endif

inline float2x2 make2x2(float a00, float a01, float a10, float a11)
{
#if defined(HOST_CODE)
    float2x2 ret;
    ret[0][0] = a00;
    ret[0][1] = a01;
    ret[1][0] = a10;
    ret[1][1] = a11;
    return ret;
#else
    return float2x2(a00, a01, a10, a11);
#endif
}

inline float3x3 zeros()
{
#if defined(HOST_CODE)
    return float3x3::zeros();
#else
    return float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
#endif
}

inline float2x2 mulT(float2 a, float2 b)
{
    return make2x2(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y);
}

inline float3x3 make3x3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
{
#if defined(HOST_CODE)
    float3x3 m;
    m[0][0] = a00;
    m[0][1] = a01;
    m[0][2] = a02;
    m[1][0] = a10;
    m[1][1] = a11;
    m[1][2] = a12;
    m[2][0] = a20;
    m[2][1] = a21;
    m[2][2] = a22;
    return m;
#else
    return float3x3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
#endif
}

inline float3x3 mulT(float3 u, float3 v)
{
    return make3x3(u.x * v.x, u.x * v.y, u.x * v.z, u.y * v.x, u.y * v.y, u.y * v.z, u.z * v.x, u.z * v.y, u.z * v.z);
}

inline float3x3 add(float3x3 a, float3x3 b)
{
#if defined(HOST_CODE)
    return a + b;
#else
    return make3x3(
        a[0][0] + b[0][0],
        a[0][1] + b[0][1],
        a[0][2] + b[0][2],
        a[1][0] + b[1][0],
        a[1][1] + b[1][1],
                   a[1][2] + b[1][2],
        a[2][0] + b[2][0],
        a[2][1] + b[2][1],
        a[2][2] + b[2][2]
    );
#endif
}

inline float3x3 sub(float3x3 a, float3x3 b)
{
    return make3x3(
        a[0][0] - b[0][0],
        a[0][1] - b[0][1],
        a[0][2] - b[0][2],
        a[1][0] - b[1][0],
        a[1][1] - b[1][1],
                   a[1][2] - b[1][2],
        a[2][0] - b[2][0],
        a[2][1] - b[2][1],
        a[2][2] - b[2][2]
    );
}

inline float3x3 mul(float k, float3x3 a)
{
    return make3x3(k * a[0][0], k * a[0][1], k * a[0][2], k * a[1][0], k * a[1][1], k * a[1][2], k * a[2][0], k * a[2][1], k * a[2][2]);
}

#if !defined(HOST_CODE)
inline float safeSqrt(float x)
{
    return sqrt(max(0, x));
}

inline float safePow(float x, float power)
{
    return pow(max(0, x), power);
}

#endif

inline float3 safeNormalize(float3 n)
{
    float len = length(n);
    if (len > 0)
        return n / len;
    return float3(0);
}
