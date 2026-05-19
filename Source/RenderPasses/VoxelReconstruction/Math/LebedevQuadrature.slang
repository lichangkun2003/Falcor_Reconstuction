#pragma once
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

#define LEBEDEV_DIRECTION_COUNT 26

static const float InvSqrt2 = 0.7071067811865475244f;
static const float InvSqrt3 = 0.5773502691896257645f;

//权重还需要乘以4pi
static const float Lebedev_AxisWeight = 1 / 21.0f;  
static const float Lebedev_EdgeWeight = 4 / 105.0f;
static const float Lebedev_CornerWeight = 9 / 280.0f;

struct LebedevSample
{
    float3 direction;
    float weight;
};

static const LebedevSample LebedevSamples[LEBEDEV_DIRECTION_COUNT] =
{
    // 6 axis points
    { float3(1, 0, 0), Lebedev_AxisWeight },
    { float3(-1, 0, 0), Lebedev_AxisWeight },
    { float3(0, 1, 0), Lebedev_AxisWeight },
    { float3(0, -1, 0), Lebedev_AxisWeight },
    { float3(0, 0, 1), Lebedev_AxisWeight },
    { float3(0, 0, -1), Lebedev_AxisWeight },

    // 12 edge-midpoints: (±a, ±a, 0) and permutations, a = 1/sqrt(2)
    { float3(InvSqrt2, InvSqrt2, 0), Lebedev_EdgeWeight },
    { float3(-InvSqrt2, -InvSqrt2, 0), Lebedev_EdgeWeight },
    { float3(InvSqrt2, -InvSqrt2, 0), Lebedev_EdgeWeight },
    { float3(-InvSqrt2, InvSqrt2, 0), Lebedev_EdgeWeight },

    { float3(InvSqrt2, 0, InvSqrt2), Lebedev_EdgeWeight },
    { float3(-InvSqrt2, 0, -InvSqrt2), Lebedev_EdgeWeight },
    { float3(InvSqrt2, 0, -InvSqrt2), Lebedev_EdgeWeight },
    { float3(-InvSqrt2, 0, InvSqrt2), Lebedev_EdgeWeight },

    { float3(0, InvSqrt2, InvSqrt2), Lebedev_EdgeWeight },
    { float3(0, -InvSqrt2, -InvSqrt2), Lebedev_EdgeWeight },
    { float3(0, InvSqrt2, -InvSqrt2), Lebedev_EdgeWeight },
    { float3(0, -InvSqrt2, InvSqrt2), Lebedev_EdgeWeight },

    // 8 corners: (±b, ±b, ±b), b = 1/sqrt(3)
    { float3(InvSqrt3, InvSqrt3, InvSqrt3), Lebedev_CornerWeight },
    { float3(-InvSqrt3, -InvSqrt3, -InvSqrt3), Lebedev_CornerWeight },
    { float3(InvSqrt3, InvSqrt3, -InvSqrt3), Lebedev_CornerWeight },
    { float3(-InvSqrt3, -InvSqrt3, InvSqrt3), Lebedev_CornerWeight },
    { float3(InvSqrt3, -InvSqrt3, InvSqrt3), Lebedev_CornerWeight },
    { float3(-InvSqrt3, InvSqrt3, -InvSqrt3), Lebedev_CornerWeight },
    { float3(InvSqrt3, -InvSqrt3, -InvSqrt3), Lebedev_CornerWeight },
    { float3(-InvSqrt3, InvSqrt3, InvSqrt3), Lebedev_CornerWeight },
};

END_NAMESPACE_FALCOR
