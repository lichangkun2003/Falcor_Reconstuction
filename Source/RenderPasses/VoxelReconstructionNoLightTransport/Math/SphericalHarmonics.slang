#pragma once
#include "../HostDeviceShared.slang"



BEGIN_NAMESPACE_FALCOR

#define SQRTPI 1.7724538509055160273f
#define SH_COUNT 16
#define EVEN_SH_COUNT 6

inline float calcSH(uint idx, float3 p)
{
    switch (idx)
    {
    // l = 0
    case 0: return 1.f / (2.f * SQRTPI);                // m = 0
    // l = 1
    case 1: return p.y * sqrt(3.f) / (2.f * SQRTPI);            // m =-1
    case 2: return p.z * sqrt(3.f) / (2.f * SQRTPI);            // m = 0
    case 3: return p.x * sqrt(3.f) / (2.f * SQRTPI);            // m = 1
    // l = 2
    case 4: return p.x * p.y * sqrt(15.f) / (2.f * SQRTPI);                 // m =-2
    case 5: return p.y * p.z * sqrt(15.f) / (2.f * SQRTPI);                 // m =-1
    case 6: return (3.f * p.z * p.z - 1.f) * sqrt(5.f) / (4.f * SQRTPI);    // m = 0
    case 7: return p.x * p.z * sqrt(15.f) / (2.f * SQRTPI);                 // m = 1
    case 8: return (p.x * p.x - p.y * p.y) * sqrt(15.f) / (4.f * SQRTPI);   // m = 2
    // l = 3
    case 9:  return p.y * (3.f * p.x * p.x - p.y * p.y) * sqrt(70.f) / (8.f * SQRTPI);  // m =-3
    case 10: return p.x * p.y * p.z * sqrt(105.f) / (2.f * SQRTPI);                     // m =-2
    case 11: return p.y * (5.f * p.z * p.z - 1.f) * sqrt(42.f) / (8.f * SQRTPI);        // m =-1
    case 12: return p.z * (5.f * p.z * p.z - 3.f) * sqrt(7.f) / (4.f * SQRTPI);         // m = 0
    case 13: return p.x * (5.f * p.z * p.z - 1.f) * sqrt(42.f) / (8.f * SQRTPI);        // m = 1
    case 14: return p.z * (p.x * p.x - p.y * p.y) * sqrt(105.f) / (4.f * SQRTPI);       // m = 2
    case 15: return p.x * (p.x * p.x - 3.f * p.y * p.y) * sqrt(70.f) / (8.f * SQRTPI);  // m = 3
    }
    // clang-format on
    return 0.f;
}

struct SphericalHarmonics
{
    float coefficients[SH_COUNT];

    MUTATING
    void init()
    {
        for (int i = 0; i < SH_COUNT;i++)
        {
            coefficients[i] = 0;
        }
    }

    MUTATING
    // 所有采样方向的权重总和应当为4PI
    void accumulate(float valueByWeight, float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] += valueByWeight * calcSH(i, direction);
        }
    }

    MUTATING
    // 不带系数统计，最后再统一乘以系数
    void accumulate(float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] += calcSH(i, direction);
        }
    }

    MUTATING
    void mul(float value)
    {
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] *= value;
        }
    }

    float calc(float3 p)
    {
        float sum = 0;
        for (int i = 0; i < SH_COUNT; i++)
        {
            sum += coefficients[i] * calcSH(i, p);
        }
        return sum;
    }
};

inline float calcEvenSH(uint idx, float3 p)
{
    switch (idx)
    {
    // l = 0
    case 0: return 1.f / (2.f * SQRTPI); // m = 0
    // l = 2
    case 1: return p.x * p.y * sqrt(15.f) / (2.f * SQRTPI); // m =-2
    case 2: return p.y * p.z * sqrt(15.f) / (2.f * SQRTPI); // m =-1
    case 3: return (3.f * p.z * p.z - 1.f) * sqrt(5.f) / (4.f * SQRTPI); // m = 0
    case 4: return p.x * p.z * sqrt(15.f) / (2.f * SQRTPI); // m = 1
    case 5: return (p.x * p.x - p.y * p.y) * sqrt(15.f) / (4.f * SQRTPI); // m = 2
    }
    // clang-format on
    return 0.f;
}

struct EvenSphericalHarmonics
{
    float coefficients[EVEN_SH_COUNT];

    MUTATING
    void init()
    {
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] = 0;
        }
    }

    MUTATING
    // 所有采样方向的权重总和应当为4PI
    void accumulate(float valueByWeight, float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] += valueByWeight * calcEvenSH(i, direction);
        }
    }

    MUTATING
    // 不带系数统计，最后再统一乘以系数
    void accumulate(float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] += calcEvenSH(i, direction);
        }
    }

    MUTATING
    void mul(float value)
    {
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] *= value;
        }
    }

    float calc(float3 p)
    {
        float sum = 0;
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            sum += coefficients[i] * calcEvenSH(i, p);
        }
        return sum;
    }
};

typedef EvenSphericalHarmonics SphericalFunc;

struct SphericalHarmonicsRGB
{
    float3 coefficients[SH_COUNT];

    MUTATING
    void init()
    {
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] = float3(0.f);
        }
    }

    MUTATING
    void accumulate(float3 valueByWeight, float3 direction)
    {
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] += valueByWeight * calcSH(i, direction);
        }
    }

    MUTATING
    void mul(float value)
    {
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] *= value;
        }
    }

    float3 calc(float3 direction)
    {
        float3 sum = float3(0.f);
        for (int i = 0; i < SH_COUNT; i++)
        {
            sum += coefficients[i] * calcSH(i, direction);
        }
        return sum;
    }
};

struct SphericalHarmonicsRGBGrad
{
    float3 coefficients[SH_COUNT];

    MUTATING
    void init()
    {
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] = float3(0.0f);
        }
    }

    MUTATING
    void addGrad(float3 dL_dI, float3 viewDir)
    {
        // viewDir 默认已经归一化
        for (int i = 0; i < SH_COUNT; i++)
        {
            float Y = calcSH(i, viewDir);
            coefficients[i] += dL_dI * Y;
        }
    }
};

END_NAMESPACE_FALCOR
