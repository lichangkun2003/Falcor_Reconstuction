#pragma once
#include "Falcor.h"
#include <random>
using namespace Falcor;

class Random
{
private:
    static std::mt19937 Generator;
    static std::uniform_real_distribution<double> Distribution;
public:
    static double Next()
    {
        return Distribution(Generator);
    }
    static float2 Next2()
    {
        return float2(Distribution(Generator), Distribution(Generator));
    }
    static float3 Next3()
    {
        return float3(Distribution(Generator), Distribution(Generator), Distribution(Generator));
    }
};
