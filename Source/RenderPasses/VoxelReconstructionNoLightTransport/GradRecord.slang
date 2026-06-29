#pragma once
#include "HostDeviceShared.slang"
#include "Defines.h"

#include "Math/SphericalHarmonics.slang"

BEGIN_NAMESPACE_FALCOR

struct GradRecord
{
    // radiance
    SphericalHarmonicsRGBGrad radianceGrad;
    uint radianceValid; 

    // geometry gradient
    float3 center;
    float B00;
    float B11;
    float B22;
    float B01;
    float B02;
    float B12;
    uint geometryValid;


    MUTATING
    void init() {
        radianceValid = 0;
        radianceGrad.init();

        center = float3(0.0f);
        B00 = 0.0f;
        B11 = 0.0f;
        B22 = 0.0f;
        B01 = 0.0f;
        B02 = 0.0f;
        B12 = 0.0f;
        geometryValid = 0;

        
    }
};

#if !defined(HOST_CODE)
bool isBadFloat(float x)
{
    return isnan(x) || isinf(x);
}

bool isBadFloat3(float3 x)
{
    return any(isnan(x)) || any(isinf(x));
}

float sanitizeFloat(float x)
{
    if (isBadFloat(x))
        return 0.0f;
    return x;
}

float3 sanitizeFloat3(float3 x)
{
    if (isBadFloat3(x))
        return float3(0.0f);
    return x;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

#endif


END_NAMESPACE_FALCOR
