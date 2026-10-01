#pragma once
#include "HostDeviceShared.slang"
#include "Defines.h"

#include "Math/SphericalHarmonics.slang"

BEGIN_NAMESPACE_FALCOR

struct GradRecord
{
    // radiance
    uint appearanceValid;
    SphericalHarmonicsRGBGrad radianceGrad;
    SphericalHarmonicsOpacity opacityGrad;

    // geometry gradient
    float3 center;
    float3 logScale;
    // Tangent-space rotation gradient. It is applied as a local axis-angle
    // update to the persistent unit quaternion.
    float3 rotation;
    uint geometryValid;
    // Number of geometry samples produced by reference pixels with alpha == 0.
    // UpdatePass uses the fraction of these samples to accelerate only
    // background space carving after Adam has normalized the gradient.
    uint backgroundCarveValid;


    MUTATING
    void init() {
        appearanceValid = 0;
        radianceGrad.init();
        opacityGrad.init();

        center = float3(0.0f);
        logScale = float3(0.0f);
        rotation = float3(0.0f);
        geometryValid = 0;
        backgroundCarveValid = 0;

        
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
