#pragma once

#include "../HostDeviceShared.slang"
#include "SphericalHarmonics.slang"

BEGIN_NAMESPACE_FALCOR

struct DensityPrimitive
{
    float3 center;
    float sigmaMax;

    EvenSphericalHarmonics falloffSH;

    float cutoff;

    MUTATING
    void init()
    {
        center = float3(0.5f);
        sigmaMax = 20.0f;

        cutoff = 0.001f;

        falloffSH.init();

        float radius = 0.5f;
        float targetFalloff = -log(cutoff / sigmaMax) / (radius * radius);
        float y0 = 1.0f / (2.0f * SQRTPI);
        falloffSH.coefficients[0] = log(targetFalloff) / y0;
    }

#if !defined(HOST_CODE)
    float evalFalloff(float3 dir)
    {
        return exp(falloffSH.calc(dir));
    }

    float evalDensity(float3 p)
    {
        float3 r = p - center;

        float r2 = dot(r, r);

        if (r2 < 1e-8f)
            return sigmaMax;

        float3 dir = normalize(r);
         
        float falloff = evalFalloff(dir);

        return sigmaMax * exp(-falloff * r2);
    }

    bool isRelevant(float density)
    {
        return density >= cutoff;
    }

#endif
};


END_NAMESPACE_FALCOR
