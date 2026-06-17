#pragma once
#include "HostDeviceShared.slang"
#include "Defines.h"

#include "Math/SphericalHarmonics.slang"

BEGIN_NAMESPACE_FALCOR

struct GradRecord
{
    // material
    float3 diffuse[LOBE_COUNT];
    float3 specular[LOBE_COUNT];
    float rough[LOBE_COUNT];
    float weight[LOBE_COUNT];
    float3 normal[LOBE_COUNT];
    SphericalFunc coverageFunc;
    SphericalFunc visibilityFunc;
    uint valid; 

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
        valid = 0;

        center = float3(0.0f);
        B00 = 0.0f;
        B11 = 0.0f;
        B22 = 0.0f;
        B01 = 0.0f;
        B02 = 0.0f;
        B12 = 0.0f;
        geometryValid = 0;

        coverageFunc.init();
        visibilityFunc.init();

        for (uint i = 0; i < LOBE_COUNT; i++) {
            diffuse[i] = float3(0.0f);
            specular[i] = float3(0.0f);
            normal[i] = float3(0.0f);
            rough[i] = float(0.0f);
            weight[i] = float(0.0f);
        }
    }
};

#if !defined(HOST_CODE)


#endif


END_NAMESPACE_FALCOR
