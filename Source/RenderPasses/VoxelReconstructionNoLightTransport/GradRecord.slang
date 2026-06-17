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


#endif


END_NAMESPACE_FALCOR
