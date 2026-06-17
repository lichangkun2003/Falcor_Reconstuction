#pragma once
#include "../HostDeviceShared.slang"
#include "ABSDF.slang"
#include "../Math/Ellipsoid.slang"
#include "../Math/SphericalHarmonics.slang"
#include "../Math/Triangle.slang"
#include "Shading.slang"

BEGIN_NAMESPACE_FALCOR



struct VoxelData
{
    uint occupied = 0;
    Ellipsoid ellipsoid;
    // SphericalHarmonics radianceR;
    // SphericalHarmonics radianceG;
    // SphericalHarmonics radianceB;
    SphericalHarmonicsRGB radiance;

    MUTATING
    void init()
    {
        // ABSDF.init();
        // primitiveProjAreaFunc.init();
        // polygonsProjAreaFunc.init();
        // totalProjAreaFunc.init();
        radiance.init();
    }


};

END_NAMESPACE_FALCOR
