#pragma once
#include "../HostDeviceShared.slang"
#include "../Math/GaussianEllipsoid.slang"
#include "../Math/SphericalHarmonics.slang"

BEGIN_NAMESPACE_FALCOR



struct VoxelData
{
    uint occupied = 0;
    GaussianEllipsoid ellipsoid;
    SphericalHarmonicsRGB radiance;
    SphericalHarmonicsOpacity opacity;

    MUTATING
    void init()
    {
        radiance.init();
        opacity.init();
    }


};

END_NAMESPACE_FALCOR
