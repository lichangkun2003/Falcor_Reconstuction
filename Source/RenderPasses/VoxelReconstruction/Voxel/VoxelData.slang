#pragma once
#include "../HostDeviceShared.slang"
#include "ABSDF.slang"
#include "../Math/Ellipsoid.slang"
#include "../Math/SphericalHarmonics.slang"
#include "../Math/Triangle.slang"
#include "../Math/DensityPrimitive.slang"
#include "Shading.slang"

BEGIN_NAMESPACE_FALCOR



struct VoxelData
{
    uint occupied = 0;
    // ABSDF ABSDF;
    // PrimitiveBSDF bsdf;
    PrimitiveBSDF_New bsdf;
    DensityPrimitive densityPrimitive;
    //Ellipsoid ellipsoid;
    //SphericalFunc primitiveProjAreaFunc;
    //SphericalFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    //SphericalFunc totalProjAreaFunc;    // 所有多边形的总投影面积

    MUTATING
    void init()
    {
        // ABSDF.init();
        // primitiveProjAreaFunc.init();
        // polygonsProjAreaFunc.init();
        // totalProjAreaFunc.init();
        densityPrimitive.init();
    }

    //bool IsSolid() { return ABSDF.area > 0; }
};

END_NAMESPACE_FALCOR
