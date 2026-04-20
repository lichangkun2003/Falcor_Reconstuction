#pragma once
#include "../HostDeviceShared.slang"
#include "ABSDF.slang"

BEGIN_NAMESPACE_FALCOR

//typedef EvenSphericalHarmonics SphericalFunc;

struct VoxelData
{
    uint occupied = 0;
    //ABSDF ABSDF;
    //Ellipsoid ellipsoid;
    //SphericalFunc primitiveProjAreaFunc;
    //SphericalFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    //SphericalFunc totalProjAreaFunc;    // 所有多边形的总投影面积

    MUTATING
    void init()
    {
        //ABSDF.init();
        //primitiveProjAreaFunc.init();
        //polygonsProjAreaFunc.init();
        //totalProjAreaFunc.init();
    }

    //bool IsSolid() { return ABSDF.area > 0; }
};

END_NAMESPACE_FALCOR
