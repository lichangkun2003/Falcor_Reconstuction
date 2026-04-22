#pragma once
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

enum class ABSDFDrawMode : uint32_t
{
    Default,
    Normal,
    Ellipsoid,
    PrimitiveProjArea,
    PolygonsProjArea,
    TotalProjArea,
    Coverage,
    InternalVisibility,
    TotalArea,
    Occupancy,
};

FALCOR_ENUM_INFO(
    ABSDFDrawMode,
    {
    { ABSDFDrawMode::Default, "Default" },
    { ABSDFDrawMode::Normal, "Normal" },
    { ABSDFDrawMode::Ellipsoid, "Ellipsoid" },
    { ABSDFDrawMode::PrimitiveProjArea, "PrimitiveProjArea" },
    { ABSDFDrawMode::PolygonsProjArea, "PolygonsProjArea" },
    { ABSDFDrawMode::TotalProjArea, "TotalProjArea" },
    { ABSDFDrawMode::Coverage, "Coverage" },
    { ABSDFDrawMode::InternalVisibility, "InternalVisibility" },
    { ABSDFDrawMode::TotalArea, "TotalArea" },
    { ABSDFDrawMode::Occupancy, "Occupancy" },
}
);
FALCOR_ENUM_REGISTER(ABSDFDrawMode);

END_NAMESPACE_FALCOR
