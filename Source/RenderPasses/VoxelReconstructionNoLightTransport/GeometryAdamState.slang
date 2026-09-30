#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

// Mode 1 stores moments only for occupied voxels. Rotation keeps its existing
// tangent-space SGD update, so no quaternion momentum is stored here.
struct GeometryAdamState
{
    float3 centerMean;
    float3 centerSquaredMean;
    float3 logScaleMean;
    float3 logScaleSquaredMean;
    uint centerSteps;
    uint logScaleSteps;
};

#if defined(HOST_CODE)
static_assert(sizeof(GeometryAdamState) == 56, "GeometryAdamState host/device layout changed.");
#endif

END_NAMESPACE_FALCOR
