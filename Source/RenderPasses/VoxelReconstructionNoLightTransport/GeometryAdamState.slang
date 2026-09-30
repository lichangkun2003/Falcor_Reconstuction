#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

// Mode 1 stores moments only for occupied voxels. Rotation moments live in the
// three-dimensional local tangent space, rather than in quaternion coordinates.
struct GeometryAdamState
{
    float3 centerMean;
    float3 centerSquaredMean;
    float3 logScaleMean;
    float3 logScaleSquaredMean;
    float3 rotationMean;
    float3 rotationSquaredMean;
    uint centerSteps;
    uint logScaleSteps;
    uint rotationSteps;
};

#if defined(HOST_CODE)
static_assert(sizeof(GeometryAdamState) == 84, "GeometryAdamState host/device layout changed.");
#endif

END_NAMESPACE_FALCOR
