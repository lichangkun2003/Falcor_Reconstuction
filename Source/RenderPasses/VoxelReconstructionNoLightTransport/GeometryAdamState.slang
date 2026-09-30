#pragma once
#include "HostDeviceShared.slang"
#include "Defines.h"

BEGIN_NAMESPACE_FALCOR

// Mode 1 stores optimizer moments only for occupied voxels. Rotation moments
// live in the three-dimensional local tangent space, rather than in quaternion
// coordinates. Radiance keeps independent moments for every RGB SH coefficient.
struct GeometryAdamState
{
    float3 centerMean;
    float3 centerSquaredMean;
    float3 logScaleMean;
    float3 logScaleSquaredMean;
    float3 rotationMean;
    float3 rotationSquaredMean;
    float3 radianceMean[SH_COUNT];
    float3 radianceSquaredMean[SH_COUNT];
    uint centerSteps;
    uint logScaleSteps;
    uint rotationSteps;
    uint radianceSteps;
};

#if defined(HOST_CODE)
static_assert(sizeof(GeometryAdamState) == 88 + 24 * SH_COUNT, "GeometryAdamState host/device layout changed.");
#endif

END_NAMESPACE_FALCOR
