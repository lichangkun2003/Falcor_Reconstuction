#pragma once
#include "HostDeviceShared.slang"
#include "Defines.h"

BEGIN_NAMESPACE_FALCOR

// Store optimizer moments only for occupied voxels. Rotation moments
// live in the three-dimensional local tangent space, rather than in quaternion
// coordinates. Opacity keeps independent moments for every SH coefficient.
struct GeometryAdamState
{
    float3 centerMean;
    float3 centerSquaredMean;
    float3 logScaleMean;
    float3 logScaleSquaredMean;
    float3 rotationMean;
    float3 rotationSquaredMean;
    float opacityMean[SH_OPACITY_COUNT];
    float opacitySquaredMean[SH_OPACITY_COUNT];
    uint centerSteps;
    uint logScaleSteps;
    uint rotationSteps;
    uint opacitySteps;
};

// Radiance state is allocated only when a voxel first receives a radiance
// gradient. A dense copy for every point-cloud-covered cell is too costly at
// 512^3, even though most cells are not visible from any one training view.
struct RadianceAdamState
{
    float3 mean[SH_COUNT];
    float3 squaredMean[SH_COUNT];
    uint steps;
};

#if defined(HOST_CODE)
static_assert(sizeof(GeometryAdamState) == 88 + 8 * SH_OPACITY_COUNT,
    "GeometryAdamState host/device layout changed.");
static_assert(sizeof(RadianceAdamState) == 4 + 24 * SH_COUNT,
    "RadianceAdamState host/device layout changed.");
#endif

END_NAMESPACE_FALCOR
