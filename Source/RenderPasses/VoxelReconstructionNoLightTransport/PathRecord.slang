#pragma once
#include "HostDeviceShared.slang"
#include "Defines.h"

BEGIN_NAMESPACE_FALCOR

struct GeometryCandidateRecord
{
    uint valid;

    uint voxelID;

    float gmin;   // ray vs ellipsoid implicit min value
    float3 xStar; // closest point on ray segment
    float sortScore; //for candidate sort

    float transmittanceBefore;
    uint frontHitCount;

    MUTATING
    void init() {
        valid = 0;
        voxelID = 0;

        gmin = 1e30f;
        xStar = float3(0.0f);
        //sortScore = 1e30f;

        transmittanceBefore = 1.0f;
        frontHitCount = 0;
    }

};

struct VoxelRecord
{
    uint voxelID;

    float3 radiance;
    float opacity;
    float transmittanceBefore;

    // Geometry information for this actual hard hit.
    float gmin;
    float3 xStar;

    MUTATING
    void init() {
        voxelID = 0;
        radiance = float3(0);
        opacity = 0.0f;
        transmittanceBefore = 1.0f;
        gmin = 1e30f;
        xStar = float3(0);
    }
};


struct PathRecord
{
    uint valid;
    float3 viewDir;  // xyz = view direction = - ray direction
    float3 backgroundColor;
    float3 finalColor;

    // Radiance & Opacity
    uint contributingVoxelCount;
    VoxelRecord voxels[MAX_CONTRIBUTING_VOXELS_PER_RAY];


    // Geometry
    uint candidateCount;
    GeometryCandidateRecord candidates[MAX_CANDIDATES];

    MUTATING
    void init() {
        valid = 0;
        viewDir = float3(0); 
        backgroundColor = float3(0);
        finalColor = float3(0);

        contributingVoxelCount = 0;
        for (uint i = 0; i < MAX_CONTRIBUTING_VOXELS_PER_RAY; i++)
        {
            voxels[i].init();
        }

        candidateCount = 0;
        for (uint i = 0; i < MAX_CANDIDATES; i++)
        {
            candidates[i].init();
        }
    }
};


#if defined(HOST_CODE)
// D3D12 limits each StructuredBuffer element to 2048 bytes.
static_assert(sizeof(PathRecord) <= 2048, "PathRecord exceeds the D3D12 structured-buffer stride limit.");
static_assert(sizeof(VoxelRecord) == 40 && sizeof(GeometryCandidateRecord) == 36 &&
    sizeof(PathRecord) == 48 + MAX_CONTRIBUTING_VOXELS_PER_RAY * sizeof(VoxelRecord) +
        MAX_CANDIDATES * sizeof(GeometryCandidateRecord),
    "PathRecord host/device layout changed.");
#else

inline uint PixelToIndex(uint2 pixel, uint2 size)
{
    return uint(pixel.x) + uint(pixel.y) * size.x;
}

#endif


END_NAMESPACE_FALCOR
