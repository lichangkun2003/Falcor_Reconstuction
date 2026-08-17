#pragma once
#include "HostDeviceShared.slang"
#include "Defines.h"

BEGIN_NAMESPACE_FALCOR

struct GeometryCandidateRecord
{
    uint valid;

    int3 cellInt; // voxel index

    float gmin;   // ray vs ellipsoid implicit min value
    float4 xStar; // closest point on ray segment

    float4 radiance; // L_candidate

    float4 behindColor; // C_next (for loss gradient)

};

struct VoxelRecord
{
    float3 hitPosition;
    int3 cellInt;

    float3 radiance;
    float opacity;
    float transmittanceBefore;

    MUTATING
    void init() {
        hitPosition = float3(0);
        cellInt = int3(0);
        radiance = float3(0);
        opacity = 0.0f;
        transmittanceBefore = 1.0f;
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
            candidates[i].valid = 0;
            candidates[i].gmin = 1e30f;
        }
    }
};


#if !defined(HOST_CODE)

inline uint PixelToIndex(uint2 pixel, uint2 size)
{
    return uint(pixel.x) + uint(pixel.y) * size.x;
}

#endif


END_NAMESPACE_FALCOR
