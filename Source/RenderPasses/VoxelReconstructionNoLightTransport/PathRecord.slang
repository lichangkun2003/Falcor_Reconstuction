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


struct PathRecord
{
    uint valid;
    int3 hitCellInt; // xyz = hit.cellInt
    float4 hitCell;     // xyz = hit.cell
    float4 viewDir;     // xyz = view direction
    float4 rayDir;     // xyz = view direction

    float4 pathContribution; // L_hit
    float4 nextContribution; // C_behind
    float4 backGroundColor;

    // Geometry
    uint candidateCount;
    GeometryCandidateRecord candidates[MAX_CANDIDATES];

    MUTATING
    void init() {
        valid = 0;
        hitCellInt = int3(0);
        hitCell = float4(0);  // xyz = hit.cell
        viewDir = float4(0);  // xyz = view direction
        rayDir = float4(0);  // xyz = ray direction

        pathContribution = float4(0);
        nextContribution = float4(0);

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
