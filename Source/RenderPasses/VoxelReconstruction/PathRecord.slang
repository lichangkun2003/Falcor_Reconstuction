#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

struct ReparamEval
{
    int3 cellInt;    // int3(-1) means invalid
    float3 xEval;    //
    float3 rayStart; // ray start
    float score;

    MUTATING
    void init()
    {
        cellInt = int3(-1, -1, -1);
        xEval = float3(0.0f);
        rayStart = float3(0.0f);
        score = 1e20f;
    }
};


struct PathRecord
{
    int3 hitCellInt; // xyz = hit.cellInt
    uint valid;
    float4 hitCell;     // xyz = hit.cell
    float4 lightDirPdf; // xyz = light direction, w = pdf
    float4 lightColor;  // rgb = Li
    float4 viewDir;     // xyz = view direction

    // Geometry
    int3 xCellInt;
    float4 xEval;    //
    float4 rayStart; // ray start
    float4 pathContribution;

    MUTATING
    void init() {
        hitCellInt = int3(0);
        valid = 0;
        hitCell = float4(0);  // xyz = hit.cell
        lightDirPdf = float4(0); // xyz = light direction, w = pdf
        lightColor = float4(0);  // rgb = Li
        viewDir = float4(0);     // xyz = view direction
        xCellInt = int3(0);
        xEval = float4(0.0f);
        rayStart = float4(0.0f);
        pathContribution = float4(0.0f);
    }
};


#if !defined(HOST_CODE)

inline uint PixelToIndex(uint2 pixel, uint2 size)
{
    return uint(pixel.x) + uint(pixel.y) * size.x;
}

#endif


END_NAMESPACE_FALCOR
