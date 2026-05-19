#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

struct PathRecord
{
    int3 hitCellInt; // xyz = hit.cellInt
    uint valid;
    float4 hitCell;     // xyz = hit.cell
    float4 lightDirPdf; // xyz = light direction, w = pdf
    float4 lightColor;  // rgb = Li
    float4 viewDir;     // xyz = view direction

    MUTATING
    void init() {
        hitCellInt = int3(0);
        valid = 0;
        hitCell = float4(0);  // xyz = hit.cell
        lightDirPdf = float4(0); // xyz = light direction, w = pdf
        lightColor = float4(0);  // rgb = Li
        viewDir = float4(0);     // xyz = view direction
    }
};

#if !defined(HOST_CODE)

inline uint PixelToIndex(uint2 pixel, uint2 size)
{
    return uint(pixel.x) + uint(pixel.y) * size.x;
}

#endif


END_NAMESPACE_FALCOR
