

#include "../PathRecord.slang"
#include "DiffVoxelIO.slang"


import DiffRendering.SceneGradientInfo;

[Differentiable]
float3 evalVoxelRadianceAD(
    SceneGradientInfo gradInfo,
    no_diff uint voxelLinearID,
    no_diff float3 viewDir
)
{
    DiffVoxelIO io;

    float3 result = float3(0.0f);

    [MaxIters(SH_COUNT)]
    for (uint i = 0; i < SH_COUNT; i++)
    {
        float basis = no_diff calcSH(i, viewDir);

        float3 coeff =
            io.loadSHCoeff(
                gradInfo,
                voxelLinearID,
                i
            );

        result += coeff * basis;
    }

    return result;
}

[Differentiable]
float3 evalPixelColorFromPathAD(
    SceneGradientInfo gradInfo,
    no_diff int3 hitCellInt,
    no_diff float4 viewDir4,
    no_diff uint3 voxelCount
)
{

    if (any(hitCellInt < int3(0)) || any(hitCellInt >= int3(voxelCount)))
        return float3(0.0f);

    // float3 viewDir =  normalizeSafe(viewDir.xyz);
    float3 viewDir = viewDir4.xyz;

    if (all(viewDir == float3(0.0f)))
        return float3(0.0f);

    uint voxelLinearID = CellToIndex(
        uint3(hitCellInt),
        voxelCount
    );

    return evalVoxelRadianceAD(
        gradInfo,
        voxelLinearID,
        viewDir
    );
}
