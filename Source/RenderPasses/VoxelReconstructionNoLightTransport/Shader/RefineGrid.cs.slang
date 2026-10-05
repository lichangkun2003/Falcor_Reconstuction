#include "../Voxel/VoxelGrid.slang"
#include "../Math/LebedevQuadrature.slang"

ParameterBlock<GridData> gCoarseGrid;
cbuffer CB
{
    uint gParentOffset;
    uint gParentCount;
};

float refinementSoftplus(float x)
{
    return max(x, 0.0f) + log(1.0f + exp(-abs(x)));
}

// Halve optical thickness: alphaChild = 1 - sqrt(1 - alphaParent).
// Project the nonlinear logit correction back to SH. For constant opacity
// this is exact; directional opacity and oblique multi-cell rays are approximate.
SphericalHarmonicsOpacity refineOpacity(SphericalHarmonicsOpacity parent)
{
    SphericalHarmonicsOpacity result = parent;
    for (uint sample = 0u; sample < LEBEDEV_DIRECTION_COUNT; ++sample)
    {
        float3 direction = LebedevSamples[sample].direction;
        float logit = parent.calcLogit(direction);
        float correction = -refinementSoftplus(0.5f * refinementSoftplus(logit));
        float weighted = correction * (4.0f * 3.14159265359f * LebedevSamples[sample].weight);
        for (uint i = 0u; i < SH_OPACITY_COUNT; ++i)
            result.coefficients[i] += weighted * calcSH(i, direction);
    }
    return result;
}

[numthreads(256, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    uint parentID = gParentOffset + threadID.x;
    if (parentID >= gParentCount) return;
    uint page = parentID / SPARSE_POOL_PAGE_SIZE;
    uint offset = parentID % SPARSE_POOL_PAGE_SIZE;
    VoxelData data = gCoarseGrid.voxelPages[NonUniformResourceIndex(page)][offset];
    uint parentCellIndex = gCoarseGrid.cellIndexPages[NonUniformResourceIndex(page)][offset];
    uint3 parentCell = uint3(IndexToCell(int(parentCellIndex), gCoarseGrid.voxelCount));
    // Preserve local center, rotation, and axis ratios in each finer cell.
    // This is a conservative all-eight spatial subdivision, not exact surface remeshing.
    data.ellipsoid.logScale -= float3(log(2.0f));
    data.opacity = refineOpacity(data.opacity);
    for (uint child = 0u; child < 8u; ++child)
    {
        uint3 bits = uint3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u);
        uint3 cell = parentCell * 2u + bits;
        uint id = parentID * 8u + child;
        VOXEL_DATA(id) = data;
        gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(id / SPARSE_POOL_PAGE_SIZE)]
            [id % SPARSE_POOL_PAGE_SIZE] = uint(CellToIndex(cell, gGridDataParamBlock.voxelCount));
        setVoxelID(cell, id);
    }
}
