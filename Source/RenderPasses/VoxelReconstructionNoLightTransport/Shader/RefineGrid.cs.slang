#include "../Voxel/VoxelGrid.slang"
#include "../Math/LebedevQuadrature.slang"

ParameterBlock<GridData> gCoarseGrid;
RWStructuredBuffer<uint2> gChildRanges; // Occupancy mask, compact output offset.
RWStructuredBuffer<uint> gChildCounter;
cbuffer CB
{
    uint gParentOffset;
    uint gParentCount;
};

// Closed ellipsoid/AABB intersection, in parent-cell coordinates. The minimum
// of the ellipsoid quadratic is at its center, a face interior, an edge, or
// a corner. Include volume containment, not just surface crossings.
bool intersectsRefinementCell(GaussianEllipsoid ellipsoid, float3 voxelSize, uint3 bits)
{
    float3 lower = 0.5f * float3(bits) - ellipsoid.center;
    float3 upper = lower + 0.5f;
    if (all(lower <= 0.0f) && all(upper >= 0.0f)) return true;
    float3 scale = gaussianEllipsoidScale(ellipsoid);
    float3x3 rotation = gaussianRotationMatrix(ellipsoid.rotation);
    float3 unitAxis[3];
    float3 scaledAxis[3];
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        unitAxis[axis] = rotation[axis] * voxelSize[axis] / scale;
        scaledAxis[axis] = rotation[axis] * scale / voxelSize[axis];
    }
    const float threshold = 1.0f + 1e-5f;
    // Six unconstrained face minima from the covariance; no unstable 2x2 inverse.
    for (uint n = 0u; n < 3u; ++n)
    {
        float variance = dot(scaledAxis[n], scaledAxis[n]);
        for (uint side = 0u; side < 2u; ++side)
        {
            float bound = side == 0u ? lower[n] : upper[n];
            if (bound * bound > threshold * variance) continue;
            float3 q;
            for (uint axis = 0u; axis < 3u; ++axis)
                q[axis] = dot(scaledAxis[axis], scaledAxis[n]) * (bound / variance);
            q[n] = bound;
            if (all(q >= lower) && all(q <= upper)) return true;
        }
    }
    // Twelve edge minima, clamped to the edge endpoints (the eight corners).
    for (uint freeAxis = 0u; freeAxis < 3u; ++freeAxis)
    {
        uint a = (freeAxis + 1u) % 3u;
        uint b = (freeAxis + 2u) % 3u;
        for (uint sides = 0u; sides < 4u; ++sides)
        {
            float qa = (sides & 1u) == 0u ? lower[a] : upper[a];
            float qb = (sides & 2u) == 0u ? lower[b] : upper[b];
            float3 fixed = unitAxis[a] * qa + unitAxis[b] * qb;
            float qf = clamp(-dot(fixed, unitAxis[freeAxis]) / dot(unitAxis[freeAxis], unitAxis[freeAxis]),
                lower[freeAxis], upper[freeAxis]);
            float3 point = fixed + unitAxis[freeAxis] * qf;
            if (dot(point, point) <= threshold) return true;
        }
    }
    return false;
}

[numthreads(256, 1, 1)]
void countChildren(uint3 threadID : SV_DispatchThreadID)
{
    uint parentID = gParentOffset + threadID.x;
    if (parentID >= gParentCount) return;
    VoxelData parent = gCoarseGrid.voxelPages[NonUniformResourceIndex(parentID / SPARSE_POOL_PAGE_SIZE)]
        [parentID % SPARSE_POOL_PAGE_SIZE];
    uint mask = 0u;
    if (parent.occupied != 0u)
        for (uint child = 0u; child < 8u; ++child)
        {
            uint3 bits = uint3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u);
            if (intersectsRefinementCell(parent.ellipsoid, gCoarseGrid.voxelSize, bits)) mask |= 1u << child;
        }
    uint offset;
    InterlockedAdd(gChildCounter[0], countbits(mask), offset);
    gChildRanges[parentID] = uint2(mask, offset);
}

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
    uint2 range = gChildRanges[parentID];
    if (range.x == 0u) return;
    VoxelData data = gCoarseGrid.voxelPages[NonUniformResourceIndex(page)][offset];
    uint parentCellIndex = gCoarseGrid.cellIndexPages[NonUniformResourceIndex(page)][offset];
    uint3 parentCell = uint3(IndexToCell(int(parentCellIndex), gCoarseGrid.voxelCount));
    // Preserve local center, rotation, and axis ratios in each finer cell.
    // The selection above uses the original parent, before shrinking. Geometry
    // inheritance remains a local-shape initialization, not exact surface remeshing.
    data.ellipsoid.logScale -= float3(log(2.0f));
    data.opacity = refineOpacity(data.opacity);
    for (uint child = 0u; child < 8u; ++child)
    {
        if ((range.x & (1u << child)) == 0u) continue;
        uint3 bits = uint3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u);
        uint3 cell = parentCell * 2u + bits;
        uint id = range.y + countbits(range.x & ((1u << child) - 1u));
        VOXEL_DATA(id) = data;
        gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(id / SPARSE_POOL_PAGE_SIZE)]
            [id % SPARSE_POOL_PAGE_SIZE] = uint(CellToIndex(cell, gGridDataParamBlock.voxelCount));
        setVoxelID(cell, id);
        InterlockedAdd(gChildCounter[0], 1u);
    }
}
