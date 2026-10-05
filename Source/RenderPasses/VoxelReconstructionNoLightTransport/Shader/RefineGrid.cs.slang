#include "../Voxel/VoxelGrid.slang"

ParameterBlock<GridData> gCoarseGrid;
RWStructuredBuffer<uint2> gChildRanges; // Occupancy mask, compact output offset.
RWStructuredBuffer<uint> gChildCounter;
cbuffer CB
{
    uint gParentOffset;
    uint gParentCount;
    float gParentOpacityThreshold;
    float gOpacityOpticalDepthScale;
    float gChildFaceOverlap;
};

// Opacity SH stores logits. A uniform DC shift lowers every directional
// logit while preserving the learned directional coefficients. Choose the
// shift so the DC-only opacity has the requested optical depth
// tauChild = scale * tauParent.
void attenuateRefinementOpacity(inout SphericalHarmonicsOpacity opacity)
{
    float scale = clamp(gOpacityOpticalDepthScale, 0.01f, 1.0f);
    if (scale >= 1.0f)
        return;
    float y0 = calcSH(0u, float3(0, 0, 1));
    float parentLogit = opacity.coefficients[0] * y0;
    float parentOpticalDepth = parentLogit > 15.0f
        ? parentLogit
        : (parentLogit < -15.0f ? exp(parentLogit) : log(1.0f + exp(parentLogit)));
    float childOpticalDepth = scale * parentOpticalDepth;
    float childLogit = childOpticalDepth > 15.0f
        ? childOpticalDepth
        : (childOpticalDepth < 1e-4f ? log(max(childOpticalDepth, 1e-30f)) : log(exp(childOpticalDepth) - 1.0f));
    // Extremely negative logits may underflow in exp(). Never let that
    // numerical case increase opacity.
    childLogit = min(childLogit, parentLogit);
    opacity.coefficients[0] += (childLogit - parentLogit) / y0;
}

// Conservative SH bound: reject only parents transparent in every direction.
bool retainRefinementParent(SphericalHarmonicsOpacity opacity)
{
    if (gParentOpacityThreshold <= 0.0f) return true;
    float upper = opacity.coefficients[0] * calcSH(0u, float3(0, 0, 1));
    for (uint i = 1u; i < SH_OPACITY_COUNT; ++i)
    {
        uint degree = uint(sqrt(float(i)));
        upper += abs(opacity.coefficients[i]) * sqrt(float(2u * degree + 1u) / (4.0f * 3.14159265359f));
    }
    return upper >= log(gParentOpacityThreshold / (1.0f - gParentOpacityThreshold));
}

// Exact box-constrained quadratic minimum in parent-cell coordinates.
// Enumerate center, face interiors, and clamped edges (including corners).
float closestParentPoint(GaussianEllipsoid parent, float3 voxelSize,
    float3 lower, float3 upper, out float3 best)
{
    float3 scale = gaussianEllipsoidScale(parent);
    float3x3 rotation = gaussianRotationMatrix(parent.rotation);
    float3 unitAxis[3];
    float3 covarianceAxis[3];
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        unitAxis[axis] = rotation[axis] * voxelSize[axis] / scale;
        covarianceAxis[axis] = rotation[axis] * scale / voxelSize[axis];
    }
    best = clamp(float3(0), lower, upper);
    float3 initial = unitAxis[0] * best.x + unitAxis[1] * best.y + unitAxis[2] * best.z;
    float minimum = dot(initial, initial);
    if (minimum == 0.0f) return minimum;
    for (uint n = 0u; n < 3u; ++n)
    {
        float variance = dot(covarianceAxis[n], covarianceAxis[n]);
        for (uint side = 0u; side < 2u; ++side)
        {
            float bound = side == 0u ? lower[n] : upper[n];
            float3 q;
            for (uint axis = 0u; axis < 3u; ++axis)
                q[axis] = dot(covarianceAxis[axis], covarianceAxis[n]) * (bound / variance);
            q[n] = bound;
            if (all(q >= lower) && all(q <= upper))
            {
                float3 unit = unitAxis[0] * q.x + unitAxis[1] * q.y + unitAxis[2] * q.z;
                float value = dot(unit, unit);
                if (value < minimum) { minimum = value; best = q; }
            }
        }
    }
    for (uint freeAxis = 0u; freeAxis < 3u; ++freeAxis)
    {
        uint a = (freeAxis + 1u) % 3u;
        uint b = (freeAxis + 2u) % 3u;
        for (uint sides = 0u; sides < 4u; ++sides)
        {
            float3 q;
            q[a] = (sides & 1u) == 0u ? lower[a] : upper[a];
            q[b] = (sides & 2u) == 0u ? lower[b] : upper[b];
            float3 fixed = unitAxis[a] * q[a] + unitAxis[b] * q[b];
            q[freeAxis] = clamp(-dot(fixed, unitAxis[freeAxis]) / dot(unitAxis[freeAxis], unitAxis[freeAxis]),
                lower[freeAxis], upper[freeAxis]);
            float3 unit = fixed + unitAxis[freeAxis] * q[freeAxis];
            float value = dot(unit, unit);
            if (value < minimum) { minimum = value; best = q; }
        }
    }
    return minimum;
}

// A child with the same rotation and semi-axis ratios is a translated,
// uniformly scaled copy of the parent. In parent-unit-sphere coordinates,
// containment is exactly |center| + scale <= 1.
bool fitRefinementChild(GaussianEllipsoid parent, float3 voxelSize, uint3 bits,
    out GaussianEllipsoid child)
{
    child = parent;
    float3 axes = gaussianEllipsoidScale(parent);
    float3x3 rotation = gaussianRotationMatrix(parent.rotation);
    float3 extent;
    for (uint axis = 0u; axis < 3u; ++axis)
        extent[axis] = length(rotation[axis] * axes) / voxelSize[axis];
    float3 lower = 0.5f * float3(bits) - parent.center;
    float3 upper = lower + 0.5f;
    float3 fineSize = 0.5f * voxelSize;
    // Respect the existing optimizer's minimum axes and [0.01,0.99] center bounds.
    // Never inflate a fitted child afterward: reject unrepresentable slivers instead.
    float3 minimumRatio = max(fineSize * 1e-4f, float3(1e-8f)) / axes;
    float lo = max(minimumRatio.x, max(minimumRatio.y, minimumRatio.z));
    // The renderer clips every primitive to its owning cell's DDA segment.
    // Bounded overlap improves coverage inside that cell without spawning
    // occupied cells beyond the parent's eight addresses.
    float overlap = 0.5f * clamp(gChildFaceOverlap, 0.0f, 0.2f);
    float3 boxLimit = (0.25f + overlap) / extent;
    float hi = min(1.0f, min(boxLimit.x, min(boxLimit.y, boxLimit.z)));
    const float margin = 1e-5f;
    // A convex quadratic reaches its box maximum at a corner. All eight
    // corners inside proves the whole child cell is inside the parent volume.
    bool interior = true;
    for (uint corner = 0u; corner < 8u; ++corner)
    {
        float3 cornerBits = float3(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
        float3 point = lower + 0.5f * cornerBits;
        float3 unit = mul(transpose(rotation), point * voxelSize) / axes;
        interior = interior && dot(unit, unit) <= 1.0f;
    }
    if (interior)
    {
        // Interior cells retain a centered shape, sized to the fine grid
        // rather than the potentially huge parent's original axes.
        float3 middle = lower + 0.25f;
        float3 unit = mul(transpose(rotation), middle * voxelSize) / axes;
        float factor = min(hi, 1.0f - margin - length(unit));
        if (factor < lo) return false;
        child.center = float3(0.5f);
        child.logScale = log(axes * factor);
        return true;
    }
    // Boundary centers stay inside their original cell. Only the ellipsoid
    // extent may cross a face, and the parent-volume containment remains exact.
    float3 inset = max(float3(0.005f), lo * extent - overlap);
    float3 best;
    if (lo > hi || any(lower + inset > upper - inset)) return false;
    float value = closestParentPoint(parent, voxelSize, lower + inset, upper - inset, best);
    if (sqrt(max(value, 0.0f)) + lo > 1.0f - margin) return false;
    // Monotone feasibility: increasing scale shrinks both permissible center sets.
    for (uint iteration = 0u; iteration < 18u; ++iteration)
    {
        float candidate = 0.5f * (lo + hi);
        inset = max(float3(0.005f), candidate * extent - overlap);
        float3 point;
        value = closestParentPoint(parent, voxelSize, lower + inset, upper - inset, point);
        if (sqrt(max(value, 0.0f)) + candidate <= 1.0f - margin)
        {
            lo = candidate;
            best = point;
        }
        else hi = candidate;
    }
    // Prefer a small shift along the inward world-space surface normal when
    // it preserves at least 90% of the maximum feasible size. Thin boundary
    // slivers may need the deeper, box-constrained center found above.
    float3 middle = lower + 0.25f;
    float3 principal = mul(transpose(rotation), middle * voxelSize);
    float3 normal = mul(rotation, principal / (axes * axes));
    float normalLength = length(normal);
    if (normalLength > 1e-20f)
    {
        float step = 0.1f * min(fineSize.x, min(fineSize.y, fineSize.z));
        float3 preferred = clamp(middle - (step / normalLength) * normal / voxelSize,
            lower + 0.005f, upper - 0.005f);
        float3 unit = mul(transpose(rotation), preferred * voxelSize) / axes;
        float factor = min(1.0f, 1.0f - margin - length(unit));
        float3 faceLimit = (min(preferred - lower, upper - preferred) + overlap) / extent;
        factor = min(factor, min(faceLimit.x, min(faceLimit.y, faceLimit.z)));
        if (factor >= max(minimumRatio.x, max(minimumRatio.y, minimumRatio.z)) && factor >= 0.9f * hi)
        {
            lo = factor;
            best = preferred;
        }
    }
    child.center = 2.0f * (parent.center + best) - float3(bits);
    child.logScale = log(axes * lo);
    return true;
}

[numthreads(256, 1, 1)]
void countChildren(uint3 threadID : SV_DispatchThreadID)
{
    uint parentID = gParentOffset + threadID.x;
    if (parentID >= gParentCount) return;
    VoxelData parent = gCoarseGrid.voxelPages[NonUniformResourceIndex(parentID / SPARSE_POOL_PAGE_SIZE)]
        [parentID % SPARSE_POOL_PAGE_SIZE];
    uint mask = 0u;
    if (parent.occupied != 0u && retainRefinementParent(parent.opacity))
        for (uint child = 0u; child < 8u; ++child)
        {
            uint3 bits = uint3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u);
            GaussianEllipsoid childEllipsoid;
            if (fitRefinementChild(parent.ellipsoid, gCoarseGrid.voxelSize, bits, childEllipsoid)) mask |= 1u << child;
        }
    uint offset;
    InterlockedAdd(gChildCounter[0], countbits(mask), offset);
    gChildRanges[parentID] = uint2(mask, offset);
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
    GaussianEllipsoid parentEllipsoid = data.ellipsoid;
    // Radiance is copied unchanged. Opacity is attenuated once per split to
    // reduce the accumulated opacity of multiple children along one ray.
    attenuateRefinementOpacity(data.opacity);
    for (uint child = 0u; child < 8u; ++child)
    {
        if ((range.x & (1u << child)) == 0u) continue;
        uint3 bits = uint3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u);
        if (!fitRefinementChild(parentEllipsoid, gCoarseGrid.voxelSize, bits, data.ellipsoid)) continue;
        uint3 cell = parentCell * 2u + bits;
        uint id = range.y + countbits(range.x & ((1u << child) - 1u));
        VOXEL_DATA(id) = data;
        gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(id / SPARSE_POOL_PAGE_SIZE)]
            [id % SPARSE_POOL_PAGE_SIZE] = uint(CellToIndex(cell, gGridDataParamBlock.voxelCount));
        setVoxelID(cell, id);
        InterlockedAdd(gChildCounter[0], 1u);
    }
}
