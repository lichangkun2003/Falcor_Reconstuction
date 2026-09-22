#include "../Voxel/VoxelData.slang"
#include "../Math/LebedevQuadrature.slang"

// Only dispatched by mode 2, with exactly twice the source resolution per axis.
StructuredBuffer<VoxelData> gSourceVoxels;
RWStructuredBuffer<VoxelData> gDestinationVoxels;

cbuffer CB
{
    uint3 gSourceVoxelCount;
    uint3 gDestinationVoxelCount;
}

uint refinementIndex(uint3 cell, uint3 count)
{
    return cell.x + count.x * (cell.y + count.y * cell.z);
}

bool finiteMatrix(float3x3 value)
{
    return all(isfinite(value[0])) && all(isfinite(value[1])) && all(isfinite(value[2]));
}

float ellipsoidValue(float3 p, float3 center, float3x3 B)
{
    float3 d = p - center;
    return dot(d, mul(B, d));
}

void considerBoxPoint(float3 candidate, float3 center, float3x3 B, inout float3 bestPoint, inout float bestValue)
{
    float value = ellipsoidValue(candidate, center, B);
    if (isfinite(value) && value < bestValue)
    {
        bestValue = value;
        bestPoint = candidate;
    }
}

// The constrained minimum of an SPD quadratic is either the unconstrained
// center, a stationary point on a face, or a minimum on an edge (including its
// vertices). Enumerating these avoids using discrete occupancy samples, which
// can miss arbitrarily thin ellipsoid/box intersections.
float3 closestBoxPoint(float3 center, float3x3 B, float3 lo, float3 hi)
{
    float3 bestPoint = clamp(center, lo, hi);
    float bestValue = ellipsoidValue(bestPoint, center, B);

    for (uint axis = 0; axis < 3; ++axis)
    {
        uint u = (axis + 1) % 3;
        uint v = (axis + 2) % 3;

        for (uint side = 0; side < 2; ++side)
        {
            float3 candidate = center;
            candidate[axis] = side == 0 ? lo[axis] : hi[axis];
            float delta = candidate[axis] - center[axis];
            float determinant2 = B[u][u] * B[v][v] - B[u][v] * B[u][v];
            if (determinant2 > 1e-12f)
            {
                candidate[u] -= delta * (B[v][v] * B[u][axis] - B[u][v] * B[v][axis]) / determinant2;
                candidate[v] -= delta * (B[u][u] * B[v][axis] - B[u][v] * B[u][axis]) / determinant2;
                considerBoxPoint(clamp(candidate, lo, hi), center, B, bestPoint, bestValue);
            }
        }

        for (uint side = 0; side < 4; ++side)
        {
            float3 candidate = center;
            candidate[u] = (side & 1) == 0 ? lo[u] : hi[u];
            candidate[v] = (side & 2) == 0 ? lo[v] : hi[v];
            candidate[axis] -= (B[axis][u] * (candidate[u] - center[u]) +
                B[axis][v] * (candidate[v] - center[v])) / max(B[axis][axis], 1e-12f);
            considerBoxPoint(clamp(candidate, lo, hi), center, B, bestPoint, bestValue);
        }
    }
    return bestPoint;
}

bool definitelyOutside(float3 p, float3 center, float3x3 B, float3 lo, float3 hi)
{
    float3 delta = p - center;
    float3 gradient = 2.0f * mul(B, delta);
    float value = dot(delta, 0.5f * gradient);
    float3 extreme = float3(gradient.x >= 0.0f ? lo.x : hi.x,
        gradient.y >= 0.0f ? lo.y : hi.y, gradient.z >= 0.0f ? lo.z : hi.z);
    float3 correction = gradient * (extreme - p);
    float lowerBound = value + correction.x + correction.y + correction.z;
    // A convex tangent plane is a lower bound everywhere in the box, even if
    // floating-point face solves above did not find its exact minimizer.
    float tolerance = 1e-4f * (1.0f + abs(value) + dot(abs(correction), float3(1.0f)));
    return isfinite(lowerBound) && lowerBound > 1.0f + tolerance;
}

float refinementSoftplus(float value)
{
    return max(value, 0.0f) + log(1.0f + exp(-abs(value)));
}

SphericalHarmonicsOpacity splitOpacity(SphericalHarmonicsOpacity source)
{
    SphericalHarmonicsOpacity destination;
    destination.init();
    for (uint sampleIndex = 0; sampleIndex < LEBEDEV_DIRECTION_COUNT; ++sampleIndex)
    {
        float3 direction = LebedevSamples[sampleIndex].direction;
        float parentLogit = source.calcLogit(direction);
        parentLogit = isfinite(parentLogit) ? clamp(parentLogit, -60.0f, 60.0f) : 0.0f;
        // Assume two child hits per former parent hit. This is only an average
        // path-length approximation: alphaChild = 1 - sqrt(1 - alphaParent).
        // Its equivalent logit expression avoids cancellation for small alpha
        // and avoids rounding alpha to 1 for large positive parent logits.
        float childLogit = parentLogit - refinementSoftplus(0.5f * refinementSoftplus(parentLogit));
        float weightedLogit = childLogit * (4.0f * PI * LebedevSamples[sampleIndex].weight);
        for (uint coefficient = 0; coefficient < SH_OPACITY_COUNT; ++coefficient)
            destination.coefficients[coefficient] += weightedLogit * calcSH(coefficient, direction);
    }
    return destination;
}

[numthreads(4, 4, 4)]
void main(uint3 cell : SV_DispatchThreadID)
{
    if (any(cell >= gDestinationVoxelCount)) return;
    uint destinationIndex = refinementIndex(cell, gDestinationVoxelCount);
    uint3 parentCell = cell / 2;
    if (any(parentCell >= gSourceVoxelCount))
    {
        gDestinationVoxels[destinationIndex] = (VoxelData)0;
        return;
    }

    VoxelData source = gSourceVoxels[refinementIndex(parentCell, gSourceVoxelCount)];
    VoxelData destination = source;
    if (source.occupied == 0)
    {
        gDestinationVoxels[destinationIndex] = (VoxelData)0;
        return;
    }

    float3 center = source.ellipsoid.center;
    float3x3 B = 0.5f * (source.ellipsoid.B + transpose(source.ellipsoid.B));
    float3x3 cholesky;
    if (!all(isfinite(center)) || !finiteMatrix(B) || !cholesky3x3SPD(B, cholesky))
    {
        // The coarse initializer normally ensures valid geometry; retain a
        // bounded occupied seed for malformed legacy data instead of NaNs.
        center = float3(0.5f);
        B = diag3(float3(1.0f / 0.36f));
    }

    float3 offset = float3(cell & uint3(1));
    float3 boxMin = 0.5f * offset;
    float3 boxMax = boxMin + 0.5f;
    float3 anchor = closestBoxPoint(center, B, boxMin, boxMax);
    // Geometric overlap determines child occupancy at every refinement stage.
    // This is independent of the periodic volume-threshold pruning during optimization.
    if (definitelyOutside(anchor, center, B, boxMin, boxMax))
    {
        gDestinationVoxels[destinationIndex] = (VoxelData)0;
        return;
    }

    // Restart retained children with full cell coverage. The farthest corners of
    // [0,1]^3 are sqrt(3)/2 from its center; .87 leaves a small numerical margin.
    // Ray intersections are still clipped to the child's own voxel segment.
    const float childRadius = 0.87f;
    destination.ellipsoid.center = float3(0.5f);
    destination.ellipsoid.B = diag3(float3(1.0f / (childRadius * childRadius)));
    destination.opacity = splitOpacity(source.opacity);
    gDestinationVoxels[destinationIndex] = destination;
}
