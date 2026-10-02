#pragma once
#include "GaussianEllipsoid.slang"

// Minimize the ellipsoid's positive-definite quadratic on the finite voxel
// face, not its infinite supporting plane or axis-aligned bounding box.
bool growthFaceContact(GaussianEllipsoid parent, float3 voxelSize, uint face, out float3 contact)
{
    uint n = face / 2u;
    uint u = (n + 1u) % 3u;
    uint v = (n + 2u) % 3u;
    float plane = (face & 1u) != 0u ? 1.0f : 0.0f;
    float3 scale = gaussianEllipsoidScale(parent);
    float3x3 rotation = gaussianRotationMatrix(parent.rotation);
    float3x3 metric;
    for (uint i = 0; i < 3u; ++i)
        for (uint j = 0; j < 3u; ++j)
            metric[i][j] = voxelSize[i] * voxelSize[j] * dot(rotation[i] / scale, rotation[j] / scale);

    float qn = plane - parent.center[n];
    float determinant = metric[u][u] * metric[v][v] - metric[u][v] * metric[u][v];
    contact = parent.center;
    contact[n] = plane;
    float best = 1e30f;
    // The constrained minimum is either interior or on one of four edges.
    // Clamping the edge minimum also includes both corners of that edge.
    for (uint candidate = 0u; candidate < 5u; ++candidate)
    {
        float3 p = parent.center;
        p[n] = plane;
        if (candidate == 0u)
        {
            if (determinant <= 1e-30f)
                continue;
            p[u] -= qn * (metric[v][v] * metric[u][n] - metric[u][v] * metric[v][n]) / determinant;
            p[v] -= qn * (metric[u][u] * metric[v][n] - metric[u][v] * metric[u][n]) / determinant;
            if (p[u] < 0.0f || p[u] > 1.0f || p[v] < 0.0f || p[v] > 1.0f)
                continue;
        }
        else
        {
            uint fixedAxis = candidate <= 2u ? u : v;
            uint freeAxis = candidate <= 2u ? v : u;
            p[fixedAxis] = float((candidate - 1u) & 1u);
            p[freeAxis] = saturate(
                parent.center[freeAxis] -
                (metric[freeAxis][n] * qn + metric[freeAxis][fixedAxis] * (p[fixedAxis] - parent.center[fixedAxis])) /
                    max(metric[freeAxis][freeAxis], 1e-30f)
            );
        }
        float3 q = p - parent.center;
        float value = dot(q, mul(metric, q));
        if (value < best)
        {
            best = value;
            contact = p;
        }
    }
    return best < 1.0f - 1e-5f;
}

bool makeGrowthEllipsoid(
    GaussianEllipsoid parent,
    float3 voxelSize,
    uint face,
    float shrink,
    float contactOffset,
    out GaussianEllipsoid child
)
{
    child = parent;
    float3 contact;
    if (!growthFaceContact(parent, voxelSize, face, contact))
        return false;
    uint n = face / 2u;
    bool positive = (face & 1u) != 0u;
    // The tangential coordinates follow the actual face intersection. Keep a
    // small margin so a contact near an edge does not create a zero-size child.
    child.center = clamp(contact, float3(0.05f), float3(0.95f));
    child.center[n] = positive ? contactOffset : 1.0f - contactOffset;
    float3 scale = gaussianEllipsoidScale(parent);
    float3x3 rotation = gaussianRotationMatrix(parent.rotation);
    float3 extent = float3(length(rotation[0] * scale), length(rotation[1] * scale), length(rotation[2] * scale)) / voxelSize;
    float factor = shrink;
    for (uint i = 0u; i < 3u; ++i)
    {
        // Only the parent-facing plane is exempt. The opposite normal plane
        // and both planes of each tangential axis bound the WHOLE rotated shape.
        float margin = i == n ? (positive ? 1.0f - child.center[i] : child.center[i]) : min(child.center[i], 1.0f - child.center[i]);
        factor = min(factor, max(margin - 0.005f, 0.0f) / max(extent[i], 1e-20f));
    }
    if (factor <= 0.0f || any(scale * factor < voxelSize * 1e-4f))
        return false;
    child.logScale = log(scale * factor);
    child.rotation = normalizeGaussianRotation(parent.rotation);
    return true;
}
