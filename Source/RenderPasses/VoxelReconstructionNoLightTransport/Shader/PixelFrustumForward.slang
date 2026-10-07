#pragma once
#include "../Math/PixelCoverage.slang"

// Compile-time switches also let GPU tests compare against the analytic path.
#ifndef FRUSTUM_CANDIDATE_BATCH_SIZE
#define FRUSTUM_CANDIDATE_BATCH_SIZE 4
#endif
#if FRUSTUM_CANDIDATE_BATCH_SIZE < 1 || FRUSTUM_CANDIDATE_BATCH_SIZE > 8
#error "Frustum candidate batches must contain between one and eight entries."
#endif

float4 worldPlaneToCell(float4 plane)
{
    return float4(plane.xyz * voxelSize, dot(plane.xyz, gridMin) + plane.w);
}

void makePixelFrustumPlanes(uint2 pixel, out float4 planes[6])
{
    float2 uvMin = float2(pixel) / float2(pixelCount);
    float2 uvMax = float2(pixel + 1u) / float2(pixelCount);
    float xMin = 2.0f * uvMin.x - 1.0f;
    float xMax = 2.0f * uvMax.x - 1.0f;
    float yMin = 1.0f - 2.0f * uvMax.y;
    float yMax = 1.0f - 2.0f * uvMin.y;
    planes[0] = worldPlaneToCell(viewProjection[0] - xMin * viewProjection[3]);
    planes[1] = worldPlaneToCell(xMax * viewProjection[3] - viewProjection[0]);
    planes[2] = worldPlaneToCell(viewProjection[1] - yMin * viewProjection[3]);
    planes[3] = worldPlaneToCell(yMax * viewProjection[3] - viewProjection[1]);
    planes[4] = worldPlaneToCell(viewProjection[2]);
    planes[5] = worldPlaneToCell(viewProjection[3] - viewProjection[2]);
}

bool cellOverlapsPixelFrustum(int3 cell, float4 planes[6])
{
    float3 center = float3(cell) + 0.5f;
    for (uint i = 0; i < 6; ++i)
    {
        float4 plane = planes[i];
        float radius = 0.5f * dot(abs(plane.xyz), float3(1));
        // Deliberately omit the whole cell if it touches/crosses the camera's
        // near (pixel) plane. Reconstruction cameras are outside the scene.
        if (i == 4 && dot(plane.xyz, center) + plane.w - radius <= 1e-5f * length(plane.xyz)) return false;
        if (dot(plane.xyz, center) + plane.w + radius < -1e-5f * length(plane.xyz)) return false;
    }
    return true;
}

float cellCameraDepth(int3 cell)
{
    float3 world = gridMin + (float3(cell) + 0.5f) * voxelSize;
    return dot(world - cameraPosition, cameraForward);
}

struct PixelFrustumRays
{
    float3 origin[4]; // Cell position at camera-forward depth zero.
    float3 slope[4]; // Cell displacement per unit camera-forward depth.
};

PixelFrustumRays makePixelFrustumRays(uint2 pixel)
{
    PixelFrustumRays rays;
    for (uint corner = 0; corner < 4; ++corner)
    {
        float2 uv = (float2(pixel) + float2(corner & 1u, corner >> 1u)) / float2(pixelCount);
        float3 nearPoint = screenCoordToCell(uv, 0.0f);
        float3 farPoint = screenCoordToCell(uv, 1.0f);
        float nearDepth = dot(gridMin + nearPoint * voxelSize - cameraPosition, cameraForward);
        float farDepth = dot(gridMin + farPoint * voxelSize - cameraPosition, cameraForward);
        rays.slope[corner] = (farPoint - nearPoint) / (farDepth - nearDepth);
        rays.origin[corner] = nearPoint - nearDepth * rays.slope[corner];
    }
    return rays;
}

void frustumSlabCellBounds(PixelFrustumRays rays, float depthMin, float depthMax, out int3 cellMin, out int3 cellMax)
{
    float3 boundsMin = float3(1e30f);
    float3 boundsMax = float3(-1e30f);
    for (uint corner = 0; corner < 4; ++corner)
    {
        float3 a = rays.origin[corner] + depthMin * rays.slope[corner];
        float3 b = rays.origin[corner] + depthMax * rays.slope[corner];
        boundsMin = min(boundsMin, min(a, b));
        boundsMax = max(boundsMax, max(a, b));
    }
    cellMin = max(int3(floor(boundsMin - 1e-4f)), int3(0));
    cellMax = min(int3(floor(boundsMax + 1e-4f)), int3(voxelCount) - 1);
}

// Certify full coverage using four interior ray hits of the convex clipped
// body. Shrink the cell and require an ellipsoid interior margin. Uncertain
// or boundary cases fall back to the double analytic contour calculation.
bool frustumPixelFullyCovered(GaussianEllipsoid ellipsoid, int3 cell, PixelFrustumRays rays)
{
    float3 scale = gaussianEllipsoidScale(ellipsoid);
    if (min(scale.x / voxelSize.x, min(scale.y / voxelSize.y, scale.z / voxelSize.z)) < 1e-4f) return false;
    float3x3 principal = transpose(gaussianRotationMatrix(ellipsoid.rotation));
    float depth = cellCameraDepth(cell);
    float halfDepth = 0.5f * dot(abs(cameraForward), voxelSize);
    for (uint corner = 0; corner < 4; ++corner)
    {
        float3 origin = rays.origin[corner] - float3(cell);
        float3 slope = rays.slope[corner];
        float lo = max(cameraDepthRange.x, depth - halfDepth);
        float hi = min(cameraDepthRange.y, depth + halfDepth);
        float3 magnitude = abs(rays.origin[corner]) + abs(float3(cell))
            + abs(slope) * max(abs(lo), abs(hi)) + float3(1);
        float cellMargin = max(1e-4f, 4e-6f * max(magnitude.x, max(magnitude.y, magnitude.z)));
        float interiorMargin = max(1e-3f, 8.0f * cellMargin
            * max(voxelSize.x, max(voxelSize.y, voxelSize.z)) / min(scale.x, min(scale.y, scale.z)));
        if (cellMargin >= 0.05f || interiorMargin >= 0.1f) return false;
        for (uint axis = 0; axis < 3; ++axis)
        {
            if (abs(slope[axis]) < 1e-12f)
            {
                if (origin[axis] <= cellMargin || origin[axis] >= 1.0f - cellMargin) return false;
            }
            else
            {
                float a = (cellMargin - origin[axis]) / slope[axis];
                float b = (1.0f - cellMargin - origin[axis]) / slope[axis];
                lo = max(lo, min(a, b));
                hi = min(hi, max(a, b));
            }
        }
        if (hi <= lo) return false;
        float3 a = mul(principal, (origin + lo * slope - ellipsoid.center) * voxelSize) / scale;
        float3 b = mul(principal, (origin + hi * slope - ellipsoid.center) * voxelSize) / scale;
        float3 delta = b - a;
        float t = saturate(-dot(a, delta) / max(dot(delta, delta), 1e-30f));
        float3 closest = a + t * delta;
        if (dot(closest, closest) >= 1.0f - interiorMargin) return false;
    }
    return true;
}

bool ellipsoidOutsidePixel(GaussianEllipsoid ellipsoid, int3 cell, float4 planes[6])
{
    float3 scale = gaussianEllipsoidScale(ellipsoid);
    float3x3 principal = transpose(gaussianRotationMatrix(ellipsoid.rotation));
    float3 center = float3(cell) + ellipsoid.center;
    for (uint i = 0; i < 4; ++i)
    {
        float3 products = planes[i].xyz * center;
        float distance = products.x + products.y + products.z + planes[i].w;
        float radius = length(mul(principal, planes[i].xyz / voxelSize) * scale);
        // Conservative roundoff margin: uncertain thin/grazing cases use the
        // exact path. Separating a complete ellipsoid also separates its clip.
        float margin = 8e-6f * (dot(abs(products), float3(1)) + abs(planes[i].w) + radius + 1.0f);
        if (distance + radius < -margin) return true;
    }
    return false;
}

float4 renderPixelFrustum(uint2 pixel)
{
    float2 centerUV = (float2(pixel) + 0.5f) / float2(pixelCount);
    float3 centerNear = screenCoordToCell(centerUV, 0.0f);
    float3 centerFar = screenCoordToCell(centerUV, 1.0f);
    float3 direction = normalize((centerFar - centerNear) * voxelSize);
    // Only an enabled, loaded environment contributes through residual T.
    // clearColor is a ray-preview setting, not environment radiance.
    float3 background = float3(0);
    #if USE_ENV_MAP
    if (renderBackGround) background = gScene.envMap.eval(direction);
    #endif
    float4 planes[6];
    makePixelFrustumPlanes(pixel, planes);
    PixelFrustumRays rays = makePixelFrustumRays(pixel);
    float3 gridCenterWorld = gridMin + 0.5f * float3(voxelCount) * voxelSize;
    float gridDepth = dot(gridCenterWorld - cameraPosition, cameraForward);
    float gridRadius = 0.5f * dot(abs(cameraForward), float3(voxelCount) * voxelSize);
    float depthMin = max(gridDepth - gridRadius, cameraDepthRange.x);
    float depthMax = min(gridDepth + gridRadius, cameraDepthRange.y);
    if (depthMin >= depthMax) return float4(background, 0.0f);
    float cellHalfDepth = 0.5f * dot(abs(cameraForward), voxelSize);
    float slabWidth = max(2.0f * cellHalfDepth, 1e-6f);
    uint slabCount = uint(ceil((depthMax - depthMin) / slabWidth));
    float3 color = float3(0);
    float transmittance = 1.0f;
    for (uint slab = 0; slab < slabCount; ++slab)
    {
        float slabMin = depthMin + float(slab) * slabWidth;
        float slabMax = min(depthMax, slabMin + slabWidth);
        int3 lo, hi;
        // A cell is owned by its center-depth slab. Expand by its depth radius
        // so that partial footprint overlaps anywhere inside it are enumerated.
        frustumSlabCellBounds(rays, max(cameraDepthRange.x, slabMin - cellHalfDepth),
            min(cameraDepthRange.y, slabMax + cellHalfDepth), lo, hi);
        float previousDepth = -1e30f;
        int previousID = -1;
        // Keep the nearest four remaining candidates in sorted order. Overflow
        // is processed in subsequent batches, never dropped or composited early.
        // Typical narrow pixel footprints need just one scan per depth slab.
        while (true)
        {
            int candidateIDs[FRUSTUM_CANDIDATE_BATCH_SIZE];
            uint candidateCells[FRUSTUM_CANDIDATE_BATCH_SIZE];
            float candidateDepths[FRUSTUM_CANDIDATE_BATCH_SIZE];
            uint count = 0;
            bool overflow = false;
            for (int z = lo.z; z <= hi.z; ++z)
                for (int y = lo.y; y <= hi.y; ++y)
                    for (int x = lo.x; x <= hi.x; ++x)
                    {
                        int3 cell = int3(x, y, z);
                        float depth = clamp(cellCameraDepth(cell), depthMin, depthMax);
                        uint owner = min(uint(max(0.0f, floor((depth - depthMin) / slabWidth))), slabCount - 1u);
                        if (owner != slab) continue;
                        int id = lookupVoxelID(cell);
                        if (id < 0 || VOXEL_DATA(uint(id)).occupied == 0u) continue;
                        if (!cellOverlapsPixelFrustum(cell, planes)) continue;
                        bool afterPrevious = depth > previousDepth || (depth == previousDepth && id > previousID);
                        if (!afterPrevious) continue;
                        if (count == FRUSTUM_CANDIDATE_BATCH_SIZE) overflow = true;
                        uint position = count;
                        while (position > 0)
                        {
                            uint before = position - 1u;
                            if (depth > candidateDepths[before]
                                || (depth == candidateDepths[before] && id >= candidateIDs[before])) break;
                            if (position < FRUSTUM_CANDIDATE_BATCH_SIZE)
                            {
                                candidateIDs[position] = candidateIDs[before];
                                candidateCells[position] = candidateCells[before];
                                candidateDepths[position] = candidateDepths[before];
                            }
                            --position;
                        }
                        if (position < FRUSTUM_CANDIDATE_BATCH_SIZE)
                        {
                            candidateIDs[position] = id;
                            candidateCells[position] = uint(x) + voxelCount.x * (uint(y) + voxelCount.y * uint(z));
                            candidateDepths[position] = depth;
                            count = min(count + 1u, uint(FRUSTUM_CANDIDATE_BATCH_SIZE));
                        }
                    }
            for (uint candidate = 0; candidate < count; ++candidate)
            {
                previousID = candidateIDs[candidate];
                previousDepth = candidateDepths[candidate];
                uint linearCell = candidateCells[candidate];
                int3 cell = int3(linearCell % voxelCount.x, (linearCell / voxelCount.x) % voxelCount.y,
                    linearCell / (voxelCount.x * voxelCount.y));
                VoxelData data = VOXEL_DATA(uint(previousID));
                float coverage;
                #if !FRUSTUM_DISABLE_FAST_PATH
                if (ellipsoidOutsidePixel(data.ellipsoid, cell, planes)) continue;
                if (frustumPixelFullyCovered(data.ellipsoid, cell, rays)) coverage = 1.0f;
                else
                #endif
                coverage = clippedEllipsoidPixelCoverage(data.ellipsoid, cell, gridMin, voxelSize,
                    viewProjection, pixelCount, pixel);
                if (coverage <= 0.0f) continue;
                float opacity = coverage * saturate(data.opacity.calcOpacity(-direction));
                float3 radiance = max(data.radiance.calc(-direction), float3(0));
                color += transmittance * opacity * radiance;
                transmittance *= 1.0f - opacity;
                if (transmittance <= transmittanceThreshold)
                    return float4(color + transmittance * background, 1.0f - transmittance);
            }
            if (!overflow) break;
        }
    }
    return float4(color + transmittance * background, 1.0f - transmittance);
}
