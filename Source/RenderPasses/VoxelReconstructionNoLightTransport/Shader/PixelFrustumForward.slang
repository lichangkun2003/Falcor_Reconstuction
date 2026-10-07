#pragma once
#include "../Math/PixelCoverage.slang"

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

void frustumSlabCellBounds(uint2 pixel, float depthMin, float depthMax, out int3 cellMin, out int3 cellMax)
{
    float3 boundsMin = float3(1e30f);
    float3 boundsMax = float3(-1e30f);
    for (uint corner = 0; corner < 4; ++corner)
    {
        float2 uv = (float2(pixel) + float2(corner & 1u, corner >> 1u)) / float2(pixelCount);
        float3 nearPoint = screenCoordToCell(uv, 0.0f);
        float3 farPoint = screenCoordToCell(uv, 1.0f);
        // Use actual unprojected endpoint depths, avoiding far-plane roundoff.
        float nearDepth = dot(gridMin + nearPoint * voxelSize - cameraPosition, cameraForward);
        float farDepth = dot(gridMin + farPoint * voxelSize - cameraPosition, cameraForward);
        float3 slope = (farPoint - nearPoint) / (farDepth - nearDepth);
        float3 a = nearPoint + (depthMin - nearDepth) * slope;
        float3 b = nearPoint + (depthMax - nearDepth) * slope;
        boundsMin = min(boundsMin, min(a, b));
        boundsMax = max(boundsMax, max(a, b));
    }
    cellMin = max(int3(floor(boundsMin - 1e-4f)), int3(0));
    cellMax = min(int3(floor(boundsMax + 1e-4f)), int3(voxelCount) - 1);
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
        frustumSlabCellBounds(pixel, max(cameraDepthRange.x, slabMin - cellHalfDepth),
            min(cameraDepthRange.y, slabMax + cellHalfDepth), lo, hi);
        float previousDepth = -1e30f;
        int previousID = -1;
        // Repeated minimum selection gives deterministic center-depth order,
        // with no bounded candidate array and no silent truncation. Debug first;
        // a block hierarchy and buffered sort can replace this search later.
        while (true)
        {
            int nextID = -1;
            int3 nextCell = int3(0);
            float nextDepth = 1e30f;
            for (int z = lo.z; z <= hi.z; ++z)
                for (int y = lo.y; y <= hi.y; ++y)
                    for (int x = lo.x; x <= hi.x; ++x)
                    {
                        int3 cell = int3(x, y, z);
                        float depth = clamp(cellCameraDepth(cell), depthMin, depthMax);
                        uint owner = min(uint(max(0.0f, floor((depth - depthMin) / slabWidth))), slabCount - 1u);
                        if (owner != slab) continue;
                        if (!cellOverlapsPixelFrustum(cell, planes)) continue;
                        int id = lookupVoxelID(cell);
                        if (id < 0 || VOXEL_DATA(uint(id)).occupied == 0u) continue;
                        bool afterPrevious = depth > previousDepth || (depth == previousDepth && id > previousID);
                        bool beforeNext = depth < nextDepth || (depth == nextDepth && (nextID < 0 || id < nextID));
                        if (afterPrevious && beforeNext)
                        {
                            nextID = id;
                            nextCell = cell;
                            nextDepth = depth;
                        }
                    }
            if (nextID < 0) break;
            previousID = nextID;
            previousDepth = nextDepth;
            VoxelData data = VOXEL_DATA(uint(nextID));
            float coverage = clippedEllipsoidPixelCoverage(data.ellipsoid, nextCell, gridMin, voxelSize,
                viewProjection, pixelCount, pixel);
            if (coverage <= 0.0f) continue;
            float opacity = coverage * saturate(data.opacity.calcOpacity(-direction));
            float3 radiance = max(data.radiance.calc(-direction), float3(0));
            color += transmittance * opacity * radiance;
            transmittance *= 1.0f - opacity;
            if (transmittance <= transmittanceThreshold)
                return float4(color + transmittance * background, 1.0f - transmittance);
        }
    }
    return float4(color + transmittance * background, 1.0f - transmittance);
}
