#pragma once
#include "../HostDeviceShared.slang"
#include "Sampling.slang"
#include "SphericalHarmonics.slang"

#if !defined(HOST_CODE)
#include "Random.slang"
#endif

BEGIN_NAMESPACE_FALCOR

// 忽略罕见的7/8/9边形情况
#define MAX_VERTEX_COUNT 6

// 记录多边形所属的面片
struct TriangleRef
{
    uint meshID;
    uint triangleID;
    uint materialID;
    MUTATING
    void init()
    {
        meshID = 0;
        triangleID = 0;
        materialID = 0;
    }
};

struct Polygon
{
    TriangleRef triRef;
    float3 vertices[MAX_VERTEX_COUNT];
    uint count;
    float3 normal;

    MUTATING
    void init() { count = 0; }

#if defined(HOST_CODE)
    // 顶点必须有序,必须至少包含三个点,必须是凸多边形，不能有重合的点
    void init(std::vector<float3>& points)
    {
        uint n = math::min(MAX_VERTEX_COUNT, (int)points.size());
        count = n;
        for (uint i = 0; i < n; i++)
        {
            vertices[i] = points[i];
        }
        triRef.init();
    }

    void addTo(std::vector<float3>& points)
    {
        for (uint i = 0; i < count; i++)
        {
            points.push_back(vertices[i]);
        }
    }
#endif

    float calcArea()
    {
        float3 s = float3(0);
        for (uint i = 0; i < count; ++i)
        {
            float3 a = vertices[i];
            float3 b = vertices[(i + 1) % count];
            s += cross(a, b);
        }
        return sqrt(dot(s, s)) / 2;
    }

    float calcProjArea(float3 direction) { return calcArea() * abs(dot(direction, normal)); }
#if defined(HOST_CODE)
    float3 calcCentroid(float& totalArea) const
#else
    float3 calcCentroid(out float totalArea)
#endif
    {
        // 用顶点0做三角扇分解
        float3 v0 = vertices[0];
        float3 sum = float3(0);
        totalArea = 0;

        for (uint i = 1; i + 1 < count; ++i)
        {
            float3 a = v0;
            float3 b = vertices[i];
            float3 c = vertices[i + 1];

            float3 ab = b - a;
            float3 ac = c - a;
            float3 s = cross(ab, ac);
            float area = sqrt(dot(s, s)) * 0.5f;
            if (area <= 0.0f)
                continue;

            float3 triCenter = (a + b + c) / 3.0f;
            sum += triCenter * area;
            totalArea += area;
        }

        if (totalArea <= 0)
            return v0;

        return sum / totalArea;
    }

    MUTATING
    void add(float3 point)
    {
        if (count < MAX_VERTEX_COUNT)
        {
            vertices[count] = point;
            count++;
        }
    }
};

struct PolygonRange
{
    uint localHead;
    uint count;
    int3 cellInt;

    MUTATING
    void init(int3 c)
    {
        localHead = 0;
        count = 0;
        cellInt = c;
    }

#if !defined(HOST_CODE)
    float3 calcCentroid(StructuredBuffer<Polygon> polygons, out float totalArea)
    {
        float3 sum = float3(0);
        totalArea = 0;

        for (uint i = 0; i < count; ++i)
        {
            float area;
            float3 c = polygons[i + localHead].calcCentroid(area);
            sum += c * area;
            totalArea += area;
        }

        if (totalArea <= 0)
            return cellInt + float3(0.5f);

        return sum / totalArea;
    }

    float calcTotalProjArea(StructuredBuffer<Polygon> polygons, float3 direction)
    {
        float sum = 0;
        for (uint i = 0; i < count; ++i)
        {
            sum += polygons[i + localHead].calcProjArea(direction);
        }
        return sum;
    }

    uint calcVisibleProjCellsRaster(StructuredBuffer<Polygon> polygons, float3 direction)
    {
        Basis2 b = orthonormal_basis(normalize(direction));

        float3 cubeCenter = float3(cellInt) + float3(0.5f);

        uint mask[PROJ_WORDS];
        [unroll]
        for (uint i = 0; i < PROJ_WORDS; ++i)
            mask[i] = 0u;

        for (uint pi = 0; pi < count; ++pi)
        {
            Polygon poly = polygons[localHead + pi];
            uint n = poly.count;
            if (n < 3)
                continue;

            float2 pv[MAX_VERTEX_COUNT];

            [unroll]
            for (uint k = 0; k < MAX_VERTEX_COUNT; ++k)
            {
                if (k >= n)
                    break;
                float3 p = poly.vertices[k] - cubeCenter;
                pv[k] = float2(dot(p, b.u), dot(p, b.w));
            }

            float2 v0 = pv[0];
            for (uint k = 1; k + 1 < n; ++k)
            {
                rasterizeTriToMask(v0, pv[k], pv[k + 1], mask);
            }
        }

        return maskCountBits(mask);
    }

    // 通过类光栅化的过程计算可见投影面积
    float calcVisibleProjAreaRaster(StructuredBuffer<Polygon> polygons, float3 direction)
    {
        uint covered = calcVisibleProjCellsRaster(polygons, direction);
        return float(covered) * (PROJ_PIXEL_SIZE * PROJ_PIXEL_SIZE);
    }
#endif
};

END_NAMESPACE_FALCOR
