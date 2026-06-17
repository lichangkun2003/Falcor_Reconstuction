#pragma once
#include "../HostDeviceShared.slang"
#include "AABB.slang"

BEGIN_NAMESPACE_FALCOR

#if !defined(HOST_CODE)
inline void buildTBN(out float3 t, out float3 b, float3 n)
{
    float3 up = (abs(n.z) < 0.999f) ? float3(0, 0, 1) : float3(1, 0, 0);
    t = normalize(cross(up, n));
    b = cross(n, t);
}
inline float3 localToWorld(float3 localDir, float3 t, float3 b, float3 n)
{
    return localDir.x * t + localDir.y * b + localDir.z * n;
}
inline float3 worldToLocal(float3 worldDir, float3 t, float3 b, float3 n)
{
    return float3(dot(worldDir, t), dot(worldDir, b), dot(worldDir, n));
}
#endif

inline float3 calcShadingNormal(float3x3 TBN, float3 normal)
{
    return safeNormalize(mul(TBN, normal));
}

inline float3 calcBarycentricCoordinates(float3 a, float3 b, float3 c, float3 p)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d00 = dot(ab, ab);
    float d01 = dot(ab, ac);
    float d11 = dot(ac, ac);
    float d20 = dot(ap, ab);
    float d21 = dot(ap, ac);

    float denom = d00 * d11 - d01 * d01;

    if (abs(denom) < 1e-8)
        return float3(1) / float3(3);

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1 - v - w;

    return float3(u, v, w);
}

// 返回点p到三角形abc的最近点，及其重心坐标
#if defined(HOST_CODE)
inline float3 closestPointOnTriangle(float3 p, float3 a, float3 b, float3 c, float3& bary)
#else
inline float3 closestPointOnTriangle(float3 p, float3 a, float3 b, float3 c, out float3 bary)
#endif
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);

    // Vertex region A
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        bary = float3(1, 0, 0);
        return a;
    }

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);

    // Vertex region B
    if (d3 >= 0.0f && d4 <= d3)
    {
        bary = float3(0, 1, 0);
        return b;
    }

    float vc = d1 * d4 - d3 * d2;
    // Edge region AB
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float v = d1 / (d1 - d3);
        bary = float3(1 - v, v, 0);
        return a + v * ab;
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);

    // Vertex region C
    if (d6 >= 0.0f && d5 <= d6)
    {
        bary = float3(0, 0, 1);
        return c;
    }

    float vb = d5 * d2 - d1 * d6;
    // Edge region AC
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float w = d2 / (d2 - d6);
        bary = float3(1 - w, 0, w);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    // Edge region BC
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        bary = float3(0, 1 - w, w);
        return b + w * (c - b);
    }

    // Inside face region
    float denom = max(va + vb + vc, 1e-20f);
    float v = vb / denom;
    float w = vc / denom;
    bary = float3(1 - v - w, v, w);
    return a + ab * v + ac * w;
}

struct Triangle
{
    float3 vertices[3];
    float3 normals[3];
    float2 uvs[3];
    float3x3 TBN;

    MUTATING
    void init()
    {
        for (int i = 0; i < 3; i++)
        {
            vertices[i] = float3(0);
            uvs[i] = float2(0);
            normals[i] = float3(0);
        }
    }

#if defined(HOST_CODE)
    void addTo(std::vector<float3>& points)
    {
        for (uint i = 0; i < 3; i++)
        {
            points.push_back(vertices[i]);
        }
    }
#endif

    // 判断点是否在三角形内（必须共面）
    bool contains2D(float3 p)
    {
        float3 normal = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
        float s0 = dot(normal, cross(vertices[0] - p, vertices[1] - p));
        float s1 = dot(normal, cross(vertices[1] - p, vertices[2] - p));
        float s2 = dot(normal, cross(vertices[2] - p, vertices[0] - p));
        return (s0 >= 0) && (s1 >= 0) && (s2 >= 0);
    }

    float3 barycentricCoordinates(float3 p) { return calcBarycentricCoordinates(vertices[0], vertices[1], vertices[2], p); }

    float2 lerpUV(float3 p)
    {
        float3 coord = barycentricCoordinates(p);
        return uvs[0] * coord.x + uvs[1] * coord.y + uvs[2] * coord.z;
    }

    float3 lerpNormal(float3 p)
    {
        float3 coord = barycentricCoordinates(p);
        return normalize(normals[0] * coord.x + normals[1] * coord.y + normals[2] * coord.z);
    }

    AABBInt calcAABBInt()
    {
        AABBInt aabb;
        aabb.init();
        for (int i = 0; i < 3; i++)
        {
            aabb.accumulate((int3)floor(vertices[i]));
        }
        aabb.complete();
        return aabb;
    }

    MUTATING
    void buildTBN()
    {
        float3 e1 = vertices[1] - vertices[0];
        float3 e2 = vertices[2] - vertices[0];
        float2 duv1 = uvs[1] - uvs[0];
        float2 duv2 = uvs[2] - uvs[0];

        float D = duv1.x * duv2.y - duv1.y * duv2.x;
        float3 N = cross(e1, e2);
        float len = length(N);
        if (len == 0)
        {
            N = normals[0] + normals[1] + normals[2];
            len = length(N);
            if (len == 0)
            {
                N = float3(0, 0, 1);
                len = 1;
            }
        }
        N /= len;
        float3 T, B;
        if (abs(D) < 1e-8)
        {
            T = normalize(e1 - N * dot(N, e1));
            B = normalize(cross(N, T));
        }
        else
        {
            float r = 1.0 / D;
            float3 Tp = (e1 * duv2.y - e2 * duv1.y) * r;
            float3 Bp = (e2 * duv1.x - e1 * duv2.x) * r;

            T = normalize(Tp - N * dot(N, Tp));
            B = normalize(Bp - N * dot(N, Bp));
        }

#if defined(HOST_CODE)
        TBN.setCol(0, T);
        TBN.setCol(1, B);
        TBN.setCol(2, N);
#else
        TBN = float3x3(T, B, N);
        TBN = transpose(TBN);
#endif
    }
};

END_NAMESPACE_FALCOR
