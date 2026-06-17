#pragma once
#include "../HostDeviceShared.slang"

#if !defined(HOST_CODE)
#define INT_MAX 1000000000
#define INT_MIN -1000000000
#endif

BEGIN_NAMESPACE_FALCOR

struct AABBInt
{
    int xMin, xMax, yMin, yMax, zMin, zMax;
    int deltaX, deltaY, deltaZ, deltaXY;

    MUTATING
    void init()
    {
        xMin = yMin = zMin = INT_MAX;
        xMax = yMax = zMax = INT_MIN;
        deltaX = deltaY = deltaZ = deltaXY = 0;
    }

    MUTATING
    void accumulate(int3 point)
    {
        xMin = min(xMin, point.x);
        yMin = min(yMin, point.y);
        zMin = min(zMin, point.z);
        xMax = max(xMax, point.x);
        yMax = max(yMax, point.y);
        zMax = max(zMax, point.z);
    }

    MUTATING
    void complete()
    {
        deltaX = xMax - xMin + 1;
        deltaY = yMax - yMin + 1;
        deltaZ = zMax - zMin + 1;
        deltaXY = deltaX * deltaY;
    }

    int count()
    {
        return deltaX * deltaY * deltaZ;
    }

    int3 indexToCell(int index)
    {
        int zLocal = index / deltaXY;
        int temp = index % deltaXY;
        int yLocal = temp / deltaX;
        int xLocal = temp % deltaX;

        return int3(
            xMin + xLocal,
            yMin + yLocal,
            zMin + zLocal
        );
    }

    int cellToIndex(int3 cell)
    {
        return (cell.z - zMin) * deltaXY + (cell.y - yMin) * deltaX + (cell.x - xMin);
    }
};

inline float2 intersect(float p1, float p2, float p0, float q)
{
    float u1 = (p1 - p0) / q;
    float u2 = (p2 - p0) / q;
    if (u1 > u2)
    {
        float temp = u1;
        u1 = u2;
        u2 = temp;
    }
    return float2(u1, u2);
}

inline float2 clip(float3 minPoint, float3 maxPoint, float3 from, float3 to)
{
    float uIn = 0, uOut = 1;
    float3 v = to - from;

    float2 u12 = intersect(minPoint.x, maxPoint.x, from.x, v.x);
    uIn = max(uIn, u12.x);
    uOut = min(uOut, u12.y);
    if (uIn > uOut)
        return float2(-1, -1);

    u12 = intersect(minPoint.y, maxPoint.y, from.y, v.y);
    uIn = max(uIn, u12.x);
    uOut = min(uOut, u12.y);
    if (uIn > uOut)
        return float2(-1, -1);

    u12 = intersect(minPoint.z, maxPoint.z, from.z, v.z);
    uIn = max(uIn, u12.x);
    uOut = min(uOut, u12.y);
    if (uIn > uOut)
        return float2(-1, -1);

    return float2(uIn, uOut);
}

END_NAMESPACE_FALCOR
