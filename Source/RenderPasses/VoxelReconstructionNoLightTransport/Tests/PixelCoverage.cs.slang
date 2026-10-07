#include "../Math/PixelCoverage.slang"
cbuffer CB
{
    GaussianEllipsoid ellipsoid;
    float4x4 vp;
}
RWStructuredBuffer<float> result;
[numthreads(1, 1, 1)]
void main()
{
    result[0] = clippedEllipsoidPixelCoverage(ellipsoid, int3(0), float3(0), float3(1), vp, uint2(1), uint2(0));
}
