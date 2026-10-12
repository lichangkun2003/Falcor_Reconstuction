import Utils.Sampling.UniformSampleGenerator;
#include "../ContourSampling.slang"
StructuredBuffer<ContourHistory> gHistory;
RWStructuredBuffer<float4> gSamples;
RWStructuredBuffer<float4> gEstimates;
cbuffer CB { uint gCount; float gUniformFraction; };

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gCount) return;
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(17, 31), id.x);
    uint bin;
    float inversePdf;
    float2 uv = sampleContour(gHistory[0], gUniformFraction, sampleNext3D(sg), bin, inversePdf);
    gSamples[id.x] = float4(uv, inversePdf, float(bin));
    // A thin rectangle occupying 1/160 of a pixel, entirely in bin zero.
    bool hit = uv.x < 0.025f && uv.y < 0.25f;
    // Test white, black, and spatially varying backgrounds simultaneously.
    float3 color = hit ? float3(0.2f, 0.4f, 0.6f) : float3(1.0f, 0.0f, uv.x);
    gEstimates[id.x] = contourEstimate(float4(color, hit ? 1.0f : 0.0f), inversePdf);
}
