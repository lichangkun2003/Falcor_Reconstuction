#include "../GradRecord.slang"
#include "../PathRecord.slang"
// Input
Texture2D<float4> gDL_dColorBuffer;
StructuredBuffer<PathRecord> gPathRecordBuffer;

// Output
RWStructuredBuffer<GradRecord> gGradBuffer;


cbuffer CB {
    uint2 gResolution;
}



[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;

    if (pixel.x >= gResolution.x || pixel.y >= gResolution.y)
        return;

    uint pixelIndex = pixel.y * gResolution.x + pixel.x;

    PathRecord rec = gPathRecordBuffer[pixelIndex];

    if (rec.valid == 0)
        return;

    float3 dL_dC = gDL_dColorBuffer.Load(int3(pixel, 0)).rgb;



}


