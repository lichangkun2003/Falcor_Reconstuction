
// Input
Texture2D<float4> gRenderedColor;
Texture2D<float4> gReferenceImage;

// Output
RWTexture2D<float> gLossBuffer;
RWTexture2D<float4> gDL_dColorBuffer;
RWTexture2D<uint> gBackgroundMaskBuffer;

cbuffer CB {
    uint2 gResolution;
}



[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;

    if (pixel.x >= gResolution.x || pixel.y >= gResolution.y)
        return;

    float3 C = gRenderedColor.Load(int3(pixel, 0)).rgb;
    float3 R = gReferenceImage.Load(int3(pixel, 0)).rgb;

    float3 diff = C - R;

    float loss = dot(diff, diff);
    float3 dL_dColor = 2.0 * diff;

    gLossBuffer[pixel] = loss;
    gBackgroundMaskBuffer[pixel] = 0;
    if (all(R == float3(1.0f)))
        gBackgroundMaskBuffer[pixel] = 1;

    gDL_dColorBuffer[pixel] = float4(dL_dColor, 0.0);
}


