
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

    gBackgroundMaskBuffer[pixel] = 0;

    float4 rendered = gRenderedColor.Load(int3(pixel, 0));
    float3 C = rendered.rgb;
    float renderAlpha = rendered.a;

    float4 ref = gReferenceImage.Load(int3(pixel, 0));
    float refAlpha = ref.a;
    float3 R = ref.rgb * refAlpha + float3(1.0f) * (1.0f - refAlpha);


    // RGB loss
    float3 diff = C - R;
    float rgbLoss = dot(diff, diff);
    float3 dL_dColor = 2.0f * diff;

    // Alpha loss
    const float alphaLossWeight = 0.3f;

    float alphaLoss = 0.0f;
    float dL_dAlpha = 0.0f;

    // Pure foreground
    if (refAlpha > 0.99f)
    {
        float alphaDiff = renderAlpha - 1.0f;

        alphaLoss =
            alphaLossWeight * alphaDiff * alphaDiff;

        dL_dAlpha =
            2.0f * alphaLossWeight * alphaDiff;
    }
    // Pure background
    else if (refAlpha < 0.01f)
    {
        float alphaDiff = renderAlpha;

        alphaLoss =
            alphaLossWeight * alphaDiff * alphaDiff;

        dL_dAlpha =
            2.0f * alphaLossWeight * alphaDiff;
    }

    // Total loss
    gLossBuffer[pixel] = rgbLoss + alphaLoss;

    gBackgroundMaskBuffer[pixel] = refAlpha < 1e-4f ? 1u : 0u;

    // xyz = RGB gradient
    // w   = alpha gradient
    gDL_dColorBuffer[pixel] = float4(dL_dColor, dL_dAlpha);
}


