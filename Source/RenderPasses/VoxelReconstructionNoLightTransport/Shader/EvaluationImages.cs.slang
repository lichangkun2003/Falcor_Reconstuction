Texture2D<float4> gColor;
RWTexture2D<float4> gRgb;
RWTexture2D<float4> gAlpha;
cbuffer CB { uint2 gResolution; }

float linearToSrgb(float x)
{
    x = saturate(x);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.0f / 2.4f) - 0.055f;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid.xy >= gResolution)) return;
    float4 color = gColor[tid.xy];
    gRgb[tid.xy] = float4(linearToSrgb(color.r), linearToSrgb(color.g), linearToSrgb(color.b), 1.0f);
    gAlpha[tid.xy] = float4(saturate(color.aaa), 1.0f);
}
