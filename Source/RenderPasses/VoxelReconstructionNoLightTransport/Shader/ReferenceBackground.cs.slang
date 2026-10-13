// The sRGB reference texture is decoded to linear RGB when loaded. Match LossPass.
Texture2D<float4> gReferenceImage;
RWTexture2D<float4> gOutput;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    uint2 outputSize;
    gOutput.GetDimensions(outputSize.x, outputSize.y);
    if (any(pixel >= outputSize))
        return;

    uint2 referenceSize;
    gReferenceImage.GetDimensions(referenceSize.x, referenceSize.y);

    // Exact texel reads at native resolution. Use nearest pixel centers when
    // previewing at another resolution, without filtering across alpha edges.
    uint2 referencePixel = min(
        uint2((float2(pixel) + 0.5f) * float2(referenceSize) / float2(outputSize)), referenceSize - 1u
    );
    float4 reference = gReferenceImage.Load(int3(referencePixel, 0));
    float3 color = reference.rgb * reference.a + float3(1.0f) * (1.0f - reference.a);

    // Already composited: RGB-only capture and UI display must agree on the background.
    gOutput[pixel] = float4(color, 1.0f);
}
