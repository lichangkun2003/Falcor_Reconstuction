
// Input
Texture2D<float4> gRenderedColor;
// 本帧（整批的最后一个采样）未乘 invSpp 的渲染结果，只用于剥出前缀平均。
Texture2D<float4> gCurrentFrameColor;
Texture2D<float4> gReferenceImage;

// Output
RWTexture2D<float> gLossBuffer;
RWTexture2D<float4> gDL_dColorBuffer;
RWTexture2D<uint> gBackgroundMaskBuffer;

cbuffer CB {
    uint2 gResolution;
    uint gSpp;
}



[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;

    if (pixel.x >= gResolution.x || pixel.y >= gResolution.y)
        return;

    gBackgroundMaskBuffer[pixel] = 0;

    // 前缀平均（leave-one-out）：
    // 累积图在整批跑完后是 (1/N) * sum_{k<N} C_k，而 gCurrentFrameColor 是未缩放的 C_{N-1}。
    // 所以 (N * accu - C_{N-1}) / (N - 1) 就是前 N-1 帧的平均。
    //
    // 这样上游梯度只依赖 xi_0..xi_{N-2}，而 gradient pass 反传的是第 N-1 帧的 PathRecord
    // （只依赖 xi_{N-1}），两者独立，E[grad] 才严格等于 grad L_avg。
    // 反之若直接用累积图，它与被反传的那一帧同源，会引入 + grad Var(C) 的偏差，
    // 该偏差的最优解就是把几何吃掉、把 alpha 摊成覆盖率。
    //
    // 不变式：本 pass 必须在一整批 gSpp 个采样全部渲染完之后才调用。
    // N == 1 时前缀为空（N * accu - C_0 == 0），必须回退到累积图本身。
    float4 rendered = gRenderedColor.Load(int3(pixel, 0));
    if (gSpp >= 2u)
    {
        float4 current = gCurrentFrameColor.Load(int3(pixel, 0));
        rendered = (float(gSpp) * rendered - current) / float(gSpp - 1u);
    }

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

    float alphaDiff = renderAlpha - refAlpha;
    float alphaLoss = alphaLossWeight * alphaDiff * alphaDiff;
    float dL_dAlpha = 2.0f * alphaLossWeight * alphaDiff;


    // Total loss
    gLossBuffer[pixel] = rgbLoss + alphaLoss;

    gBackgroundMaskBuffer[pixel] = refAlpha < 1e-4f ? 1u : 0u;

    // xyz = RGB gradient
    // w   = alpha gradient
    gDL_dColorBuffer[pixel] = float4(dL_dColor, dL_dAlpha);
}


