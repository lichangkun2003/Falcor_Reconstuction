
// Input
// 累积图：第 k 帧时是 (1/N) * sum_{j<=k} C_j。
Texture2D<float4> gRenderedColor;
// 本帧未乘 invSpp 的渲染结果 C_k，用于从累积图里剥出前缀平均。
Texture2D<float4> gCurrentFrameColor;
Texture2D<float4> gReferenceImage;

// Output
RWTexture2D<float> gLossBuffer;
RWTexture2D<float4> gDL_dColorBuffer;
RWTexture2D<uint> gBackgroundMaskBuffer;

cbuffer CB {
    uint2 gResolution;
    // 一整批的目标采样数 N，也是累积图的缩放分母（invSpp == 1/N）。
    uint gSpp;
    // 到当前帧为止这一批里已经累积的采样数，第 k 帧时等于 k + 1。
    uint gSampleCount;
}



[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;

    if (pixel.x >= gResolution.x || pixel.y >= gResolution.y)
        return;

    gBackgroundMaskBuffer[pixel] = 0;

    // 前缀平均（leave-one-out），整批内逐帧计算：
    // 第 k 帧时累积图是 (1/N) * sum_{j<=k} C_j（gSpp == N，累积器永远按 1/N 累加），
    // 而 gCurrentFrameColor 是未缩放的 C_k，所以 (N * accu - C_k) / k 就是前 k 帧的平均，
    // 也就是不含当前帧的前缀平均；k == N-1 时退化成 (N * accu - C_{N-1}) / (N - 1)，
    // 与只调一次的那个实现逐字节一致。
    //
    // 这样上游梯度只依赖 xi_0..xi_{k-1}，而 gradient pass 反传的是第 k 帧的 PathRecord
    // （只依赖 xi_k），两者独立，E[grad] 才严格等于 grad L_avg。
    // 整批内每一帧都这样算一次，梯度在 gradBuffer 里原子累加，最后一次按计数求平均。
    // 反之若直接用累积图，它与被反传的那一帧同源，会引入 + grad Var(C) 的偏差，
    // 该偏差的最优解就是把几何吃掉、把 alpha 摊成覆盖率。
    //
    // 不变式：第 k 帧调用时 gSampleCount == k + 1（调用方传的就是本帧渲染完之后
    // 已经自增过的 mSampleIndex），所以除数 gSampleCount - 1 就是 k。
    // k == 0 时前缀为空（N * accu - C_0 == 0，除数为 0），直接用本帧的 C_0；
    // 累积图此时是 C_0 / N，只有 N == 1 才等于 C_0，所以回退不能用累积图。
    float4 current = gCurrentFrameColor.Load(int3(pixel, 0));
    float4 rendered = current;
    if (gSampleCount >= 2u)
    {
        rendered = (float(gSpp) * gRenderedColor.Load(int3(pixel, 0)) - current) / float(gSampleCount - 1u);
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


