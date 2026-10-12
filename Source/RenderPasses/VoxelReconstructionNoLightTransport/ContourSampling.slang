#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

// Only reference pixels with 0 < alpha < 1 allocate persistent history.
struct ContourHistory
{
    uint pixelIndex;
    float meanAlpha[16];
};

// Shared scratch for the currently visited camera, never part of its proposal.
struct ContourObservation
{
    float alphaSum;
    uint count;
};

#if defined(HOST_CODE)
static_assert(sizeof(ContourHistory) == 68 && sizeof(ContourObservation) == 8, "Contour history layout changed.");
#else
bool isContourReference(float alpha)
{
    return alpha > 0.0f && alpha < 1.0f;
}

void contourProbabilities(ContourHistory history, float uniformFraction, out float probabilities[16])
{
    float sum = 0.0f;
    for (uint b = 0u; b < 16u; ++b) sum += max(history.meanAlpha[b], 0.0f);
    float epsilon = clamp(uniformFraction, 0.05f, 1.0f);
    for (uint b = 0u; b < 16u; ++b)
        probabilities[b] = sum > 0.0f ? epsilon / 16.0f + (1.0f - epsilon) * max(history.meanAlpha[b], 0.0f) / sum : 1.0f / 16.0f;
}

float2 sampleContour(ContourHistory history, float uniformFraction, float3 random,
    out uint bin, out float inversePdf)
{
    float probabilities[16];
    contourProbabilities(history, uniformFraction, probabilities);
    float cumulative = 0.0f;
    bin = 15u;
    for (uint b = 0u; b < 16u; ++b)
    {
        cumulative += probabilities[b];
        if (random.x < cumulative) { bin = b; break; }
    }
    // PDF with respect to unit pixel area, including the uniform mixture.
    inversePdf = 1.0f / (16.0f * probabilities[bin]);
    return (float2(bin % 4u, bin / 4u) + random.yz) / 4.0f;
}

float4 contourEstimate(float4 raw, float inversePdf)
{
    if (inversePdf == 1.0f) return raw;
    // A fixed white control variate has a known pixel integral of one, even
    // when the actual envmap varies. Never substitute a sampled env direction.
    // Do not clamp this estimator: corrected RGB/alpha may lie outside [0,1].
    return float4(float3(1.0f) + (raw.rgb - float3(1.0f)) * inversePdf, raw.a * inversePdf);
}
#endif

END_NAMESPACE_FALCOR
