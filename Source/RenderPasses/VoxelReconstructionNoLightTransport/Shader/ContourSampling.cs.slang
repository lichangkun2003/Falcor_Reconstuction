#include "../ContourSampling.slang"

Texture2D<float4> gReference;
RWStructuredBuffer<uint> gCounter;
RWStructuredBuffer<uint> gPixels;
RWStructuredBuffer<ContourHistory> gHistory;
RWStructuredBuffer<ContourObservation> gObservations;
RWTexture2D<uint> gLookup;
cbuffer CB
{
    uint2 gResolution;
    uint gCount;
    float gHistoryRate;
};

[numthreads(8, 8, 1)]
void findContours(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gResolution)) return;
    if (!isContourReference(gReference[id.xy].a)) return;
    uint slot;
    InterlockedAdd(gCounter[0], 1u, slot);
    gPixels[slot] = id.x + id.y * gResolution.x;
}

[numthreads(256, 1, 1)]
void initializeHistory(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gCount) return;
    uint index = gPixels[id.x];
    float prior = gReference[uint2(index % gResolution.x, index / gResolution.x)].a;
    ContourHistory history;
    history.pixelIndex = index;
    for (uint b = 0u; b < 16u; ++b) history.meanAlpha[b] = prior;
    gHistory[id.x] = history;
}

[numthreads(256, 1, 1)]
void prepareBatch(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gCount) return;
    uint index = gHistory[id.x].pixelIndex;
    gLookup[uint2(index % gResolution.x, index / gResolution.x)] = id.x + 1u;
    for (uint b = 0u; b < 16u; ++b)
        gObservations[id.x * 16u + b] = (ContourObservation)0;
}

[numthreads(256, 1, 1)]
void finishBatch(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gCount) return;
    ContourHistory history = gHistory[id.x];
    for (uint b = 0u; b < 16u; ++b)
    {
        ContourObservation observation = gObservations[id.x * 16u + b];
        // Unvisited bins retain their old estimate. Visit count does not
        // increase importance; use the conditional mean of RAW ray alpha.
        if (observation.count > 0u)
            history.meanAlpha[b] = lerp(history.meanAlpha[b], observation.alphaSum / float(observation.count), gHistoryRate);
    }
    gHistory[id.x] = history;
}
