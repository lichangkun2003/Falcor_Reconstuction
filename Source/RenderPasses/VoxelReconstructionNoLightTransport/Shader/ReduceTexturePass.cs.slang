Texture2D<float> gLossBuffer;
RWStructuredBuffer<float> gPartialLoss;

cbuffer CB
{
    uint2 gResolution;
    uint2 gNumGroups;
};

groupshared float sLoss[256];

[numthreads(16, 16, 1)]
void main(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint3 groupThreadId    : SV_GroupThreadID,
    uint3 groupId          : SV_GroupID)
{
    uint localIndex = groupThreadId.y * 16 + groupThreadId.x;
    uint2 pixel = dispatchThreadId.xy;

    float loss = 0.0f;
    if (pixel.x < gResolution.x && pixel.y < gResolution.y)
    {
        loss = gLossBuffer.Load(int3(pixel, 0));
    }

    sLoss[localIndex] = loss;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (localIndex < stride)
        {
            sLoss[localIndex] += sLoss[localIndex + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0)
    {
        uint groupIndex = groupId.y * gNumGroups.x + groupId.x;
        gPartialLoss[groupIndex] = sLoss[0];
    }
}
