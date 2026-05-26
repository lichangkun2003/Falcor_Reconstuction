StructuredBuffer<float> gInput;
RWStructuredBuffer<float> gOutput;

cbuffer CB
{
    uint gInputCount;
};

groupshared float sData[256];

[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint3 groupThreadId    : SV_GroupThreadID,
    uint3 groupId          : SV_GroupID)
{
    uint tid = groupThreadId.x;
    uint index = dispatchThreadId.x;

    float v = 0.0f;
    if (index < gInputCount)
    {
        v = gInput[index];
    }

    sData[tid] = v;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sData[tid] += sData[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        gOutput[groupId.x] = sData[0];
    }
}
