#include "../Voxel/VoxelGrid.slang"

// Must match PointCloudInitialization::Seed on the host (4 bytes).
struct PointCloudSeed
{
    uint voxelIndex;
};
StructuredBuffer<PointCloudSeed> gSeeds;
cbuffer CB
{
    uint gSeedCount;
    uint gSeedOffset;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint seedIndex = gSeedOffset + dispatchThreadId.x;
    if (seedIndex >= gSeedCount) return;
    PointCloudSeed seed = gSeeds[seedIndex];
    uint3 count = gGridDataParamBlock.voxelCount;
    if (seed.voxelIndex >= count.x * count.y * count.z) return;

    // Empty cells were cleared on the host. Each occupied index occurs exactly once.
    VoxelData data = (VoxelData)0;
    data.occupied = 1;
    data.ellipsoid.center = float3(0.5f);
    float invR2 = 1.f / (0.6f * 0.6f);
    data.ellipsoid.B = float3x3(invR2, 0.f, 0.f, 0.f, invR2, 0.f, 0.f, 0.f, invR2);
    // Point clouds determine occupancy only. All radiance SH coefficients start at zero (black).
    data.radiance.init();
    // Same opacity.init() as mode 3: zero logit SH, sigmoid(0) = 0.5.
    // Initialize independently of the optimization learning rates.
    data.opacity.init();
    gGridDataParamBlock.gridDataBuffer[seed.voxelIndex] = data;
}
