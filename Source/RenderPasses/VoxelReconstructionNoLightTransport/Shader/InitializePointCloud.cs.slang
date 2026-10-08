#include "../Voxel/VoxelGrid.slang"

// Must match PointCloudInitialization::Seed on the host (56 bytes).
struct PointCloudSeed
{
    uint voxelIndex;
    float3 center;
    float3 scale;
    float4 rotation;
    float3 color;
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
    if (seedIndex >= gGridDataParamBlock.voxelCapacity) return;

    // Empty cells were cleared on the host. Each occupied index occurs exactly once.
    VoxelData data = (VoxelData)0;
    data.occupied = 1;
    data.ellipsoid.center = seed.center;
    float3 initialScale = max(seed.scale, float3(1e-8f));
    data.ellipsoid.logScale = log(initialScale);
    data.ellipsoid.rotation = seed.rotation;
    data.radiance.init();
    data.radiance.coefficients[0] = seed.color / calcSH(0u, float3(0, 0, 1));
    // Same opacity.init() as mode 3: zero logit SH, sigmoid(0) = 0.5.
    // Initialize independently of the optimization learning rates.
    data.opacity.init();
    uint3 cell = uint3(IndexToCell(int(seed.voxelIndex), count));
    VOXEL_DATA(seedIndex) = data;
    gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(seedIndex / SPARSE_POOL_PAGE_SIZE)]
        [seedIndex % SPARSE_POOL_PAGE_SIZE] = seed.voxelIndex;
    setVoxelID(cell, seedIndex);
}
