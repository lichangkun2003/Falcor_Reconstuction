#include "../Voxel/VoxelGrid.slang"
#include "../GeometryAdamState.slang"

RWStructuredBuffer<uint> gHoleIDs;
RWStructuredBuffer<uint> gDonorIDs;
// [0] candidate count, [1] holes in the retained prefix, [2] live donors in the removed suffix.
RWStructuredBuffer<uint> gDeletionCounters;
RWStructuredBuffer<GeometryAdamState> gGeometryAdamPages[SPARSE_POOL_MAX_PAGES];

cbuffer CB
{
    uint gSparseOffset;
    uint gOldActiveCount;
    uint gNewActiveCount;
    uint gListCapacity;
    uint gPairOffset;
    uint gPairCount;
    uint gCooldownUntilIteration;
}

[numthreads(256, 1, 1)]
void buildDeletionLists(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gSparseOffset + dispatchThreadId.x;
    if (voxelID >= gOldActiveCount)
        return;

    bool candidate = isDeletionCandidate(voxelID);
    uint ignored;
    if (candidate)
        InterlockedAdd(gDeletionCounters[0], 1u, ignored);

    if (candidate && voxelID < gNewActiveCount)
    {
        uint listIndex;
        InterlockedAdd(gDeletionCounters[1], 1u, listIndex);
        if (listIndex < gListCapacity)
            gHoleIDs[listIndex] = voxelID;
    }
    else if (!candidate && voxelID >= gNewActiveCount)
    {
        uint listIndex;
        InterlockedAdd(gDeletionCounters[2], 1u, listIndex);
        if (listIndex < gListCapacity)
            gDonorIDs[listIndex] = voxelID;
    }
}

[numthreads(256, 1, 1)]
void clearDeletionIndices(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gSparseOffset + dispatchThreadId.x;
    if (voxelID >= gOldActiveCount || !isDeletionCandidate(voxelID))
        return;

    uint3 cell = uint3(getVoxelCell(voxelID));
    // This negative value remains empty to lookupVoxelID(), but inhibits growth
    // until its absolute training-iteration expiry, independent of compact IDs.
    uint3 page = cell / SPARSE_INDEX_PAGE_EDGE;
    uint pageID = page.x + page.y * gGridDataParamBlock.indexPageCount.x +
        page.z * gGridDataParamBlock.indexPageCount.x * gGridDataParamBlock.indexPageCount.y;
    gGridDataParamBlock.indexPages[NonUniformResourceIndex(pageID)][cell % SPARSE_INDEX_PAGE_EDGE] =
        -2 - int(gCooldownUntilIteration);
}

[numthreads(256, 1, 1)]
void moveDeletionSurvivors(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint pairID = gPairOffset + dispatchThreadId.x;
    if (pairID >= gPairCount)
        return;

    uint destination = gHoleIDs[pairID];
    uint source = gDonorIDs[pairID];
    uint destinationPage = destination / SPARSE_POOL_PAGE_SIZE;
    uint destinationOffset = destination % SPARSE_POOL_PAGE_SIZE;
    uint sourcePage = source / SPARSE_POOL_PAGE_SIZE;
    uint sourceOffset = source % SPARSE_POOL_PAGE_SIZE;

    VOXEL_DATA(destination) = VOXEL_DATA(source);
    TOPOLOGY_DATA(destination) = TOPOLOGY_DATA(source);
    gGeometryAdamPages[NonUniformResourceIndex(destinationPage)][destinationOffset] =
        gGeometryAdamPages[NonUniformResourceIndex(sourcePage)][sourceOffset];
    uint cellIndex = gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(sourcePage)][sourceOffset];
    gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(destinationPage)][destinationOffset] = cellIndex;
    setVoxelID(uint3(IndexToCell(int(cellIndex), gGridDataParamBlock.voxelCount)), destination);
}

[numthreads(256, 1, 1)]
void clearDeletionTail(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gSparseOffset + dispatchThreadId.x;
    if (voxelID < gNewActiveCount || voxelID >= gOldActiveCount)
        return;

    VOXEL_DATA(voxelID).occupied = 0u;
}
