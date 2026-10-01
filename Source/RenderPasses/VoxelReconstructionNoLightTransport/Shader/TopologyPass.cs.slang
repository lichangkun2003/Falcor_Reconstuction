#include "../Voxel/VoxelGrid.slang"

RWStructuredBuffer<uint> gTopologySummary;

cbuffer CB
{
    uint gSparseOffset;
    uint gMinConflictViews;
    uint gMaxSupportViews;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gSparseOffset + dispatchThreadId.x;
    if (voxelID >= gGridDataParamBlock.activeVoxelCount)
        return;

    TopologyEvidence evidence = TOPOLOGY_DATA(voxelID);
    uint packed = evidence.packedCountsAndFlags;
    uint foregroundViews = topologyForegroundViews(packed);
    uint backgroundViews = topologyBackgroundViews(packed);
    bool qualified = VOXEL_DATA(voxelID).occupied != 0 &&
        foregroundViews <= gMaxSupportViews &&
        backgroundViews >= gMinConflictViews;
    bool candidate = qualified && topologyWasQualified(packed);

    evidence.packedCountsAndFlags =
        (qualified ? kTopologyEvidencePreviousQualified : 0u) |
        (candidate ? kTopologyEvidenceDeletionCandidate : 0u);
    // Begin a fresh window. Camera IDs may vote again in the next window, but
    // repeated iterations of the same camera cannot inflate this window.
    evidence.lastForegroundView = 0u;
    evidence.lastBackgroundView = 0u;
    TOPOLOGY_DATA(voxelID) = evidence;

    if (candidate)
        InterlockedAdd(gTopologySummary[0], 1u);
    else if (qualified)
        InterlockedAdd(gTopologySummary[1], 1u);
    else if (foregroundViews > gMaxSupportViews)
        InterlockedAdd(gTopologySummary[2], 1u);
    else if (backgroundViews > 0u)
        InterlockedAdd(gTopologySummary[3], 1u);
}
