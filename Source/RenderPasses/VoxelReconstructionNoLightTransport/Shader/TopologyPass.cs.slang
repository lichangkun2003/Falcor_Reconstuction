#include "../Voxel/VoxelGrid.slang"

RWStructuredBuffer<uint> gTopologySummary;

cbuffer CB
{
    uint gSparseOffset;
    uint gMinConflictViews;
    uint gMaxSupportViews;
    uint gCurrentIteration;
}

[numthreads(256, 1, 1)]
void resetEvidence(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gSparseOffset + dispatchThreadId.x;
    if (voxelID >= gGridDataParamBlock.activeVoxelCount) return;
    TOPOLOGY_DATA(voxelID).lastForegroundView = 0u;
    TOPOLOGY_DATA(voxelID).lastBackgroundView = 0u;
    TOPOLOGY_DATA(voxelID).packedCountsAndFlags = 0u;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gSparseOffset + dispatchThreadId.x;
    if (voxelID >= gGridDataParamBlock.activeVoxelCount)
        return;

    TopologyEvidence evidence = TOPOLOGY_DATA(voxelID);
    if (gCurrentIteration < evidence.deletionEligibleIteration)
    {
        evidence.lastForegroundView = 0u;
        evidence.lastBackgroundView = 0u;
        evidence.packedCountsAndFlags = 0u;
        TOPOLOGY_DATA(voxelID) = evidence;
        return;
    }
    uint packed = evidence.packedCountsAndFlags;
    uint foregroundViews = topologyForegroundViews(packed);
    uint backgroundViews = topologyBackgroundViews(packed);
    bool occupied = VOXEL_DATA(voxelID).occupied != 0;
    bool rejectedByForeground = foregroundViews > gMaxSupportViews;
    bool qualified = occupied && !rejectedByForeground &&
        foregroundViews <= gMaxSupportViews &&
        backgroundViews >= gMinConflictViews;
    bool wasQualified = topologyWasQualified(packed);
    bool wasCandidate = topologyIsDeletionCandidate(packed);

    // A missing stochastic hit is absence of evidence, not evidence that the
    // voxel should be kept. Preserve a previous background strike (and a
    // confirmed candidate) through empty/weak windows. A reliable foreground
    // support vote explicitly clears both states. This makes the decision
    // stable for both 1 SPP and multi-SPP while remaining conservative at the
    // silhouette.
    bool candidate = occupied && !rejectedByForeground &&
        (wasCandidate || (qualified && wasQualified));
    bool keepQualification = occupied && !rejectedByForeground &&
        (wasQualified || qualified || candidate);

    evidence.packedCountsAndFlags =
        (keepQualification ? kTopologyEvidencePreviousQualified : 0u) |
        (candidate ? kTopologyEvidenceDeletionCandidate : 0u);
    // Begin a fresh window. Camera IDs may vote again in the next window, but
    // repeated iterations of the same camera cannot inflate this window.
    evidence.lastForegroundView = 0u;
    evidence.lastBackgroundView = 0u;
    TOPOLOGY_DATA(voxelID) = evidence;

    if (candidate)
        InterlockedAdd(gTopologySummary[0], 1u);
    else if (keepQualification)
        InterlockedAdd(gTopologySummary[1], 1u);
    else if (foregroundViews > gMaxSupportViews)
        InterlockedAdd(gTopologySummary[2], 1u);
    else if (backgroundViews > 0u)
        InterlockedAdd(gTopologySummary[3], 1u);
}
