#include "../Voxel/VoxelGrid.slang"

RWStructuredBuffer<uint> gTopologySummary;

cbuffer CB
{
    uint gSparseOffset;
    uint gMinConflictViews;
    uint gConflictSupportRatio;
    uint gCurrentIteration;
    bool gPreserveGrowthProtection;
}

[numthreads(256, 1, 1)]
void resetEvidence(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint voxelID = gSparseOffset + dispatchThreadId.x;
    if (voxelID >= gGridDataParamBlock.activeVoxelCount) return;
    for (uint word = 0u; word < TOPOLOGY_EVIDENCE_VIEW_WORDS; ++word)
    {
        TOPOLOGY_DATA(voxelID).foregroundViewMask[word] = 0u;
        TOPOLOGY_DATA(voxelID).backgroundViewMask[word] = 0u;
    }
    TOPOLOGY_DATA(voxelID).packedCountsAndFlags = 0u;
    if (!gPreserveGrowthProtection)
    {
        TOPOLOGY_DATA(voxelID).deletionEligibleIteration = 0u;
        TOPOLOGY_DATA(voxelID).growthWaitStartIterationPlusOne = 0u;
    }
    // growthBirthIterationPlusOne is lineage, not deletion evidence.
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
        for (uint word = 0u; word < TOPOLOGY_EVIDENCE_VIEW_WORDS; ++word)
        {
            evidence.foregroundViewMask[word] = 0u;
            evidence.backgroundViewMask[word] = 0u;
        }
        evidence.packedCountsAndFlags = 0u;
        TOPOLOGY_DATA(voxelID) = evidence;
        return;
    }
    uint packed = evidence.packedCountsAndFlags;
    uint foregroundViews = topologyForegroundViews(packed);
    uint backgroundViews = topologyBackgroundViews(packed);
    bool occupied = VOXEL_DATA(voxelID).occupied != 0;
    uint ratio = max(gConflictSupportRatio, 1u);
    bool conflictDominates = backgroundViews >= ratio * foregroundViews;
    // Sparse windows cannot cancel history on one incidental foreground hit.
    // Require enough support to challenge a minimally qualifying conflict set.
    uint minReliableSupport = max(2u, (gMinConflictViews + ratio - 1u) / ratio);
    bool rejectedByForeground = foregroundViews >= minReliableSupport && !conflictDominates;
    bool qualified = occupied && backgroundViews >= gMinConflictViews && conflictDominates;
    bool wasQualified = topologyWasQualified(packed);
    bool wasCandidate = topologyIsDeletionCandidate(packed);

    // A missing stochastic hit is absence of evidence, not evidence that the
    // voxel should be kept. Preserve a previous background strike (and a
    // confirmed candidate) through empty/weak windows. Sufficient foreground
    // support without dominant background conflict clears both states. This makes the decision
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
    for (uint word = 0u; word < TOPOLOGY_EVIDENCE_VIEW_WORDS; ++word)
    {
        evidence.foregroundViewMask[word] = 0u;
        evidence.backgroundViewMask[word] = 0u;
    }
    TOPOLOGY_DATA(voxelID) = evidence;

    if (candidate)
        InterlockedAdd(gTopologySummary[0], 1u);
    else if (keepQualification)
        InterlockedAdd(gTopologySummary[1], 1u);
    else if (rejectedByForeground)
        InterlockedAdd(gTopologySummary[2], 1u);
    else if (backgroundViews > 0u)
        InterlockedAdd(gTopologySummary[3], 1u);
}
