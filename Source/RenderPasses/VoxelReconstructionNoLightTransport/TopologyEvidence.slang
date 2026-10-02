#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

// One compact persistent record per active voxel. View stamps ensure that one
// physical camera contributes at most one foreground vote and one background
// vote in each evidence window, regardless of iterations, resolution, or SPP.
struct TopologyEvidence
{
    uint lastForegroundView;
    uint lastBackgroundView;
    uint packedCountsAndFlags;
    // Full training rounds must finish before a newborn can collect deletion votes.
    uint deletionEligibleIteration;
};

// Negative spatial-index values are empty cells. Cooldown is -(expiry + 2),
// while temporary growth claims occupy a disjoint, much more negative range.
static const int kGrowthClaimBase = -1073741824;
static const uint kMaximumTopologyIteration = 1000000000u;

#if defined(HOST_CODE)
static_assert(uint64_t(SPARSE_POOL_PAGE_SIZE) * SPARSE_POOL_MAX_PAGES * 6u < (uint64_t(1) << 30u),
    "Growth claim encoding cannot represent this sparse-pool capacity.");
#endif

static const uint kTopologyEvidenceCountBits = 15u;
static const uint kTopologyEvidenceCountMask = (1u << kTopologyEvidenceCountBits) - 1u;
static const uint kTopologyEvidenceBackgroundShift = kTopologyEvidenceCountBits;
static const uint kTopologyEvidencePreviousQualified = 1u << 30u;
static const uint kTopologyEvidenceDeletionCandidate = 1u << 31u;

inline uint topologyForegroundViews(uint packed)
{
    return packed & kTopologyEvidenceCountMask;
}

inline uint topologyBackgroundViews(uint packed)
{
    return (packed >> kTopologyEvidenceBackgroundShift) & kTopologyEvidenceCountMask;
}

inline bool topologyWasQualified(uint packed)
{
    return (packed & kTopologyEvidencePreviousQualified) != 0u;
}

inline bool topologyIsDeletionCandidate(uint packed)
{
    return (packed & kTopologyEvidenceDeletionCandidate) != 0u;
}

END_NAMESPACE_FALCOR
