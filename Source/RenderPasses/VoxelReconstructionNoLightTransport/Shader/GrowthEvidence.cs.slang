#include "GrowthPass.cs.slang"

// Frozen for one complete training round. Parent addresses, not compact IDs,
// survive deletion/compaction before growth commits at the round boundary.
struct GrowthCandidate
{
    GaussianEllipsoid ellipsoid;
    uint parentCell;
    uint neighborIndex;
    uint foregroundViews;
    uint backgroundViews;
};
RWStructuredBuffer<GrowthCandidate> gGrowthCandidates;
Texture2D<float4> gGrowthReference;
Texture2D<float4> gGrowthRendered;
cbuffer EvidenceCB
{
    uint gCandidateCount;
    uint gMinSupportingParents;
    bool gUse26Neighbors;
    bool gWriteCandidates;
    uint gMinForegroundViews;
    uint gBackgroundVetoViews;
    float gMinAlphaDeficit;
    float gForegroundAlphaMin;
    float gBackgroundAlphaMax;
    uint2 gEvidenceResolution;
    float4x4 gEvidenceVP;
};

bool growthCellAvailable(uint3 cell)
{
    int previous = growthRawIndex(cell);
    return previous < 0 && previous > kGrowthClaimBase &&
        (previous == -1 || gCurrentIteration >= uint(-previous - 2));
}

uint evidenceNeighborCount() { return gUse26Neighbors ? 26u : 6u; }

int3 evidenceNeighborOffset(uint index)
{
    if (gUse26Neighbors) return growthNeighborOffset(index);
    int3 offset = int3(0);
    offset[index / 2u] = (index & 1u) != 0u ? 1 : -1;
    return offset;
}

bool evidenceGrowthChild(uint parentID, int3 offset, out uint3 cell, out GaussianEllipsoid child)
{
    child = VOXEL_DATA(parentID).ellipsoid;
    cell = uint3(0);
    if (!growthParentEligible(parentID)) return false;
    int3 neighbor = getVoxelCell(parentID) + offset;
    if (any(neighbor < 0) || any(neighbor >= int3(gGridDataParamBlock.voxelCount))) return false;
    cell = uint3(neighbor);
    return makeNeighborGrowthEllipsoid(child, gGridDataParamBlock.voxelSize, offset,
        gFacePenetration, gShrink, gContactOffset, child);
}

// Count distinct, mature neighbors that can actually grow toward this cell.
// Recheck at commit because supporting parents can be removed during evidence collection.
bool hasSupportingParents(uint3 cell)
{
    uint support = 0u;
    for (uint index = 0u; index < evidenceNeighborCount(); ++index)
    {
        int3 offset = evidenceNeighborOffset(index);
        int3 neighbor = int3(cell) + offset;
        if (any(neighbor < 0) || any(neighbor >= int3(gGridDataParamBlock.voxelCount))) continue;
        int parentID = lookupVoxelID(neighbor);
        if (parentID < 0 || uint(parentID) >= gParentCount) continue;
        uint3 target;
        GaussianEllipsoid child;
        if (evidenceGrowthChild(uint(parentID), -offset, target, child) && all(target == cell))
            if (++support >= max(gMinSupportingParents, 1u)) return true;
    }
    return false;
}

[numthreads(256, 1, 1)]
void buildCandidates(uint3 threadID : SV_DispatchThreadID)
{
    uint parentID = gSparseOffset + threadID.x;
    if (parentID >= gParentCount) return;
    for (uint index = 0u; index < evidenceNeighborCount(); ++index)
    {
        uint3 cell;
        GaussianEllipsoid child;
        if (!evidenceGrowthChild(parentID, evidenceNeighborOffset(index), cell, child) || !growthCellAvailable(cell)) continue;
        if (!hasSupportingParents(cell)) continue;
        uint slot;
        InterlockedAdd(gGrowthCounter[0], 1u, slot);
        if (!gWriteCandidates || slot >= gCandidateCount) continue;
        GrowthCandidate candidate;
        candidate.ellipsoid = child;
        candidate.parentCell = uint(CellToIndex(uint3(getVoxelCell(parentID)), gGridDataParamBlock.voxelCount));
        candidate.neighborIndex = index;
        candidate.foregroundViews = 0u;
        candidate.backgroundViews = 0u;
        gGrowthCandidates[slot] = candidate;
    }
}

uint3 candidateCell(GrowthCandidate candidate)
{
    int3 cell = IndexToCell(int(candidate.parentCell), gGridDataParamBlock.voxelCount);
    cell += evidenceNeighborOffset(candidate.neighborIndex);
    return uint3(cell);
}

[numthreads(256, 1, 1)]
void evaluateCandidates(uint3 threadID : SV_DispatchThreadID)
{
    uint id = gSparseOffset + threadID.x;
    if (id >= gCandidateCount) return;
    GrowthCandidate candidate = gGrowthCandidates[id];
    uint3 cell = candidateCell(candidate);
    bool foreground = false, background = false;
    float3 axes = gaussianEllipsoidScale(candidate.ellipsoid);
    float3x3 rotation = gaussianRotationMatrix(candidate.ellipsoid.rotation);
    // A small footprint probe, not an exact projection. Background tests are
    // independent of rendered opacity; a wrong occluder must not protect a hole.
    // The host dispatches each distinct camera exactly once per evidence round.
    for (uint sample = 0u; sample < 7u; ++sample)
    {
        float3 local = candidate.ellipsoid.center;
        if (sample > 0u)
        {
            uint axis = (sample - 1u) / 2u;
            float3 offset = float3(0);
            offset[axis] = ((sample & 1u) != 0u ? -0.5f : 0.5f) * axes[axis];
            local += mul(rotation, offset) / gGridDataParamBlock.voxelSize;
        }
        // Clipping toward the interior center keeps probes inside the child.
        float3 delta = local - candidate.ellipsoid.center;
        float t = 1.0f;
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            if (delta[axis] > 0.0f) t = min(t, (0.999f - candidate.ellipsoid.center[axis]) / delta[axis]);
            if (delta[axis] < 0.0f) t = min(t, (0.001f - candidate.ellipsoid.center[axis]) / delta[axis]);
        }
        local = candidate.ellipsoid.center + saturate(t) * delta;
        float3 world = gGridDataParamBlock.gridMin + (float3(cell) + local) * gGridDataParamBlock.voxelSize;
        float4 projected = mul(gEvidenceVP, float4(world, 1));
        if (projected.w <= 1e-8f) continue;
        float3 ndc = projected.xyz / projected.w;
        float2 uv = float2(0.5f * ndc.x + 0.5f, 0.5f - 0.5f * ndc.y);
        if (ndc.z < 0.0f || ndc.z > 1.0f || any(uv < 0.0f) || any(uv >= 1.0f)) continue;
        uint2 pixel = uint2(uv * float2(gEvidenceResolution));
        float referenceAlpha = gGrowthReference.Load(int3(pixel, 0)).a;
        float renderedAlpha = gGrowthRendered.Load(int3(pixel, 0)).a;
        background = background || referenceAlpha <= gBackgroundAlphaMax;
        foreground = foreground || (referenceAlpha >= gForegroundAlphaMin &&
            referenceAlpha - renderedAlpha >= gMinAlphaDeficit);
    }
    if (background) candidate.backgroundViews++;
    else if (foreground) candidate.foregroundViews++;
    gGrowthCandidates[id] = candidate;
}

bool eligibleCandidate(uint id, out uint parentID, out uint3 cell)
{
    GrowthCandidate candidate = gGrowthCandidates[id];
    cell = candidateCell(candidate);
    parentID = 0u;
    if (candidate.foregroundViews < max(gMinForegroundViews, 1u) ||
        candidate.backgroundViews >= max(gBackgroundVetoViews, 1u)) return false;
    int parent = lookupVoxelID(IndexToCell(int(candidate.parentCell), gGridDataParamBlock.voxelCount));
    if (parent < 0 || VOXEL_DATA(uint(parent)).occupied == 0u || isDeletionCandidate(uint(parent))) return false;
    parentID = uint(parent);
    uint3 target;
    GaussianEllipsoid child;
    return evidenceGrowthChild(parentID, evidenceNeighborOffset(candidate.neighborIndex), target, child) &&
        all(target == cell) && hasSupportingParents(cell);
}

[numthreads(256, 1, 1)]
void proposeEvidenceGrowth(uint3 threadID : SV_DispatchThreadID)
{
    uint id = gSparseOffset + threadID.x;
    if (id >= gCandidateCount) return;
    uint parentID;
    uint3 cell;
    GrowthCandidate candidate = gGrowthCandidates[id];
    if (candidate.foregroundViews >= max(gMinForegroundViews, 1u)) InterlockedAdd(gGrowthCounter[1], 1u);
    if (candidate.backgroundViews >= max(gBackgroundVetoViews, 1u)) InterlockedAdd(gGrowthCounter[2], 1u);
    if (!eligibleCandidate(id, parentID, cell)) return;
    int previous = growthRawIndex(cell), original;
    if (previous >= 0 || previous <= kGrowthClaimBase ||
        (previous != -1 && gCurrentIteration < uint(-previous - 2))) return;
    InterlockedCompareExchange(
        gGridDataParamBlock.indexPages[NonUniformResourceIndex(growthIndexPage(cell))][cell % SPARSE_INDEX_PAGE_EDGE],
        previous, kGrowthClaimBase - int(id), original);
    if (original == previous) InterlockedAdd(gGrowthCounter[0], 1u);
}

[numthreads(256, 1, 1)]
void initializeEvidenceGrowth(uint3 threadID : SV_DispatchThreadID)
{
    uint id = gSparseOffset + threadID.x;
    if (id >= gCandidateCount) return;
    uint parentID;
    uint3 cell;
    if (!eligibleCandidate(id, parentID, cell) || growthRawIndex(cell) != kGrowthClaimBase - int(id)) return;
    uint slot;
    InterlockedAdd(gGrowthCounter[0], 1u, slot);
    if (slot < gNewCount) writeGrowthChild(parentID, gParentCount + slot, gGrowthCandidates[id].ellipsoid, cell);
}

[numthreads(256, 1, 1)]
void clearEvidenceClaims(uint3 threadID : SV_DispatchThreadID)
{
    uint id = gSparseOffset + threadID.x;
    if (id >= gCandidateCount) return;
    uint3 cell = candidateCell(gGrowthCandidates[id]);
    int ignored;
    InterlockedCompareExchange(
        gGridDataParamBlock.indexPages[NonUniformResourceIndex(growthIndexPage(cell))][cell % SPARSE_INDEX_PAGE_EDGE],
        kGrowthClaimBase - int(id), -1, ignored);
}
