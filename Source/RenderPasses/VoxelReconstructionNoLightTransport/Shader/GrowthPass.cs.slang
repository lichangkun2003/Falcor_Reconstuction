#include "../Voxel/VoxelGrid.slang"
#include "../GeometryAdamState.slang"
#include "../Math/GrowthEllipsoid.slang"

RWStructuredBuffer<uint> gGrowthCounter;
RWStructuredBuffer<GeometryAdamState> gGeometryAdamPages[SPARSE_POOL_MAX_PAGES];
RWStructuredBuffer<uint> gRadianceAdamIndexPages[SPARSE_POOL_MAX_PAGES];

cbuffer CB
{
    uint gSparseOffset;
    uint gParentCount;
    uint gNewCount;
    uint gCurrentIteration;
    uint gEligibleIteration;
    uint gGrowthWaitIterations;
    float gFacePenetration;
    float gShrink;
    float gContactOffset;
    float gInitialOpacity;
}

uint growthIndexPage(uint3 cell)
{
    uint3 page = cell / SPARSE_INDEX_PAGE_EDGE;
    return page.x + page.y * gGridDataParamBlock.indexPageCount.x +
           page.z * gGridDataParamBlock.indexPageCount.x * gGridDataParamBlock.indexPageCount.y;
}

int growthRawIndex(uint3 cell)
{
    return gGridDataParamBlock.indexPages[NonUniformResourceIndex(growthIndexPage(cell))][cell % SPARSE_INDEX_PAGE_EDGE];
}

int growthClaim(uint parentID, uint face)
{
    return kGrowthClaimBase - int(parentID * 6u + face);
}

bool growthParentEligible(uint parentID)
{
    if (VOXEL_DATA(parentID).occupied == 0u || isDeletionCandidate(parentID))
        return false;
    uint waitStart = TOPOLOGY_DATA(parentID).growthWaitStartIterationPlusOne;
    if (waitStart != 0u &&
        (gCurrentIteration < waitStart - 1u || gCurrentIteration - (waitStart - 1u) < gGrowthWaitIterations))
        return false;
    return true;
}

bool growthChild(uint parentID, uint face, out uint3 cell, out GaussianEllipsoid child)
{
    cell = uint3(0);
    child = VOXEL_DATA(parentID).ellipsoid;
    if (!growthParentEligible(parentID)) return false;
    int3 neighbor = getVoxelCell(parentID);
    neighbor[face / 2u] += (face & 1u) != 0u ? 1 : -1;
    if (any(neighbor < 0) || any(neighbor >= int3(gGridDataParamBlock.voxelCount)))
        return false;
    cell = uint3(neighbor);
    return makeGrowthEllipsoid(child, gGridDataParamBlock.voxelSize, face, gFacePenetration, gShrink, gContactOffset, child);
}

[numthreads(256, 1, 1)] void proposeGrowth(uint3 threadID
                                           : SV_DispatchThreadID)
{
    uint parentID = gSparseOffset + threadID.x;
    if (parentID >= gParentCount)
        return;
    for (uint face = 0u; face < 6u; ++face)
    {
        uint3 cell;
        GaussianEllipsoid child;
        if (!growthChild(parentID, face, cell, child))
            continue;
        int previous = growthRawIndex(cell);
        if (previous >= 0 || previous <= kGrowthClaimBase)
            continue;
        if (previous != -1 && gCurrentIteration < uint(-previous - 2))
            continue;
        int original;
        InterlockedCompareExchange(
            gGridDataParamBlock.indexPages[NonUniformResourceIndex(growthIndexPage(cell))][cell % SPARSE_INDEX_PAGE_EDGE],
            previous,
            growthClaim(parentID, face),
            original
        );
        if (original == previous)
            InterlockedAdd(gGrowthCounter[0], 1u);
    }
}

void writeGrowthChild(uint parentID, uint id, GaussianEllipsoid child, uint3 cell)
{
    VoxelData data = VOXEL_DATA(parentID);
    data.ellipsoid = child;
    data.occupied = 1u;
    // Start with the parent's DC radiance; learn directional variation locally.
    for (uint coefficient = 1u; coefficient < SH_COUNT; ++coefficient)
        data.radiance.coefficients[coefficient] = float3(0.0f);
    // Initialize the requested alpha uniformly in every direction.
    float y0 = calcSH(0u, float3(0, 0, 1));
    for (uint i = 1u; i < SH_OPACITY_COUNT; ++i)
        data.opacity.coefficients[i] = 0.0f;
    float desiredLogit = log(gInitialOpacity / (1.0f - gInitialOpacity));
    data.opacity.coefficients[0] = desiredLogit / y0;
    VOXEL_DATA(id) = data;
    GradRecord gradient;
    gradient.init();
    GRAD_DATA(id) = gradient;
    TopologyEvidence evidence = {};
    evidence.deletionEligibleIteration = gEligibleIteration;
    evidence.growthBirthIterationPlusOne = gCurrentIteration + 1u;
    evidence.growthWaitStartIterationPlusOne = gCurrentIteration + 1u;
    TOPOLOGY_DATA(id) = evidence;
    uint page = id / SPARSE_POOL_PAGE_SIZE;
    uint offset = id % SPARSE_POOL_PAGE_SIZE;
    GeometryAdamState adam = {};
    gGeometryAdamPages[NonUniformResourceIndex(page)][offset] = adam;
    gRadianceAdamIndexPages[NonUniformResourceIndex(page)][offset] = 0xffffffffu;
    gGridDataParamBlock.cellIndexPages[NonUniformResourceIndex(page)][offset] = uint(CellToIndex(cell, gGridDataParamBlock.voxelCount));
}

    [numthreads(256, 1, 1)] void initializeGrowth(uint3 threadID
                                                  : SV_DispatchThreadID)
{
    uint parentID = gSparseOffset + threadID.x;
    if (parentID >= gParentCount)
        return;
    for (uint face = 0u; face < 6u; ++face)
    {
        uint3 cell;
        GaussianEllipsoid child;
        if (!growthChild(parentID, face, cell, child) || growthRawIndex(cell) != growthClaim(parentID, face))
            continue;
        uint slot;
        InterlockedAdd(gGrowthCounter[0], 1u, slot);
        if (slot >= gNewCount)
            continue;
        uint id = gParentCount + slot;
        writeGrowthChild(parentID, id, child, cell);
        // Leave the index negative until all children have initialized. Failed
        // initialization can be rolled back without exposing a partial layer.
    }
}

[numthreads(256, 1, 1)] void commitGrowth(uint3 threadID
                                          : SV_DispatchThreadID)
{
    uint slot = gSparseOffset + threadID.x;
    if (slot >= gNewCount)
        return;
    uint id = gParentCount + slot;
    setVoxelID(uint3(getVoxelCell(id)), id);
}

    [numthreads(256, 1, 1)] void clearGrowthClaims(uint3 threadID
                                                   : SV_DispatchThreadID)
{
    uint parentID = gSparseOffset + threadID.x;
    if (parentID >= gParentCount)
        return;
    int3 parentCell = getVoxelCell(parentID);
    for (uint face = 0u; face < 6u; ++face)
    {
        int3 cell = parentCell;
        cell[face / 2u] += (face & 1u) != 0u ? 1 : -1;
        if (any(cell < 0) || any(cell >= int3(gGridDataParamBlock.voxelCount)))
            continue;
        int ignored;
        InterlockedCompareExchange(
            gGridDataParamBlock.indexPages[NonUniformResourceIndex(growthIndexPage(uint3(cell)))][uint3(cell) % SPARSE_INDEX_PAGE_EDGE],
            growthClaim(parentID, face),
            -1,
            ignored
        );
    }
}

[numthreads(256, 1, 1)] void rollbackGrowth(uint3 threadID
                                            : SV_DispatchThreadID)
{
    uint slot = gSparseOffset + threadID.x;
    if (slot >= gNewCount)
        return;
    uint id = gParentCount + slot;
    uint3 cell = uint3(getVoxelCell(id));
    int ignored;
    InterlockedCompareExchange(
        gGridDataParamBlock.indexPages[NonUniformResourceIndex(growthIndexPage(cell))][cell % SPARSE_INDEX_PAGE_EDGE], int(id), -1, ignored
    );
    VOXEL_DATA(id).occupied = 0u;
}

    [numthreads(256, 1, 1)] void clearGrowthCooldown(uint3 threadID
                                                     : SV_DispatchThreadID)
{
    uint index = gSparseOffset + threadID.x;
    uint total = gGridDataParamBlock.voxelCount.x * gGridDataParamBlock.voxelCount.y * gGridDataParamBlock.voxelCount.z;
    if (index >= total)
        return;
    uint3 cell = uint3(IndexToCell(int(index), gGridDataParamBlock.voxelCount));
    if (growthRawIndex(cell) < -1)
        clearVoxelID(cell);
}
