// Headless regression test. Build with VOXEL_RECONSTRUCTION_BUILD_GROWTH_TESTS=ON.
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Debug/PixelDebug.h"
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Scene/SceneBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <set>
#include "../VoxelReconstructionNoLightTransport.h"

void runCoarseToFineTests(const ref<Device>& device);

struct NeighborGrowthTestAccess
{
    static void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    static uint32_t cell(uint32_t x, uint32_t y, uint32_t z) { return x + 8u * y + 64u * z; }
    static int3 cellCoordinates(uint32_t index) { return int3(index % 8u, (index / 8u) % 8u, index / 64u); }

    static VoxelData parentVoxel()
    {
        VoxelData result = {};
        result.occupied = 1u;
        result.ellipsoid.center = float3(0.5f);
        result.ellipsoid.logScale = float3(std::log(4.0f), std::log(2.0f), std::log(0.8f));
        result.ellipsoid.rotation = float4(std::cos(0.3f), 0, 0, std::sin(0.3f));
        result.radiance.coefficients[0] = float3(0.3f, 0.5f, 0.7f);
        result.opacity.coefficients[1] = 0.03f;
        return result;
    }

    static void seed(
        VoxelReconstructionNoLightTransport& pass,
        RenderContext* ctx,
        const std::vector<uint32_t>& cells,
        const std::vector<VoxelData>& voxels,
        float3 voxelSize = float3(1.0f)
    )
    {
        GridData grid = {};
        grid.voxelCount = uint3(8u);
        grid.voxelSize = voxelSize;
        grid.activeVoxelCount = grid.solidVoxelCount = uint32_t(cells.size());
        auto resources = pass.allocateSparseGrid(ctx, grid, grid.activeVoxelCount);
        auto block = pass.createSparseGridBlock(resources);
        pass.commitSparseGrid(std::move(resources), block, 8u);
        pass.uploadSparseBatch(ctx, block, 0, cells.data(), uint32_t(cells.size()), voxels.data());
        pass.barrierSparseVoxels(ctx);
        for (const auto& page : pass.mGridResources.indexPages)
            ctx->uavBarrier(page.get());
        ctx->submit(true);
        pass.mOptimizerParams.currentIteration = 50u;
        pass.mTopologySettings.enableGrowth = true;
        pass.mTopologySettings.growthFacePenetration = 0.05f;
        // Geometry/lifecycle fixtures explicitly retain per-round scheduling.
        pass.mTopologySettings.growthInterval = 1u;
        pass.mTopologySettings.growthWaitIterations = 5u;
        pass.mGrowthCooldownPresent = false;
        pass.resetDeletionEvidence(ctx);
    }

    template<typename T>
    static T read(const std::vector<ref<Buffer>>& pages, uint32_t id)
    {
        T value;
        pages[id / SPARSE_POOL_PAGE_SIZE]->getBlob(&value, size_t(id % SPARSE_POOL_PAGE_SIZE) * sizeof(T), sizeof(T));
        return value;
    }

    template<typename T>
    static void write(const std::vector<ref<Buffer>>& pages, uint32_t id, const T& value)
    {
        pages[id / SPARSE_POOL_PAGE_SIZE]->setBlob(&value, size_t(id % SPARSE_POOL_PAGE_SIZE) * sizeof(T), sizeof(T));
    }

    static std::vector<int32_t> indices(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        auto bytes = ctx->readTextureSubresource(pass.mGridResources.indexPages[0].get(), 0);
        std::vector<int32_t> result(512);
        require(bytes.size() == result.size() * sizeof(int32_t), "Unexpected index texture readback layout");
        std::memcpy(result.data(), bytes.data(), bytes.size());
        return result;
    }

    static float3 extents(const GaussianEllipsoid& e)
    {
        const float w = e.rotation.x, x = e.rotation.y, y = e.rotation.z, z = e.rotation.w;
        const float r[3][3] = {
            {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
            {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
            {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
        float3 a(std::exp(e.logScale.x), std::exp(e.logScale.y), std::exp(e.logScale.z));
        float3 extent;
        for (uint32_t i = 0; i < 3; ++i)
            extent[i] = std::sqrt(r[i][0] * r[i][0] * a.x * a.x + r[i][1] * r[i][1] * a.y * a.y + r[i][2] * r[i][2] * a.z * a.z);
        return extent;
    }

    static void checkChild(const VoxelData& parent, const VoxelData& child, uint32_t childCell, uint32_t parentCell)
    {
        require(child.occupied == 1u, "Child is not occupied");
        require(std::memcmp(&parent.radiance, &child.radiance, sizeof(parent.radiance)) == 0, "Radiance was not inherited");
        for (uint32_t i = 0; i < 4; ++i)
            require(std::abs(parent.ellipsoid.rotation[i] - child.ellipsoid.rotation[i]) < 1e-6f, "Parent rotation changed");
        float3 ratios = child.ellipsoid.logScale - parent.ellipsoid.logScale;
        require(
            std::abs(ratios.x - ratios.y) < 2e-5f && std::abs(ratios.x - ratios.z) < 2e-5f, "Child did not preserve three-axis proportions"
        );
        require(ratios.x < 0, "Child did not shrink");
        const int3 delta = cellCoordinates(childCell) - cellCoordinates(parentCell);
        require(std::abs(delta.x) + std::abs(delta.y) + std::abs(delta.z) == 1, "More than one layer grew");
        const float3 extent = extents(child.ellipsoid);
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            if (delta[axis] <= 0)
                require(child.ellipsoid.center[axis] - extent[axis] >= -2e-5f, "Child crossed a non-contact lower face");
            if (delta[axis] >= 0)
                require(child.ellipsoid.center[axis] + extent[axis] <= 1.0f + 2e-5f, "Child crossed a non-contact upper face");
        }
        for (uint32_t sample = 0; sample < 128; ++sample)
        {
            const float z = 1.0f - 2.0f * (float(sample) + 0.5f) / 128.0f;
            const float phi = 2.39996323f * float(sample);
            const float radius = std::sqrt(1.0f - z * z);
            auto opacity = child.opacity;
            const float alpha = opacity.calcOpacity(float3(radius * std::cos(phi), radius * std::sin(phi), z));
            require(alpha >= 0.01f - 1e-6f && alpha <= 0.1f + 1e-6f, "Initial opacity is not low and trainable");
        }
    }

    static void checkDeletionProtection(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(GradientPassShaderFilePath).csEntry("main");
        auto gradientPass = ComputePass::create(pass.mpDevice, desc, pass.getReconstructionDefines(), true);
        auto var = gradientPass->getRootVar();
        var["gGridDataParamBlock"] = pass.mpGridBlock;
        PathRecord path = {};
        path.valid = 1u;
        path.contributingVoxelCount = 1u;
        path.viewDir = float3(0, 0, 1);
        path.voxels[0].voxelID = 1u;
        path.voxels[0].opacity = 0.1f;
        path.voxels[0].transmittanceBefore = 1.0f;
        path.voxels[0].gmin = -1.0f;
        auto voxel = read<VoxelData>(pass.mGridResources.voxelPages, 1);
        path.voxels[0].xStar = float3(cellCoordinates(read<uint32_t>(pass.mGridResources.cellIndexPages, 1))) + voxel.ellipsoid.center;
        var["gPathRecordBuffer"] = pass.mpDevice->createStructuredBuffer(
            sizeof(PathRecord), 1u, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &path
        );
        const float4 derivative(0, 0, 0, 1);
        const uint32_t background = 1u;
        var["gDL_dColorBuffer"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, &derivative);
        var["gBackgroundMaskBuffer"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::R32Uint, 1, 1, &background);
        var["dummy"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess);
        auto cb = var["CB"];
        cb["gResolution"] = uint2(1);
        cb["gVoxelCount"] = uint3(8);
        cb["gGeometryTauWorld"] = 0.12f;
        cb["gGeometryGradClamp"] = 5.0f;
        cb["gAlphaGeometryWeight"] = 1.0f;
        cb["gBackgroundCarveWeight"] = 0.01f;
        cb["gBinaryOpacityWeight"] = 0.0f;
        cb["gCollectDeletionEvidence"] = true;
        cb["gEvidenceViewID"] = 1u;
        cb["gMinRemovalLossDelta"] = 0.0f;
        cb["gMinEvidenceTransmittance"] = 0.0f;
        cb["gAlphaLossWeight"] = 1.0f;
        pass.clearSparseGradients(ctx);
        cb["gCurrentIteration"] = 54u;
        gradientPass->execute(ctx, uint3(1));
        pass.barrierSparseGradients(ctx);
        pass.barrierTopologyEvidence(ctx);
        ctx->submit(true);
        require(
            topologyBackgroundViews(read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 1).packedCountsAndFlags) == 0u,
            "Protected newborn collected deletion evidence"
        );
        require(
            read<GradRecord>(pass.mGridResources.gradPages, 1).appearanceValid > 0u, "Protection blocked normal appearance optimization"
        );
        cb["gCurrentIteration"] = 55u;
        for (uint32_t sample = 0; sample < 4; ++sample)
        {
            gradientPass->execute(ctx, uint3(1));
            pass.barrierSparseGradients(ctx);
            pass.barrierTopologyEvidence(ctx);
        }
        ctx->submit(true);
        require(
            topologyBackgroundViews(read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 1).packedCountsAndFlags) == 1u,
            "Expired protection did not permit one unique view vote"
        );
        // A protected voxel also cannot retain a stale/fabricated candidate flag.
        auto evidence = read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 1);
        evidence.packedCountsAndFlags = kTopologyEvidenceDeletionCandidate;
        write(pass.mGridResources.topologyEvidencePages, 1, evidence);
        const uint32_t interval = pass.mTopologySettings.evidenceInterval;
        pass.mTopologySettings.evidenceInterval = 1u;
        pass.mOptimizerParams.currentIteration = 54u;
        pass.evaluateDeletionEvidence(ctx);
        evidence = read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 1);
        require(
            evidence.packedCountsAndFlags == 0u && evidence.deletionEligibleIteration == 55u,
            "Protected newborn qualified for deletion or lost its expiry"
        );
        pass.mTopologySettings.evidenceInterval = interval;
        pass.mOptimizerParams.currentIteration = 50u;
    }
    static void checkDeletionVoteRatio(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        // Exercise a known ratio without constraining user-tunable experiment defaults.
        const uint32_t originalMinViews = pass.mTopologySettings.minDeletionConflictViews;
        const uint32_t originalRatio = pass.mTopologySettings.deletionConflictSupportRatio;
        pass.mTopologySettings.minDeletionConflictViews = 5u;
        pass.mTopologySettings.deletionConflictSupportRatio = 3u;
        seed(pass, ctx, {cell(3, 3, 3)}, {parentVoxel()});
        uint32_t history = 0u;
        const auto window = [&](uint32_t background, uint32_t foreground)
        {
            auto evidence = read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 0u);
            evidence.packedCountsAndFlags = history | foreground | (background << kTopologyEvidenceBackgroundShift);
            write(pass.mGridResources.topologyEvidencePages, 0u, evidence);
            pass.mOptimizerParams.currentIteration += pass.mTopologySettings.evidenceInterval;
            pass.evaluateDeletionEvidence(ctx);
            evidence = read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 0u);
            history = evidence.packedCountsAndFlags;
            require(evidence.lastForegroundView == 0u && evidence.lastBackgroundView == 0u &&
                topologyForegroundViews(history) == 0u && topologyBackgroundViews(history) == 0u,
                "Evidence window did not reset camera stamps/counts");
        };
        window(4u, 0u);
        require(history == 0u, "Insufficient background qualified for deletion");
        window(5u, 1u);
        require(topologyWasQualified(history) && !topologyIsDeletionCandidate(history), "Minority support vetoed first strike");
        window(0u, 1u);
        require(topologyWasQualified(history) && !topologyIsDeletionCandidate(history), "Sparse support erased history");
        window(0u, 0u);
        require(topologyWasQualified(history), "Stochastic empty window erased history");
        window(5u, 1u);
        require(topologyIsDeletionCandidate(history), "Two qualifying windows did not confirm deletion");
        window(0u, 1u);
        require(topologyIsDeletionCandidate(history), "Weak window unlatched candidate");
        window(5u, 2u);
        require(history == 0u, "Reliable non-dominated foreground did not cancel candidate");
        window(6u, 2u);
        require(topologyWasQualified(history) && !topologyIsDeletionCandidate(history), "Inclusive ratio boundary failed");
        window(6u, 2u);
        require(topologyIsDeletionCandidate(history), "Dominant background did not override multiple support votes");
        pass.growNeighborVoxels(ctx);
        require(pass.mGridResources.gridData.activeVoxelCount == 1u && pass.mTopologySettings.lastGrowthCount == 0u,
            "Ratio-qualified deletion candidate still produced children");
        window(0u, 2u);
        require(history == 0u, "Strong foreground-only evidence did not clear history");
        auto evidence = read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 0u);
        evidence.deletionEligibleIteration = pass.mOptimizerParams.currentIteration + 10u;
        write(pass.mGridResources.topologyEvidencePages, 0u, evidence);
        window(20u, 0u);
        require(history == 0u, "Vote ratio bypassed newborn deletion protection");
        pass.mTopologySettings.minDeletionConflictViews = originalMinViews;
        pass.mTopologySettings.deletionConflictSupportRatio = originalRatio;
    }

    static void checkGrowthPenetration(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        const uint32_t origin = cell(3, 3, 3);
        auto sphere = parentVoxel();
        sphere.ellipsoid.rotation = float4(1, 0, 0, 0);
        sphere.ellipsoid.logScale = float3(std::log(0.6f));
        seed(pass, ctx, {origin}, {sphere});
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "A 0.05 depth threshold did not grow a 0.6-voxel sphere");
        require(pass.mGridResources.gridData.activeVoxelCount == 7u, "Depth-triggered growth cascaded within one round");

        seed(pass, ctx, {origin}, {sphere});
        pass.mTopologySettings.growthFacePenetration = 0.2f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "0.1-voxel penetration passed a 0.2-voxel threshold");
        seed(pass, ctx, {origin}, {sphere});
        pass.mTopologySettings.growthFacePenetration = 0.1f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "Mere tangency at the depth threshold triggered growth");
        pass.mTopologySettings.growthFacePenetration = 0.099f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "Lowering the depth threshold did not enable growth");

        for (float radius : {0.549f, 0.551f})
        {
            sphere.ellipsoid.logScale = float3(std::log(radius));
            seed(pass, ctx, {origin}, {sphere});
            pass.growNeighborVoxels(ctx);
            require(
                pass.mTopologySettings.lastGrowthCount == (radius > 0.55f ? 6u : 0u),
                "Growth did not distinguish depths just below/above the 0.05 threshold"
            );
        }
        sphere.ellipsoid.logScale = float3(std::log(0.5f));
        seed(pass, ctx, {origin}, {sphere});
        pass.mTopologySettings.growthFacePenetration = 0.0f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "Zero threshold accepted a tangent shared face");

        sphere.ellipsoid.logScale = float3(std::log(0.6f));
        for (float x : {0.25f, 0.75f})
        {
            sphere.ellipsoid.center = float3(x, 0.5f, 0.5f);
            seed(pass, ctx, {origin}, {sphere});
            pass.mTopologySettings.growthFacePenetration = 0.2f;
            pass.growNeighborVoxels(ctx);
            require(pass.mTopologySettings.lastGrowthCount == 1u, "Depth threshold was not evaluated separately for each face");
            const auto map = indices(pass, ctx);
            require(map[cell(x < 0.5f ? 2u : 4u, 3, 3)] >= 0, "Growth selected the wrong outward face direction");
        }
        auto small = sphere;
        small.ellipsoid.center = float3(0.5f);
        small.ellipsoid.logScale = float3(std::log(0.1f));
        seed(pass, ctx, {origin, cell(4, 3, 3)}, {sphere, small});
        pass.mTopologySettings.growthFacePenetration = 0.2f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "Growth duplicated an already occupied face neighbor");

        const float3 widths(2.0f, 3.0f, 0.5f);
        sphere.ellipsoid.center = float3(0.5f);
        sphere.ellipsoid.logScale = float3(std::log(0.6f * widths.x), std::log(0.6f * widths.y), std::log(0.6f * widths.z));
        seed(pass, ctx, {origin}, {sphere}, widths);
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "Depth was not normalized by the face axis's voxel width");

        // A 45-degree skinny shape crosses each XY face, but its AABB's 0.070
        // outward depth occurs outside the neighbor footprint. True depth in
        // that footprint is about 0.052, so 0.05 passes and 0.06 must fail.
        auto diagonal = parentVoxel();
        diagonal.ellipsoid.logScale = float3(std::log(0.8f), std::log(0.1f), std::log(0.1f));
        diagonal.ellipsoid.rotation = float4(std::cos(3.14159265359f / 8.0f), 0, 0, std::sin(3.14159265359f / 8.0f));
        seed(pass, ctx, {origin}, {diagonal});
        pass.mTopologySettings.growthFacePenetration = 0.06f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "AABB-only penetration outside the neighbor footprint triggered growth");
        pass.mTopologySettings.growthFacePenetration = 0.05f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 4u, "Valid rotated finite-face penetration did not grow XY neighbors");

        sphere.ellipsoid.logScale = float3(std::log(2.0f));
        seed(pass, ctx, {origin}, {sphere});
        pass.mTopologySettings.growthFacePenetration = 2.0f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "Delta 2 was silently clamped to 1 during GPU upload");
        pass.mTopologySettings.growthFacePenetration = 1.0f;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "Delta 1 and delta 2 did not produce distinct growth decisions");
        for (float radius : {2.5f, 2.6f})
        {
            sphere.ellipsoid.logScale = float3(std::log(radius));
            seed(pass, ctx, {origin}, {sphere});
            pass.mTopologySettings.growthFacePenetration = 2.0f;
            pass.growNeighborVoxels(ctx);
            require(
                pass.mTopologySettings.lastGrowthCount == (radius > 2.5f ? 6u : 0u),
                "Delta 2 did not require strictly more than two voxel widths beyond the shared face"
            );
        }
        for (float radius : {3.5f, 3.6f})
        {
            sphere.ellipsoid.logScale = float3(std::log(radius));
            seed(pass, ctx, {origin}, {sphere});
            pass.mTopologySettings.growthFacePenetration = 3.0f;
            pass.growNeighborVoxels(ctx);
            require(
                pass.mTopologySettings.lastGrowthCount == (radius > 3.5f ? 6u : 0u),
                "Growth penetration above delta 2 was clamped or used a non-strict boundary"
            );
        }
    }

    static void checkCoarseModeGrowth(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        const uint32_t origin = cell(3, 3, 3);
        auto sphere = parentVoxel();
        sphere.ellipsoid.rotation = float4(1, 0, 0, 0);
        sphere.ellipsoid.logScale = float3(std::log(2.6f));
        seed(pass, ctx, {origin}, {sphere});
        pass.mReconstructionMode = 1u;
        pass.mCoarseToFine.targetResolution = 16u;
        pass.mCoarseToFine.levelStartIteration = 40u;
        pass.mCoarseToFine.enableCoarseGrowth = true;
        pass.mCoarseToFine.coarseGrowthFacePenetration = 2.0f;
        pass.mCoarseToFine.coarseGrowthInterval = 20u;
        pass.mOptimizerParams.currentIteration = 59u;
        pass.growNeighborVoxels(ctx);
        require(pass.mGridResources.gridData.activeVoxelCount == 1u, "Mode1 coarse growth ignored its level-local cadence");
        pass.mOptimizerParams.currentIteration = 60u;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "Mode1 coarse growth did not run at delta 2 after twenty rounds");

        seed(pass, ctx, {origin}, {sphere});
        pass.mCoarseToFine.targetResolution = 8u;
        pass.mCoarseToFine.levelStartIteration = 40u;
        pass.mOptimizerParams.currentIteration = 60u;
        pass.growNeighborVoxels(ctx);
        require(pass.mGridResources.gridData.activeVoxelCount == 1u, "Mode1 growth remained enabled at the target level");
        pass.mReconstructionMode = 0u;
    }

    static void checkGrowthWaiting(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        const auto parent = parentVoxel();
        seed(pass, ctx, {cell(3, 3, 3)}, {parent});
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "Growth wait blocked an initialized parent");
        const uint32_t firstGenerationCount = pass.mGridResources.gridData.activeVoxelCount;
        auto newborn = read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 1);
        require(newborn.growthWaitStartIterationPlusOne == 51u, "Newborn growth-wait clock was not initialized");
        // Deliberately enlarge one child as if unstable updates had pushed it
        // through its other faces. Remove deletion protection to prove the
        // parent waiting gate is independent of the deletion gate.
        newborn.deletionEligibleIteration = 0u;
        write(pass.mGridResources.topologyEvidencePages, 1, newborn);
        write(pass.mGridResources.voxelPages, 1, parent);
        pass.barrierSparseVoxels(ctx);
        for (uint32_t iteration : {51u, 54u})
        {
            pass.mOptimizerParams.currentIteration = iteration;
            pass.growNeighborVoxels(ctx);
            require(pass.mTopologySettings.lastGrowthCount == 0u, "Unoptimized newborn grew before completing its wait");
        }
        pass.resetDeletionEvidence(ctx, true);
        pass.mEnableReconstruction = true;
        pass.mOptimizerParams.isRunning = true;
        pass.stopReconstruction();
        ctx->submit(true);
        require(
            read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, 1).growthWaitStartIterationPlusOne == 51u,
            "Evidence reset or stopping training erased the newborn wait"
        );

        pass.mOptimizerParams.currentIteration = 55u;
        pass.mTopologySettings.growthWaitIterations = 10u;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "Increasing the wait did not apply to an existing newborn");
        pass.mTopologySettings.growthWaitIterations = 5u;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount > 0u, "Newborn could not grow at the exact end of its five-round wait");
        const uint32_t secondGenerationCount = pass.mGridResources.gridData.activeVoxelCount;
        for (uint32_t id = firstGenerationCount; id < secondGenerationCount; ++id)
            require(
                read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, id).growthWaitStartIterationPlusOne == 56u,
                "Second-generation child inherited an old waiting clock"
            );
        write(pass.mGridResources.voxelPages, firstGenerationCount, parent);
        pass.barrierSparseVoxels(ctx);
        pass.mOptimizerParams.currentIteration = 56u;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 0u, "A new generation bypassed its own waiting period");
        pass.mTopologySettings.growthWaitIterations = 0u;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount > 0u, "Zero waiting period did not restore next-round growth");

        // Conversely, deletion protection must not block a mature growth
        // parent. Its growth age and its deletion expiry are separate clocks.
        seed(pass, ctx, {cell(3, 3, 3)}, {parent});
        newborn = {};
        newborn.growthBirthIterationPlusOne = 41u;
        newborn.growthWaitStartIterationPlusOne = 41u;
        newborn.deletionEligibleIteration = 100u;
        write(pass.mGridResources.topologyEvidencePages, 0, newborn);
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "Deletion protection prevented an otherwise mature parent from growing");
    }

    static void checkGrowthInterval(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        auto sphere = parentVoxel();
        sphere.ellipsoid.rotation = float4(1, 0, 0, 0);
        sphere.ellipsoid.logScale = float3(std::log(2.0f));
        seed(pass, ctx, {cell(3, 3, 3)}, {sphere});
        pass.mTopologySettings.growthFacePenetration = 1.0f;
        pass.mTopologySettings.growthInterval = 10u;
        pass.mOptimizerParams.currentIteration = 49u;
        pass.growNeighborVoxels(ctx);
        require(pass.mGridResources.gridData.activeVoxelCount == 1u, "Interval allowed growth before warm-up ended");
        pass.mOptimizerParams.currentIteration = 50u;
        pass.growNeighborVoxels(ctx);
        require(pass.mTopologySettings.lastGrowthCount == 6u, "First scheduled delta-1 growth did not run at round 50");
        const auto firstMap = indices(pass, ctx);
        const auto firstStatus = pass.mTopologySettings.growthStatus;
        // Even an enlarged child that passes the depth and five-round age
        // gates must wait until the next global ten-round boundary.
        write(pass.mGridResources.voxelPages, 1, sphere);
        pass.barrierSparseVoxels(ctx);
        for (uint32_t iteration = 51u; iteration < 60u; ++iteration)
        {
            pass.mOptimizerParams.currentIteration = iteration;
            pass.growNeighborVoxels(ctx);
            require(pass.mGridResources.gridData.activeVoxelCount == 7u, "Growth ran between scheduled boundaries");
            require(pass.mTopologySettings.lastGrowthCount == 6u && pass.mTopologySettings.growthStatus == firstStatus,
                "Skipped round erased the last growth result");
        }
        require(indices(pass, ctx) == firstMap, "Skipped growth rounds modified spatial indices");
        pass.mOptimizerParams.currentIteration = 60u;
        pass.growNeighborVoxels(ctx);
        require(pass.mGridResources.gridData.activeVoxelCount > 7u, "Child did not grow after ten complete rounds");
        for (uint32_t id = 7u; id < pass.mGridResources.gridData.activeVoxelCount; ++id)
            require(read<TopologyEvidence>(pass.mGridResources.topologyEvidencePages, id).growthWaitStartIterationPlusOne == 61u,
                "Scheduled newborn has an incorrect birth clock");

        // Changing the UI interval keeps its phase anchored to warm-up, not
        // absolute iteration multiples (53 is 50 + 3).
        seed(pass, ctx, {cell(3, 3, 3)}, {sphere});
        pass.mTopologySettings.growthInterval = 3u;
        pass.mTopologySettings.growthFacePenetration = 1.0f;
        pass.mOptimizerParams.currentIteration = 52u;
        pass.growNeighborVoxels(ctx);
        require(pass.mGridResources.gridData.activeVoxelCount == 1u, "Changed interval grew off cadence");
        pass.mOptimizerParams.currentIteration = 53u;
        pass.growNeighborVoxels(ctx);
        require(pass.mGridResources.gridData.activeVoxelCount == 7u, "Changed interval lost its warm-up anchor");
    }

    static void checkGrowthPreview(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        auto voxel = parentVoxel();
        voxel.ellipsoid.center = float3(0.01f);
        voxel.ellipsoid.logScale = float3(std::log(0.001f));
        voxel.opacity.coefficients[0] = -100.0f;
        seed(pass, ctx, {cell(3, 3, 2), cell(3, 3, 4), cell(2, 3, 2)}, {voxel, voxel, voxel});
        TopologyEvidence grown = {};
        grown.growthBirthIterationPlusOne = 51u;
        write(pass.mGridResources.topologyEvidencePages, 1, grown);

        // Compile and execute the production forward shader, not a test-only
        // approximation. The dummy scene supplies its required scene defines.
        auto scene = SceneBuilder(pass.mpDevice, Settings()).getScene();
        ProgramDesc desc;
        desc.addShaderLibrary(RayMarchingShaderFilePath).psEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        desc.addTypeConformances(scene->getTypeConformances());
        auto defines = scene->getSceneDefines();
        defines.add(pass.getReconstructionDefines());
        defines.add("CHECK_PRIMITIVE", "1");
        defines.add("USE_ENV_MAP", "0");
        auto preview = FullScreenPass::create(pass.mpDevice, desc, defines);
        auto var = preview->getRootVar();
        scene->bindShaderData(var["gScene"]);
        var["gGridDataParamBlock"] = pass.mpGridBlock;
        var["gPathRecordBuffer"] = pass.mpDevice->createStructuredBuffer(
            sizeof(PathRecord), 1u, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        auto accumulation = pass.mpDevice->createTexture2D(
            1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        var["gAccuColor"] = accumulation;
        var["GridData"]["gridMin"] = float3(0);
        var["GridData"]["voxelSize"] = float3(1);
        var["GridData"]["voxelCount"] = uint3(8);
        var["GridData"]["solidVoxelCount"] = 3u;
        auto cb = var["CB"];
        cb["pixelCount"] = uint2(1);
        cb["drawMode"] = uint32_t(ABSDFDrawMode::TopologyDebug);
        cb["enableReconstruction"] = false;
        cb["renderBackGround"] = false;
        cb["invSpp"] = 1.0f;
        cb["frameIndex"] = 0u;
        auto output = pass.mpDevice->createTexture2D(
            1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
        );
        auto fbo = Fbo::create(pass.mpDevice);
        fbo->attachColorTarget(output, 0);
        const auto render = [&](float x, TopologyDebugLayer layer, bool context, const float3& expected)
        {
            auto invVP = float4x4::identity();
            invVP[0][0] = invVP[1][1] = 0.001f;
            invVP[2][2] = 10.0f;
            invVP[0][3] = x;
            invVP[1][3] = 3.5f;
            invVP[2][3] = -1.0f;
            cb["invVP"] = invVP;
            cb["topologyDebugLayer"] = uint32_t(layer);
            cb["showOccupiedContext"] = context;
            ctx->clearUAV(accumulation->getUAV().get(), float4(0));
            preview->execute(ctx, fbo);
            auto bytes = ctx->readTextureSubresource(output.get(), 0);
            float4 pixel;
            require(bytes.size() == sizeof(pixel), "Unexpected preview readback layout");
            std::memcpy(&pixel, bytes.data(), sizeof(pixel));
            require(length(float3(pixel.x, pixel.y, pixel.z) - expected) < 1e-5f, "Topology preview color/filter is incorrect");
        };
        render(3.5f, TopologyDebugLayer::Growth, false, float3(0.1f, 1.0f, 0.25f));
        render(3.5f, TopologyDebugLayer::Growth, true, float3(0.1f, 1.0f, 0.25f));
        render(2.5f, TopologyDebugLayer::Growth, true, float3(0.18f, 0.2f, 0.23f));
        render(2.5f, TopologyDebugLayer::Growth, false, float3(0.02f, 0.02f, 0.025f));
        render(3.5f, TopologyDebugLayer::Occupied, false, float3(0.15f, 0.65f, 1.0f));
        grown.packedCountsAndFlags = kTopologyEvidenceDeletionCandidate;
        write(pass.mGridResources.topologyEvidencePages, 1, grown);
        render(3.5f, TopologyDebugLayer::Deletion, false, float3(1.0f, 0.12f, 0.04f));
    }

    static int run()
    {
        try
        {
            setErrorDiagnosticFlags(ErrorDiagnosticFlags::None);
            Device::Desc desc;
            desc.type = Device::Type::D3D12;
            desc.enableDebugLayer = true;
            auto device = make_ref<Device>(desc);
            auto ctx = device->getRenderContext();
            Properties properties;
            properties["reconstructionMode"] = 0u;
            auto pass = VoxelReconstructionNoLightTransport::create(device, properties);
            require(pass->mTopologySettings.growthFacePenetration == 1.0f, "Default growth depth is not delta 1");
            require(pass->mTopologySettings.growthInterval == 10u, "Default growth interval is not ten full rounds");
            require(
                pass->mCoarseToFine.enableCoarseGrowth && pass->mCoarseToFine.coarseGrowthFacePenetration == 2.0f &&
                    pass->mCoarseToFine.coarseGrowthInterval == 20u,
                "Default mode1 coarse-growth settings are incorrect"
            );
            pass->mUpdatePass.init();
            pass->createTopologyPassResource(ctx);
            pass->createDeletionPassResources();
            checkDeletionVoteRatio(*pass, ctx);
            std::cout << "PASS: deletion vote dominance, two-window confirmation, weak-window history, foreground recovery, protection\n";
            checkGrowthPenetration(*pass, ctx);
            std::cout << "PASS: face-depth trigger, strict thresholds, per-face direction, occupied neighbors, non-cubic widths, rotated "
                         "footprint\n";
            checkCoarseModeGrowth(*pass, ctx);
            std::cout << "PASS: mode1 grows every twenty level-local rounds at delta 2 and stops growth at the target\n";
            checkGrowthWaiting(*pass, ctx);
            std::cout << "PASS: newborn wait blocks unstable children, expires after full rounds, resets per generation, supports UI "
                         "changes, independent of deletion\n";
            checkGrowthInterval(*pass, ctx);
            std::cout << "PASS: delta-1 growth at rounds 50/60, no growth in between, last-result retention, configurable cadence\n";
            const auto parent = parentVoxel();
            const uint32_t origin = cell(3, 3, 3);
            seed(*pass, ctx, {origin}, {parent});
            GeometryAdamState parentAdam = {};
            parentAdam.centerSteps = 17;
            parentAdam.centerMean = float3(0.3f);
            write(pass->mGridResources.adamPages, 0, parentAdam);
            pass->mOptimizerParams.currentIteration = 49;
            pass->growNeighborVoxels(ctx);
            require(pass->mGridResources.gridData.activeVoxelCount == 1, "Growth started before opacity ramp ended");
            pass->mOptimizerParams.currentIteration = 50;
            pass->growNeighborVoxels(ctx);
            require(pass->mTopologySettings.lastGrowthCount == 6u, "One parent should produce exactly six face neighbors");
            auto retained = read<VoxelData>(pass->mGridResources.voxelPages, 0);
            require(std::memcmp(&retained, &parent, sizeof(parent)) == 0, "Growth modified parent data");
            auto retainedAdam = read<GeometryAdamState>(pass->mGridResources.adamPages, 0);
            require(std::memcmp(&retainedAdam, &parentAdam, sizeof(parentAdam)) == 0, "Growth reset parent Adam");
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).growthBirthIterationPlusOne == 0u,
                "Growth marked the original parent as grown"
            );
            for (uint32_t id = 1; id < 7; ++id)
            {
                checkChild(
                    parent,
                    read<VoxelData>(pass->mGridResources.voxelPages, id),
                    read<uint32_t>(pass->mGridResources.cellIndexPages, id),
                    origin
                );
                auto adam = read<GeometryAdamState>(pass->mGridResources.adamPages, id);
                const GeometryAdamState zero = {};
                require(std::memcmp(&adam, &zero, sizeof(adam)) == 0, "Newborn Adam was not zeroed");
                require(
                    read<uint32_t>(pass->mGridResources.radianceAdamIndexPages, id) == 0xffffffffu, "Newborn inherited a radiance Adam slot"
                );
                require(
                    read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, id).deletionEligibleIteration == 55u,
                    "Newborn protection expiry is incorrect"
                );
                require(
                    read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, id).growthBirthIterationPlusOne == 51u,
                    "Newborn did not retain its growth birth round"
                );
                require(
                    read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, id).growthWaitStartIterationPlusOne == 51u,
                    "Newborn did not receive its own growth-wait clock"
                );
            }
            std::cout << "PASS: trigger timing, single layer, rotated five-face bounds, inheritance, low opacity, fresh Adam\n";
            checkDeletionProtection(*pass, ctx);
            std::cout << "PASS: newborn evidence gate preserves gradients; expiry restores unique-view votes\n";

            TopologyEvidence deletion = {};
            deletion.packedCountsAndFlags = kTopologyEvidenceDeletionCandidate;
            write(pass->mGridResources.topologyEvidencePages, 0, deletion);
            pass->mOptimizerParams.currentIteration = 51;
            pass->mTopologySettings.candidateCount = 1;
            pass->deleteAndCompactCandidates(ctx);
            require(pass->mGridResources.gridData.activeVoxelCount == 6u, "Compaction did not delete the parent");
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).deletionEligibleIteration == 55u,
                "Compaction lost newborn protection"
            );
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).growthBirthIterationPlusOne == 51u,
                "Compaction/evidence reset lost the moved newborn's growth marker"
            );
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).growthWaitStartIterationPlusOne == 51u,
                "Compaction/evidence reset lost the moved newborn's growth wait"
            );
            auto map = indices(*pass, ctx);
            require(map[origin] == -58, "Cooldown is not bound to the deleted spatial cell");
            pass->mTopologySettings.growthFacePenetration = 0.0f;
            pass->growNeighborVoxels(ctx);
            require(indices(*pass, ctx)[origin] < 0, "Deleted cell grew back during cooldown");
            pass->mOptimizerParams.currentIteration = 55;
            pass->growNeighborVoxels(ctx);
            require(indices(*pass, ctx)[origin] < 0, "Cooldown expired early");
            pass->mOptimizerParams.currentIteration = 56;
            pass->growNeighborVoxels(ctx);
            require(indices(*pass, ctx)[origin] >= 0, "Cell did not become available after cooldown");
            std::cout << "PASS: deletion compaction preserves protection, spatial cooldown blocks and expires\n";

            seed(*pass, ctx, {cell(2, 3, 3), cell(4, 3, 3)}, {parent, parent});
            pass->growNeighborVoxels(ctx);
            require(pass->mTopologySettings.lastGrowthCount == 11u, "Shared neighbor was not deduplicated");
            std::cout << "PASS: two parents create their shared neighbor only once\n";

            auto diagonal = parent;
            diagonal.ellipsoid.center = float3(0.05f, 0.95f, 0.5f);
            diagonal.ellipsoid.logScale = float3(std::log(4.0f), std::log(0.01f), std::log(0.01f));
            diagonal.ellipsoid.rotation = float4(std::cos(3.14159265359f / 8.0f), 0, 0, std::sin(3.14159265359f / 8.0f));
            seed(*pass, ctx, {origin}, {diagonal});
            pass->growNeighborVoxels(ctx);
            require(
                pass->mTopologySettings.lastGrowthCount == 2u && indices(*pass, ctx)[cell(4, 3, 3)] == -1,
                "Bounding-box face crossing was mistaken for a finite-face intersection"
            );
            std::cout << "PASS: finite-face test rejects a rotated bounding-box false positive\n";

            std::vector<uint32_t> cells;
            std::vector<VoxelData> voxels;
            auto small = parent;
            small.ellipsoid.logScale = float3(std::log(0.1f));
            for (uint32_t i = 0; i < 29; ++i)
            {
                cells.push_back(i);
                voxels.push_back(small);
            }
            cells.push_back(cell(5, 5, 5));
            voxels.push_back(parent);
            seed(*pass, ctx, cells, voxels);
            auto oldPage = pass->mGridResources.voxelPages[0];
            pass->growNeighborVoxels(ctx);
            require(
                pass->mGridResources.gridData.activeVoxelCount == 36u && pass->mGridResources.voxelPages.size() == 2u,
                "Growth did not expand across the page boundary"
            );
            require(pass->mGridResources.voxelPages[0] == oldPage, "Expansion replaced the old parameter page");
            std::cout << "PASS: on-demand page extension retains existing pages\n";

            cells.clear();
            voxels.clear();
            for (uint32_t i = 0; i < 62; ++i)
            {
                cells.push_back(i);
                voxels.push_back(small);
            }
            cells.push_back(cell(5, 5, 5));
            voxels.push_back(parent);
            seed(*pass, ctx, cells, voxels);
            pass->growNeighborVoxels(ctx);
            require(
                !pass->mTopologySettings.enableGrowth && pass->mGridResources.gridData.activeVoxelCount == 63u,
                "Pool overflow did not pause growth and retain existing voxels"
            );
            map = indices(*pass, ctx);
            std::set<int32_t> ids;
            for (int32_t id : map)
            {
                require(id > kGrowthClaimBase, "Failed growth left a temporary claim");
                if (id >= 0)
                    ids.insert(id);
            }
            require(ids.size() == 63u && *ids.rbegin() == 62, "Failed growth damaged the spatial index");
            std::cout << "PASS: capacity overflow rolls back all claims without partial growth\n";

            seed(*pass, ctx, {origin}, {parent});
            pass->growNeighborVoxels(ctx);
            write(pass->mGridResources.topologyEvidencePages, 0, deletion);
            pass->mOptimizerParams.currentIteration = 51;
            pass->mTopologySettings.candidateCount = 1;
            pass->deleteAndCompactCandidates(ctx);
            const auto beforeRestart = read<VoxelData>(pass->mGridResources.voxelPages, 0);
            pass->mEnableReconstruction = true;
            pass->mOptimizerParams.isRunning = true;
            pass->stopReconstruction();
            require(
                !pass->mEnableReconstruction && !pass->mOptimizerParams.isRunning &&
                    read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).growthBirthIterationPlusOne == 51u,
                "Stopping optimization lost the growth marker"
            );
            pass->resetPointCloudOptimization(ctx);
            ctx->submit(true);
            require(indices(*pass, ctx)[origin] == -1, "Optimization restart retained old cooldown");
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).deletionEligibleIteration == 0u,
                "Optimization restart retained old protection expiry"
            );
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).growthWaitStartIterationPlusOne == 0u,
                "Optimization restart retained a waiting clock from the previous training timeline"
            );
            const auto afterRestart = read<VoxelData>(pass->mGridResources.voxelPages, 0);
            require(std::memcmp(&beforeRestart, &afterRestart, sizeof(beforeRestart)) == 0, "Restart changed voxel parameters");
            std::cout << "PASS: restart clears transient protection/cooldown without changing geometry\n";
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).growthBirthIterationPlusOne == 51u,
                "Optimization restart lost persistent growth lineage"
            );
            seed(*pass, ctx, {origin}, {parent});
            require(
                read<TopologyEvidence>(pass->mGridResources.topologyEvidencePages, 0).growthBirthIterationPlusOne == 0u,
                "Fresh initialization retained an old growth marker"
            );
            std::cout << "PASS: growth lineage survives stop/restart/compaction and clears on fresh initialization\n";
            checkGrowthPreview(*pass, ctx);
            std::cout << "PASS: production preview highlights low-opacity grown cells, filters originals, retains "
                         "context/deletion/occupied layers\n";
            runCoarseToFineTests(device);
            device->wait();
            std::cout << "All neighbor-growth GPU regression tests passed.\n";
            return 0;
        }
        catch (const std::exception& error)
        {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
};

int main()
{
    return NeighborGrowthTestAccess::run();
}
