#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace PointCloudInitialization
{
struct Seed
{
    uint32_t voxelIndex;
};
static_assert(sizeof(Seed) == 4, "Point-cloud seeds must match the shader buffer layout.");

struct Statistics
{
    uint64_t inputPoints = 0;
    uint64_t invalidPoints = 0;
    uint64_t outsidePoints = 0;
    // 3DGS input only. Coverage is clipped to the grid, never to a fixed per-Gaussian radius.
    uint64_t droppedByOpacity = 0; // peak opacity below the visibility threshold
    uint64_t testedBlocks = 0;
    uint64_t skippedFullBlocks = 0;
    uint64_t testedCells = 0;     // exact ellipsoid/AABB tests at the boundary
    uint64_t coveredCells = 0;    // newly occupied cells, excluding repeated coverage
    uint64_t maxCoveredCells = 0; // largest number of newly occupied cells from one Gaussian
};

struct Result
{
    std::vector<Seed> seeds;
    Statistics statistics;
};

// Input positions use the NeRF world coordinates of the reference dataset.
// The renderer uses (x, z, -y).
//
// A 3DGS reconstruction carries the per-Gaussian properties (opacity, scale_0..2, rot_0..3), and
// Every voxel whose closed AABB intersects the Gaussian's threshold ellipsoid is occupied.
// This preserves thin walls even when the covered region misses every voxel center.
// Without those properties a vertex only marks the voxel containing it.
// `opacityThreshold` is compared against the peak opacity sigmoid(opacity) and must be in (0, 1).
Result load(
    const std::filesystem::path& path,
    const std::array<uint32_t, 3>& voxelCount,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& voxelSize,
    double opacityThreshold
);
} // namespace PointCloudInitialization
