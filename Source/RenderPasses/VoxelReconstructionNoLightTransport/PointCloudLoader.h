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
    // 3DGS input only. By-product of the coverage test, useful to judge the cost guard.
    uint64_t droppedByOpacity = 0; // peak opacity below the visibility threshold
    uint64_t cappedCoverage = 0;   // coverage box clipped by the per-axis cost guard
    uint64_t coveredCells = 0;     // cell marks written by the coverage test
    uint64_t maxCoveredCells = 0;  // largest per-Gaussian coverage
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
// then the ellipsoid decides occupancy: every voxel whose center the Gaussian still covers above
// `opacityThreshold` is occupied, which fills the regions a raw point cloud leaves hollow.
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
