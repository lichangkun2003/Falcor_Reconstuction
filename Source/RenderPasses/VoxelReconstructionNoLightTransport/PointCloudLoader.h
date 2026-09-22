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
};

struct Result
{
    std::vector<Seed> seeds;
    Statistics statistics;
};

// Input positions use the NeRF world coordinates of the reference dataset.
// The renderer uses (x, z, -y). Only positions affect occupancy; other properties
// are skipped. No neighboring cells are populated.
Result load(
    const std::filesystem::path& path,
    const std::array<uint32_t, 3>& voxelCount,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& voxelSize
);
} // namespace PointCloudInitialization
