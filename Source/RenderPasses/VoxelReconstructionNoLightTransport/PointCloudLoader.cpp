#include "PointCloudLoader.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace PointCloudInitialization
{
namespace
{
enum class ScalarType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };

struct Property
{
    std::string name;
    ScalarType valueType;
    bool isList = false;
    ScalarType countType = ScalarType::UInt8;
};

struct Element
{
    std::string name;
    uint64_t count;
    std::vector<Property> properties;
};

constexpr size_t kMissingProperty = std::numeric_limits<size_t>::max();

struct Header
{
    bool binary = false;
    std::vector<Element> elements;
    size_t vertexElement = 0;
    std::array<size_t, 3> positions;
    // Optional per-Gaussian properties of a 3DGS reconstruction. Only when all of them are
    // present does the ellipsoid decide occupancy.
    size_t opacity = kMissingProperty;
    std::array<size_t, 3> scales = {kMissingProperty, kMissingProperty, kMissingProperty};
    std::array<size_t, 4> rotations = {kMissingProperty, kMissingProperty, kMissingProperty, kMissingProperty};
    bool hasGaussianProperties = false;
};

// Cost guard: one Gaussian never sweeps more than this many voxels along one axis. The number of
// Gaussians whose coverage was clipped is reported, so a binding limit is visible.
constexpr double kMaxCoverageVoxels = 8.0;

bool anyMissing(const size_t* indices, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        if (indices[i] == kMissingProperty) return true;
    return false;
}

// Rotation matrix of a (w, x, y, z) quaternion, the convention of gaussianRotationMatrix in the
// shaders. A degenerate quaternion falls back to the identity.
void makeRotation(const std::array<double, 4>& quaternion, double matrix[3][3])
{
    double norm = 0;
    for (size_t i = 0; i < 4; ++i) norm += quaternion[i] * quaternion[i];
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm < 1e-12)
    {
        for (size_t row = 0; row < 3; ++row)
            for (size_t column = 0; column < 3; ++column) matrix[row][column] = row == column ? 1.0 : 0.0;
        return;
    }
    const double w = quaternion[0] / norm;
    const double x = quaternion[1] / norm;
    const double y = quaternion[2] / norm;
    const double z = quaternion[3] / norm;
    matrix[0][0] = 1.0 - 2.0 * (y * y + z * z);
    matrix[0][1] = 2.0 * (x * y - z * w);
    matrix[0][2] = 2.0 * (x * z + y * w);
    matrix[1][0] = 2.0 * (x * y + z * w);
    matrix[1][1] = 1.0 - 2.0 * (x * x + z * z);
    matrix[1][2] = 2.0 * (y * z - x * w);
    matrix[2][0] = 2.0 * (x * z - y * w);
    matrix[2][1] = 2.0 * (y * z + x * w);
    matrix[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error("Point-cloud PLY: " + message);
}

ScalarType scalarType(const std::string& name)
{
    if (name == "char" || name == "int8") return ScalarType::Int8;
    if (name == "uchar" || name == "uint8") return ScalarType::UInt8;
    if (name == "short" || name == "int16") return ScalarType::Int16;
    if (name == "ushort" || name == "uint16") return ScalarType::UInt16;
    if (name == "int" || name == "int32") return ScalarType::Int32;
    if (name == "uint" || name == "uint32") return ScalarType::UInt32;
    if (name == "float" || name == "float32") return ScalarType::Float32;
    if (name == "double" || name == "float64") return ScalarType::Float64;
    fail("unsupported property type '" + name + "'.");
}

bool isInteger(ScalarType type)
{
    return type != ScalarType::Float32 && type != ScalarType::Float64;
}

size_t scalarBytes(ScalarType type)
{
    switch (type)
    {
    case ScalarType::Int8: case ScalarType::UInt8: return 1;
    case ScalarType::Int16: case ScalarType::UInt16: return 2;
    case ScalarType::Int32: case ScalarType::UInt32: case ScalarType::Float32: return 4;
    case ScalarType::Float64: return 8;
    }
    fail("invalid scalar type.");
}

uint64_t parseCount(const std::string& text)
{
    uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
        fail("invalid element count '" + text + "'.");
    return value;
}

void requireEnd(std::istringstream& line)
{
    std::string extra;
    if (line >> extra) fail("unexpected token in the header: '" + extra + "'.");
}

Header readHeader(std::ifstream& input, uint64_t fileBytes)
{
    Header header;
    std::string line;
    if (!std::getline(input, line)) fail("empty file.");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply") fail("missing PLY magic.");

    bool hasFormat = false;
    bool hasEnd = false;
    uint64_t headerBytes = 4;
    while (std::getline(input, line))
    {
        headerBytes += line.size() + 1;
        if (headerBytes > 1024 * 1024) fail("header exceeds 1 MiB.");
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword.empty() || keyword == "comment" || keyword == "obj_info") continue;
        if (keyword == "format")
        {
            std::string format, version;
            if (hasFormat || !(tokens >> format >> version) || version != "1.0") fail("invalid PLY format declaration.");
            if (format != "ascii" && format != "binary_little_endian")
                fail("unsupported format '" + format + "'; expected ascii or binary_little_endian.");
            header.binary = format == "binary_little_endian";
            hasFormat = true;
            requireEnd(tokens);
        }
        else if (keyword == "element")
        {
            std::string name, count;
            if (!(tokens >> name >> count)) fail("invalid element declaration.");
            requireEnd(tokens);
            for (const auto& element : header.elements)
                if (element.name == name) fail("duplicate element '" + name + "'.");
            const uint64_t parsedCount = parseCount(count);
            // Every supported nonempty element consumes at least one byte per item.
            if (parsedCount > fileBytes) fail("element count exceeds the file size (truncated or invalid file).");
            header.elements.push_back({name, parsedCount, {}});
        }
        else if (keyword == "property")
        {
            if (header.elements.empty()) fail("property appears before its element.");
            std::string type, name;
            if (!(tokens >> type)) fail("invalid property declaration.");
            Property property;
            if (type == "list")
            {
                std::string countType, valueType;
                if (!(tokens >> countType >> valueType >> name)) fail("invalid list property declaration.");
                property.isList = true;
                property.countType = scalarType(countType);
                property.valueType = scalarType(valueType);
                if (!isInteger(property.countType)) fail("list count type must be an integer.");
            }
            else
            {
                if (!(tokens >> name)) fail("missing property name.");
                property.valueType = scalarType(type);
            }
            requireEnd(tokens);
            property.name = name;
            auto& properties = header.elements.back().properties;
            for (const auto& previous : properties)
                if (previous.name == name) fail("duplicate property '" + name + "'.");
            properties.push_back(std::move(property));
        }
        else if (keyword == "end_header")
        {
            requireEnd(tokens);
            hasEnd = true;
            break;
        }
        else fail("unsupported header declaration '" + keyword + "'.");
    }
    if (!hasFormat || !hasEnd) fail("missing format or end_header.");

    header.positions.fill(kMissingProperty);
    bool hasVertex = false;
    const std::array<std::string, 3> positionNames = {"x", "y", "z"};
    const std::array<std::string, 3> scaleNames = {"scale_0", "scale_1", "scale_2"};
    const std::array<std::string, 4> rotationNames = {"rot_0", "rot_1", "rot_2", "rot_3"};
    for (size_t elementIndex = 0; elementIndex < header.elements.size(); ++elementIndex)
    {
        const auto& element = header.elements[elementIndex];
        if (element.count != 0 && element.properties.empty()) fail("nonempty element has no properties.");
        if (element.name != "vertex") continue;
        hasVertex = true;
        header.vertexElement = elementIndex;
        for (size_t propertyIndex = 0; propertyIndex < element.properties.size(); ++propertyIndex)
        {
            const auto& property = element.properties[propertyIndex];
            for (size_t axis = 0; axis < 3; ++axis)
            {
                if (property.name == positionNames[axis])
                {
                    if (property.isList) fail("vertex positions must be scalar properties.");
                    header.positions[axis] = propertyIndex;
                }
            }
            if (property.isList) continue;
            if (property.name == "opacity") header.opacity = propertyIndex;
            for (size_t axis = 0; axis < 3; ++axis)
                if (property.name == scaleNames[axis]) header.scales[axis] = propertyIndex;
            for (size_t i = 0; i < 4; ++i)
                if (property.name == rotationNames[i]) header.rotations[i] = propertyIndex;
        }
    }
    if (!hasVertex) fail("no vertex element.");
    for (const auto index : header.positions)
        if (index == kMissingProperty) fail("vertex element must contain x, y and z.");
    // Per-Gaussian properties are optional: a plain point cloud only marks its containing voxel.
    header.hasGaussianProperties = header.opacity != kMissingProperty &&
        !anyMissing(header.scales.data(), header.scales.size()) &&
        !anyMissing(header.rotations.data(), header.rotations.size());
    return header;
}

class PayloadReader
{
public:
    PayloadReader(std::ifstream& input, bool binary, uint64_t fileBytes) : mInput(input), mBinary(binary), mFileBytes(fileBytes) {}

    double scalar(ScalarType type)
    {
        if (!mBinary) return asciiScalar(type);
        unsigned char bytes[8] = {};
        const size_t size = scalarBytes(type);
        if (!mInput.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(size))) fail("truncated binary payload.");
        uint64_t bits = 0;
        for (size_t i = 0; i < size; ++i) bits |= uint64_t(bytes[i]) << (8 * i);
        switch (type)
        {
        case ScalarType::UInt8: case ScalarType::UInt16: case ScalarType::UInt32: return static_cast<double>(bits);
        case ScalarType::Int8: return bits < 128 ? double(bits) : double(int64_t(bits) - 256);
        case ScalarType::Int16: return bits < 32768 ? double(bits) : double(int64_t(bits) - 65536);
        case ScalarType::Int32: return bits < 2147483648ull ? double(bits) : double(int64_t(bits) - 4294967296ll);
        case ScalarType::Float32:
        {
            const uint32_t word = static_cast<uint32_t>(bits);
            float value;
            static_assert(sizeof(value) == sizeof(word), "PLY reader requires 32-bit float.");
            std::memcpy(&value, &word, sizeof(value));
            return value;
        }
        case ScalarType::Float64:
        {
            double value;
            static_assert(sizeof(value) == sizeof(bits), "PLY reader requires 64-bit double.");
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        }
        fail("invalid scalar type.");
    }

    void skip(const Property& property)
    {
        if (!property.isList)
        {
            scalar(property.valueType);
            return;
        }
        const double countValue = scalar(property.countType);
        if (countValue < 0 || countValue > double(mFileBytes)) fail("invalid list length.");
        const uint64_t count = static_cast<uint64_t>(countValue);
        if (mBinary)
        {
            const uint64_t bytes = count * scalarBytes(property.valueType);
            if (bytes > mFileBytes || bytes > uint64_t(std::numeric_limits<std::streamsize>::max())) fail("list exceeds file size.");
            mInput.ignore(static_cast<std::streamsize>(bytes));
            if (uint64_t(mInput.gcount()) != bytes) fail("truncated binary list.");
        }
        else
            for (uint64_t i = 0; i < count; ++i) scalar(property.valueType);
    }

private:
    double asciiScalar(ScalarType type)
    {
        std::string token;
        if (!(mInput >> token)) fail("truncated ASCII payload.");
        if (isInteger(type))
        {
            int64_t value = 0;
            const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
            if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size()) fail("invalid ASCII integer '" + token + "'.");
            int64_t lo = 0, hi = 0;
            switch (type)
            {
            case ScalarType::Int8: lo = -128; hi = 127; break;
            case ScalarType::UInt8: hi = 255; break;
            case ScalarType::Int16: lo = -32768; hi = 32767; break;
            case ScalarType::UInt16: hi = 65535; break;
            case ScalarType::Int32: lo = -2147483647ll - 1; hi = 2147483647ll; break;
            case ScalarType::UInt32: hi = 4294967295ll; break;
            default: break;
            }
            if (value < lo || value > hi) fail("ASCII integer is outside its declared type range.");
            return static_cast<double>(value);
        }
        double value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value, std::chars_format::general);
        if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size()) fail("invalid ASCII number '" + token + "'.");
        return value;
    }

    std::ifstream& mInput;
    bool mBinary;
    uint64_t mFileBytes;
};

} // namespace

Result load(
    const std::filesystem::path& path,
    const std::array<uint32_t, 3>& voxelCount,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& voxelSize,
    double opacityThreshold
)
{
    // Occupancy of a 3DGS point cloud: a Gaussian contributes alpha * exp(-r^2 / 2) at a point,
    // with r^2 = q^T Sigma^-1 q measured from its center, so a cell is occupied while that value
    // stays above the threshold, i.e. while r^2 <= 2 ln(alpha / opacityThreshold).
    if (!(opacityThreshold > 0.0 && opacityThreshold < 1.0))
        fail("opacity threshold must be in (0, 1).");

    uint64_t totalVoxelCount = 1;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        if (voxelCount[axis] == 0 || !std::isfinite(gridMin[axis]) || !std::isfinite(voxelSize[axis]) || voxelSize[axis] <= 0)
            fail("invalid voxel grid dimensions or bounds.");
        totalVoxelCount *= voxelCount[axis];
        if (totalVoxelCount > std::numeric_limits<uint32_t>::max()) fail("voxel grid exceeds the 32-bit index range.");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open '" + path.string() + "'.");
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) fail("cannot determine file size.");
    const uint64_t fileBytes = static_cast<uint64_t>(end);
    input.seekg(0);
    const Header header = readHeader(input, fileBytes);
    PayloadReader reader(input, header.binary, fileBytes);

    Result result;
    result.statistics.inputPoints = header.elements[header.vertexElement].count;

    // Occupancy bitmask: Gaussian coverage overlaps heavily, and 128^3 cells is only 256 KiB.
    std::vector<uint64_t> occupancy(static_cast<size_t>((totalVoxelCount + 63) / 64), 0);
    const auto mark = [&occupancy](uint64_t index) { occupancy[static_cast<size_t>(index >> 6)] |= uint64_t(1) << (index & 63); };
    const auto cellOfWorld = [&](const std::array<double, 3>& world, std::array<uint32_t, 3>& cell)
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const double coordinate = (world[axis] - double(gridMin[axis])) / double(voxelSize[axis]);
            if (!(coordinate >= 0 && coordinate < double(voxelCount[axis]))) return false;
            cell[axis] = static_cast<uint32_t>(std::floor(coordinate));
        }
        return true;
    };
    const auto cellIndexOf = [&](const std::array<uint32_t, 3>& cell)
    {
        return static_cast<uint32_t>(uint64_t(cell[0]) + uint64_t(cell[1]) * voxelCount[0] +
            uint64_t(cell[2]) * voxelCount[0] * voxelCount[1]);
    };

    for (size_t elementIndex = 0; elementIndex < header.elements.size(); ++elementIndex)
    {
        const auto& element = header.elements[elementIndex];
        const bool vertex = elementIndex == header.vertexElement;
        for (uint64_t item = 0; item < element.count; ++item)
        {
            std::array<double, 3> position = {};
            double opacity = 0.0;
            std::array<double, 3> logScale = {};
            std::array<double, 4> rotation = {};
            for (size_t propertyIndex = 0; propertyIndex < element.properties.size(); ++propertyIndex)
            {
                const auto& property = element.properties[propertyIndex];
                bool consumed = false;
                if (vertex)
                {
                    for (size_t axis = 0; axis < 3; ++axis)
                    {
                        if (header.positions[axis] == propertyIndex)
                        {
                            position[axis] = reader.scalar(property.valueType);
                            consumed = true;
                            break;
                        }
                    }
                    if (!consumed && header.opacity == propertyIndex)
                    {
                        opacity = reader.scalar(property.valueType);
                        consumed = true;
                    }
                    if (!consumed)
                        for (size_t axis = 0; axis < 3 && !consumed; ++axis)
                        {
                            if (header.scales[axis] == propertyIndex)
                            {
                                logScale[axis] = reader.scalar(property.valueType);
                                consumed = true;
                            }
                        }
                    if (!consumed)
                        for (size_t i = 0; i < 4 && !consumed; ++i)
                        {
                            if (header.rotations[i] == propertyIndex)
                            {
                                rotation[i] = reader.scalar(property.valueType);
                                consumed = true;
                            }
                        }
                }
                if (!consumed) reader.skip(property);
            }
            if (!vertex) continue;

            bool valid = true;
            for (size_t axis = 0; axis < 3; ++axis)
                valid = valid && std::isfinite(position[axis]);
            if (!valid)
            {
                ++result.statistics.invalidPoints;
                continue;
            }
            const std::array<double, 3> world = {position[0], position[2], -position[1]};

            if (!header.hasGaussianProperties)
            {
                // Plain point cloud: mark the containing voxel only.
                std::array<uint32_t, 3> cell = {};
                if (!cellOfWorld(world, cell))
                {
                    ++result.statistics.outsidePoints;
                    continue;
                }
                mark(cellIndexOf(cell));
                continue;
            }

            // 3DGS Gaussian: opacity is a logit, scales are log of the world-unit sigma, the
            // quaternion is (w, x, y, z).
            for (size_t axis = 0; axis < 3; ++axis)
                valid = valid && std::isfinite(logScale[axis]);
            for (size_t i = 0; i < 4; ++i)
                valid = valid && std::isfinite(rotation[i]);
            // Drop NaN rows and the very transparent tail: an invisible Gaussian claims no voxels.
            const double alpha = 1.0 / (1.0 + std::exp(-opacity));
            if (!valid || !std::isfinite(alpha))
            {
                ++result.statistics.invalidPoints;
                continue;
            }
            if (alpha < opacityThreshold)
            {
                ++result.statistics.droppedByOpacity;
                continue;
            }

            std::array<double, 3> sigma = {};
            for (size_t axis = 0; axis < 3; ++axis) sigma[axis] = std::exp(logScale[axis]);
            if (!(std::isfinite(sigma[0]) && sigma[0] > 0 && std::isfinite(sigma[1]) && sigma[1] > 0 &&
                    std::isfinite(sigma[2]) && sigma[2] > 0))
            {
                ++result.statistics.invalidPoints;
                continue;
            }

            double rotationMatrix[3][3];
            makeRotation(rotation, rotationMatrix);

            // Sigma = R diag(sigma^2) R^T, so its diagonal gives the squared extent along each NeRF
            // axis, and the covered region is the ellipsoid r <= radius with r^2 <= threshold.
            const double radius = std::sqrt(2.0 * std::log(alpha / opacityThreshold));
            const double threshold = radius * radius;
            std::array<double, 3> half = {};
            for (size_t axis = 0; axis < 3; ++axis)
            {
                double variance = 0;
                for (size_t i = 0; i < 3; ++i)
                {
                    const double component = rotationMatrix[axis][i] * sigma[i];
                    variance += component * component;
                }
                half[axis] = radius * std::sqrt(variance);
            }
            // (x, y, z) -> (x, z, -y) is a 90 degree rotation, so the extents are simply permuted.
            half = {half[0], half[2], half[1]};
            bool capped = false;
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const double limit = kMaxCoverageVoxels * double(voxelSize[axis]);
                if (half[axis] > limit)
                {
                    half[axis] = limit;
                    capped = true;
                }
            }
            if (capped) ++result.statistics.cappedCoverage;

            // Candidate cells: the axis-aligned box of the ellipsoid, clipped to the grid.
            std::array<uint32_t, 3> lo = {}, hi = {};
            bool overlaps = true;
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const double scale = double(voxelSize[axis]);
                const double low = (world[axis] - half[axis] - double(gridMin[axis])) / scale;
                const double high = (world[axis] + half[axis] - double(gridMin[axis])) / scale;
                if (high < 0 || low >= double(voxelCount[axis]))
                {
                    overlaps = false;
                    break;
                }
                lo[axis] = static_cast<uint32_t>(std::max(0.0, std::floor(low)));
                hi[axis] = static_cast<uint32_t>(std::min(double(voxelCount[axis]) - 1.0, std::floor(high)));
            }
            if (!overlaps)
            {
                ++result.statistics.outsidePoints;
                continue;
            }

            // A Gaussian thinner than a cell can miss every cell center, so it always claims the
            // voxel containing its own center.
            std::array<uint32_t, 3> centerCell = {};
            if (cellOfWorld(world, centerCell)) mark(cellIndexOf(centerCell));

            uint64_t covered = 0;
            for (uint32_t z = lo[2]; z <= hi[2]; ++z)
                for (uint32_t y = lo[1]; y <= hi[1]; ++y)
                    for (uint32_t x = lo[0]; x <= hi[0]; ++x)
                    {
                        // Cell center in the renderer frame, mapped back into the NeRF frame of the
                        // PLY: (x, y, z) = (forward.x, -forward.z, forward.y).
                        const double forwardX = double(gridMin[0]) + (double(x) + 0.5) * double(voxelSize[0]);
                        const double forwardY = double(gridMin[1]) + (double(y) + 0.5) * double(voxelSize[1]);
                        const double forwardZ = double(gridMin[2]) + (double(z) + 0.5) * double(voxelSize[2]);
                        const double q[3] = {
                            forwardX - position[0], -forwardZ - position[1], forwardY - position[2]};

                        double r2 = 0;
                        for (size_t axis = 0; axis < 3; ++axis)
                        {
                            // principle = R^T q, so this is the component along principal axis `axis`.
                            const double principal = rotationMatrix[0][axis] * q[0] +
                                rotationMatrix[1][axis] * q[1] + rotationMatrix[2][axis] * q[2];
                            const double normalized = principal / sigma[axis];
                            r2 += normalized * normalized;
                        }
                        if (r2 <= threshold)
                        {
                            mark(cellIndexOf({x, y, z}));
                            ++covered;
                        }
                    }
            result.statistics.coveredCells += covered;
            result.statistics.maxCoveredCells = std::max(result.statistics.maxCoveredCells, covered);
        }
    }

    // Scan order is ascending, so the seeds come out sorted and each occupied index appears once.
    for (size_t word = 0; word < occupancy.size(); ++word)
        for (uint32_t bit = 0; bit < 64; ++bit)
            if ((occupancy[word] >> bit) & 1ull)
            {
                const uint64_t index = uint64_t(word) * 64 + bit;
                if (index < totalVoxelCount) result.seeds.push_back({static_cast<uint32_t>(index)});
            }
    if (result.seeds.empty())
        fail("no occupied voxels inside the voxel grid (input=" + std::to_string(result.statistics.inputPoints) +
            ", invalid=" + std::to_string(result.statistics.invalidPoints) + ", outside=" +
            std::to_string(result.statistics.outsidePoints) + "). Check the point-cloud coordinates and grid bounds.");
    return result;
}
} // namespace PointCloudInitialization
