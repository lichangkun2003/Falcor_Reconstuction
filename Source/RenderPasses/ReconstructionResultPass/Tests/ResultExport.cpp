#include "../ReconstructionResultPass.h"
#include "../ResultMetadata.h"
#include "RenderGraph/ResourceCache.h"
#include "Core/Platform/OS.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace Falcor;
namespace fs = std::filesystem;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
{
    try
    {
        setErrorDiagnosticFlags(ErrorDiagnosticFlags::None);
        Device::Desc desc;
        desc.type = Device::Type::D3D12;
        desc.enableDebugLayer = true;
        auto device = make_ref<Device>(desc);
        auto context = device->getRenderContext();
        const auto directory = getProjectDirectory() / "build" / "result_export_tests" /
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        Properties props;
        props["outputDirectory"] = directory.string();
        auto pass = ReconstructionResultPass::create(device, props);
        const std::array<float4, 4> linear = {
            float4(0.25f, 0.5f, 1.f, 0.f), float4(0.f, 1.f, 0.f, 0.25f),
            float4(1.f, 0.f, 0.f, 1.f), float4(-1.f, 2.f, 0.0031308f, 1.f)
        };
        const std::array<uint8_t, 16> heatmap = {13, 51, 222, 255, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255};
        auto image = device->createTexture2D(2, 2, ResourceFormat::RGBA32Float, 1, 1, linear.data());
        auto flip = device->createTexture2D(2, 2, ResourceFormat::RGBA8UnormSrgb, 1, 1, heatmap.data());
        ResourceCache cache;
        Dictionary dictionary;
        const std::string name = "ResultSave";
        cache.registerExternalResource(name + ".output", image);
        cache.registerExternalResource(name + ".ref", image);
        cache.registerExternalResource(name + ".flip", flip);
        struct TestRenderData : RenderData
        {
            TestRenderData(const std::string& name, ResourceCache& cache, Dictionary& dict)
                : RenderData(name, cache, dict, uint2(2), ResourceFormat::RGBA32Float) {}
        } data(name, cache, dictionary);
        dictionary["VoxelReconstruction" + std::string(ReconstructionResult::kScene)] = std::string("ship");
        dictionary["VoxelReconstruction" + std::string(ReconstructionResult::kCamera)] = 7u;
        dictionary["VoxelReconstruction" + std::string(ReconstructionResult::kReady)] = true;
        dictionary["FLIPPass" + std::string(ReconstructionResult::kFlipValid)] = false;
        pass->requestSave();
        pass->execute(context, data);
        require(!fs::exists(directory), "Disabled FLIP must not save a stale heatmap");
        dictionary["FLIPPass" + std::string(ReconstructionResult::kFlipValid)] = true;
        pass->execute(context, data);
        require(!fs::exists(directory), "A failed click must not be deferred to a later frame");
        pass->requestSave();
        pass->execute(context, data);

        fs::path datedDirectory;
        for (auto& entry : fs::directory_iterator(directory)) if (entry.is_directory()) datedDirectory = entry.path();
        require(!datedDirectory.empty() && datedDirectory.filename().string().size() == 5, "Missing MM.DD output directory");
        auto check = [&](const char* suffix, bool isFlip)
        {
            const auto path = datedDirectory / (std::string("ship_007_") + suffix + ".png");
            auto bitmap = Bitmap::createFromFile(path, true);
            require(bitmap && bitmap->getWidth() == 2 && bitmap->getHeight() == 2, "PNG dimensions changed");
            const bool bgra = bitmap->getFormat() == ResourceFormat::BGRA8Unorm || bitmap->getFormat() == ResourceFormat::BGRX8Unorm;
            require(bgra || bitmap->getFormat() == ResourceFormat::RGBA8Unorm, "Expected an 8-bit PNG");
            auto encode = [](float x)
            {
                x = std::clamp(x, 0.f, 1.f);
                return int(std::round(255.f * (x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f)));
            };
            for (uint32_t y = 0; y < 2; ++y)
                for (uint32_t x = 0; x < 2; ++x)
                    for (uint32_t c = 0; c < 3; ++c)
                    {
                        const uint32_t i = y * 2 + x;
                        const uint32_t channel = bgra ? 2u - c : c;
                        const int actual = bitmap->getData()[y * bitmap->getRowPitch() + x * 4 + channel];
                        const int expected = isFlip ? heatmap[i * 4 + c] : encode(linear[i][c]);
                        require(std::abs(actual - expected) <= 1, "Wrong sRGB encoding, channel order, alpha handling or vertical orientation");
                    }
        };
        check("output", false);
        check("ref", false);
        check("flip", true);
        std::cout << "PASS: PNG colors, clipping, orientation and exact sRGB heatmap preservation\n";
        pass->requestSave();
        pass->execute(context, data);
        require(fs::exists(datedDirectory / "ship_007_001_output.png") && fs::exists(datedDirectory / "ship_007_001_ref.png") &&
            fs::exists(datedDirectory / "ship_007_001_flip.png"), "Repeated save overwrote the original triplet");
        dictionary["VoxelReconstruction" + std::string(ReconstructionResult::kReady)] = false;
        pass->requestSave();
        pass->execute(context, data);
        require(!fs::exists(datedDirectory / "ship_007_002_output.png"), "Saved unavailable reconstruction data");
        std::cout << "PASS: one shot, invalid-frame rejection, camera filenames and collision-safe triplets\n";
        device->wait();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
