#include "ReconstructionResultPass.h"
#include "ResultMetadata.h"
#include "Core/Platform/OS.h"
#include <array>
#include <ctime>

namespace
{
const std::array<const char*, 3> kImages = {"output", "ref", "flip"};

std::string safeFilename(std::string name)
{
    for (char& c : name)
        if (static_cast<unsigned char>(c) < 32u || std::string("<>:\"/\\|?*").find(c) != std::string::npos) c = '_';
    while (!name.empty() && (name.back() == '.' || name.back() == ' ')) name.pop_back();
    return name.empty() ? "scene" : name;
}
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReconstructionResultPass>();
    ScriptBindings::registerBinding([](pybind11::module& m)
    {
        pybind11::class_<ReconstructionResultPass, RenderPass, ref<ReconstructionResultPass>> pass(m, "ReconstructionResultPass");
        pass.def("requestSave", &ReconstructionResultPass::requestSave);
        pass.def_property_readonly("status", &ReconstructionResultPass::getStatus);
    });
}

ReconstructionResultPass::ReconstructionResultPass(ref<Device> device, const Properties& props) : RenderPass(device)
{
    std::filesystem::path directory = "picture";
    for (const auto& [key, value] : props)
    {
        if (key == "outputDirectory") directory = props.get<std::string>(key);
        else if (key == "reconstructionPass") mReconstructionPass = props.get<std::string>(key);
        else if (key == "flipPass") mFlipPass = props.get<std::string>(key);
        else logWarning("Unknown ReconstructionResultPass property: {}", key);
    }
    mOutputDirectory = (directory.is_absolute() ? directory : getProjectDirectory() / directory).lexically_normal();
}

Properties ReconstructionResultPass::getProperties() const
{
    Properties props;
    props["outputDirectory"] = mOutputDirectory.string();
    props["reconstructionPass"] = mReconstructionPass;
    props["flipPass"] = mFlipPass;
    return props;
}

RenderPassReflection ReconstructionResultPass::reflect(const CompileData&)
{
    RenderPassReflection reflection;
    reflection.addInput("output", "Accumulated reconstruction (linear RGB)").bindFlags(ResourceBindFlags::ShaderResource);
    reflection.addInput("ref", "White-composited reference (linear RGB)").bindFlags(ResourceBindFlags::ShaderResource);
    reflection.addInput("flip", "FLIP display heatmap (sRGB texture)").bindFlags(ResourceBindFlags::ShaderResource);
    return reflection;
}

void ReconstructionResultPass::execute(RenderContext* context, const RenderData& data)
{
    auto& dict = data.getDictionary();
    mSceneName = dict.getValue(mReconstructionPass + ReconstructionResult::kScene, std::string{});
    mCameraIndex = dict.getValue(mReconstructionPass + ReconstructionResult::kCamera, 0u);
    mReady = dict.getValue(mReconstructionPass + ReconstructionResult::kReady, false) &&
        dict.getValue(mFlipPass + ReconstructionResult::kFlipValid, false) && !mSceneName.empty();
    if (!mSaveRequested) return;
    mSaveRequested = false; // Exactly one attempt per click, including failed/unavailable frames.
    if (!mReady)
    {
        mStatus = "Save skipped: finish loading, stop reconstruction, enable ReferenceCamera, Default draw mode and FLIP.";
        return;
    }
    try
    {
        saveImages(context, data);
    }
    catch (const std::exception& e)
    {
        mStatus = "Save failed: " + std::string(e.what());
        logWarning(mStatus);
    }
}

void ReconstructionResultPass::saveImages(RenderContext* context, const RenderData& data)
{
    const auto output = data.getTexture("output");
    FALCOR_CHECK(output, "Missing reconstruction input.");
    for (const auto* name : kImages)
    {
        const auto texture = data.getTexture(name);
        FALCOR_CHECK(texture && texture->getWidth() == output->getWidth() && texture->getHeight() == output->getHeight(),
            "All three result images must have matching dimensions ({}).", name);
    }

    const auto now = std::time(nullptr);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif
    const auto directory = mOutputDirectory / fmt::format("{:02}.{:02}", localTime.tm_mon + 1, localTime.tm_mday);
    std::filesystem::create_directories(directory);
    const std::string base = fmt::format("{}_{:03}", safeFilename(mSceneName), mCameraIndex);
    std::string stem = base;
    std::array<std::filesystem::path, 3> paths;
    for (uint32_t attempt = 0; ; ++attempt)
    {
        stem = attempt == 0 ? base : fmt::format("{}_{:03}", base, attempt);
        bool exists = false;
        for (size_t i = 0; i < kImages.size(); ++i)
        {
            paths[i] = directory / (stem + "_" + kImages[i] + ".png");
            exists |= std::filesystem::exists(paths[i]);
        }
        if (!exists) break;
    }

    if (!mpPngTexture || mpPngTexture->getWidth() != output->getWidth() || mpPngTexture->getHeight() != output->getHeight())
        mpPngTexture = mpDevice->createTexture2D(output->getWidth(), output->getHeight(), ResourceFormat::RGBA8UnormSrgb,
            1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);

    for (size_t i = 0; i < kImages.size(); ++i)
    {
        // Writing an sRGB render target encodes linear RGB once and clamps to the PNG's LDR range.
        // An sRGB FLIP input is decoded on read and re-encoded (or copied), preserving its display colors.
        // Equal dimensions + point filtering preserve edge texels; the input images are already composited.
        context->blit(data.getTexture(kImages[i])->getSRV(), mpPngTexture->getRTV(),
            RenderContext::kMaxRect, RenderContext::kMaxRect, TextureFilteringMode::Point);
        mpPngTexture->captureToFile(0, 0, paths[i], Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
        FALCOR_CHECK(std::filesystem::exists(paths[i]) && std::filesystem::file_size(paths[i]) > 0,
            "Could not write {}", paths[i].string());
    }
    mStatus = "Saved: " + (directory / (stem + "_{output,ref,flip}.png")).string();
    logInfo(mStatus);
}

void ReconstructionResultPass::renderUI(Gui::Widgets& widget)
{
    widget.text("PNG directory: " + mOutputDirectory.string() + "/MM.DD");
    if (mReady)
    {
        widget.text(fmt::format("Scene: {} | Camera Index: {} (zero-based)", mSceneName, mCameraIndex));
        if (widget.button("Save output / ref / flip PNG")) requestSave();
        widget.text("Saves the current accumulation. Repeated saves keep earlier files.");
    }
    else widget.text("Load input data, stop reconstruction, enable ReferenceCamera, Default draw mode and FLIP to save.");
    if (!mStatus.empty()) widget.text(mStatus);
}
