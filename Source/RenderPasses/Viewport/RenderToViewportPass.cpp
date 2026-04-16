#include "RenderToViewportPass.h"
#include "Utils/Color/ColorUtils.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
    const std::string kShaderFile = "RenderPasses/Viewport/RenderToViewport.ps.slang";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, RenderToViewportPass>();
}

RenderToViewportPass::RenderToViewportPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mEnable = false;
    mOptionsChanged = false;
    mOutputMin = uint2(600, 30);
    mOutputSize = uint2(1024, 1024);

    mpDevice = pDevice;

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point)
        .setAddressingMode(TextureAddressingMode::Border, TextureAddressingMode::Border, TextureAddressingMode::Border)
        .setBorderColor(float4(0));
    mpSampler = mpDevice->createSampler(samplerDesc);

    mpFullScreenPass = FullScreenPass::create(mpDevice, kShaderFile);
}

RenderPassReflection RenderToViewportPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput("input", "input").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput("output", "output")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0, 1, 1);
    return reflector;
}

void RenderToViewportPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpFullScreenPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile).psEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        mpFullScreenPass = FullScreenPass::create(mpDevice, desc);
    }

    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    ref<Texture> pInput = renderData.getTexture("input");
    ref<Texture> pOutput = renderData.getTexture("output");
    pRenderContext->clearRtv(pOutput->getRTV().get(), float4(0));

    auto var = mpFullScreenPass->getRootVar();
    var["input"] = pInput;
    var["sampler"] = mpSampler;
    auto cb = var["CB"];
    cb["enable"] = mEnable;
    cb["outputMin"] = mOutputMin;
    cb["outputSize"] = mOutputSize;
    cb["windowSize"] = uint2(pOutput->getWidth(), pOutput->getHeight());
    ref<Fbo> fbo = Fbo::create(mpDevice);
    fbo->attachColorTarget(pOutput, 0);
    mpFullScreenPass->execute(pRenderContext, fbo);
}

void RenderToViewportPass::renderUI(Gui::Widgets& widget)
{
    if (widget.checkbox("Enable", mEnable))
        mOptionsChanged = true;
}
