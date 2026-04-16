#include "VoxelizationPass.h"
#include "Image/ImageLoader.h"
#include <fstream>

namespace
{
const std::string kAnalyzePolygonProgramFile = "RenderPasses/Voxelization/AnalyzePolygon.cs.slang";
}; // namespace

VoxelizationPass::VoxelizationPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), polygonGroup(pDevice, VoxelizationBase::GlobalGridData), gridData(VoxelizationBase::GlobalGridData)
{
    mSceneNameIndex = 0;
    mSceneName = "Auto";
    mVoxelizationComplete = true;
    mSamplingComplete = true;
    mCompleteTimes = 0;
    pVBuffer_CPU = nullptr;

    mSampleFrequency = 1024;
    mVoxelResolution = 512;

    VoxelizationBase::UpdateVoxelGrid(nullptr, mVoxelResolution);

    mpDevice = pDevice;

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpSampler = pDevice->createSampler(samplerDesc);
}

RenderPassReflection VoxelizationPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0, 1, 1);
    return reflector;
}

void VoxelizationPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene || mVoxelizationComplete && mSamplingComplete)
        return;

    if (!mVoxelizationComplete)
    {
        voxelize(pRenderContext, renderData);
        mVoxelizationComplete = true;

        blockMap = mpDevice->createStructuredBuffer(sizeof(uint), 4 * gridData.totalBlockCount(), ResourceBindFlags::UnorderedAccess);
        pRenderContext->clearUAV(blockMap->getUAV().get(), uint4(0));
    }
    else
    {
        if (mCompleteTimes < polygonGroup.size())
        {
            if (mCompleteTimes == 0)
                Tools::Profiler::BeginSample("Analyze Polygon");
            sample(pRenderContext, renderData);
            mCompleteTimes++;
        }
        else
        {
            pRenderContext->submit(true);
            Tools::Profiler::EndSample("Analyze Polygon");

            Tools::Profiler::BeginSample("Write File");
            ref<Buffer> cpuGBuffer = copyToCpu(mpDevice, pRenderContext, gBuffer);
            ref<Buffer> cpuBlockMap = copyToCpu(mpDevice, pRenderContext, blockMap);
            pRenderContext->submit(true);
            void* pGBuffer_CPU = cpuGBuffer->map();
            void* pBlockMap_CPU = cpuBlockMap->map();
            write(getFileName(), pGBuffer_CPU, pVBuffer_CPU, pBlockMap_CPU);
            cpuGBuffer->unmap();
            cpuBlockMap->unmap();
            mSamplingComplete = true;
            Tools::Profiler::EndSample("Write File");
            Tools::Profiler::Print();
            Tools::Profiler::Reset();
        }
    }
}

void VoxelizationPass::compile(RenderContext* pRenderContext, const CompileData& compileData) {}

void VoxelizationPass::renderUI(Gui::Widgets& widget)
{
    static const uint resolutions[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 1024};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(resolutions) / sizeof(uint); i++)
        {
            list.push_back({resolutions[i], std::to_string(resolutions[i])});
        }
        widget.dropdown("Voxel Resolution", list, mVoxelResolution);
    }

    static const std::string sceneNames[] = {"Auto", "Arcade", "Azalea", "BoxBunny", "Box", "Chandelier", "Colosseum"};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(sceneNames) / sizeof(std::string); i++)
        {
            list.push_back({i, sceneNames[i]});
        }
        if (widget.dropdown("Scene Name", list, mSceneNameIndex))
        {
            mSceneName = sceneNames[mSceneNameIndex];
        }
    }

    static const uint sampleFrequencies[] = {0, 64, 256, 512, 1024, 2048, 4096};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(sampleFrequencies) / sizeof(uint); i++)
        {
            list.push_back({sampleFrequencies[i], std::to_string(sampleFrequencies[i])});
        }
        widget.dropdown("Sample Frequency", list, mSampleFrequency);
    }

    static const uint polygonPerFrames[] = {1000, 4000, 16000, 64000, 128000, 256000, 512000, 1024000};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(polygonPerFrames) / sizeof(uint); i++)
        {
            list.push_back({polygonPerFrames[i], std::to_string(polygonPerFrames[i])});
        }
        widget.dropdown("Polygon Per Frame", list, polygonGroup.maxPolygonCount);
    }

    widget.checkbox("LerpNormal", mLerpNormal);

    if (mpScene && mVoxelizationComplete && mSamplingComplete && widget.button("Generate"))
    {
        VoxelizationBase::UpdateVoxelGrid(mpScene, mVoxelResolution);
        ImageLoader::Instance().setSceneName(mSceneName);
        mVoxelizationComplete = false;
        mSamplingComplete = false;
        mCompleteTimes = 0;
        requestRecompile();
    }
}

void VoxelizationPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mAnalyzePolygonPass = nullptr;
    VoxelizationBase::UpdateVoxelGrid(mpScene, mVoxelResolution);
}

void VoxelizationPass::voxelize(RenderContext* pRenderContext, const RenderData& renderData) {}

void VoxelizationPass::sample(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mAnalyzePolygonPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kAnalyzePolygonProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        mAnalyzePolygonPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mAnalyzePolygonPass->addDefine("LERP_NORMAL", mLerpNormal ? "1" : "0");

    ShaderVar var = mAnalyzePolygonPass->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["sampler"] = mpSampler;
    var[kGBuffer] = gBuffer;
    var[kPolygonRangeBuffer] = polygonRangeBuffer;
    var[kPolygonBuffer] = polygonGroup.get(mCompleteTimes);
    var[kBlockMap] = blockMap;

    uint groupVoxelCount = polygonGroup.getVoxelCount(mCompleteTimes);
    auto cb = var["CB"];
    cb["groupVoxelCount"] = groupVoxelCount;
    cb["sampleFrequency"] = mSampleFrequency;
    cb["gBufferOffset"] = polygonGroup.getVoxelOffset(mCompleteTimes);
    cb["blockCount"] = gridData.blockCount();

    auto cb_grid = var["GridData"];
    cb_grid["gridMin"] = gridData.gridMin;
    cb_grid["voxelSize"] = gridData.voxelSize;
    cb_grid["voxelCount"] = gridData.voxelCount;

    mAnalyzePolygonPass->execute(pRenderContext, uint3(groupVoxelCount, 1, 1)); // 每个体素执行一次，没有同步问题
}

std::string trim_non_alnum_ends(std::string s)
{
    auto is_alnum = [](unsigned char c) { return std::isalnum(c) != 0; };

    size_t b = 0;
    while (b < s.size() && !is_alnum((unsigned char)s[b]))
        ++b;

    size_t e = s.size();
    while (e > b && !is_alnum((unsigned char)s[e - 1]))
        --e;

    return s.substr(b, e - b);
}

std::string VoxelizationPass::getFileName()
{
    std::ostringstream oss;
    if (mSceneName == "Auto")
        oss << trim_non_alnum_ends(mpScene->getPath().stem().string());
    else
        oss << mSceneName;
    oss << "_";
    oss << ToString((int3)gridData.voxelCount);
    oss << "_";
    oss << std::to_string(mSampleFrequency);
    oss << ".bin";
    return oss.str();
}

void VoxelizationPass::write(std::string fileName, void* pGBuffer, void* pVBuffer, void* pBlockMap)
{
    std::ofstream f;
    std::string s = VoxelizationBase::ResourceFolder + fileName;
    f.open(s, std::ios::binary);
    f.write(reinterpret_cast<char*>(&gridData), sizeof(GridData));

    f.write(reinterpret_cast<const char*>(pVBuffer), gridData.totalVoxelCount() * sizeof(int));
    f.write(reinterpret_cast<const char*>(pGBuffer), gridData.solidVoxelCount * sizeof(VoxelData));
    f.write(reinterpret_cast<const char*>(pBlockMap), gridData.totalBlockCount() * sizeof(uint4));

    f.close();
    VoxelizationBase::FileUpdated = true;
}
