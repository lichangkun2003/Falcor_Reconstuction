#include "RayMarchingPass.h"
#include "Shading.slang"
#include "Math/SphericalHarmonics.slang"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const std::string kShaderFile = "RenderPasses/Voxelization/RayMarching.ps.slang";
const std::string kDisplayShaderFile = "RenderPasses/Voxelization/DisplayNDF.ps.slang";
const std::string kBakeShaderFile = "RenderPasses/Voxelization/BakeReconstruction.cs.slang";
const std::string kOutputColor = "color";

// 烘焙输出的 CPU 侧镜像，必须与 BakeReconstruction.cs.slang 里的 ReconVoxelData
// 以及重建端 VoxelReconstructionNoLightTransport 的 VoxelData 三者逐字节一致。
// Ellipsoid 直接复用 Math/Ellipsoid.slang 的宿主版本（48 字节，与设备端相同）。
struct BakeVoxelData
{
    uint32_t occupied;
    Ellipsoid ellipsoid;
    float3 radiance[9];
    float opacity[9];
};
static_assert(sizeof(BakeVoxelData) == 196, "Bake layout must match VoxelReconstructionNoLightTransport::VoxelData");

// 重建端 v1 文件头的魔数，必须与 DataProcess.cpp 里的 kReconstructionMagic 相同
constexpr uint32_t kBakeMagic = 0x56525831; // "VRX1"
} // namespace

RayMarchingPass::RayMarchingPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), gridData(VoxelizationBase::GlobalGridData)
{
    mpDevice = pDevice;
    mShadowBias100 = 0.01f;
    mMinPdf100 = 0.1f;
    mTrasmittanceThreshold100 = 5.f;
    mUseEmissiveLight = false;
    mDebug = false;
    mCheckEllipsoid = true;
    mCheckVisibility = true;
    mCheckCoverage = true;
    mUseMipmap = true;
    mDrawMode = 0;
    mMaxBounce = 3;
    mRenderBackGround = true;
    mClearColor = float3(0);
    mSelectedResolution = 0;
    mOutputResolution = uint2(1920, 1080);

    mDisplayNDF = false;
    mSelectedUV = float2(0);
    mSelectedPixel = uint2(0);

    mOptionsChanged = false;
    mFrameIndex = 0;

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpPointSampler = mpDevice->createSampler(samplerDesc);
}

RenderPassReflection RayMarchingPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput(kVBuffer, kVBuffer)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::R32Uint)
        .texture3D(gridData.voxelCount.x, gridData.voxelCount.y, gridData.voxelCount.z, 1);

    reflector.addInput(kGBuffer, kGBuffer)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::Unknown)
        .rawBuffer(gridData.solidVoxelCount * sizeof(PrimitiveBSDF));

    reflector.addInput(kPBuffer, kPBuffer)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::Unknown)
        .rawBuffer(gridData.solidVoxelCount * sizeof(Ellipsoid));

    reflector.addInput(kBlockMap, kBlockMap)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Uint)
        .texture2D(gridData.blockCount().x, gridData.blockCount().y);

    reflector.addOutput(kOutputColor, "Color")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mOutputResolution.x, mOutputResolution.y, 1, 1);
    return reflector;
}

void RayMarchingPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    if (mBake.requested)
    {
        mBake.requested = false;
        bakeReconstruction(pRenderContext, renderData);
    }

    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    FALCOR_PROFILE(pRenderContext, "RayMarching");
    ref<Camera> pCamera = mpScene->getCamera();
    ref<Texture> pOutputColor = renderData.getTexture(kOutputColor);
    if (!mSelectedVoxel)
    {
        mSelectedVoxel =
            mpDevice->createStructuredBuffer(sizeof(float4), 1, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    }

    pRenderContext->clearRtv(pOutputColor->getRTV().get(), float4(0));

    mSelectedPixel = uint2(mSelectedUV.x * pOutputColor->getWidth(), mSelectedUV.y * pOutputColor->getHeight());

    if (!mDisplayNDF)
    {
        if (!mpFullScreenPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kShaderFile).psEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            desc.addTypeConformances(mpScene->getTypeConformances());
            mpFullScreenPass = FullScreenPass::create(mpDevice, desc, mpScene->getSceneDefines());
        }
        pRenderContext->clearUAV(mSelectedVoxel->getUAV().get(), float4(-1));

        mpFullScreenPass->addDefine("CHECK_ELLIPSOID", mCheckEllipsoid ? "1" : "0");
        mpFullScreenPass->addDefine("CHECK_VISIBILITY", mCheckVisibility ? "1" : "0");
        mpFullScreenPass->addDefine("CHECK_COVERAGE", mCheckCoverage ? "1" : "0");
        mpFullScreenPass->addDefine("USE_MIP_MAP", mUseMipmap ? "1" : "0");
        mpFullScreenPass->addDefine("DEBUG", mDebug ? "1" : "0");

        ref<EnvMap> pEnvMap = mpScene->getEnvMap();
        mpFullScreenPass->addDefine("USE_ENV_MAP", pEnvMap ? "1" : "0");
        if (pEnvMap)
        {
            if (!mpEnvMapSampler || mpEnvMapSampler->getEnvMap() != pEnvMap)
                mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, pEnvMap);
        }
        if (mUseEmissiveLight)
        {
            if (VoxelizationBase::LightChanged)
            {
                mpScene->getILightCollection(pRenderContext);
                mpFullScreenPass->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
                VoxelizationBase::LightChanged = false;
                pRenderContext->submit(true);
                return;
            }
        }
        else
        {
            mpFullScreenPass->addDefine("USE_EMISSIVE_LIGHTS", "0");
        }

        // 必须在addDefine之后获取var
        auto var = mpFullScreenPass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        if (pEnvMap)
            mpEnvMapSampler->bindShaderData(var["gEnvMapSampler"]);

        var[kVBuffer] = renderData.getTexture(kVBuffer);
        var[kGBuffer] = renderData.getResource(kGBuffer)->asBuffer();
        var[kPBuffer] = renderData.getResource(kPBuffer)->asBuffer();
        var[kBlockMap] = renderData.getTexture(kBlockMap);
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb_GridData = var["GridData"];
        cb_GridData["gridMin"] = gridData.gridMin;
        cb_GridData["voxelSize"] = gridData.voxelSize;
        cb_GridData["voxelCount"] = gridData.voxelCount;
        cb_GridData["solidVoxelCount"] = (uint)gridData.solidVoxelCount;

        auto cb = var["CB"];
        cb["pixelCount"] = mOutputResolution;
        cb["blockCount"] = gridData.blockCount3D();
        cb["invVP"] = math::inverse(pCamera->getViewProjMatrixNoJitter());
        cb["shadowBias"] = mShadowBias100 / 100 / gridData.voxelSize.x;
        cb["drawMode"] = mDrawMode;
        cb["maxBounce"] = mMaxBounce;
        cb["frameIndex"] = mFrameIndex;
        cb["minPdf"] = mMinPdf100 / 100;
        cb["trasmittanceThreshold"] = mTrasmittanceThreshold100 / 100;
        cb["selectedPixel"] = mSelectedPixel;
        cb["renderBackGround"] = mRenderBackGround;
        cb["clearColor"] = float4(mClearColor, 0);
        mFrameIndex++;

        ref<Fbo> fbo = Fbo::create(mpDevice);
        fbo->attachColorTarget(pOutputColor, 0);
        mpFullScreenPass->execute(pRenderContext, fbo);
    }
    else
    {
        if (!mpDisplayNDFPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kDisplayShaderFile).psEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            mpDisplayNDFPass = FullScreenPass::create(mpDevice, desc);
        }
        auto var = mpDisplayNDFPass->getRootVar();
        var[kGBuffer] = renderData.getResource(kGBuffer)->asBuffer();
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb = var["CB"];
        cb["clearColor"] = float4(mClearColor, 0);

        ref<Fbo> fbo = Fbo::create(mpDevice);
        fbo->attachColorTarget(pOutputColor, 0);
        mpDisplayNDFPass->execute(pRenderContext, fbo);
    }



}

void RayMarchingPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mUseEmissiveLight = false;
    VoxelizationBase::LightChanged = true;
}

void RayMarchingPass::bakeReconstruction(RenderContext* pRenderContext, const RenderData& renderData)
{
    try
    {
        // 网格必须是等大的立方体。网格空间与世界空间方向一致这个前提依赖它，
        // 否则 Lebedev 方向不能同时喂给 DDA 和 BRDF。
        if (gridData.voxelSize.x != gridData.voxelSize.y || gridData.voxelSize.x != gridData.voxelSize.z ||
            gridData.voxelCount.x != gridData.voxelCount.y || gridData.voxelCount.x != gridData.voxelCount.z)
        {
            throw std::runtime_error(
                "Baking needs an isotropic cubic grid. In the Voxelization pass, tick \"Match Reconstruction Grid\" "
                "(AABB [-1.3, 1.3]^3, resolution 128), then Generate and Read before baking."
            );
        }
        if (gridData.solidVoxelCount == 0)
            throw std::runtime_error("No solid voxels loaded. Press Read in ReadVoxelPass first.");

        if (!mBake.pass)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kBakeShaderFile).csEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            desc.addTypeConformances(mpScene->getTypeConformances());

            DefineList defines;
            defines.add(mpScene->getSceneDefines());
            // 这三个必须为 1：关掉的话 calcCoverage / calcInternalVisibility / 椭球裁剪
            // 会退化成常量，烘焙出来的 alpha 和 radiance 就全是错的。
            defines.add("CHECK_ELLIPSOID", "1");
            defines.add("CHECK_COVERAGE", "1");
            defines.add("CHECK_VISIBILITY", "1");
            defines.add("USE_MIP_MAP", "0"); // 用 DDA，逐格精确
            mBake.pass = ComputePass::create(mpDevice, desc, defines, true);
        }

        const uint64_t elementCount = gridData.totalVoxelCount();
        const uint64_t byteSize = elementCount * sizeof(BakeVoxelData);
        if (!mBake.output || mBake.output->getSize() < byteSize)
        {
            mBake.output = mpDevice->createStructuredBuffer(
                sizeof(BakeVoxelData), elementCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
        }

        auto var = mBake.pass->getRootVar();
        var[kVBuffer] = renderData.getTexture(kVBuffer);
        var[kGBuffer] = renderData.getResource(kGBuffer)->asBuffer();
        var[kPBuffer] = renderData.getResource(kPBuffer)->asBuffer();
        var[kBlockMap] = renderData.getTexture(kBlockMap);
        var["bakeOutput"] = mBake.output;

        auto cbGrid = var["GridData"];
        cbGrid["gridMin"] = gridData.gridMin;
        cbGrid["voxelSize"] = gridData.voxelSize;
        cbGrid["voxelCount"] = gridData.voxelCount;
        cbGrid["solidVoxelCount"] = (uint)gridData.solidVoxelCount;

        auto cb = var["BakeCB"];
        cb["coverageEps"] = mBake.coverageEps;

        // 着色器对每个下标都会写一次（空体素写 occupied=0），所以不需要预先 clear。
        //
        // 注意 ComputePass::execute 的这个重载收的是线程数, 内部会自己
        // div_round_up(threads, threadGroupSize) 换算成 group 数 (见 ComputePass.cpp)。
        // 这里必须传体素总数本身, 不能再手动除 [numthreads] 的 64, 否则只会有
        // elementCount/64 个线程真正跑起来, 网格大部分区域根本没被写过。
        mBake.pass->execute(pRenderContext, uint3((uint)elementCount, 1, 1));
        pRenderContext->submit(true);

        std::vector<uint8_t> data((size_t)byteSize);
        mBake.output->getBlob(data.data(), 0, data.size());

        // 输出到 <project>/resource/new/，和 Voxelization 写出的 resource/*.bin_CPU 放在一起。
        // 用 getProjectDirectory() 而不是 VoxelizationBase::ResourceFolder：后者是硬编码的绝对路径，
        // 工程换个位置就断了，而 getProjectDirectory() 来自 CMake 的 FALCOR_PROJECT_DIR。
        //
        // 注意：重建端 loadReconstruction 的 requireModeFile() 只认
        // <project>/Reconstruction_Output/mode3 之内的文件，所以这里产出的文件
        // 暂时不能直接在重建 pass 里加载。
        const std::filesystem::path directory = Falcor::getProjectDirectory() / "resource" / "new";
        std::filesystem::create_directories(directory);
        const std::filesystem::path path = directory / fmt::format(
            "{}_bake_{}x{}x{}.bin",
            mpScene->getPath().stem().string(),
            gridData.voxelCount.x, gridData.voxelCount.y, gridData.voxelCount.z
        );

        std::ofstream out(path, std::ios::binary);
        if (!out)
            throw std::runtime_error("Cannot open for writing: " + path.string());
        const uint32_t version = 1;
        const uint32_t voxelDataSize = (uint32_t)sizeof(BakeVoxelData);
        out.write(reinterpret_cast<const char*>(&kBakeMagic), sizeof(kBakeMagic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&gridData.voxelCount), sizeof(gridData.voxelCount));
        out.write(reinterpret_cast<const char*>(&voxelDataSize), sizeof(voxelDataSize));
        out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)byteSize);
        out.close();
        if (!out)
            throw std::runtime_error("Write failed: " + path.string());

        // 写出的 occupied 数。重建端加载时会报它自己数出来的 occupied 数，两者必须相同；
        // 不同就说明布局对不上。
        const BakeVoxelData* voxels = reinterpret_cast<const BakeVoxelData*>(data.data());
        uint64_t occupied = 0;
        for (uint64_t i = 0; i < elementCount; i++)
            occupied += voxels[i].occupied != 0 ? 1 : 0;

        mBake.outputDirectory = directory.string();
        mBake.status = fmt::format("Baked {} voxels -> {}", occupied, path.filename().string());
        logInfo(
            "Bake done: {} (source solid={}, written occupied={}, bytes={})",
            path.string(), gridData.solidVoxelCount, occupied, byteSize
        );
    }
    catch (const std::exception& e)
    {
        mBake.status = std::string("Bake failed: ") + e.what();
        logError("{}", mBake.status);
    }
}

void RayMarchingPass::renderBakeUI(Gui::Widgets& widget)
{
    widget.text("--- Bake ---");
    bool clicked = widget.button("Bake To Reconstruction Format");
    // 字符串字面量必须保持 ASCII：本工程按 GBK(936) 编译，源码里的 UTF-8 中文
    // 一旦出现在字面量中就可能有字节被当成转义符，导致 C2001。中文只写进注释。
    widget.tooltip(
        "Bake the loaded ABSDF voxels into the VoxelReconstructionNoLightTransport\n"
        "representation: pure white sky, direct light only, alpha = coverage,\n"
        "radiance projected onto 9-coefficient SH.\n"
        "Output goes to <project>/resource/new/."
    );
    if (clicked)
        mBake.requested = true;

    widget.slider("Coverage Eps", mBake.coverageEps, 1e-5f, 1e-1f);
    widget.text("Status: " + mBake.status);
}

void RayMarchingPass::renderUI(Gui::Widgets& widget)
{
    if (widget.checkbox("Debug", mDebug))
        mOptionsChanged = true;
    if (widget.checkbox("Use Emissive Light", mUseEmissiveLight))
        mOptionsChanged = true;
    if (widget.checkbox("Check Ellipsoid", mCheckEllipsoid))
        mOptionsChanged = true;
    if (widget.checkbox("Check Visibility", mCheckVisibility))
        mOptionsChanged = true;
    if (widget.checkbox("Check Coverage", mCheckCoverage))
        mOptionsChanged = true;
    if (widget.checkbox("Use Mipmap", mUseMipmap))
        mOptionsChanged = true;
    if (widget.slider("Shadow Bias(x100)", mShadowBias100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("Min Pdf(x100)", mMinPdf100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("T Threshold(x100)", mTrasmittanceThreshold100, 0.0f, 10.0f))
        mOptionsChanged = true;
    if (widget.dropdown("Draw Mode", reinterpret_cast<ABSDFDrawMode&>(mDrawMode)))
        mOptionsChanged = true;
    if (widget.slider("Max Bounce", mMaxBounce, 0u, 4u))
        mOptionsChanged = true;
    if (widget.checkbox("Display NDF", mDisplayNDF))
        mOptionsChanged = true;
    if (widget.rgbColor("Clear Color", mClearColor))
        mOptionsChanged = true;
    if (widget.checkbox("Render Background", mRenderBackGround))
        mOptionsChanged = true;

    static const uint resolutions[] = {0, 32, 64, 128, 256, 512, 1024};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(resolutions) / sizeof(uint); i++)
        {
            list.push_back({resolutions[i], std::to_string(resolutions[i])});
        }
        if (widget.dropdown("Output Resolution", list, mSelectedResolution))
        {
            if (mSelectedResolution == 0)
                mOutputResolution = uint2(1920, 1080);
            else
                mOutputResolution = uint2(mSelectedResolution, mSelectedResolution);
            ref<Camera> camera = mpScene->getCamera();
            if (camera)
                camera->setAspectRatio(mOutputResolution.x / (float)mOutputResolution.y);
            requestRecompile();
        }
    }

    widget.text("Selected Pixel: " + ToString(mSelectedPixel));

    renderBakeUI(widget);
}

void RayMarchingPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpFullScreenPass = nullptr;
    mpDisplayNDFPass = nullptr;
    mBake.pass = nullptr;
    mDebug = false;
    mUseEmissiveLight = false;
}

bool RayMarchingPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
    {
        mSelectedUV = mouseEvent.pos;

        return true;
    }
    return false;
}
