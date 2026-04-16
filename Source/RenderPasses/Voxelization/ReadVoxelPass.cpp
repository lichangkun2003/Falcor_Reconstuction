#include "ReadVoxelPass.h"
#include "Shading.slang"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const std::string kPrepareProgramFile = "RenderPasses/Voxelization/PrepareShadingData.cs.slang";

}; // namespace

ReadVoxelPass::ReadVoxelPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice), gridData(VoxelizationBase::GlobalGridData)
{
    mComplete = true;
    mOptionsChanged = false;
    selectedFile = 0;
    mpDevice = pDevice;
}

RenderPassReflection ReadVoxelPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0, 1, 1);

    reflector.addOutput(kVBuffer, kVBuffer)
        .bindFlags(ResourceBindFlags::None)
        .format(ResourceFormat::R32Uint)
        .texture3D(gridData.voxelCount.x, gridData.voxelCount.y, gridData.voxelCount.z, 1);

    reflector.addOutput(kGBuffer, kGBuffer)
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::Unknown)
        .rawBuffer(gridData.solidVoxelCount * sizeof(PrimitiveBSDF));

    reflector.addOutput(kPBuffer, kPBuffer)
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::Unknown)
        .rawBuffer(gridData.solidVoxelCount * sizeof(Ellipsoid));

    reflector.addOutput(kBlockMap, kBlockMap)
        .bindFlags(ResourceBindFlags::None)
        .format(ResourceFormat::RGBA32Uint)
        .texture2D(gridData.blockCount().x, gridData.blockCount().y);

    return reflector;
}

void ReadVoxelPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mComplete)
        return;

    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    if (!mPreparePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPrepareProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        mPreparePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    size_t voxelCount = gridData.totalVoxelCount();

    std::ifstream f;
    size_t offset = 0;

    f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
    size_t fileSize = std::filesystem::file_size(filePaths[selectedFile]);
    tryRead(f, offset, sizeof(GridData), nullptr, fileSize);

    ref<Texture> pVBuffer = renderData.getTexture(kVBuffer);
    uint* vBuffer = new uint[gridData.totalVoxelCount()];
    tryRead(f, offset, gridData.totalVoxelCount() * sizeof(uint), vBuffer, fileSize);
    pVBuffer->setSubresourceBlob(0, vBuffer, gridData.totalVoxelCount() * sizeof(uint));

    mpVoxelDataBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), gridData.solidVoxelCount, ResourceBindFlags::ShaderResource);
    VoxelData* voxelDataBuffer = new VoxelData[gridData.solidVoxelCount];
    tryRead(f, offset, gridData.solidVoxelCount * sizeof(VoxelData), voxelDataBuffer, fileSize);
    mpVoxelDataBuffer->setBlob(voxelDataBuffer, 0, gridData.solidVoxelCount * sizeof(VoxelData));
    pRenderContext->submit(true);

    ref<Texture> pBlockMap = renderData.getTexture(kBlockMap);
    uint4* blockMap = new uint4[gridData.totalBlockCount()];
    tryRead(f, offset, gridData.totalBlockCount() * sizeof(uint4), blockMap, fileSize);
    pBlockMap->setSubresourceBlob(0, blockMap, gridData.totalBlockCount() * sizeof(uint4));

    // VoxelData将拆分成PrimitiveBSDF和Ellipsoid
    ref<Buffer> pGBuffer = renderData.getResource(kGBuffer)->asBuffer();
    ref<Buffer> pPBuffer = renderData.getResource(kPBuffer)->asBuffer();

    ShaderVar var = mPreparePass->getRootVar();
    var["voxelDataBuffer"] = mpVoxelDataBuffer;
    var[kGBuffer] = pGBuffer;
    var[kPBuffer] = pPBuffer;

    auto cb = var["CB"];
    cb["voxelCount"] = (uint)gridData.solidVoxelCount;
    mPreparePass->execute(pRenderContext, uint3((uint)gridData.solidVoxelCount, 1, 1));
    pRenderContext->submit(true);
    mComplete = true;
}

void ReadVoxelPass::compile(RenderContext* pRenderContext, const CompileData& compileData) {}

void ReadVoxelPass::renderUI(Gui::Widgets& widget)
{
    if (VoxelizationBase::FileUpdated)
    {
        filePaths.clear();
        for (const auto& entry : std::filesystem::directory_iterator(VoxelizationBase::ResourceFolder))
        {
            if (std::filesystem::is_regular_file(entry))
            {
                filePaths.push_back(entry.path());
            }
        }
        VoxelizationBase::FileUpdated = false;
    }
    Gui::DropdownList list;
    for (uint i = 0; i < filePaths.size(); i++)
    {
        list.push_back({i, filePaths[i].filename().string()});
    }
    widget.dropdown("File", list, selectedFile);

    if (mpScene && widget.button("Read"))
    {
        std::ifstream f;
        size_t offset = 0;

        f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
        if (!f.is_open())
            return;

        size_t fileSize = std::filesystem::file_size(filePaths[selectedFile]);
        tryRead(f, offset, sizeof(GridData), &gridData, fileSize);

        f.close();

        requestRecompile();
        mComplete = false;
        mOptionsChanged = true;
    }

    GridData& data = VoxelizationBase::GlobalGridData;
    widget.text("Voxel Size: " + ToString(data.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)data.voxelCount));
    widget.text("Block Count: " + ToString((int3)data.blockCount3D()));
    widget.text("Grid Min: " + ToString(data.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(data.solidVoxelCount));
    widget.text("Solid Rate: " + std::to_string(data.solidVoxelCount / (float)data.totalVoxelCount()));
    widget.text("Max Polygon Count: " + std::to_string(data.maxPolygonCount));
    widget.text("Total Polygon Count: " + std::to_string(data.totalPolygonCount));
}

void ReadVoxelPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

bool ReadVoxelPass::tryRead(std::ifstream& f, size_t& offset, size_t bytes, void* dst, size_t fileSize)
{
    if (offset + bytes > fileSize)
        return false;
    if (dst)
    {
        f.seekg(offset, std::ios::beg);
        f.read(reinterpret_cast<char*>(dst), bytes);
    }
    offset += bytes;
    return true;
}
