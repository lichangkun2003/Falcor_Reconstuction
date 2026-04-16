#include "VoxelizationPass_GPU.h"
#include "Shading.slang"

namespace
{
const std::string kSampleMeshProgramFile = "RenderPasses/Voxelization/SampleMesh.cs.slang";
const std::string kClipMeshProgramFile = "RenderPasses/Voxelization/ClipMesh.cs.slang";

}; // namespace

VoxelizationPass_GPU::VoxelizationPass_GPU(ref<Device> pDevice, const Properties& props) : VoxelizationPass(pDevice, props)
{
    mSolidRate = 0.05;
    mSampleFrequency = 256;
}

void VoxelizationPass_GPU::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    VoxelizationPass::setScene(pRenderContext, pScene);
    mSampleMeshPass = nullptr;
    mClipPolygonPass = nullptr;
}

void VoxelizationPass_GPU::voxelize(RenderContext* pRenderContext, const RenderData& renderData)
{
    double maxCapacity = min(mSolidRate * gridData.totalVoxelCount() * sizeof(VoxelData), 4294967296.0); // 缓冲区最大容量为4G
    maxSolidVoxelCount = (uint)ceil(maxCapacity / sizeof(VoxelData));
    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), maxSolidVoxelCount, ResourceBindFlags::UnorderedAccess);
    ref<Buffer> gBufferLock = mpDevice->createStructuredBuffer(sizeof(uint), maxSolidVoxelCount, ResourceBindFlags::UnorderedAccess);
    vBuffer = mpDevice->createStructuredBuffer(sizeof(int), gridData.totalVoxelCount(), ResourceBindFlags::UnorderedAccess);
    ref<Buffer> solidVoxelCount = mpDevice->createStructuredBuffer(sizeof(uint), 1, ResourceBindFlags::UnorderedAccess);
    polygonCountBuffer = mpDevice->createStructuredBuffer(sizeof(uint), maxSolidVoxelCount, ResourceBindFlags::UnorderedAccess);
    polygonRangeBuffer = mpDevice->createStructuredBuffer(
        sizeof(PolygonRange), maxSolidVoxelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    pRenderContext->clearUAV(gBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(gBufferLock->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(vBuffer->getUAV().get(), uint4(0xFFFFFFFFu));
    pRenderContext->clearUAV(solidVoxelCount->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(polygonCountBuffer->getUAV().get(), uint4(0));

    if (!mSampleMeshPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kSampleMeshProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());

        mSampleMeshPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    ShaderVar var = mSampleMeshPass->getRootVar();
    var["s"] = mpSampler;
    mpScene->bindShaderData(var["gScene"]);
    var["gBufferLock"] = gBufferLock;
    var["gBuffer"] = gBuffer;
    var[kVBuffer] = vBuffer;
    var["polygonCountBuffer"] = polygonCountBuffer;
    var["solidVoxelCount"] = solidVoxelCount;

    auto cb_grid = var["GridData"];
    cb_grid["gridMin"] = gridData.gridMin;
    cb_grid["voxelSize"] = gridData.voxelSize;
    cb_grid["voxelCount"] = gridData.voxelCount;

    uint meshCount = mpScene->getMeshCount();
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        uint triangleCount = meshDesc.getTriangleCount();

        auto cb_mesh = mSampleMeshPass->getRootVar()["MeshData"];
        cb_mesh["vertexCount"] = meshDesc.vertexCount;
        cb_mesh["vbOffset"] = meshDesc.vbOffset;
        cb_mesh["triangleCount"] = triangleCount;
        cb_mesh["ibOffset"] = meshDesc.ibOffset;
        cb_mesh["use16BitIndices"] = meshDesc.use16BitIndices();
        cb_mesh["materialID"] = meshDesc.materialID;
        mSampleMeshPass->execute(pRenderContext, uint3(triangleCount, 1, 1));
        pRenderContext->uavBarrier(vBuffer.get());
        pRenderContext->uavBarrier(solidVoxelCount.get());
        pRenderContext->uavBarrier(polygonCountBuffer.get());
        pRenderContext->uavBarrier(gBuffer.get());
        pRenderContext->uavBarrier(gBufferLock.get());
    }
    Tools::Profiler::BeginSample("Sample Texture");
    pRenderContext->submit(true);
    Tools::Profiler::EndSample("Sample Texture");

    ref<Buffer> cpuSolidVoxelCount = copyToCpu(mpDevice, pRenderContext, solidVoxelCount);
    ref<Buffer> cpuVBuffer = copyToCpu(mpDevice, pRenderContext, vBuffer);
    ref<Buffer> cpuPolygonCountBuffer = copyToCpu(mpDevice, pRenderContext, polygonCountBuffer);
    pRenderContext->submit(true);

    uint* pSolidVoxelCount = reinterpret_cast<uint*>(cpuSolidVoxelCount->map());
    void* pVbuffer = cpuVBuffer->map();
    void* pPolygonCount = cpuPolygonCountBuffer->map();

    gridData.solidVoxelCount = pSolidVoxelCount[0];

    std::vector<uint> polygonCounts;
    polygonCounts.resize(gridData.solidVoxelCount);
    memcpy(polygonCounts.data(), pPolygonCount, sizeof(uint) * gridData.solidVoxelCount);

    vBuffer_CPU.resize(gridData.totalVoxelCount());
    memcpy(vBuffer_CPU.data(), pVbuffer, sizeof(int) * gridData.totalVoxelCount());
    pVBuffer_CPU = vBuffer_CPU.data();

    std::vector<PolygonRange> polygonRanges;
    polygonRanges.resize(gridData.solidVoxelCount);
    polygonGroup.reserve(polygonCounts, polygonRanges);
    for (size_t i = 0; i < vBuffer_CPU.size(); i++)
    {
        int offset = vBuffer_CPU[i];
        if (offset >= 0)
        {
            int3 cellInt = IndexToCell(i, gridData.voxelCount);
            polygonRanges[offset].cellInt = cellInt;
        }
    }
    polygonRangeBuffer->setBlob(polygonRanges.data(), 0, sizeof(PolygonRange) * gridData.solidVoxelCount);

    cpuSolidVoxelCount->unmap();
    cpuPolygonCountBuffer->unmap();
    cpuVBuffer->unmap();
    pRenderContext->clearUAV(polygonCountBuffer->getUAV().get(), uint4(0));
    pRenderContext->submit(true);
}

void VoxelizationPass_GPU::sample(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mClipPolygonPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kClipMeshProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        mClipPolygonPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    ShaderVar var = mClipPolygonPass->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var[kPolygonRangeBuffer] = polygonRangeBuffer;
    var[kPolygonBuffer] = polygonGroup.get(mCompleteTimes);
    var[kVBuffer] = vBuffer;

    uint groupVoxelCount = polygonGroup.getVoxelCount(mCompleteTimes);
    auto cb = var["CB"];
    cb["groupVoxelCount"] = groupVoxelCount;
    cb["gBufferOffset"] = polygonGroup.getVoxelOffset(mCompleteTimes);

    auto cb_grid = var["GridData"];
    cb_grid["gridMin"] = gridData.gridMin;
    cb_grid["voxelSize"] = gridData.voxelSize;
    cb_grid["voxelCount"] = gridData.voxelCount;
    var["polygonCountBuffer"] = polygonCountBuffer;

    Tools::Profiler::BeginSample("Clip");
    uint meshCount = mpScene->getMeshCount();
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        uint triangleCount = meshDesc.getTriangleCount();

        auto cb_mesh = var["MeshData"];
        cb_mesh["vertexCount"] = meshDesc.vertexCount;
        cb_mesh["vbOffset"] = meshDesc.vbOffset;
        cb_mesh["triangleCount"] = triangleCount;
        cb_mesh["ibOffset"] = meshDesc.ibOffset;
        cb_mesh["use16BitIndices"] = meshDesc.use16BitIndices();
        cb_mesh["materialID"] = meshDesc.materialID;
        mClipPolygonPass->execute(pRenderContext, uint3(triangleCount, 1, 1));
        pRenderContext->uavBarrier(polygonCountBuffer.get());
    }
    pRenderContext->uavBarrier(polygonGroup.get(mCompleteTimes).get());
    pRenderContext->submit(true);
    Tools::Profiler::EndSample("Clip");

    VoxelizationPass::sample(pRenderContext, renderData);
}

void VoxelizationPass_GPU::renderUI(Gui::Widgets& widget)
{
    VoxelizationPass::renderUI(widget);
    widget.var("Solid Rate", mSolidRate, 0.01, 1.0);
}

std::string VoxelizationPass_GPU::getFileName()
{
    return VoxelizationPass::getFileName() + "_GPU";
}
