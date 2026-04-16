#include "VoxelizationPass_CPU.h"

namespace
{
const std::string kLoadMeshProgramFile = "RenderPasses/Voxelization/LoadMesh.cs.slang";
}; // namespace

VoxelizationPass_CPU::VoxelizationPass_CPU(ref<Device> pDevice, const Properties& props) : VoxelizationPass(pDevice, props)
{
}

void VoxelizationPass_CPU::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    VoxelizationPass::setScene(pRenderContext, pScene);
    mLoadMeshPass = nullptr;
}

void VoxelizationPass_CPU::voxelize(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mLoadMeshPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kLoadMeshProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());

        mLoadMeshPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    uint meshCount = mpScene->getMeshCount();
    uint vertexCount = 0;
    uint triangleCount = 0;
    for (MeshID meshID{ 0 }; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        vertexCount += meshDesc.vertexCount;
        triangleCount += meshDesc.getTriangleCount();
    }

    ref<Buffer> positions = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, ResourceBindFlags::UnorderedAccess);
    ref<Buffer> normals = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, ResourceBindFlags::UnorderedAccess);
    ref<Buffer> texCoords = mpDevice->createStructuredBuffer(sizeof(float2), vertexCount, ResourceBindFlags::UnorderedAccess);
    ref<Buffer> triangles = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, ResourceBindFlags::UnorderedAccess);

    std::vector<MeshHeader> meshList;

    ShaderVar var = mLoadMeshPass->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["positions"] = positions;
    var["normals"] = normals;
    var["texCoords"] = texCoords;
    var["triangles"] = triangles;
    uint triangleOffset = 0;
    for (MeshID meshID{ 0 }; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        MeshHeader mesh = { meshID.get(), meshDesc.materialID, meshDesc.vertexCount, meshDesc.getTriangleCount(), triangleOffset};
        meshList.push_back(mesh);

        auto meshData = mLoadMeshPass->getRootVar()["MeshData"];
        meshData["vertexCount"] = meshDesc.vertexCount;
        meshData["vbOffset"] = meshDesc.vbOffset;
        meshData["triangleCount"] = meshDesc.getTriangleCount();
        meshData["ibOffset"] = meshDesc.ibOffset;
        meshData["triOffset"] = triangleOffset;
        meshData["use16BitIndices"] = meshDesc.use16BitIndices();
        mLoadMeshPass->execute(pRenderContext, uint3(meshDesc.getTriangleCount(), 1, 1));

        triangleOffset += meshDesc.getTriangleCount();
    }

    ref<Buffer> cpuPositions = copyToCpu(mpDevice, pRenderContext, positions);
    ref<Buffer> cpuNormals = copyToCpu(mpDevice, pRenderContext, normals);
    ref<Buffer> cpuTexCoords = copyToCpu(mpDevice, pRenderContext, texCoords);
    ref<Buffer> cpuTriangles = copyToCpu(mpDevice, pRenderContext, triangles);
    pRenderContext->submit(true);

    float3* pPos = reinterpret_cast<float3*>(cpuPositions->map());
    float3* pNormal = reinterpret_cast<float3*>(cpuNormals->map());
    float2* pUV = reinterpret_cast<float2*>(cpuTexCoords->map());
    uint3* pTri = reinterpret_cast<uint3*>(cpuTriangles->map());
    SceneHeader header = { meshCount, vertexCount, triangleCount };

    Tools::Profiler::BeginSample("Clip");
    polygonGenerator.reset();
    polygonGenerator.clipAll(header, meshList, pPos, pNormal, pUV, pTri);
    Tools::Profiler::EndSample("Clip");

    polygonGroup.setBlob(polygonGenerator.polygonArrays, polygonGenerator.polygonRangeBuffer);
    cpuPositions->unmap();
    cpuNormals->unmap();
    cpuTexCoords->unmap();
    cpuTriangles->unmap();

    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), gridData.solidVoxelCount, ResourceBindFlags::UnorderedAccess);
    gBuffer->setBlob(polygonGenerator.gBuffer.data(), 0, gridData.solidVoxelCount * sizeof(VoxelData));

    polygonRangeBuffer = mpDevice->createStructuredBuffer(
        sizeof(PolygonRange),
        gridData.solidVoxelCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    polygonRangeBuffer->setBlob(polygonGenerator.polygonRangeBuffer.data(), 0, gridData.solidVoxelCount * sizeof(PolygonRange));

    pVBuffer_CPU = polygonGenerator.vBuffer.data();

    pRenderContext->submit(true);
}

void VoxelizationPass_CPU::sample(RenderContext* pRenderContext, const RenderData& renderData)
{
    VoxelizationPass::sample(pRenderContext, renderData);
}

void VoxelizationPass_CPU::renderUI(Gui::Widgets& widget)
{
    VoxelizationPass::renderUI(widget);
}

std::string VoxelizationPass_CPU::getFileName()
{
    return VoxelizationPass::getFileName() + "_CPU";
}
