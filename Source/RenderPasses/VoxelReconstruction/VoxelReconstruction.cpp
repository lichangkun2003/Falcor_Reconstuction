/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "VoxelReconstruction.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VoxelReconstruction>();
}


// 常用命名（shader路径）
namespace VoxelPrime
{
const std::string ReflectTypesShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/ReflectTypes.cs.slang";
const std::string ProcessXuDataShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/ProcessXuData.cs.slang";



inline std::string kGBuffer = "gBuffer";
inline std::string kVBuffer = "vBuffer";
inline std::string kPBuffer = "pBuffer";
inline std::string kBlockMap = "blockMap";
}

VoxelReconstruction::VoxelReconstruction(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mpDevice = pDevice;

    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);


    // Create Grid pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(VoxelPrime::ReflectTypesShaderFilePath).csEntry("main");
        DefineList defines;
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, true);
    }

    // Create ProcessXuData pass
    {
        ProgramDesc desc;
        desc.addShaderLibrary(VoxelPrime::ProcessXuDataShaderFilePath).csEntry("main");
        DefineList defines;
        mpProcessXuDataPass = ComputePass::create(mpDevice, desc, defines, true);
    }

}

Properties VoxelReconstruction::getProperties() const
{
    return {};
}

RenderPassReflection VoxelReconstruction::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;

    
    // Input
    reflector.addInput(VoxelPrime::kVBuffer, VoxelPrime::kVBuffer)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::R32Uint)
        .texture3D();

    //reflector.addInput(kGBuffer, kGBuffer)
    //    .bindFlags(ResourceBindFlags::ShaderResource)
    //    .format(ResourceFormat::Unknown)
    //    .rawBuffer(VoxelizationBase::GlobalGridData.solidVoxelCount * sizeof(PrimitiveBSDF));

    //reflector.addInput(kPBuffer, kPBuffer)
    //    .bindFlags(ResourceBindFlags::ShaderResource)
    //    .format(ResourceFormat::Unknown)
    //    .rawBuffer(VoxelizationBase::GlobalGridData.solidVoxelCount * sizeof(Ellipsoid));

    reflector.addInput(VoxelPrime::kBlockMap, VoxelPrime::kBlockMap)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Uint)
        .texture2D();


    // Output
    reflector.addOutput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0, 1, 1);

    return reflector;
}

void VoxelReconstruction::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData.getTexture("src");       
    if (mpScene == nullptr)
        return;
    mFrameDim = renderData.getDefaultTextureDims();
    mInvFrameDim = 1.0f / float2(mFrameDim);
    beginFrame(pRenderContext);

    proccessXuData(pRenderContext, renderData);


    endFrame(pRenderContext);
}

void VoxelReconstruction::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene){
    mpScene = pScene;
    UpdateVoxelGrid(mpScene, mVoxelResolution);
    setupGridResouce(pRenderContext, true);
}

void VoxelReconstruction::renderUI(Gui::Widgets& widget) {
    //widget.text("kBlockMap Size: " + ToString(BlockMapSize));








    widget.text("Voxel Size: " + ToString(mGridResources.gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)mGridResources.gridData.voxelCount));
    widget.text("Block Count: " + ToString((int3)mGridResources.gridData.blockCount3D()));
    widget.text("Grid Min: " + ToString(mGridResources.gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(mGridResources.gridData.solidVoxelCount));
    widget.text(
        "Solid Rate: " + std::to_string(mGridResources.gridData.solidVoxelCount / (float)mGridResources.gridData.totalVoxelCount())
    );
    //widget.text("Max Polygon Count: " + std::to_string(mGridResources.gridData.maxPolygonCount));
    //widget.text("Total Polygon Count: " + std::to_string(mGridResources.gridData.totalPolygonCount));
}

void VoxelReconstruction::beginFrame(RenderContext* pRenderContext, bool forceReset)
{
    mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
    setupGridResouce(pRenderContext, forceReset);
}

void VoxelReconstruction::endFrame(RenderContext* pRenderContext) {

    mpPixelDebug->endFrame(pRenderContext);
    mFrameCount++;
}



void VoxelReconstruction::setupGridResouce(RenderContext* pRenderContext, bool forceReset) {
    if (!mpGridBlock)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock");

        if (!reflector)
            std::cout << "ComputerPass : ReflectTypes Error !!!!\n ";

        mpGridBlock = ParameterBlock::create(mpDevice, reflector);
    }
    ShaderVar gridBlock = mpGridBlock->getRootVar();

    // we only fully initialize resource once
    const bool initializeResource = !mGridResources.gridDataBuffer;

    // -----------------------------------------------------------------------------
    // Resource setup
    // -----------------------------------------------------------------------------
    if (initializeResource || forceReset)
    {
        mGridResources.gridDataBuffer = mpDevice->createStructuredBuffer(
            sizeof(VoxelData), mGridResources.gridData.totalVoxelCount(),
            ResourceBindFlags::UnorderedAccess);
        mGridResources.blockOM = mpDevice->createTexture2D(
            mGridResources.gridData.blockGridSizeXY().x,
            mGridResources.gridData.blockGridSizeXY().y,
            ResourceFormat::RGBA32Uint,
            1u,
            1u,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        pRenderContext->clearUAV(mGridResources.gridDataBuffer->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(mGridResources.blockOM->getUAV().get(), uint4(0));
    }


    gridBlock["gridDataBuffer"] = mGridResources.gridDataBuffer;
    gridBlock["blockOM"] = mGridResources.blockOM;
    gridBlock["voxelCount"] = mGridResources.gridData.voxelCount;

}


void VoxelReconstruction::proccessXuData(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->clearUAV(mGridResources.gridDataBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mGridResources.blockOM->getUAV().get(), uint4(0));


    auto var = mpProcessXuDataPass->getRootVar();
    var[VoxelPrime::kVBuffer] = renderData.getTexture(VoxelPrime::kVBuffer);
    var[VoxelPrime::kBlockMap] = renderData.getTexture(VoxelPrime::kBlockMap);

    mpProcessXuDataPass->execute(pRenderContext, mGridResources.gridData.voxelCount);

    // TODO
    //mGridResources.gridData.solidVoxelCount = 0;
}

void VoxelReconstruction::UpdateVoxelGrid(ref<Scene> scene, uint voxelResolution)
{
    float3 diag;
    float length;
    float3 center;
    if (scene)
    {
        AABB aabb = scene->getSceneBounds();
        diag = aabb.maxPoint - aabb.minPoint;
        length = std::max(diag.z, std::max(diag.x, diag.y));
        center = aabb.center();
        diag *= 1.02f;
        length *= 1.02f;
    }
    else
    {
        diag = float3(1);
        length = 1.f;
        center = float3(0);
    }

    mGridResources.gridData.voxelSize = float3(length / voxelResolution);
    float3 temp = diag / mGridResources.gridData.voxelSize;

    mGridResources.gridData.voxelCount = uint3(
        (uint)math::ceil(temp.x / MinFactor.x) * MinFactor.x,
        (uint)math::ceil(temp.y / MinFactor.y) * MinFactor.y,
        (uint)math::ceil(temp.z / MinFactor.z) * MinFactor.z
    );
    mGridResources.gridData.gridMin = center - 0.5f * mGridResources.gridData.voxelSize * float3(mGridResources.gridData.voxelCount);
    mGridResources.gridData.solidVoxelCount = 0;
}
