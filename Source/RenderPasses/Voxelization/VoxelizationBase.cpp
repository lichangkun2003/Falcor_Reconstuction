#include "VoxelizationBase.h"
#include "VoxelizationPass.h"
#include "RayMarchingPass.h"
#include "VoxelizationPass_CPU.h"
#include "VoxelizationPass_GPU.h"
#include "ReadVoxelPass.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, RayMarchingPass>();
    registry.registerClass<RenderPass, VoxelizationPass_CPU>();
    registry.registerClass<RenderPass, VoxelizationPass_GPU>();
    registry.registerClass<RenderPass, ReadVoxelPass>();
}

GridData VoxelizationBase::GlobalGridData = {};
uint3 VoxelizationBase::MinFactor = uint3(1, 1, 1);
bool VoxelizationBase::FileUpdated = true;
bool VoxelizationBase::LightChanged = true;
std::string VoxelizationBase::ResourceFolder = "D:/lck/vs/Falcor_Reconstuction/resource/";

std::random_device rd;
std::mt19937 Random::Generator{ rd() };
std::uniform_real_distribution<double> Random::Distribution{ 0.0, 1.0 };

const std::string kInputDiffuse = "diffuse";
const std::string kInputSpecular = "specular";
const std::string kInputRoughness = "roughness";
const std::string kInputArea = "area";
