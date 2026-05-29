add_plugin(VoxelReconstruction)

target_sources(VoxelReconstruction PRIVATE
    HostDeviceShared.slang

    Voxel/ABSDF.slang
    Voxel/VoxelGrid.slang
    Voxel/VoxelData.slang
    Voxel/Shading.slang

    Math/AABB.slang
    Math/Projection.slang
    Math/Ellipsoid.slang
    Math/Polygon.slang
    Math/Triangle.slang
    Math/Sampling.slang
    Math/SphericalHarmonics.slang
    Math/LebedevQuadrature.slang
    Math/Random.h
    Math/Random.slang
    Math/DensityPrimitive.slang

    VoxelReconstruction.cpp
    VoxelReconstruction.h
    LossPass.cpp
    RayMarchingPass.cpp
    Defines.h
    GradientPass.cpp
    UpdatePass.cpp
    ReducePass.cpp
    DataProcess.cpp

    Shader/ReflectTypes.cs.slang
    Shader/ProcessXuData.cs.slang
    Shader/RayMarchingPass.ps.slang
    Shader/LossPass.cs.slang
    Shader/GradientPass.cs.slang
    Shader/UpdatePass.cs.slang
    Shader/ReduceTexturePass.cs.slang
    Shader/ReduceBufferPass.cs.slang

    MixLightSampler.slang
    PathRecord.slang
    GradRecord.slang
    
)


target_copy_shaders(VoxelReconstruction RenderPasses/VoxelReconstruction)
target_source_group(VoxelReconstruction "RenderPasses")
