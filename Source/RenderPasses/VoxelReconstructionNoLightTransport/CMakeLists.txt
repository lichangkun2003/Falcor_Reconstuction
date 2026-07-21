add_plugin(VoxelReconstructionNoLightTransport)

target_sources(VoxelReconstructionNoLightTransport PRIVATE
    HostDeviceShared.slang

    VoxelReconstructionNoLightTransport.cpp
    VoxelReconstructionNoLightTransport.h
    Defines.h
    RayMarchingPass.cpp
    LossPass.cpp
    ReducePass.cpp
    GradientPass.cpp
    UpdatePass.cpp
    DataProcess.cpp

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

    Shader/ReflectTypes.cs.slang
    Shader/ProcessXuData.cs.slang
    Shader/RayMarchingPass.ps.slang
    Shader/LossPass.cs.slang
    Shader/ReduceTexturePass.cs.slang
    Shader/ReduceBufferPass.cs.slang
    Shader/GradientPass.cs.slang
    Shader/UpdatePass.cs.slang
    Shader/DiffVoxelIO.slang
    Shader/DiffTools.slang
    Shader/DiffShading.slang

    PathRecord.slang
    GradRecord.slang

)

target_source_group(VoxelReconstructionNoLightTransport "RenderPasses")
