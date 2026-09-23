add_plugin(Voxelization)

target_sources(Voxelization PRIVATE
    HostDeviceShared.slang

    Voxel/ABSDF.slang
    Voxel/VoxelGrid.slang
    
    Math/AABB.slang
    Math/Projection.slang
    Math/Ellipsoid.slang
    Math/Polygon.slang
    Math/Triangle.slang
    Math/Sampling.slang
    Math/SphericalHarmonics.slang
    Math/LebedevQuadrature.slang
    Math/VoxelizationUtility.h
    Math/VoxelizationUtility.slang
    Math/Random.h
    Math/Random.slang

    Image/CPUTexture.h
    Image/ImageLoader.cpp
    Image/ImageLoader.h

    VoxelizationBase.h
    VoxelizationBase.cpp

    VoxelizationShared.slang
    Shading.slang
    MixLightSampler.slang

    VoxelizationPass.cpp
    VoxelizationPass.h
    AnalyzePolygon.cs.slang

    VoxelizationPass_CPU.cpp
    VoxelizationPass_CPU.h
    MeshSampler.h
    LoadMesh.cs.slang

    VoxelizationPass_GPU.cpp
    VoxelizationPass_GPU.h
    SampleMesh.cs.slang
    ClipMesh.cs.slang

    RayMarchingPass.cpp
    RayMarchingPass.h
    PrepareShadingData.cs.slang
    RayMarchingCommon.slang
    RayMarching.ps.slang
    BakeReconstruction.cs.slang
    DisplayNDF.ps.slang

    ReadVoxelPass.cpp
    ReadVoxelPass.h
)
target_copy_shaders(Voxelization RenderPasses/Voxelization)
target_source_group(Voxelization "RenderPasses")
