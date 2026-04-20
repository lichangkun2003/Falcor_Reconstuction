add_plugin(VoxelReconstruction)

target_sources(VoxelReconstruction PRIVATE
    HostDeviceShared.slang

    Voxel/ABSDF.slang
    Voxel/VoxelGrid.slang
    Voxel/VoxelData.slang

    VoxelReconstruction.cpp
    VoxelReconstruction.h
    Defines.h

    Shader/ReflectTypes.cs.slang
)


target_copy_shaders(VoxelReconstruction RenderPasses/VoxelReconstruction)
target_source_group(VoxelReconstruction "RenderPasses")
