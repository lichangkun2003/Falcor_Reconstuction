add_plugin(VoxelReconstruction)

target_sources(VoxelReconstruction PRIVATE
    VoxelReconstruction.cpp
    VoxelReconstruction.h
)

target_source_group(VoxelReconstruction "RenderPasses")
