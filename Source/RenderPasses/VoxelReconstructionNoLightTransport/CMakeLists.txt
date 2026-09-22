add_plugin(VoxelReconstructionNoLightTransport)

set(voxel_reconstruction_sources
    VoxelReconstructionNoLightTransport.cpp
    Initialization.cpp
    PointCloudInitialization.cpp
    CoarseToFine.cpp
    RayMarchingPass.cpp
    LossPass.cpp
    ReducePass.cpp
    GradientPass.cpp
    UpdatePass.cpp
    DataProcess.cpp
)

set(voxel_reconstruction_shared_headers
    VoxelReconstructionNoLightTransport.h
    Defines.h
    HostDeviceShared.slang
    PathRecord.slang
    GradRecord.slang
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
)

target_sources(VoxelReconstructionNoLightTransport PRIVATE
    ${voxel_reconstruction_sources}
    ${voxel_reconstruction_shared_headers}
    PointCloudLoader.cpp
    PointCloudLoader.h
    Math/LebedevQuadrature.slang
    Math/Random.h
    Math/Random.slang

    Shader/ReflectTypes.cs.slang
    Shader/InitializeData.cs.slang
    Shader/InitializePointCloud.cs.slang
    Shader/RefineVoxelGrid.cs.slang
    Shader/EvaluationImages.cs.slang
    Shader/RayMarchingPass.ps.slang
    Shader/LossPass.cs.slang
    Shader/ReduceTexturePass.cs.slang
    Shader/ReduceBufferPass.cs.slang
    Shader/GradientPass.cs.slang
    Shader/UpdatePass.cs.slang
    Shader/DiffVoxelIO.slang
    Shader/DiffTools.slang
    Shader/DiffShading.slang

)

# Keep this pass's C++ dependencies explicit: localized MSVC /showIncludes output
# can fail to match Ninja's recorded prefix. A successful build must still pick
# up changed experiment settings and shared CPU/GPU layouts. Keep the list above
# in sync with the local C++ include graph; GPU-only shaders need no object rebuild.
set(voxel_reconstruction_object_dependencies ${voxel_reconstruction_shared_headers})
list(TRANSFORM voxel_reconstruction_object_dependencies PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
foreach(source IN LISTS voxel_reconstruction_sources)
    set_property(SOURCE "${source}" APPEND PROPERTY OBJECT_DEPENDS ${voxel_reconstruction_object_dependencies})
endforeach()
set_property(SOURCE PointCloudLoader.cpp PointCloudInitialization.cpp APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/PointCloudLoader.h"
)

target_source_group(VoxelReconstructionNoLightTransport "RenderPasses")
