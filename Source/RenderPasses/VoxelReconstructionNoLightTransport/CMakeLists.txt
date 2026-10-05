add_plugin(VoxelReconstructionNoLightTransport)

set(voxel_reconstruction_sources
    VoxelReconstructionNoLightTransport.cpp
    PointCloudInitialization.cpp
    RayMarchingPass.cpp
    LossPass.cpp
    ReducePass.cpp
    GradientPass.cpp
    UpdatePass.cpp
    TopologyPass.cpp
    TopologyDeletion.cpp
    TopologyGrowth.cpp
    CoarseToFine.cpp
    DataProcess.cpp
)

set(voxel_reconstruction_shared_headers
    VoxelReconstructionNoLightTransport.h
    Defines.h
    HostDeviceShared.slang
    PathRecord.slang
    GradRecord.slang
    GeometryAdamState.slang
    TopologyEvidence.slang
    Voxel/ABSDF.slang
    Voxel/VoxelGrid.slang
    Voxel/VoxelData.slang
    Voxel/Shading.slang

    Math/AABB.slang
    Math/Projection.slang
    Math/Ellipsoid.slang
    Math/GaussianEllipsoid.slang
    Math/GrowthEllipsoid.slang
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
    Shader/InitializePointCloud.cs.slang
    Shader/BuildSparseIndex.cs.slang
    Shader/RayMarchingPass.ps.slang
    Shader/LossPass.cs.slang
    Shader/ReduceTexturePass.cs.slang
    Shader/ReduceBufferPass.cs.slang
    Shader/GradientPass.cs.slang
    Shader/UpdatePass.cs.slang
    Shader/TopologyPass.cs.slang
    Shader/DeleteCompactPass.cs.slang
    Shader/GrowthPass.cs.slang
    Shader/RefineGrid.cs.slang
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

option(VOXEL_RECONSTRUCTION_BUILD_GROWTH_TESTS "Build the headless neighbor-growth GPU regression test" OFF)
if(VOXEL_RECONSTRUCTION_BUILD_GROWTH_TESTS)
    add_falcor_executable(VoxelNeighborGrowthTests)
    target_sources(VoxelNeighborGrowthTests PRIVATE
        Tests/NeighborGrowth.cpp
        Tests/CoarseToFine.cpp
        ${voxel_reconstruction_sources}
        ${voxel_reconstruction_shared_headers}
        PointCloudLoader.cpp
        PointCloudLoader.h
    )
    # Small pages exercise real extension/overflow paths without allocating
    # millions of voxels. getReconstructionDefines() forwards these to Slang.
    target_compile_definitions(VoxelNeighborGrowthTests PRIVATE SPARSE_POOL_PAGE_SIZE=32u SPARSE_POOL_MAX_PAGES=2)
    set_property(SOURCE Tests/NeighborGrowth.cpp APPEND PROPERTY OBJECT_DEPENDS ${voxel_reconstruction_object_dependencies})
    set_property(SOURCE Tests/CoarseToFine.cpp APPEND PROPERTY OBJECT_DEPENDS ${voxel_reconstruction_object_dependencies})
endif()
