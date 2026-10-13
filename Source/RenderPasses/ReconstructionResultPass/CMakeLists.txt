add_plugin(ReconstructionResultPass)
target_sources(ReconstructionResultPass PRIVATE ReconstructionResultPass.cpp ReconstructionResultPass.h ResultMetadata.h)
target_source_group(ReconstructionResultPass "RenderPasses")

option(RECONSTRUCTION_RESULT_BUILD_TESTS "Build headless PNG export regression tests" OFF)
if(RECONSTRUCTION_RESULT_BUILD_TESTS)
    add_falcor_executable(ReconstructionResultTests)
    target_sources(ReconstructionResultTests PRIVATE Tests/ResultExport.cpp ReconstructionResultPass.cpp)
endif()
