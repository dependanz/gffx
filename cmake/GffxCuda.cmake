include_guard(GLOBAL)

# The CUDA provider is a separate shared object loaded only by an explicit full capability probe.
# It uses the Driver API and deliberately has no CUDA Runtime, framework, allocator, or graphics
# library dependency.
if(APPLE)
    message(FATAL_ERROR "The Phase 1 CUDA plugin supports Windows/Linux x86-64 only")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
    message(FATAL_ERROR "The Phase 1 CUDA plugin requires Windows/Linux x86-64")
endif()

find_package(CUDAToolkit 12.8 REQUIRED)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    message(FATAL_ERROR
        "The single-plugin feasibility target is CUDA Toolkit 12.8; CUDA 13+ builds are not admitted"
    )
elseif(NOT CUDAToolkit_VERSION VERSION_EQUAL 12.8)
    message(WARNING
        "Building the Step 9 scaffold with CUDA Toolkit ${CUDAToolkit_VERSION}. "
        "This is developer evidence only; release artifacts require CUDA Toolkit 12.8."
    )
endif()

add_library(gffx_cuda12 MODULE ${CMAKE_CURRENT_SOURCE_DIR}/native/cuda/plugin.c)
target_compile_features(gffx_cuda12 PRIVATE c_std_11)
target_include_directories(gffx_cuda12 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/native/cuda
)
target_compile_definitions(gffx_cuda12 PRIVATE
    GFFX_BUILDING_CUDA_PLUGIN=1
    GFFX_CUDA_BUILD_ID="gffx-cuda12/${GFFX_PACKAGE_VERSION_VALUE},toolkit=${CUDAToolkit_VERSION},driver-api"
)
target_link_libraries(gffx_cuda12 PRIVATE CUDA::cuda_driver)
set_target_properties(gffx_cuda12 PROPERTIES
    OUTPUT_NAME "gffx_cuda12"
    C_VISIBILITY_PRESET hidden
)
if(WIN32)
    set_target_properties(gffx_cuda12 PROPERTIES PREFIX "")
endif()

install(TARGETS gffx_cuda12
    RUNTIME DESTINATION gffx
    LIBRARY DESTINATION gffx
)
