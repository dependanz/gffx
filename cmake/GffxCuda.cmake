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

# Any toolkit from 12.0 accepted, which follows from the dependency principle rather than from
# convenience. A user needs a driver, never a toolkit, so the toolkit version is a property of how
# an artifact was produced. The earlier recipe required 12.8 exactly and refused 13.x outright,
# which would have failed on a developer machine carrying a newer toolkit for no benefit.
#
# The version does matter for one thing, and it is counter-intuitive: PTX ISA is set by the
# toolkit, so a newer toolkit raises the *driver* floor of whatever it builds while the
# architecture floor stays where -arch puts it. Released artifacts are therefore built at the
# bottom of this range, not the top, and that is recorded in PYTORCH_CUDA_MATRIX_V0_1.md section 0
# rather than enforced here, because a local build should never fail over it.
find_package(CUDAToolkit 12.0 REQUIRED)

if(NOT CUDAToolkit_VERSION VERSION_EQUAL 12.0)
    message(STATUS
        "Building the CUDA plugin with CUDA Toolkit ${CUDAToolkit_VERSION}. Local builds accept "
        "any toolkit from 12.0; release artifacts are built at the range floor so the published "
        "driver minimum is as low as the range allows."
    )
endif()

# ---------------------------------------------------------------------------------------------
# Device kernels: compiled offline to PTX, embedded in the plugin, JIT-compiled by the driver.
#
# nvcc is invoked directly rather than through enable_language(CUDA), which would introduce a CUDA
# Runtime dependency the isolation gate forbids. Only a text artifact is produced here; nothing is
# linked.
# ---------------------------------------------------------------------------------------------
set(GFFX_CUDA_ARCH_FLOOR "compute_75"
    CACHE STRING "Virtual architecture the embedded PTX targets; the driver JITs forward from it")

set(gffx_cuda_ptx "${CMAKE_CURRENT_BINARY_DIR}/gffx_kernels.ptx")
set(gffx_cuda_ptx_header "${CMAKE_CURRENT_BINARY_DIR}/generated/gffx_cuda_ptx.h")

add_custom_command(
    OUTPUT "${gffx_cuda_ptx}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
            -arch=${GFFX_CUDA_ARCH_FLOOR}
            # Conformance over local accuracy: contraction into fused multiply-add would make the
            # GPU disagree with the CPU reference the acceptance fixtures pin.
            -fmad=false
            -ptx "${CMAKE_CURRENT_SOURCE_DIR}/native/cuda/kernels.cu"
            -o "${gffx_cuda_ptx}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/native/cuda/kernels.cu"
    COMMENT "Compiling GFFX CUDA kernels to PTX (${GFFX_CUDA_ARCH_FLOOR})"
    VERBATIM
)

add_custom_command(
    OUTPUT "${gffx_cuda_ptx_header}"
    COMMAND ${CMAKE_COMMAND}
            -DGFFX_PTX_INPUT=${gffx_cuda_ptx}
            -DGFFX_PTX_OUTPUT=${gffx_cuda_ptx_header}
            -DGFFX_PTX_SYMBOL=gffx_cuda_embedded_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GffxEmbedPtx.cmake"
    DEPENDS "${gffx_cuda_ptx}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GffxEmbedPtx.cmake"
    COMMENT "Embedding GFFX PTX into the plugin"
    VERBATIM
)

add_custom_target(gffx_cuda_ptx_target DEPENDS "${gffx_cuda_ptx_header}")

add_library(gffx_cuda12 MODULE ${CMAKE_CURRENT_SOURCE_DIR}/native/cuda/plugin.c)
target_compile_features(gffx_cuda12 PRIVATE c_std_11)
target_include_directories(gffx_cuda12 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/native/cuda
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
add_dependencies(gffx_cuda12 gffx_cuda_ptx_target)
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
