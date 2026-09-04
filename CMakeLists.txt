cmake_minimum_required(VERSION 3.26)  # Python_add_library(USE_SABI) requires 3.26

project(gffx LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GffxOptions)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" GFFX_VERSION_LINE
    REGEX "^version[ \t]*="
    LIMIT_COUNT 1
)
string(REGEX REPLACE
    "^version[ \t]*=[ \t]*\"([^\"]+)\".*$"
    "\\1"
    GFFX_PACKAGE_VERSION_VALUE
    "${GFFX_VERSION_LINE}"
)
if(GFFX_PACKAGE_VERSION_VALUE STREQUAL GFFX_VERSION_LINE)
    message(FATAL_ERROR "Unable to read the static project version from pyproject.toml")
endif()

add_library(gffx_core SHARED
    native/core/abi.c
    native/core/capabilities.c
    native/core/cuda_loader.c
    native/core/io_ply.c
    native/core/mesh_edge_topology.c
    native/core/mesh_face_geometry.c
    native/core/mesh_gather_faces.c
    native/core/mesh_sample_surface.c
    native/core/mesh_validate.c
    native/core/mesh_vertex_normals.c
    native/io/ply_file.c
    native/core/points_proximity.c
    native/core/render_rasterize.c
    native/core/render_texture.c
    native/core/status.c
    native/core/transforms.c
)

target_compile_features(gffx_core PUBLIC c_std_11)
target_compile_definitions(gffx_core PRIVATE
    GFFX_BUILDING_LIBRARY=1
    GFFX_PACKAGE_VERSION="${GFFX_PACKAGE_VERSION_VALUE}"
)
target_include_directories(gffx_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/native/core
        ${CMAKE_CURRENT_SOURCE_DIR}/native/cuda
)

# Three shared libraries are built from the core sources: gffx_core itself, and the two providers
# in tests/abi that mirror it. All three needed the same runtime dependencies and all three were
# missing libm, but the failure surfaced one target at a time, because each is linked by a
# different executable and only the executable's link fails. Fixing one and pushing cost a full CI
# round; declaring the rule once is the actual repair.
#
# glibc keeps the math functions in a separate library, so a core calling floor, log2 or sqrt has
# to say so. The omission is invisible in three of four contexts: a shared object may carry
# unresolved symbols so the library builds, and the interpreter has already loaded libm so the
# wheel imports and every Python gate passes. Only a directly linked executable fails, which is
# what the native CTest build does. Windows and macOS carry these in their default C library.
function(gffx_link_core_runtime target)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    if(UNIX)
        target_link_libraries(${target} PRIVATE m)
    endif()
    if(UNIX AND NOT APPLE)
        # Report a missing dependency at the library that has it, rather than at whichever
        # executable links that library later. Verified against these sources on Linux: they link
        # cleanly with the flag, and with libm removed the flag names the six math symbols at the
        # library itself instead of deferring them to a consumer.
        target_link_options(${target} PRIVATE "LINKER:--no-undefined")
    endif()
endfunction()

gffx_link_core_runtime(gffx_core)

set_target_properties(gffx_core PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# Step 7 packaging: the wheel carries gffx_core beside the Python package, where
# the private CPython loader finds it on demand.
install(TARGETS gffx_core
    RUNTIME DESTINATION gffx
    LIBRARY DESTINATION gffx
    # The Windows import library (.lib) is build-only and is excluded from the runtime wheel.
    ARCHIVE DESTINATION gffx EXCLUDE_FROM_ALL
)

# The private CPython Limited-API loader is built only when a Python target is requested, which
# keeps the pure C11 core configurable and buildable with no Python present.
if(GFFX_BUILD_PYTHON)
    include(GffxPython)
endif()

# The adapter is an opt-in build component: ordinary CPU/core builds need no C++ compiler or
# PyTorch installation, while PyTorch-ready wheels enable it explicitly.
if(GFFX_BUILD_PYTORCH)
    enable_language(CXX)
    include(GffxPyTorch)
endif()

# CUDA is a separately built plugin. The default core path never enables the CUDA language.
if(GFFX_ENABLE_CUDA)
    include(GffxCuda)
endif()

# Developer demo. Not a test, not installed, and not part of the library surface; it exists so
# the implemented kernels can be seen producing an image. Build with --target gffx_demo_cube.
add_executable(gffx_demo_cube EXCLUDE_FROM_ALL examples/render_cube.c)
target_link_libraries(gffx_demo_cube PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_demo_cube PRIVATE m)
endif()
if(WIN32)
    add_custom_command(TARGET gffx_demo_cube POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core> $<TARGET_FILE_DIR:gffx_demo_cube>
        VERBATIM
    )
endif()

add_executable(gffx_demo_panels EXCLUDE_FROM_ALL examples/render_panels.c)
target_link_libraries(gffx_demo_panels PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_demo_panels PRIVATE m)
endif()
if(WIN32)
    add_custom_command(TARGET gffx_demo_panels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core> $<TARGET_FILE_DIR:gffx_demo_panels>
        VERBATIM
    )
endif()

# Emits the committed parity oracle that the PyTorch adapter fixtures compare against, so the C
# kernel stays the single source of truth for their expected values. Regenerate by running it.
add_executable(gffx_emit_face_geometry_oracle EXCLUDE_FROM_ALL examples/emit_face_geometry_oracle.c)
target_link_libraries(gffx_emit_face_geometry_oracle PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_emit_face_geometry_oracle PRIVATE m)
endif()
if(WIN32)
    add_custom_command(TARGET gffx_emit_face_geometry_oracle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core> $<TARGET_FILE_DIR:gffx_emit_face_geometry_oracle>
        VERBATIM
    )
endif()

include(CTest)

if(BUILD_TESTING)
    enable_language(CXX)
    add_subdirectory(tests/abi)
    add_subdirectory(tests/cuda)
    add_subdirectory(tests/geometry)
    add_subdirectory(tests/io)
    add_subdirectory(tests/sweep)
endif()
