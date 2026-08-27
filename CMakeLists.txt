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
    native/core/mesh_edge_topology.c
    native/core/mesh_face_geometry.c
    native/core/mesh_gather_faces.c
    native/core/mesh_sample_surface.c
    native/core/mesh_validate.c
    native/core/mesh_vertex_normals.c
    native/core/points_proximity.c
    native/core/render_rasterize.c
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

target_link_libraries(gffx_core PRIVATE ${CMAKE_DL_LIBS})

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

include(CTest)

if(BUILD_TESTING)
    enable_language(CXX)
    add_subdirectory(tests/abi)
    add_subdirectory(tests/cuda)
    add_subdirectory(tests/geometry)
endif()
