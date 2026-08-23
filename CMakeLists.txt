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
    native/core/status.c
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
)

set_target_properties(gffx_core PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# Step 7 packaging: the wheel carries gffx_core beside the Python package, where
# src/gffx/_capabilities.py looks for it. No CPython extension is installed here.
install(TARGETS gffx_core
    RUNTIME DESTINATION gffx
    LIBRARY DESTINATION gffx
    # The Windows import library (.lib) is needed only to link against the core at build time and
    # is deliberately excluded from the runtime wheel, which loads the DLL through ctypes.
    ARCHIVE DESTINATION gffx EXCLUDE_FROM_ALL
)

# The private CPython Limited-API loader is built only when a Python target is requested, which
# keeps the pure C11 core configurable and buildable with no Python present.
if(GFFX_BUILD_PYTHON)
    include(GffxPython)
endif()

include(CTest)

if(BUILD_TESTING)
    enable_language(CXX)
    add_subdirectory(tests/abi)
endif()
