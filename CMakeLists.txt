cmake_minimum_required(VERSION 3.25)

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

include(CTest)

if(BUILD_TESTING)
    enable_language(CXX)
    add_subdirectory(tests/abi)
endif()
