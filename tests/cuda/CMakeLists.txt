add_executable(gffx_test_cuda_plugin_api test_plugin_api.cpp)
target_compile_features(gffx_test_cuda_plugin_api PRIVATE cxx_std_11)
target_include_directories(gffx_test_cuda_plugin_api PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/native/cuda
)
add_test(NAME cuda.plugin_api COMMAND gffx_test_cuda_plugin_api)

add_test(NAME cuda.source_isolation
    COMMAND ${CMAKE_COMMAND}
        -DGFFX_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/source_isolation.cmake
)

if(WIN32)
    get_filename_component(gffx_tool_directory "${CMAKE_LINKER}" DIRECTORY)
    find_program(GFFX_BINARY_INSPECTOR NAMES dumpbin HINTS "${gffx_tool_directory}")
    set(GFFX_BINARY_MODE windows)
elseif(UNIX AND NOT APPLE)
    find_program(GFFX_BINARY_INSPECTOR NAMES readelf llvm-readelf)
    set(GFFX_BINARY_MODE elf)
endif()

if(GFFX_BINARY_INSPECTOR)
    add_test(NAME cuda.binary_core_isolation
        COMMAND ${CMAKE_COMMAND}
            -DGFFX_BINARY_INSPECTOR=${GFFX_BINARY_INSPECTOR}
            -DGFFX_BINARY_MODE=${GFFX_BINARY_MODE}
            -DGFFX_CORE_BINARY=$<TARGET_FILE:gffx_core>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/binary_isolation.cmake
    )
    if(TARGET gffx_cuda12)
        add_test(NAME cuda.binary_plugin_isolation
            COMMAND ${CMAKE_COMMAND}
                -DGFFX_BINARY_INSPECTOR=${GFFX_BINARY_INSPECTOR}
                -DGFFX_BINARY_MODE=${GFFX_BINARY_MODE}
                -DGFFX_CORE_BINARY=$<TARGET_FILE:gffx_core>
                -DGFFX_PLUGIN_BINARY=$<TARGET_FILE:gffx_cuda12>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/binary_isolation.cmake
        )
    endif()
else()
    message(WARNING "No supported binary inspector found; CUDA dependency closure is untested")
endif()

function(gffx_add_synthetic_cuda_plugin target mode)
    add_library(${target} MODULE synthetic_plugin.c)
    target_compile_features(${target} PRIVATE c_std_11)
    target_compile_definitions(${target} PRIVATE
        GFFX_BUILDING_CUDA_PLUGIN=1
        GFFX_SYNTHETIC_PLUGIN_MODE=${mode}
    )
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/native/cuda
    )
    set_target_properties(${target} PROPERTIES
        C_VISIBILITY_PRESET hidden
        PREFIX ""
    )
endfunction()

gffx_add_synthetic_cuda_plugin(gffx_cuda_synthetic_good 0)
gffx_add_synthetic_cuda_plugin(gffx_cuda_synthetic_incompatible 1)
gffx_add_synthetic_cuda_plugin(gffx_cuda_synthetic_missing_symbol 2)
gffx_add_synthetic_cuda_plugin(gffx_cuda_synthetic_must_not_load 3)
# Operation-table negotiation, added to plugin ABI v1 alongside dispatch.
gffx_add_synthetic_cuda_plugin(gffx_cuda_synthetic_null_operations 4)
gffx_add_synthetic_cuda_plugin(gffx_cuda_synthetic_operations 5)

# Calls a CUDA operation through the negotiated table. Needs the real plugin and a device, so it
# is registered only when the CUDA build is enabled and skips itself when no device is present.
if(TARGET gffx_cuda12)
    add_executable(gffx_test_cuda_operation_dispatch test_operation_dispatch.c)
    target_compile_features(gffx_test_cuda_operation_dispatch PRIVATE c_std_11)
    target_include_directories(gffx_test_cuda_operation_dispatch PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/native/cuda
    )
    target_link_libraries(gffx_test_cuda_operation_dispatch PRIVATE gffx_core CUDA::cuda_driver)
    add_test(NAME cuda.operation_dispatch
        COMMAND gffx_test_cuda_operation_dispatch $<TARGET_FILE:gffx_cuda12>
    )
endif()

add_executable(gffx_test_cuda_plugin_probe test_plugin_probe.c)
target_compile_features(gffx_test_cuda_plugin_probe PRIVATE c_std_11)
target_link_libraries(gffx_test_cuda_plugin_probe PRIVATE gffx_core)

if(WIN32)
    add_custom_command(TARGET gffx_test_cuda_plugin_probe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core>
            $<TARGET_FILE_DIR:gffx_test_cuda_plugin_probe>
        VERBATIM
    )
endif()
if(TARGET gffx_cuda12)
    add_custom_command(TARGET gffx_test_cuda_plugin_probe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_cuda12>
            $<TARGET_FILE_DIR:gffx_test_cuda_plugin_probe>
        VERBATIM
    )
endif()

add_executable(gffx_test_cuda_invalid_probe_isolation test_invalid_probe_isolation.c)
target_compile_features(gffx_test_cuda_invalid_probe_isolation PRIVATE c_std_11)
target_link_libraries(gffx_test_cuda_invalid_probe_isolation PRIVATE gffx_core)
if(WIN32)
    add_custom_command(TARGET gffx_test_cuda_invalid_probe_isolation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core>
            $<TARGET_FILE_DIR:gffx_test_cuda_invalid_probe_isolation>
        VERBATIM
    )
endif()
add_test(NAME cuda.invalid_probe_isolation
    COMMAND gffx_test_cuda_invalid_probe_isolation
        $<TARGET_FILE:gffx_cuda_synthetic_must_not_load>
)

add_test(NAME cuda.plugin.compatible
    COMMAND gffx_test_cuda_plugin_probe
        $<TARGET_FILE:gffx_cuda_synthetic_good> loaded compatible
)
# A plugin publishing a well-formed but entirely empty operation table is compatible: advertising
# dispatch says the path exists, never that any kernel does.
add_test(NAME cuda.plugin.operations
    COMMAND gffx_test_cuda_plugin_probe
        $<TARGET_FILE:gffx_cuda_synthetic_operations> loaded compatible
)
# Advertising operations while publishing no table is a build mismatch, not a capability to
# route around, so the host must reject the plugin rather than dispatch through a null table.
add_test(NAME cuda.plugin.null_operations
    COMMAND gffx_test_cuda_plugin_probe
        $<TARGET_FILE:gffx_cuda_synthetic_null_operations> "published no table" partial
)

add_test(NAME cuda.plugin.incompatible
    COMMAND gffx_test_cuda_plugin_probe
        $<TARGET_FILE:gffx_cuda_synthetic_incompatible> incompatible partial
)
add_test(NAME cuda.plugin.missing_symbol
    COMMAND gffx_test_cuda_plugin_probe
        $<TARGET_FILE:gffx_cuda_synthetic_missing_symbol> "missing handshake symbol" partial
)
add_test(NAME cuda.plugin.corrupt
    COMMAND gffx_test_cuda_plugin_probe
        ${CMAKE_CURRENT_SOURCE_DIR}/corrupt_plugin.txt "load failed" partial
)
add_test(NAME cuda.plugin.absent
    COMMAND gffx_test_cuda_plugin_probe
        ${CMAKE_CURRENT_BINARY_DIR}/definitely_missing_gffx_cuda12 "not found" absent
)

add_test(NAME cuda.plugin.removed
    COMMAND ${CMAKE_COMMAND}
        -DTEST_PROGRAM=$<TARGET_FILE:gffx_test_cuda_plugin_probe>
        -DPLUGIN_SOURCE=$<TARGET_FILE:gffx_cuda_synthetic_good>
        -DPLUGIN_COPY=${CMAKE_CURRENT_BINARY_DIR}/removed_plugin/$<TARGET_FILE_NAME:gffx_cuda_synthetic_good>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_removed_plugin.cmake
)

if(GFFX_ENABLE_CUDA AND GFFX_CUDA_RUN_DEVICE_TESTS)
    add_test(NAME cuda.plugin.real_device_probe
        COMMAND gffx_test_cuda_plugin_probe $<TARGET_FILE:gffx_cuda12> loaded real
    )
    add_test(NAME cuda.plugin.default_discovery
        COMMAND gffx_test_cuda_plugin_probe DEFAULT loaded real
    )
endif()
