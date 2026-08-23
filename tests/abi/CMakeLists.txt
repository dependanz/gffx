add_executable(gffx_test_abi_layout test_abi_layout.c)
target_link_libraries(gffx_test_abi_layout PRIVATE gffx_core)
add_test(NAME abi.layout COMMAND gffx_test_abi_layout)

add_executable(gffx_test_abi_runtime test_abi_runtime.c)
target_link_libraries(gffx_test_abi_runtime PRIVATE gffx_core)
add_test(NAME abi.runtime COMMAND gffx_test_abi_runtime)

add_executable(gffx_test_abi_capabilities test_abi_capabilities.c)
target_link_libraries(gffx_test_abi_capabilities PRIVATE gffx_core)
add_test(NAME abi.capabilities COMMAND gffx_test_abi_capabilities)

add_executable(gffx_test_abi_symbols test_abi_symbols.c)
target_link_libraries(gffx_test_abi_symbols PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME abi.symbols COMMAND gffx_test_abi_symbols $<TARGET_FILE:gffx_core>)

add_executable(gffx_test_abi_cpp_headers test_abi_cpp_headers.cpp)
target_compile_features(gffx_test_abi_cpp_headers PRIVATE cxx_std_11)
target_link_libraries(gffx_test_abi_cpp_headers PRIVATE gffx_core)
add_test(NAME abi.cpp_headers COMMAND gffx_test_abi_cpp_headers)

add_library(gffx_core_test_provider SHARED
    ${PROJECT_SOURCE_DIR}/native/core/abi.c
    ${PROJECT_SOURCE_DIR}/native/core/capabilities.c
    ${PROJECT_SOURCE_DIR}/native/core/status.c
)
target_compile_features(gffx_core_test_provider PUBLIC c_std_11)
target_compile_definitions(gffx_core_test_provider PRIVATE
    GFFX_BUILDING_LIBRARY=1
    GFFX_ENABLE_TEST_PROVIDER=1
    GFFX_PACKAGE_VERSION="${GFFX_PACKAGE_VERSION_VALUE}"
)
target_include_directories(gffx_core_test_provider PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(gffx_core_test_provider PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

add_executable(gffx_test_abi_test_provider test_abi_test_provider.c)
target_link_libraries(gffx_test_abi_test_provider PRIVATE gffx_core_test_provider)
add_test(NAME abi.test_provider COMMAND gffx_test_abi_test_provider)

if(WIN32)
    foreach(test_target IN ITEMS
        gffx_test_abi_layout
        gffx_test_abi_runtime
        gffx_test_abi_capabilities
        gffx_test_abi_cpp_headers
    )
        add_custom_command(TARGET ${test_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:gffx_core>
                $<TARGET_FILE_DIR:${test_target}>
            VERBATIM
        )
    endforeach()
    add_custom_command(TARGET gffx_test_abi_test_provider POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core_test_provider>
            $<TARGET_FILE_DIR:gffx_test_abi_test_provider>
        VERBATIM
    )
endif()
