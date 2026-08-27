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
    ${PROJECT_SOURCE_DIR}/native/core/cuda_loader.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_edge_topology.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_face_geometry.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_gather_faces.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_sample_surface.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_validate.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_vertex_normals.c
    ${PROJECT_SOURCE_DIR}/native/core/points_proximity.c
    ${PROJECT_SOURCE_DIR}/native/core/render_rasterize.c
    ${PROJECT_SOURCE_DIR}/native/core/status.c
    ${PROJECT_SOURCE_DIR}/native/core/transforms.c
)
target_compile_features(gffx_core_test_provider PUBLIC c_std_11)
target_compile_definitions(gffx_core_test_provider PRIVATE
    GFFX_BUILDING_LIBRARY=1
    GFFX_ENABLE_TEST_PROVIDER=1
    GFFX_PACKAGE_VERSION="${GFFX_PACKAGE_VERSION_VALUE}"
)
target_include_directories(gffx_core_test_provider
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/native/core ${PROJECT_SOURCE_DIR}/native/cuda
)
target_link_libraries(gffx_core_test_provider PRIVATE ${CMAKE_DL_LIBS})
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

# Phase 1 Step 5: private calling-path self-test.
#
# The self-test entry point is compiled ONLY into this private build of the core, never into the
# shipped gffx_core target, so the public ABI v1 export set is unchanged.
add_library(gffx_core_self_test SHARED
    ${PROJECT_SOURCE_DIR}/native/core/abi.c
    ${PROJECT_SOURCE_DIR}/native/core/capabilities.c
    ${PROJECT_SOURCE_DIR}/native/core/cuda_loader.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_edge_topology.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_face_geometry.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_gather_faces.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_sample_surface.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_validate.c
    ${PROJECT_SOURCE_DIR}/native/core/mesh_vertex_normals.c
    ${PROJECT_SOURCE_DIR}/native/core/points_proximity.c
    ${PROJECT_SOURCE_DIR}/native/core/render_rasterize.c
    ${PROJECT_SOURCE_DIR}/native/core/status.c
    ${PROJECT_SOURCE_DIR}/native/core/transforms.c
    ${PROJECT_SOURCE_DIR}/native/core/self_test.c
)
target_compile_features(gffx_core_self_test PUBLIC c_std_11)
target_compile_definitions(gffx_core_self_test PRIVATE
    GFFX_BUILDING_LIBRARY=1
    GFFX_ENABLE_SELF_TEST=1
    GFFX_PACKAGE_VERSION="${GFFX_PACKAGE_VERSION_VALUE}"
)
target_include_directories(gffx_core_self_test
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/native/core ${PROJECT_SOURCE_DIR}/native/cuda
)
target_link_libraries(gffx_core_self_test PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(gffx_core_self_test PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

add_executable(gffx_test_self_test_calling_path test_self_test_calling_path.c)
target_include_directories(gffx_test_self_test_calling_path PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/native/core
)
target_link_libraries(gffx_test_self_test_calling_path PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME runtime.self_test_calling_path
    COMMAND gffx_test_self_test_calling_path
        $<TARGET_FILE:gffx_core_self_test>
        $<TARGET_FILE:gffx_core>
)

# Phase 1 Step 5: dependency and ownership inspection over the CPU runtime scaffold.
add_test(NAME runtime.dependency_inspection
    COMMAND ${CMAKE_COMMAND}
        -DGFFX_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/dependency_inspection.cmake
)
