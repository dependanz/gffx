# Phase 2 geometry-operation acceptance fixtures. Each suite is dependency-free C linked
# against the shipped core, mirroring the ABI test conventions.

add_executable(gffx_test_face_geometry test_face_geometry.c)
target_link_libraries(gffx_test_face_geometry PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_face_geometry PRIVATE m)
endif()
add_test(NAME geometry.face_geometry COMMAND gffx_test_face_geometry)

add_executable(gffx_test_vertex_normals test_vertex_normals.c)
target_link_libraries(gffx_test_vertex_normals PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_vertex_normals PRIVATE m)
endif()
add_test(NAME geometry.vertex_normals COMMAND gffx_test_vertex_normals)

add_executable(gffx_test_gather_faces test_gather_faces.c)
target_link_libraries(gffx_test_gather_faces PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_gather_faces PRIVATE m)
endif()
add_test(NAME geometry.gather_faces COMMAND gffx_test_gather_faces)

add_executable(gffx_test_transforms test_transforms.c)
target_link_libraries(gffx_test_transforms PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_transforms PRIVATE m)
endif()
add_test(NAME geometry.transforms COMMAND gffx_test_transforms)

add_executable(gffx_test_edge_topology test_edge_topology.c)
target_link_libraries(gffx_test_edge_topology PRIVATE gffx_core)
add_test(NAME geometry.edge_topology COMMAND gffx_test_edge_topology)

add_executable(gffx_test_proximity test_proximity.c)
target_link_libraries(gffx_test_proximity PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_proximity PRIVATE m)
endif()
add_test(NAME geometry.proximity COMMAND gffx_test_proximity)

add_executable(gffx_test_sample_surface test_sample_surface.c)
target_link_libraries(gffx_test_sample_surface PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_sample_surface PRIVATE m)
endif()
add_test(NAME geometry.sample_surface COMMAND gffx_test_sample_surface)

if(WIN32)
    foreach(test_target IN ITEMS
        gffx_test_face_geometry
        gffx_test_vertex_normals
        gffx_test_gather_faces
        gffx_test_transforms
        gffx_test_edge_topology
        gffx_test_proximity
        gffx_test_sample_surface
    )
        add_custom_command(TARGET ${test_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:gffx_core>
                $<TARGET_FILE_DIR:${test_target}>
            VERBATIM
        )
    endforeach()
endif()
