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

if(WIN32)
    foreach(test_target IN ITEMS
        gffx_test_face_geometry
        gffx_test_vertex_normals
        gffx_test_gather_faces
        gffx_test_transforms
    )
        add_custom_command(TARGET ${test_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:gffx_core>
                $<TARGET_FILE_DIR:${test_target}>
            VERBATIM
        )
    endforeach()
endif()
