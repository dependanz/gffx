# Phase 2 geometry-operation acceptance fixtures. Each suite is dependency-free C linked
# against the shipped core, mirroring the ABI test conventions.

add_executable(gffx_test_face_geometry test_face_geometry.c)
target_link_libraries(gffx_test_face_geometry PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_face_geometry PRIVATE m)
endif()
add_test(NAME geometry.face_geometry COMMAND gffx_test_face_geometry)

if(WIN32)
    add_custom_command(TARGET gffx_test_face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core>
            $<TARGET_FILE_DIR:gffx_test_face_geometry>
        VERBATIM
    )
endif()
