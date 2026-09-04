# Phase 2 Step 4 cross-cutting verification sweep. Drives every operation through one table so
# each property is applied by identical code rather than by per-operation reasoning.

add_executable(gffx_test_crosscutting test_crosscutting.c)
target_link_libraries(gffx_test_crosscutting PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_crosscutting PRIVATE m)
endif()
add_test(NAME sweep.crosscutting COMMAND gffx_test_crosscutting)

if(WIN32)
    add_custom_command(TARGET gffx_test_crosscutting POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gffx_core>
            $<TARGET_FILE_DIR:gffx_test_crosscutting>
        VERBATIM
    )
endif()
