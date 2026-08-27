# Phase 2 Step 3 interchange-format acceptance fixtures. Unlike the geometry suites this test
# generates its own fixture bytes in-process, so the repository carries no binary PLY assets.

add_executable(gffx_test_ply test_ply.c)
target_link_libraries(gffx_test_ply PRIVATE gffx_core)
if(UNIX)
    target_link_libraries(gffx_test_ply PRIVATE m)
endif()
add_test(NAME io.ply COMMAND gffx_test_ply)
