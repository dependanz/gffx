# Phase 2 Step 3 isolation inspection for the setup-class file layer.
#
# native/io is deliberately outside the allocation-free geometry scaffold that
# dependency_inspection.cmake governs: reading a file needs the platform C library, exactly as the
# CUDA probe loader does. Excluding it from that gate must not mean leaving it unpoliced, so this
# inspection permits stdio and the allocator while still forbidding ambient environment access,
# process control, concurrency, process-wide mutable state, and any framework dependency. Run with:
#
#   cmake -DGFFX_SOURCE_DIR=<repo> -P io_isolation.cmake
#
# Scanning uses narrow file(STRINGS ... REGEX) filters rather than a full line walk, for the reason
# documented at the top of dependency_inspection.cmake.

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED GFFX_SOURCE_DIR)
    message(FATAL_ERROR "GFFX_SOURCE_DIR must be defined")
endif()

set(allowed_includes
    "<stdint.h>"
    "<stddef.h>"
    "<string.h>"
    "<limits.h>"
    "<math.h>"
    # The two facilities this layer exists to use, and the reason it is inspected separately.
    "<stdio.h>"
    "<stdlib.h>"
)

# Permitted here: fopen, fread, fclose, malloc, free. Still forbidden: anything that reaches
# outside the call, blocks, or persists between calls.
set(forbidden_identifiers
    "getenv" "putenv" "system" "popen" "execv" "fork"
    "pthread" "_Atomic" "atomic_" "CreateThread" "thrd_create" "mtx_"
    "abort" "assert" "longjmp" "setjmp"
    "srand" "rand" "clock" "localtime"
    "remove" "rename" "tmpnam" "freopen"
    "printf" "fprintf" "sprintf" "puts" "fputs" "perror"
    "torch" "numpy" "Py_" "PyObject"
)

set(scan_globs
    "${GFFX_SOURCE_DIR}/native/io/*.c"
    "${GFFX_SOURCE_DIR}/native/io/*.h"
)

file(GLOB scan_files ${scan_globs})
list(SORT scan_files)
if(scan_files STREQUAL "")
    message(FATAL_ERROR "io isolation inspection found no source files to scan")
endif()

set(violations "")
set(scanned_count 0)

foreach(source_file IN LISTS scan_files)
    math(EXPR scanned_count "${scanned_count} + 1")
    get_filename_component(short_name "${source_file}" NAME)

    file(STRINGS "${source_file}" include_lines REGEX "^[ \t]*#[ \t]*include")
    foreach(line IN LISTS include_lines)
        if(line MATCHES "^[ \t]*#[ \t]*include[ \t]+(.+)$")
            string(STRIP "${CMAKE_MATCH_1}" include_target)
            set(include_ok FALSE)
            if(include_target MATCHES "^<gffx/[A-Za-z0-9_]+[.]h>$")
                set(include_ok TRUE)
            endif()
            foreach(allowed IN LISTS allowed_includes)
                if(include_target STREQUAL allowed)
                    set(include_ok TRUE)
                endif()
            endforeach()
            if(NOT include_ok)
                list(APPEND violations "${short_name}: disallowed include ${include_target}")
            endif()
        endif()
    endforeach()

    foreach(identifier IN LISTS forbidden_identifiers)
        file(STRINGS "${source_file}" hits REGEX "(^|[^A-Za-z0-9_])${identifier}")
        foreach(line IN LISTS hits)
            if(NOT line MATCHES "^[ \t]*(//|/[*]|[*])")
                list(APPEND violations
                    "${short_name}: forbidden facility '${identifier}' in:${line}")
            endif()
        endforeach()
    endforeach()

    file(STRINGS "${source_file}" static_lines REGEX "^[ \t]*static[ \t]")
    foreach(line IN LISTS static_lines)
        if(NOT line MATCHES "(^|[^A-Za-z0-9_])const([^A-Za-z0-9_]|$)")
            if(line MATCHES "=" AND NOT line MATCHES "[(]")
                list(APPEND violations "${short_name}: file-scope mutable static state:${line}")
            endif()
        endif()
    endforeach()
endforeach()

if(NOT violations STREQUAL "")
    list(REMOVE_DUPLICATES violations)
    string(REPLACE ";" "\n  " violation_text "${violations}")
    message(FATAL_ERROR
        "io isolation inspection failed across ${scanned_count} files:\n  ${violation_text}")
endif()

message(STATUS "io isolation inspection passed across ${scanned_count} files")
