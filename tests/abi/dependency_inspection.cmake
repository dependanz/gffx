# Phase 1 Step 5 dependency and ownership inspection.
#
# Enforces, as a reproducible gate rather than a one-time reading, that the CPU runtime scaffold
# depends only on the required platform C runtime and math facilities, and that it owns no hidden
# process-wide state. Run with:
#
#   cmake -DGFFX_SOURCE_DIR=<repo> -P dependency_inspection.cmake

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
    "\"internal.h\""
    "\"self_test.h\""
)

# Facilities the scaffold must not reach for: allocation, process-wide I/O, abort paths,
# concurrency, ambient environment, and nondeterminism.
set(forbidden_identifiers
    "malloc" "calloc" "realloc" "aligned_alloc"
    "printf" "fprintf" "sprintf" "puts" "fputs" "perror"
    "abort" "assert" "longjmp" "setjmp"
    "pthread" "_Atomic" "atomic_" "CreateThread" "thrd_create" "mtx_"
    "getenv" "putenv" "system"
    "fopen" "freopen" "remove" "rename"
    "srand" "clock" "localtime"
)

set(scan_globs
    "${GFFX_SOURCE_DIR}/native/core/*.c"
    "${GFFX_SOURCE_DIR}/native/core/*.h"
    "${GFFX_SOURCE_DIR}/include/gffx/*.h"
)

file(GLOB scan_files ${scan_globs})
list(SORT scan_files)
if(scan_files STREQUAL "")
    message(FATAL_ERROR "dependency inspection found no source files to scan")
endif()

set(violations "")
set(scanned_count 0)

foreach(source_file IN LISTS scan_files)
    math(EXPR scanned_count "${scanned_count} + 1")
    file(STRINGS "${source_file}" lines)
    get_filename_component(short_name "${source_file}" NAME)
    set(line_number 0)
    foreach(line IN LISTS lines)
        math(EXPR line_number "${line_number} + 1")

        # Strip whole-line comments so prose cannot trip the identifier scan.
        string(REGEX REPLACE "^[ \t]*(//|/[*]|[*]).*$" "" code_line "${line}")

        if(code_line MATCHES "^[ \t]*#[ \t]*include[ \t]+(.+)$")
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
                list(APPEND violations
                    "${short_name}:${line_number}: disallowed include ${include_target}")
            endif()
        endif()

        foreach(identifier IN LISTS forbidden_identifiers)
            if(code_line MATCHES "(^|[^A-Za-z0-9_])${identifier}")
                list(APPEND violations
                    "${short_name}:${line_number}: forbidden facility '${identifier}'")
            endif()
        endforeach()

        # 'free' and 'exit' need a call-shaped match so words like 'dependency-free' pass.
        if(code_line MATCHES "(^|[^A-Za-z0-9_])(free|exit)[ \t]*[(]")
            list(APPEND violations
                "${short_name}:${line_number}: forbidden facility '${CMAKE_MATCH_2}('")
        endif()

        # File-scope mutable static state. Function definitions and const tables are permitted.
        if(code_line MATCHES "^[ \t]*static[ \t]")
            if(NOT code_line MATCHES "(^|[^A-Za-z0-9_])const([^A-Za-z0-9_]|$)")
                if(code_line MATCHES "=" AND NOT code_line MATCHES "[(]")
                    list(APPEND violations
                        "${short_name}:${line_number}: file-scope mutable static state")
                endif()
            endif()
        endif()
    endforeach()
endforeach()

if(NOT violations STREQUAL "")
    list(REMOVE_DUPLICATES violations)
    string(REPLACE ";" "\n  " violation_text "${violations}")
    message(FATAL_ERROR
        "dependency/ownership inspection failed across ${scanned_count} files:\n  ${violation_text}")
endif()

message(STATUS "dependency/ownership inspection passed across ${scanned_count} files")
