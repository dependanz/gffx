# Phase 1 Step 5 dependency and ownership inspection.
#
# Enforces, as a reproducible gate rather than a one-time reading, that the CPU runtime scaffold
# depends only on the required platform C runtime and math facilities, and that it owns no hidden
# process-wide state. Run with:
#
#   cmake -DGFFX_SOURCE_DIR=<repo> -P dependency_inspection.cmake
#
# SCANNING NOTE (2026-08-27). This gate previously iterated each file's full line list, which is
# not reliable in CMake: for native/core/mesh_sample_surface.c, a 702-line file, that yields 76
# elements, the remainder having collapsed into a single one. Unanchored identifier checks still
# matched inside the collapsed blob, so forbidden facilities kept being caught, but the two
# anchored checks here, disallowed includes and file-scope mutable statics, silently stopped
# applying past the collapse point. Every check now uses file(STRINGS ... REGEX <narrow pattern>),
# whose results were verified against grep ground truth. Line numbers are no longer reported,
# because a filtered result set does not carry them; the offending line text is reported instead,
# which locates the site at least as precisely.

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
    "\"cuda_loader.h\""
    # Phase 2: shared header-only validation helpers for the mesh operations.
    "\"mesh_common.h\""
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
# The explicit full-probe loader is setup/diagnostic infrastructure and necessarily uses the
# platform dynamic-loader and environment APIs. Step 9 gives it a separate isolation inspection;
# it is not part of the allocation-free geometry/runtime scaffold governed by this Step 5 gate.
list(FILTER scan_files EXCLUDE REGEX "/cuda_loader[.][ch]$")
if(scan_files STREQUAL "")
    message(FATAL_ERROR "dependency inspection found no source files to scan")
endif()

set(violations "")
set(scanned_count 0)

foreach(source_file IN LISTS scan_files)
    math(EXPR scanned_count "${scanned_count} + 1")
    get_filename_component(short_name "${source_file}" NAME)

    # --- includes ---------------------------------------------------------------------------
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

    # --- forbidden facilities, one filtered pass per identifier -----------------------------
    foreach(identifier IN LISTS forbidden_identifiers)
        file(STRINGS "${source_file}" hits REGEX "(^|[^A-Za-z0-9_])${identifier}")
        foreach(line IN LISTS hits)
            # Skip whole-line comments so prose naming a facility cannot trip the scan.
            if(NOT line MATCHES "^[ \t]*(//|/[*]|[*])")
                list(APPEND violations
                    "${short_name}: forbidden facility '${identifier}' in:${line}")
            endif()
        endforeach()
    endforeach()

    # 'free' and 'exit' need a call-shaped match so words like 'dependency-free' pass.
    file(STRINGS "${source_file}" call_hits REGEX "(^|[^A-Za-z0-9_])(free|exit)[ \t]*[(]")
    foreach(line IN LISTS call_hits)
        if(NOT line MATCHES "^[ \t]*(//|/[*]|[*])")
            list(APPEND violations
                "${short_name}: forbidden facility 'free(' or 'exit(' in:${line}")
        endif()
    endforeach()

    # --- file-scope mutable static state ----------------------------------------------------
    # Function definitions and const tables are permitted.
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
        "dependency/ownership inspection failed across ${scanned_count} files:\n  ${violation_text}")
endif()

message(STATUS "dependency/ownership inspection passed across ${scanned_count} files")
