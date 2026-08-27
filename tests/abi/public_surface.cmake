# Phase 2 Step 5 public-surface closure gate (MB-01..MB-05).
#
# Enforces that the committed public C surface and its classification stay in agreement with the
# headers, in both directions. A symbol must not be able to join the compatibility surface by
# accident, and a classified symbol must not silently disappear from it. Run with:
#
#   cmake -DGFFX_SOURCE_DIR=<repo> -P public_surface.cmake
#
# SCANNING NOTE, which applies to every gate in this directory.
#
# All inspection here uses file(STRINGS ... REGEX <narrow pattern>) and never iterates a file's
# full line list. Reading a whole file into a CMake list is not reliable: for include/gffx/mesh.h,
# a 354-line file, plain file(STRINGS) yields 21 elements and file(STRINGS ... REGEX ".") yields
# 19, with the remainder collapsed into a single element. file(READ) followed by an explicit
# newline split, with or without semicolon masking, collapses identically, so this is a property
# of CMake list handling rather than of file(STRINGS). native/core/mesh_sample_surface.c collapses
# the same way, at 76 of 702 lines.
#
# Why that matters, and it is subtle: an unanchored regex still matches inside a collapsed blob, so
# identifier checks appear to keep working, while every anchored check silently stops applying past
# the collapse point. A gate can report success having inspected a tenth of a file. Narrow filters
# return small exact result sets, and every pattern used here was verified against grep ground
# truth before being relied on.

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED GFFX_SOURCE_DIR)
    message(FATAL_ERROR "GFFX_SOURCE_DIR must be defined")
endif()

set(manifest_file "${GFFX_SOURCE_DIR}/tests/abi/public_surface_manifest.txt")
if(NOT EXISTS "${manifest_file}")
    message(FATAL_ERROR "public surface manifest not found: ${manifest_file}")
endif()

set(violations "")

# ---------------------------------------------------------------- read the classified manifest
file(STRINGS "${manifest_file}" manifest_entries REGEX "^(infra|op|setup|private)[ \t]")
set(manifest_symbols "")
set(manifest_ops "")
set(manifest_private "")
foreach(entry IN LISTS manifest_entries)
    if(NOT entry MATCHES "^(infra|op|setup|private)[ \t]+(gffx_[A-Za-z0-9_]+)[ \t]*$")
        list(APPEND violations
            "manifest line is not '<infra|op|setup|private> <gffx_symbol>': ${entry}")
        continue()
    endif()
    set(entry_class "${CMAKE_MATCH_1}")
    set(entry_symbol "${CMAKE_MATCH_2}")
    if(entry_symbol IN_LIST manifest_symbols)
        list(APPEND violations "manifest lists ${entry_symbol} more than once")
    endif()
    list(APPEND manifest_symbols "${entry_symbol}")
    if(entry_class STREQUAL "op")
        list(APPEND manifest_ops "${entry_symbol}")
    elseif(entry_class STREQUAL "private")
        list(APPEND manifest_private "${entry_symbol}")
    endif()
endforeach()

# ---------------------------------------------- collect GFFX_API declarations from public headers
file(GLOB public_headers "${GFFX_SOURCE_DIR}/include/gffx/*.h")
list(SORT public_headers)
if(public_headers STREQUAL "")
    message(FATAL_ERROR "public surface inspection found no headers to scan")
endif()

set(declared_symbols "")
foreach(header IN LISTS public_headers)
    get_filename_component(short_name "${header}" NAME)
    file(STRINGS "${header}" api_lines REGEX "^GFFX_API")
    foreach(line IN LISTS api_lines)
        if(line MATCHES "^GFFX_API[ \t]+.*GFFX_CALL[ \t]+(gffx_[A-Za-z0-9_]+)[ \t]*\\(")
            list(APPEND declared_symbols "${CMAKE_MATCH_1}")
        else()
            list(APPEND violations
                "${short_name}: GFFX_API line is not a recognised declaration: ${line}")
        endif()
    endforeach()
endforeach()

# ------------------------------------------------------------- MB-01 / MB-02 closure both ways
foreach(symbol IN LISTS declared_symbols)
    if(NOT symbol IN_LIST manifest_symbols)
        list(APPEND violations
            "${symbol} is declared GFFX_API in a public header but is not classified in the "
            "manifest. Classify it here and in MODULE_BOUNDARY_V0_1.md section 3, or keep it "
            "internal by moving it out of include/gffx and dropping GFFX_API.")
    endif()
endforeach()

foreach(symbol IN LISTS manifest_symbols)
    if(symbol IN_LIST manifest_private)
        continue()   # private symbols live in internal headers and are checked below
    endif()
    if(NOT symbol IN_LIST declared_symbols)
        list(APPEND violations
            "${symbol} is classified in the manifest but is no longer declared GFFX_API in any "
            "public header. Removing a published symbol is a breaking change.")
    endif()
endforeach()

# ------------------------------------------------------ MB-04 operation surface self-consistency
foreach(symbol IN LISTS manifest_ops)
    if(symbol MATCHES "_backward$")
        string(REGEX REPLACE "_backward$" "" base "${symbol}")
        if(NOT "${base}" IN_LIST manifest_ops)
            list(APPEND violations "${symbol} has no forward entry point ${base}")
        endif()
        if(NOT "${base}_workspace" IN_LIST manifest_ops)
            list(APPEND violations
                "${symbol} exists but ${base}_workspace does not; every operation publishes a "
                "workspace query")
        endif()
    endif()
endforeach()

# ---------------------------------------------------------- MB-03 internal headers stay internal
#
# An internal header may declare GFFX_API only for a symbol classified 'private': one that must sit
# in the export table for a mechanical reason but carries no compatibility promise. The self-test
# entry point is the only such symbol, exported because tests/abi resolves it by name at runtime.
# Classifying it is the point of this check. An exported symbol that no promise covers should be
# visible in the record rather than ambient.
file(GLOB internal_headers "${GFFX_SOURCE_DIR}/native/core/*.h" "${GFFX_SOURCE_DIR}/native/io/*.h")
list(SORT internal_headers)
foreach(header IN LISTS internal_headers)
    get_filename_component(short_name "${header}" NAME)
    file(STRINGS "${header}" api_lines REGEX "^GFFX_API")
    foreach(line IN LISTS api_lines)
        if(line MATCHES "^GFFX_API[ \t]+.*GFFX_CALL[ \t]+(gffx_[A-Za-z0-9_]+)[ \t]*\\(")
            set(internal_symbol "${CMAKE_MATCH_1}")
            if(NOT internal_symbol IN_LIST manifest_private)
                list(APPEND violations
                    "${short_name}: ${internal_symbol} is declared GFFX_API in an internal header "
                    "without being classified 'private' in the manifest")
            endif()
        else()
            list(APPEND violations
                "${short_name}: GFFX_API line is not a recognised declaration: ${line}")
        endif()
    endforeach()
endforeach()

foreach(symbol IN LISTS manifest_private)
    if(symbol IN_LIST declared_symbols)
        list(APPEND violations
            "${symbol} is classified private but is declared in a public header; a private symbol "
            "must not appear in include/gffx")
    endif()
endforeach()

# ------------------------- MB-05 build_edge_topology is the only operation taking no vertex input
#
# A targeted MATCH over the file content rather than a line walk, for the reason in the scanning
# note. The argument list contains no ')' before its close, so [^)]* spans exactly that list.
file(READ "${GFFX_SOURCE_DIR}/include/gffx/mesh.h" mesh_content)
string(REGEX MATCH "gffx_mesh_build_edge_topology\\([^)]*\\)" edge_signature "${mesh_content}")
if(edge_signature STREQUAL "")
    list(APPEND violations
        "could not locate the gffx_mesh_build_edge_topology signature to check MB-05")
elseif(edge_signature MATCHES "vertices")
    list(APPEND violations
        "gffx_mesh_build_edge_topology now takes a vertex input, which invalidates the persistence "
        "table in MODULE_BOUNDARY_V0_1.md section 2.1: its outputs may no longer be treated as "
        "topology-derived state that a streaming host can hold across frames")
endif()

if(NOT violations STREQUAL "")
    list(REMOVE_DUPLICATES violations)
    string(REPLACE ";" "\n  " violation_text "${violations}")
    message(FATAL_ERROR "public surface inspection failed:\n  ${violation_text}")
endif()

list(LENGTH manifest_symbols manifest_count)
list(LENGTH declared_symbols declared_count)
message(STATUS
    "public surface inspection passed: ${declared_count} public declarations, "
    "${manifest_count} classified symbols")
