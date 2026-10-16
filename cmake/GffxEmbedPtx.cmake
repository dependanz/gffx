# Embeds a compiled PTX module into the plugin as generated C source.
#
# The plugin carries its kernels inside its own binary rather than loading a .ptx file beside it at
# runtime. A file would be a second artifact to install, to find, to version against the plugin, and
# to fail on when absent or stale; embedding removes all four failure modes and keeps the plugin a
# single self-describing object. It also means the kernels cannot drift from the loader that
# launches them, because they ship as one build product.
#
# Fixed2048-byte rows use escaped C string literals, avoiding millions of
# integer initializer tokens while staying below C11 minimum literal limits.
# Full rows omit implicit terminators; an explicit final NUL ends the payload.
#
# Run in script mode:
#   cmake -DGFFX_PTX_INPUT=<file> -DGFFX_PTX_OUTPUT=<file> -DGFFX_PTX_SYMBOL=<name>
#         -P GffxEmbedPtx.cmake

cmake_minimum_required(VERSION 3.25)

foreach(required GFFX_PTX_INPUT GFFX_PTX_OUTPUT GFFX_PTX_SYMBOL)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} must be defined")
    endif()
endforeach()

if(NOT EXISTS "${GFFX_PTX_INPUT}")
    message(FATAL_ERROR "PTX input not found: ${GFFX_PTX_INPUT}")
endif()

file(READ "${GFFX_PTX_INPUT}" ptx_hex HEX)
string(LENGTH "${ptx_hex}" hex_length)
math(EXPR byte_count "${hex_length} / 2")
if(byte_count EQUAL 0)
    message(FATAL_ERROR "PTX input is empty: ${GFFX_PTX_INPUT}")
endif()

# Extract the .version and .target directives so the generated file records what it carries. These
# are the two properties that decide which drivers can parse the module and which GPUs it reaches,
# so they belong in the artifact rather than only in a build log.
file(STRINGS "${GFFX_PTX_INPUT}" ptx_version REGEX "^\\.version")
file(STRINGS "${GFFX_PTX_INPUT}" ptx_target REGEX "^\\.target")
if(ptx_version STREQUAL "")
    set(ptx_version "unknown")
endif()
if(ptx_target STREQUAL "")
    set(ptx_target "unknown")
endif()

# C11 arrays are contiguous. Explicit byte escapes preserve all input bytes;
# bounded rows avoid implementation limits on a single concatenated literal.
file(WRITE "${GFFX_PTX_OUTPUT}"
"/* Generated PTX ${ptx_version}, ${ptx_target}; exact payload, do not edit. */\n"
"static const unsigned char ${GFFX_PTX_SYMBOL}[][2048] = {\n")
string(APPEND ptx_hex "00")
string(LENGTH "${ptx_hex}" terminated_length)
set(offset 0)
while(offset LESS terminated_length)
    math(EXPR remaining "${terminated_length} - ${offset}")
    if(remaining GREATER 4096)
        set(length 4096)
    else()
        set(length ${remaining})
    endif()
    string(SUBSTRING "${ptx_hex}" ${offset} ${length} chunk)
    string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "\\\\x\\1" chunk "${chunk}")
    file(APPEND "${GFFX_PTX_OUTPUT}" "\"${chunk}\",\n")
    math(EXPR offset "${offset} + ${length}")
endwhile()
file(APPEND "${GFFX_PTX_OUTPUT}" "};\nstatic const unsigned long ${GFFX_PTX_SYMBOL}_size = ${byte_count}UL;\n")
message(STATUS "embedded ${byte_count} PTX bytes in bounded C literal rows (${ptx_version}, ${ptx_target})")
