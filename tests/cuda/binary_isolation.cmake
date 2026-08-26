cmake_minimum_required(VERSION 3.25)

foreach(required IN ITEMS GFFX_BINARY_INSPECTOR GFFX_BINARY_MODE GFFX_CORE_BINARY)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} must be defined")
    endif()
endforeach()

if(GFFX_BINARY_MODE STREQUAL "windows")
    execute_process(
        COMMAND "${GFFX_BINARY_INSPECTOR}" /DEPENDENTS "${GFFX_CORE_BINARY}"
        RESULT_VARIABLE core_result
        OUTPUT_VARIABLE core_dependencies
        ERROR_VARIABLE core_error
    )
elseif(GFFX_BINARY_MODE STREQUAL "elf")
    execute_process(
        COMMAND "${GFFX_BINARY_INSPECTOR}" -d "${GFFX_CORE_BINARY}"
        RESULT_VARIABLE core_result
        OUTPUT_VARIABLE core_dependencies
        ERROR_VARIABLE core_error
    )
else()
    message(FATAL_ERROR "unsupported binary inspection mode: ${GFFX_BINARY_MODE}")
endif()
if(NOT core_result EQUAL 0)
    message(FATAL_ERROR "core dependency inspection failed: ${core_error}")
endif()
string(TOLOWER "${core_dependencies}" core_dependencies_lower)
if(core_dependencies_lower MATCHES "nvcuda|libcuda|cudart|cublas|cudnn|nccl|torch_cuda")
    message(FATAL_ERROR "CPU core acquired a CUDA/framework dependency:\n${core_dependencies}")
endif()

if(DEFINED GFFX_PLUGIN_BINARY AND NOT GFFX_PLUGIN_BINARY STREQUAL "")
    if(GFFX_BINARY_MODE STREQUAL "windows")
        execute_process(
            COMMAND "${GFFX_BINARY_INSPECTOR}" /DEPENDENTS "${GFFX_PLUGIN_BINARY}"
            RESULT_VARIABLE plugin_result
            OUTPUT_VARIABLE plugin_dependencies
            ERROR_VARIABLE plugin_error
        )
        execute_process(
            COMMAND "${GFFX_BINARY_INSPECTOR}" /EXPORTS "${GFFX_PLUGIN_BINARY}"
            RESULT_VARIABLE export_result
            OUTPUT_VARIABLE plugin_exports
            ERROR_VARIABLE export_error
        )
        set(required_driver "nvcuda[.]dll")
    else()
        execute_process(
            COMMAND "${GFFX_BINARY_INSPECTOR}" -d "${GFFX_PLUGIN_BINARY}"
            RESULT_VARIABLE plugin_result
            OUTPUT_VARIABLE plugin_dependencies
            ERROR_VARIABLE plugin_error
        )
        execute_process(
            COMMAND "${GFFX_BINARY_INSPECTOR}" -Ws "${GFFX_PLUGIN_BINARY}"
            RESULT_VARIABLE export_result
            OUTPUT_VARIABLE plugin_exports
            ERROR_VARIABLE export_error
        )
        set(required_driver "libcuda[.]so")
    endif()
    if(NOT plugin_result EQUAL 0 OR NOT export_result EQUAL 0)
        message(FATAL_ERROR "plugin binary inspection failed: ${plugin_error} ${export_error}")
    endif()
    string(TOLOWER "${plugin_dependencies}" plugin_dependencies_lower)
    if(NOT plugin_dependencies_lower MATCHES "${required_driver}")
        message(FATAL_ERROR "CUDA plugin does not depend on the driver library")
    endif()
    if(plugin_dependencies_lower MATCHES "cudart|cublas|cudnn|nccl|torch_cuda|torch_cpu")
        message(FATAL_ERROR "CUDA plugin acquired a forbidden dependency:\n${plugin_dependencies}")
    endif()
    if(NOT plugin_exports MATCHES "gffx_cuda_plugin_handshake_v1")
        message(FATAL_ERROR "CUDA plugin does not export the versioned handshake")
    endif()
    if(GFFX_BINARY_MODE STREQUAL "windows" AND
       NOT plugin_exports MATCHES "1 number of names")
        message(FATAL_ERROR "CUDA plugin exports more than the one private handshake")
    endif()
endif()

message(STATUS "CUDA binary isolation inspection passed")
