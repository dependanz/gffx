include_guard(GLOBAL)

# Build the private adapter as both a CPython 3.10 Limited-API module and a LibTorch Stable-ABI
# extension targeting the PyTorch 2.10 floor. PyTorch is a build input only for this opt-in target;
# the dependency-free base package never imports it.

if(NOT COMMAND Python_add_library)
    if(DEFINED Python_EXECUTABLE AND NOT DEFINED Python_FIND_VIRTUALENV)
        set(Python_FIND_VIRTUALENV STANDARD)
    endif()

    if(DEFINED SKBUILD_SABI_COMPONENT AND NOT SKBUILD_SABI_COMPONENT STREQUAL "")
        set(GFFX_PYTORCH_SABI_COMPONENT "${SKBUILD_SABI_COMPONENT}")
    else()
        set(GFFX_PYTORCH_SABI_COMPONENT "Development.SABIModule")
    endif()

    find_package(Python REQUIRED COMPONENTS
        Interpreter
        Development.Module
        ${GFFX_PYTORCH_SABI_COMPONENT}
    )
endif()

# TorchConfig.cmake ships inside the Python distribution. Respect an explicit Torch_DIR; otherwise
# ask the selected build interpreter for the canonical CMake prefix instead of guessing a path.
if(NOT Torch_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE GFFX_TORCH_PREFIX_RESULT
        OUTPUT_VARIABLE GFFX_TORCH_CMAKE_PREFIX
        ERROR_VARIABLE GFFX_TORCH_PREFIX_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GFFX_TORCH_PREFIX_RESULT EQUAL 0 OR GFFX_TORCH_CMAKE_PREFIX STREQUAL "")
        message(FATAL_ERROR
            "GFFX_BUILD_PYTORCH=ON requires PyTorch 2.10 or newer in the selected build "
            "interpreter (${Python_EXECUTABLE}). Import failed: ${GFFX_TORCH_PREFIX_ERROR}"
        )
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${GFFX_TORCH_CMAKE_PREFIX}")
endif()

find_package(Torch 2.10 REQUIRED CONFIG)

python_add_library(gffx_pytorch_adapter MODULE USE_SABI 3.10 WITH_SOABI
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/pytorch/register.cpp
)

set_target_properties(gffx_pytorch_adapter PROPERTIES
    OUTPUT_NAME "_torch"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

target_compile_definitions(gffx_pytorch_adapter PRIVATE
    Py_LIMITED_API=0x030A0000
    TORCH_TARGET_VERSION=0x020a000000000000
)

if(TORCH_CXX_FLAGS)
    separate_arguments(GFFX_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(gffx_pytorch_adapter PRIVATE ${GFFX_TORCH_CXX_FLAGS})
endif()

target_link_libraries(gffx_pytorch_adapter PRIVATE ${TORCH_LIBRARIES})

# The adapter is imported only after Python has imported torch, so the framework's own runtime
# libraries are already loaded and remain owned by the framework distribution.
install(TARGETS gffx_pytorch_adapter
    RUNTIME DESTINATION gffx
    LIBRARY DESTINATION gffx
)
