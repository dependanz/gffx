include_guard(GLOBAL)

option(GFFX_BUILD_PYTHON "Build the CPython Limited-API loader" OFF)
option(GFFX_BUILD_PYTORCH "Build the LibTorch Stable-ABI adapter" OFF)
option(GFFX_ENABLE_CUDA "Build the optional CUDA plugin" OFF)

# CPU-only, framework-free configuration remains the default. The owning implementation steps
# will include GffxPython.cmake and GffxCuda.cmake when their targets exist.
