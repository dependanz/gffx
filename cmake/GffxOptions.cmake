include_guard(GLOBAL)

option(GFFX_BUILD_PYTHON "Build the CPython Limited-API loader" OFF)
option(GFFX_BUILD_PYTORCH "Build the LibTorch Stable-ABI adapter" OFF)
option(GFFX_ENABLE_CUDA "Build the optional CUDA plugin" OFF)
option(
    GFFX_CUDA_RUN_DEVICE_TESTS
    "Run explicit CUDA-driver/device probes (requires a trusted GPU host)"
    OFF
)

# CPU-only, framework-free configuration remains the default. The owning implementation steps
# include adapter/plugin build modules only when their explicit options are enabled.
