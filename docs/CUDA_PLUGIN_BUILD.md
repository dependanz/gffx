# Isolated CUDA plugin build recipes

The base GFFX build is CPU-only. CUDA is enabled only by `GFFX_ENABLE_CUDA=ON`, produces the
separate `gffx_cuda12` shared library, and is discovered only by an explicit full capability
probe. `import gffx`, ordinary CPU configuration, and CPU wheels do not enable CUDA.

Release-candidate artifacts use CUDA Toolkit 12.8. CMake accepts a CUDA 12.x toolkit for local
scaffold development but embeds its exact version in the plugin build identity and warns unless it
is 12.8; CUDA 13+ plugin builds are rejected. The Step 9 host plugin is C11, links the CUDA Driver
API only, never enables the CUDA language, and must not acquire the CUDA Runtime, cuBLAS, cuDNN,
NCCL, framework, allocator, or
graphics-library dependencies.

## Windows x86-64 / Visual Studio 2022

```powershell
cmake -S . -B build/cuda-win-x64 -A x64 `
  -DGFFX_ENABLE_CUDA=ON `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" `
  -DBUILD_TESTING=ON
cmake --build build/cuda-win-x64 --config Release --parallel
ctest --test-dir build/cuda-win-x64 -C Release --output-on-failure
```

Use `-DGFFX_CUDA_RUN_DEVICE_TESTS=ON` only on a trusted GPU host. It makes the explicit test probe
load the plugin and driver and enumerate devices; it is unsuitable for public pull-request code.

## Linux manylinux 2.28 x86-64

```bash
cmake -S . -B build/cuda-linux-x64 \
  -DGFFX_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build/cuda-linux-x64 --parallel
ctest --test-dir build/cuda-linux-x64 --output-on-failure
```

The Linux release build must run inside the selected manylinux 2.28-compatible environment. GPU
execution evidence additionally requires the separately approved native Ubuntu 24.04/Tesla T4
environment; a successful compile alone is not a functional CUDA support claim.

Functional GPU kernels are intentionally absent in Step 9. Phase 3 will compile C+CUDA device
artifacts for the admitted architecture set and load them through the Driver API; those artifacts
must preserve the no-CUDA-Runtime dependency boundary. Until then, load/driver/device enumeration
is infrastructure evidence only.

## Explicit discovery override

Installed probes look only beside `gffx_core` for `gffx_cuda12.dll` on Windows or
`libgffx_cuda12.so` on Linux. Tests and diagnostics may set `GFFX_CUDA_PLUGIN_PATH` to one absolute
plugin path. The loader never searches the current working directory or the process `PATH`.
