# Building the optional CUDA provider

The provider implements GPU operations and first derivatives. It is compiled
offline to PTX, embedded in `gffx_cuda12.dll`, then loaded beside the CPU core
only when explicitly requested. It links the NVIDIA Driver API. Installed users
need a suitable NVIDIA driver and GPU, plus CUDA-enabled PyTorch for that adapter;
they do not install a toolkit or compiler to use a wheel.

## Experimental 0.2.0a2 Windows build

The exact candidate uses CUDA Toolkit/nvcc 12.6.20, PTX ISA 8.5, `compute_75`
(`sm_75` PTX), Visual Studio 2022/MSVC 14.44 and a CPU-only Torch 2.10.0 builder.
Building the adapter against CPU-only Torch avoids a mandatory torch_cuda.dll
dependency; CUDA stream access is resolved from the user's loaded framework.

```powershell
py -3.10 -m venv C:/path/to/cpu-builder
C:/path/to/cpu-builder/Scripts/python.exe -m pip install --upgrade pip
C:/path/to/cpu-builder/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
C:/path/to/cpu-builder/Scripts/python.exe -m pip install scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0 delvewheel==1.13.2
python tools/gpu_wheel.py build --python C:/path/to/cpu-builder/Scripts/python.exe --source . --work-dir C:/path/to/new-output --toolkit "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
```

The guard checks the exact compiler and CPU-only Torch builder, uses one build
worker, repairs runtime DLLs while excluding framework/NVIDIA DLLs, then audits
PE dependencies, provider ABI export, embedded PTX, metadata and RECORD hashes.
The tool never publishes and refuses an existing output directory.

The declared Windows driver baseline is 560.76 from NVIDIA's CUDA 12.6 GA table;
the GPU must also be supported by that driver. Only the named actual driver/GPU
profile receives test evidence. CUDA 12.x minor compatibility alone does not
establish support for a newer PTX ISA. See the [exact alpha profile](ALPHA_0_2_0A2.md).

## Broader source and supported-release targets

Generic CMake source builds accept a toolkit from 12.0 upward and target virtual
architecture `compute_75` by default. A later toolkit changes emitted PTX and its
driver requirement. The broader fully supported release target builds at 12.0
to keep that floor low; the explicit 12.6 experimental alpha does not claim that
older driver range or completion of the full platform/framework/hardware matrix.
No architecture-specific cubins or user-side CUDA Runtime dependency are added.

Linux source builds retain their own CMake/manylinux and device qualification
requirements. This alpha publishes one Windows x64 artifact only.

## Discovery and validation

The loader normally searches beside `gffx_core.dll` for `gffx_cuda12.dll`.
`GFFX_CUDA_PLUGIN_PATH` is an explicit diagnostic/test override, never required
for the packaged alpha. The loader does not search the current directory or PATH.
Base import and static CPU capabilities never probe the driver or provider.

The release checks use the installed wheel, real CUDA tensors, matched CPU/GPU
forward values and VJPs, independent local numerical gradients and caller-stream
tests. See [fixture methods and limits](../tests/packaging/GPU_ALPHA.md).
No skipped GPU case counts as acceptance. The filtered opaque-renderer reference
campaign remains separately unqualified and paused indefinitely.
