# Building GFFX

## Build layers

GFFX keeps its build graph explicit:

| Layer | Default | Toolchain | Output |
|---|---:|---|---|
| C11 core | on | CMake 4.4.2 and a 64-bit C11 compiler | `gffx_core` |
| CPython bridge | wheel on / direct CMake off | CPython `Development.SABIModule` | `gffx._core` |
| PyTorch adapter | off | C++17, CPython Limited API, PyTorch Stable ABI 2.10 | `gffx._torch` |
| CUDA provider | off | CUDA Toolkit 12.x Driver API | `gffx_cuda12` |
| Tests | direct CMake on / wheel off | CTest, C++ compiler; pytest separately | native fixtures |

The PEP 517 backend is `scikit-build-core==1.0.3`. It pins CMake `4.4.2` and Ninja `1.13.0` as
build-only tools. Installed wheels do not depend on those packages.

## CMake options

| Option | Default | Meaning |
|---|---:|---|
| `GFFX_BUILD_PYTHON` | `OFF` in direct CMake | Build the CPython 3.10 Limited-API bridge |
| `GFFX_BUILD_PYTORCH` | `OFF` | Build the private PyTorch loading scaffold |
| `GFFX_ENABLE_CUDA` | `OFF` | Build the isolated CUDA diagnostic provider |
| `GFFX_CUDA_RUN_DEVICE_TESTS` | `OFF` | Permit explicit trusted-host driver/device probes |
| `BUILD_TESTING` | CTest default | Build native ABI and isolation tests |

The wheel configuration overrides `GFFX_BUILD_PYTHON=ON` and `BUILD_TESTING=OFF`; it leaves both
optional providers off.

## Direct CPU build

Windows x86-64 with Visual Studio 2022:

```powershell
cmake -S . -B build/cpu-win-x64 -A x64 -DGFFX_BUILD_PYTHON=OFF
cmake --build build/cpu-win-x64 --config Release --parallel
ctest --test-dir build/cpu-win-x64 -C Release --output-on-failure
```

Linux x86-64 or ARM64:

```bash
cmake -S . -B build/cpu-linux -DCMAKE_BUILD_TYPE=Release -DGFFX_BUILD_PYTHON=OFF
cmake --build build/cpu-linux --parallel
ctest --test-dir build/cpu-linux --output-on-failure
```

## Python tests and development groups

The top-level PEP 735 groups keep source-development packages out of runtime metadata. With a
frontend that supports dependency groups:

```powershell
python -m pip install --group test
python -m pytest -q tests/python tests/pytorch tests/packaging
```

`development` includes `test` and `packaging`. Empty groups are deliberate placeholders; they do
not install unfinished framework, accelerator, visualization, format, example, or benchmark
features.

## Optional builds

PyTorch-ready builds require an explicitly selected PyTorch environment and
`GFFX_BUILD_PYTORCH=ON`. CUDA provider builds require `GFFX_ENABLE_CUDA=ON`; device execution tests
also require `GFFX_CUDA_RUN_DEVICE_TESTS=ON` on trusted hardware. See
[INSTALLATION.md](INSTALLATION.md) and [CUDA_PLUGIN_BUILD.md](CUDA_PLUGIN_BUILD.md).

## Reproducible Windows CPU alpha wheel

Use a fresh Windows x64 environment with CPU-only PyTorch 2.10.0. The alpha
profile includes the optional adapter while keeping the gffx CUDA provider off.
A GPU-enabled build environment can create an unintended torch_cuda.dll import
even with the gffx CUDA provider disabled; the guard below refuses that input.

```powershell
python -m venv C:/path/to/cpu-builder
C:/path/to/cpu-builder/Scripts/python.exe -m pip install --upgrade pip
C:/path/to/cpu-builder/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
C:/path/to/cpu-builder/Scripts/python.exe -m pip install scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0 delvewheel==1.13.2
python tools/cpu_wheel.py build --python C:/path/to/cpu-builder/Scripts/python.exe --work-dir C:/path/to/new-output --generator "Visual Studio 17 2022"
```

The tool uses one build worker and no dependency downloads during the build,
repairs the wheel without bundling PyTorch, and audits native imports, tags,
metadata and RECORD hashes. It never uploads. Output directories must be new.
The build report and repaired wheel live under the selected output directory.
The actual selected Python/framework/tool versions are recorded in the report.
The existing broad foundation/RC workflows retain their own matrix and claims.

## Package-foundation verification

After installing the `development` dependency group, run the eleven-gate verifier with a new work
directory and a separately prepared environment containing a PyTorch 2.10+ adapter-ready wheel:

```powershell
python -m pip install --group development
python tools/verify_foundation.py `
  --work-dir build/foundation-verification `
  --adapter-python C:/path/to/pytorch-2.10-plus-environment/python.exe
```

The verifier copies only Git-tracked and non-ignored intended source into a clean snapshot, scans
that inventory without printing matched secret values, runs the Python and native contracts,
builds and inspects one base wheel plus one sdist, and installs the wheel into a fresh environment
with compiler sentinels. It explicitly checks missing PyTorch, absent/corrupt CUDA providers, a
real opt-in PyTorch adapter load, uninstall cleanup, and deterministic missing compiler/PyTorch/
CUDA-toolkit source-configuration failures.

A run passes only when all eleven gates pass. `report.json` beneath the work directory contains the
machine-readable result and artifact hashes. The generated evidence is ignored and is not a public
support claim; cross-platform and full version-matrix execution remains the CI work in Step 12.

All build and test output belongs beneath ignored `build/`, `dist/`, or tool cache directories. No
generated binary, cache, environment, credential, or machine-specific path should be committed.
