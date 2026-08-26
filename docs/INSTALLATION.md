# Installation

## Current distribution status

GFFX `0.2.0.dev0` is pre-alpha foundation software and has not been published to a package index.
PyPI currently serves the unsupported inherited prototype through version `0.1.4`; `pip install
gffx` therefore does not select the source described by this document.

Install the current foundation only from this checkout or an explicitly supplied internal wheel.
Internal artifacts prove a bounded build or loading path; they do not create a public support tier.

## Install an internal CPU wheel

An ordinary wheel needs Python 3.10 or newer and no compiler or mandatory third-party Python
package:

```powershell
python -m pip install C:/path/to/gffx-0.2.0.dev0-cp310-abi3-win_amd64.whl
```

Use the wheel whose platform tag matches the target. Do not rename or retag an artifact.

Verify the dependency-free entry point:

```powershell
python -c "import gffx; print(gffx.__version__); print(gffx.capabilities())"
```

`import gffx` loads neither the native core, an autodiff framework, nor a GPU library. The static
capability call loads the private native core but still never probes a GPU.

## Build and install the default CPU package from source

A source build requires a platform C11 compiler and network or local-cache access for the pinned
build-only tools. The resulting installation has zero mandatory third-party Python runtime
dependencies.

```powershell
python -m pip wheel . --no-deps --wheel-dir dist/local
python -m pip install --force-reinstall C:/path/to/generated-wheel.whl
```

The PEP 517 configuration explicitly sets `GFFX_BUILD_PYTHON=ON`, `GFFX_BUILD_PYTORCH=OFF`,
`GFFX_ENABLE_CUDA=OFF`, and `BUILD_TESTING=OFF`.

## Optional PyTorch loading scaffold

The current adapter is infrastructure only: it registers a private foundation probe and no public
operation. Build it only in a controlled environment containing a selected PyTorch 2.10-2.13
installation and the pinned build tools:

```powershell
python -m pip install scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0 "torch==2.10.*"
python -m pip wheel . --no-deps --no-build-isolation `
  --config-settings cmake.define.GFFX_BUILD_PYTORCH=ON `
  --wheel-dir dist/pytorch-local
```

PyTorch is intentionally selected outside GFFX metadata so CPU/CUDA variants and indexes remain
under user control. `GFFX_BUILD_PYTORCH=ON` does not create a supported operation or validate a new
Python/platform combination.

## Optional CUDA diagnostic provider

`GFFX_ENABLE_CUDA=ON` builds the separate driver-facing provider. It requires a system CUDA toolkit
and is never enabled by the default package build:

```powershell
python -m pip wheel . --no-deps `
  --config-settings cmake.define.GFFX_ENABLE_CUDA=ON `
  --wheel-dir dist/cuda-local
```

Release-candidate builds target CUDA Toolkit 12.8. Local builds using another CUDA 12.x toolkit are
development evidence only. See [CUDA_PLUGIN_BUILD.md](CUDA_PLUGIN_BUILD.md) for exact Windows and
Linux recipes and trusted-device test controls.

## Uninstall

```powershell
python -m pip uninstall gffx
```

Uninstall completeness was verified by the local Phase 1 package-foundation gates. That evidence
remains internal until the hosted release-candidate matrix succeeds and the resulting artifacts
are published under the support policy.
