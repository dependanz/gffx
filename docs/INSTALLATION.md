# Installing gffx 0.2.0a1

The first `0.2.0a1` prerelease is an experimental Windows x64 CPU wheel. Its
versioned instructions apply when the release is available on PyPI; preparation
and publication are separate. See [the exact alpha profile](ALPHA_0_2_0A1.md).
The earlier public `0.1.x` package is the inherited prototype with a different API.

## Base package

Use ordinary-GIL CPython 3.10 or newer on Windows x64. The wheel uses the CPython
3.10 stable ABI; that tag alone does not establish every Python/framework profile.
No compiler, CUDA toolkit, GPU or PyTorch installation is needed for base use.

```powershell
python -m pip install --only-binary=gffx "gffx==0.2.0a1"
python -c "import gffx; print(gffx.__version__); print(gffx.capabilities())"
```

`import gffx` is lazy and framework-free. `gffx.capabilities()` loads the native
core and reports static capabilities without probing a GPU. `--only-binary=gffx`
prevents an unsupported platform from unexpectedly attempting a source build.

## PyTorch operations and gradients

The alpha wheel includes the optional PyTorch adapter. Install the chosen PyTorch
runtime separately; gffx neither chooses its CPU/CUDA variant nor bundles it.
The adapter targets the PyTorch 2.10 stable ABI. Exact tested combinations are
listed in [the alpha profile](ALPHA_0_2_0A1.md); newer versions are not automatically
promoted to a supported combination.

For the selected CPU-only runtime:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=gffx "gffx==0.2.0a1"
python -c "import torch, gffx.torch; print(torch.__version__)"
```

A CUDA-enabled PyTorch installation may be retained for the CPU operations in
the verified profile. It does not enable the separately built gffx CUDA provider.
This alpha has no gffx GPU-execution claim. Python free-threaded builds, PyPy,
other architectures and other operating systems are outside this first wheel.

## Building from source

Default source builds contain the CPU core and CPython bridge with the optional
adapter disabled. Building a wheel equivalent to this alpha requires the explicit
CPU-only PyTorch build recipe in [BUILDING.md](BUILDING.md); a default source build
is not an adapter-enabled release artifact. Build tools and compilers are build
inputs, never dependencies of an installed wheel.

## Uninstall

```powershell
python -m pip uninstall gffx
```
