# Installing gffx0.2

The first stable release targets Windows x64 and ordinary-GIL CPython3.10-3.14.
During candidate evaluation, use the published0.2.0a2 alpha explicitly until
0.2.0 publication is verified on PyPI. [Release status](RELEASE_0_2_0.md) and
[the exact support contract](RELEASE_POLICY_0_2_0.md) distinguish actual admission
from packaging compatibility. Linux/macOS/other architectures need separately
qualified wheels; no automatic source-build or universal support is promised.

## Base package

After stable publication, a fresh compatible Windows environment can use:

```powershell
python -m pip install --only-binary=gffx gffx
```

For an existing installation use `python -m pip install --upgrade --only-binary=gffx gffx`.
Pin `gffx==0.2.0` for reproducibility. The inherited public0.1.x API is unrelated;
see [migration](MIGRATING_0_2_0.md). Prereleases need a version pin or `--pre`.
Base import/static CPU capabilities need no framework, GPU probe, toolkit or compiler.

## CPU operations

Install separately selected CPU-only PyTorch in the tested2.10 line:

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=gffx gffx
python -c "import gffx, gffx.torch; print(gffx.__version__); print(gffx.capabilities())"
```

The exact stable eager operation list excludes higher-order, compiled/exported,
streaming, renderer/performance and broad framework/platform claims. Optional
PyTorch activation reports missing/incompatible frameworks explicitly.

## Optional CUDA

The same Windows wheel includes the lazy CUDA Driver-API provider. It is an
experimental exact profile, not a GPU fleet guarantee. The initial actual-device
profile is Python3.14.6/Torch2.10.0+cu126/RTX3080/driver610.88; the declared Windows
driver baseline is560.76+, also subject to GPU support. PTX8.5 targetssm_75 and is
compiled with nvcc12.6.20. No toolkit/compiler is required to use the wheel.

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --only-binary=gffx gffx
python -c "import torch, gffx.torch, gffx.cuda; print(torch.cuda.is_available()); print(gffx.cuda.capabilities())"
```

Use CUDA input tensors to select GPU execution. All tensor operands, including
sampling RNG tensors, must share the device. No implicit CPU fallback or hidden
device copy is added. A CPU-only Torch installation cannot create CUDA tensors.
`gffx[gpu]` is not declared; the provider is already in the normal wheel and
framework/driver selection remains explicit. No framework or NVIDIA DLL is bundled.
