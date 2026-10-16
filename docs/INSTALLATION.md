# Installing the gffx alphas

The 0.2.x alphas package the M1 development snapshot for ordinary-GIL CPython
3.10+ on Windows x64. The `cp310-abi3` tag is an ABI floor, not a complete tested
Python/framework/GPU matrix. Public 0.1.x releases are the inherited prototype
with different APIs. Use an explicit alpha version.

| Version | Artifact scope |
|---|---|
| `0.2.0a1` | Published CPU core and optional PyTorch adapter; no CUDA provider |
| `0.2.0a2` | CPU core, optional adapter and optional CUDA provider; availability is verified separately by publication |

## Base and CPU use

After the selected version is published:

```powershell
python -m pip install --only-binary=gffx "gffx==0.2.0a2"
python -c "import gffx; print(gffx.__version__); print(gffx.capabilities())"
```

Base use needs no PyTorch, GPU, driver probe, toolkit or compiler. The a2 wheel's
provider remains unloaded during base/CPU capability checks. `--only-binary=gffx`
prevents an unsupported platform from attempting an unintended source build.
For CPU operations with PyTorch, install the separately chosen framework:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

The CPU-only a1 remains available by pinning `gffx==0.2.0a1` instead.

## GPU operations with a2

The exact validation profile uses Python 3.14.6, Torch 2.10.0+cu126, RTX 3080
(compute 8.6) and NVIDIA driver 610.88. The candidate's PTX 8.5 targets compute
7.5; its declared Windows driver baseline is 560.76, subject to GPU support.
Older drivers, other GPUs and other framework combinations are not qualified by
the one local profile. CUDA-enabled PyTorch and an NVIDIA driver are selected
separately; no Torch, CUDA Runtime or driver DLL is bundled in gffx.

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --only-binary=gffx "gffx==0.2.0a2"
```

Check the actual operation, using explicit CUDA tensors:

```python
import torch
import gffx
import gffx.torch as gt

assert torch.cuda.is_available()
vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], device="cuda", requires_grad=True)
faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
normals, areas, valid = gt.mesh.face_geometry(vertices, faces)
areas.sum().backward()
print(areas, vertices.grad)
print(gffx.cuda.capabilities())  # explicit provider/driver probe
```

Every tensor operand, including surface-sampling RNG key/counter tensors, must
be on the caller's device. gffx does not silently copy data or fall back to CPU.
See [the exact a2 validation profile](ALPHA_0_2_0A2.md) for tested operations,
first derivatives, exclusions and retained limitations. No full support tier or
photograph-fit quality claim follows from installing an alpha.

## Source build and removal

Default source builds contain the CPU core and CPython bridge, with optional
adapter/provider disabled. See [BUILDING.md](BUILDING.md) for CPU builds and
[CUDA_PLUGIN_BUILD.md](CUDA_PLUGIN_BUILD.md) for the guarded a2 build.
Build tools are never installed-wheel runtime dependencies.

```powershell
python -m pip uninstall gffx
```
