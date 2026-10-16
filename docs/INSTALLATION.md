# Installing gffx 0.2.1

The 0.2.1 distribution targets Windows x64 and ordinary-GIL Python 3.10–3.14:

```powershell
python -m pip install gffx
python -m gffx.examples.cornell_box
```

The example is included in the wheel and writes `cornell_box.png` in the current
directory. Existing output is protected; use a fresh directory for reruns.
Use `python -m pip install --upgrade gffx` for an existing installation, or pin
`gffx==0.2.1`. The inherited 0.1.x API is unrelated.

## CPU and GPU selection

The same wheel contains the native CPU core and lazy CUDA provider. Its one
direct Python dependency is `torch>=2.10.0,<=2.14.1`. Fresh installs select
2.14.1; an already installed compatible CPU/CUDA build is reused. The bounded
range is not a tested-support claim for every intermediate framework version.

On Windows standard PyPI Torch is CPU-only, so plain fresh installation renders
on CPU even if the PC has an NVIDIA GPU. Pip has no GPU-hardware dependency
marker and cannot switch indexes by detecting a graphics card. Package import
does not install or replace frameworks. For a fresh CUDA environment use:

```powershell
python -m pip install gffx --extra-index-url https://download.pytorch.org/whl/cu126
python -m gffx.examples.cornell_box
```

To switch an existing CPU-only framework to the current CUDA build explicitly:

```powershell
python -m pip install --upgrade torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

`Scene.to('auto')` and `Camera.look_at(..., device='auto')` select CUDA only when
both CUDA Torch and the gffx provider are usable; otherwise they select CPU at
setup. Explicit-device operations never silently fall back or copy devices.

GPU use needs a compatible NVIDIA device/driver, with no toolkit/compiler.
Exact actual-device profiles are Python 3.14.6/Torch 2.14.1+cu126 and retained
2.10.0+cu126 on RTX3080/compute8.6/driver610.88; PTX8.5/sm_75. This is not a GPU
fleet guarantee. Linux/macOS/other architectures need separately qualified
wheels. See [policy](RELEASE_POLICY_0_2_1.md) and [simple scenes](SIMPLE_SCENES.md).

## Retained runtime and native-only use

The older 2.10 eager compatibility profile is retained to avoid replacing
working CUDA environments. It has the upstream TorchScript advisory
[GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47); gffx does
not invoke TorchScript and compilation/export is outside admission. Older Torch
is not claimed upstream-security-maintained. New environments use patched2.14.1.

The C11 core and bare import remain independent of Torch and do not probe a GPU.
Advanced native-only consumers may deliberately use `pip install --no-deps gffx`;
rendering requires the declared framework. Such an intentionally incomplete
Python installation fails `pip check` until Torch is installed. Older published
0.2.0a1/a2/0.2.0 artifacts keep their previous dependency-free-install policy.
