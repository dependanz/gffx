# Installing GFFX

## Published release: 0.2.1 (Windows x64 only)

The currently published GFFX 0.2.1 distribution has a Windows x64 wheel for
ordinary-GIL CPython 3.10–3.14. Linux and macOS users cannot install its new
scene API through that release. Historical Windows instructions remain:

```powershell
python -m pip install "gffx==0.2.1"
python -m gffx.examples.cornell_box
```

The inherited 0.1.x API is unrelated to the new renderer.

## Cross-platform 0.2.2 candidate (not yet published)

GFFX 0.2.2 targets **binary wheels** for these exact supported platforms:

| Platform | CPU installation | Native GFFX GPU provider |
| --- | --- | --- |
| Windows x86-64 | yes | NVIDIA CUDA 12, subject to hardware qualification |
| Linux x86-64, glibc 2.28+ | yes | NVIDIA CUDA 12, subject to hardware qualification |
| Linux aarch64, glibc 2.28+ | yes | no (CPU only in this release) |
| macOS 14+, Apple Silicon arm64 | yes | no (CPU only in this release) |

macOS Intel, musllinux (including Alpine), FreeBSD and other Unix targets are
not part of the 0.2.2 binary distribution contract. In particular, neither
macOS CUDA nor Metal/MPS kernels are implemented in this release.

**Only after 0.2.2 is published to PyPI**, ordinary users will install with:

```bash
python -m pip install "gffx==0.2.2"
python -m gffx.examples.cornell_box
```

No local C/C++ compiler, NVCC, CMake or Git is required to install a supported
binary wheel. Use a supported CPython 3.10–3.14 installation on the target
platform. The example writes `cornell_box.png` in the current directory and
protects existing files from accidental overwrite.

### CPU and PyTorch installation behavior

The Python scene API requires `torch>=2.10.0,<=2.14.1`. A fresh ordinary
`pip install gffx` pulls a compatible PyTorch distribution. If a compatible
CPU or CUDA PyTorch build is already installed, pip can reuse it.

The GFFX native C11 core and bare `import gffx` remain lazy and independent
of CUDA; GPU discovery is an explicit operation. When a CUDA provider is
present in a wheel but no compatible device is available, CPU operations still
work. Explicit CUDA operations do not silently migrate tensors to CPU.

### NVIDIA GPU selection (Windows and Linux x64)

NVIDIA execution requires a compatible driver/device, an appropriate
CUDA-enabled PyTorch distribution, and the GFFX provider embedded in the
published Windows/Linux x64 wheel. The **CUDA toolkit is a build-time
requirement only**, not a requirement for ordinary wheel installation.

GPU-enabled PyTorch wheels are obtained from the official PyTorch indexes.
Select the compatible wheel family for your device and driver by following
[PyTorch's installation guide](https://pytorch.org/get-started/locally/).
If the runtime's CUDA-enabled PyTorch version is in GFFX's accepted range,
install GFFX with its regular PyPI index without overwriting PyTorch:

```bash
python -m pip install "gffx==0.2.2"
```

Installing GFFX does not automatically discover NVIDIA hardware and switch
PyTorch indexes; pip cannot express that choice in package metadata.

The current Linux CUDA build pilot was produced using a CUDA 12.9 toolkit.
Its actual driver compatibility floor must be qualified on NVIDIA hardware
before 0.2.2 release. The older Windows 0.2.1 provider has its own recorded
hardware evidence in [the prior release policy](RELEASE_POLICY_0_2_1.md).

### Source installation for contributors before PyPI publication

Build the candidate source from an exact commit, not a moving release name.
Use a Python/PyTorch version in the declared support range. The default
source configuration is CPU/native-only and does not build the optional
PyTorch adapter; a scene-capable source install must request it:

```bash
python -m pip install scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0
python -m pip install --no-build-isolation --no-deps \
  --config-settings=cmake.define.GFFX_BUILD_PYTORCH=ON \
  "git+https://github.com/dependanz/gffx.git@<exact-commit-sha>"
```

For CUDA provider builds, `GFFX_ENABLE_CUDA=ON` additionally requires the
CUDA toolkit and supported platform. Experimental source installs are not
equivalent to the qualified final wheel.

For the candidate's release admission procedure, see
[MULTIPLATFORM_RELEASE_ADMISSION.md](MULTIPLATFORM_RELEASE_ADMISSION.md).

## Legacy policies and security boundaries

The 0.2.1 Windows-only release evidence and its Python/CUDA hardware profiles
remain in [RELEASE_POLICY_0_2_1.md](RELEASE_POLICY_0_2_1.md).
Older 0.2.0 and alpha packages retain their historical installation policy.

Torch 2.10 remains within the retained compatibility range, but it is not
claimed upstream-security-maintained. The historical TorchScript advisory
[GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
affected earlier PyTorch versions. GFFX itself does not invoke TorchScript,
and training compilation/export are outside the bounded release claim.
