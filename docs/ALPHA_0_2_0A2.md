# Experimental Windows GPU-capable alpha: 0.2.0a2

This candidate contains the CPU core, CPU-built optional PyTorch adapter and
existing CUDA Driver-API provider in one Windows x64 `cp310-abi3` wheel. The
published CPU-only 0.2.0a1 remains unchanged. GPU execution is selected by CUDA
input tensors; base import and CPU use remain lazy and require no GPU probing.
PyTorch is selected separately. No Torch, CUDA Runtime, or NVIDIA driver DLL is
bundled. CUDA Toolkit is a build input, not a user installation requirement.

## Exact experimental profile

| Dimension | Candidate profile |
|---|---|
| Build | Windows x64; CPython 3.10; CPU-only Torch 2.10.0 |
| Provider | CUDA Toolkit 12.6.20; Driver API only; PTX 8.5 targeting sm_75 |
| Declared NVIDIA baseline | Windows driver 560.76 or newer, also supporting the user's GPU |
| Actual GPU validation | RTX 3080, compute 8.6, driver 610.88 |
| Actual GPU framework | CPython 3.14.6, Torch 2.10.0+cu126 |
| Verified CPU compatibility | CPython 3.10/3.14 with CPU-only Torch 2.10.0 |
| Results | 245 selected GPU-cohort checks passed, no skips; CPU profiles each 232 passed |

The driver baseline follows NVIDIA's CUDA 12.6 GA table and is not an old-driver
test result. PTX JIT needs a driver that understands this compiler's PTX ISA;
the CUDA 12.x minor-compatibility floor alone does not establish that. The broader
fully supported release target builds at CUDA 12.0. This first experimental
candidate uses the available 12.6 toolchain and must be approved with that explicit
higher-toolkit scope. No full Python/framework/GPU/OS matrix is claimed.

## Installation after publication

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --only-binary=gffx "gffx==0.2.0a2"
python -c "import torch, gffx.torch, gffx.cuda; print(torch.__version__); print(gffx.cuda.capabilities())"
```

For CPU-only use, retain CPU-only Torch or omit Torch for the base capabilities
surface. `import gffx` and `gffx.capabilities()` do not probe the GPU;
`gffx.cuda.capabilities()` explicitly probes the optional provider and driver.
The wheel cannot enable CUDA tensors in a CPU-only PyTorch distribution.

## Verification boundary

Exact wheel installation, base/CPU behavior, native import isolation and embedded
PTX identity precede finite actual-device operation, first-gradient, caller-stream,
storage/refusal and deterministic parity checks. The selected test inventory and
actual results are recorded below. GPU tests must not silently
skip or fall back to CPU. Numerical tolerances are never weakened for packaging.

The independent filtered opaque-renderer reference campaign remains unqualified
and paused indefinitely. No photograph fit, external model, long-running reference
worker, performance qualification or broader support promotion is included.
CPU-only mesh topology/validation and exact route-specific refusals remain visible.

See [GPU build recipe](CUDA_PLUGIN_BUILD.md) and [publication procedure](PUBLISHING_ALPHA.md).
NVIDIA provenance: [CUDA 12.6 release notes](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html).


## Exact artifact and completed checks

The repaired candidate wheel is `gffx-0.2.0a2-cp310-abi3-win_amd64.whl`,
3,185,317 bytes, SHA-256
`756cff3b2fc1045bda29afffca5a51fce7c8e7481ad4a1d63a0fe69c5ba34fba`.
The release manifest fixes those bytes. All installed package payloads in the
three fresh validation environments match that wheel. The package has no
mandatory Python dependencies. PE imports, provider ABI, embedded PTX, wheel
RECORD, strict Twine checks and positive/negative release-verifier checks pass.
The CPU audit still accepts the published a1 wheel and rejects this GPU profile.
Build guards reject CUDA-enabled Torch builders and a different CUDA toolkit.

| Installed-wheel profile | Selected results |
|---|---|
| Python 3.10.11, Torch 2.10.0+cpu | 232 passed, 1 CUDA-device skip, 10 explicit deselections, 9 subtests |
| Python 3.14.6, Torch 2.10.0+cpu | 232 passed, 1 CUDA-device skip, 10 explicit deselections, 9 subtests |
| Python 3.14.6, Torch 2.10.0+cu126, RTX 3080 | 245 passed, 0 failed, 0 skipped; 98.34 seconds of pytest time |

The GPU cohort includes 17 independent numerical gradient checks (12 primitive
and 5 animation cases), 24 float32/float64 primitive forward and matched-VJP
parity cases, and four public/raw sampler device-validation regressions. Existing
operation, animation, image/compositor, SH/deferred, caller-stream, storage and
refusal controls complete the finite selection; some controls deliberately use
the CPU adapter. GPU tests cannot silently skip. This is not a GPU performance
benchmark or the full support matrix. The 53 GPU-run warnings are retained in
the evidence, principally pytest record-property/JUnit-format warnings and a
tensor-to-scalar warning; they are not failed assertions.

Both CPU profiles include the saved gffit consumer tests plus camera, image and
SH controls. Their sole skip needs an actual CUDA device; their 10 deselections
exclude CUDA cases and the separately recorded, unqualified filtered-opaque
case. That historical one-ULP equality failure is not converted into a pass or
rerun as part of this release. Actual GPU execution was tested on Python 3.14
only; the abi3 tag and CPU Python 3.10 check do not establish every GPU/Python
combination. Base probes before installing a framework confirm that neither
`gffx_cuda12.dll` nor `nvcuda.dll` loads during import/static capabilities.

Candidate construction used one build worker and test runs used one CPU thread
and one GPU process. The corrected build took 57.355 seconds with 455,200,768
bytes sampled child-process-tree RSS. The final GPU run took 102.384 supervised
seconds, peaked at 1,934,901,248 sampled host bytes and 641,728,512 incremental
global GPU bytes; Torch peaks were 21,192,192 allocated and 39,845,888 reserved
bytes. All preparation guards passed. Sampled observations are not hard peak
measurements or a general execution budget.

## Preserved failure and verified repair

The first candidate's new sampler tests incorrectly put RNG states on CPU with
CUDA geometry. This also exposed a missing argument check: native dispatch could
try a device read and return an internal error. The corrected Python surface
rejects mixed devices before dispatch, and the CUDA provider validates RNG views
and shapes before copies. The fixture now follows the existing same-device
contract. Four regressions cover the refusals. Valid sampling arithmetic,
device kernels and floating-point flags are unchanged. Initial failed artifacts
and test reports remain preserved; the corrected wheel was freshly rebuilt and
all selected profiles passed against those exact bytes.
