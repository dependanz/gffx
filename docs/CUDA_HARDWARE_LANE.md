# The CUDA hardware lane

Every other GFFX workflow runs on a disposable GitHub-hosted runner. None of them has a GPU, and
`tools/verify_foundation.py` configures `-DGFFX_ENABLE_CUDA=OFF`, so continuous integration has
never executed a CUDA kernel. This lane is the exception, and it exists because the claim it
guards cannot be checked anywhere else.

## What is missing without it

| Configuration | CTest inventory |
|---|---|
| Hosted runner, CUDA off | 25 |
| Reference host, CUDA on with device tests | 40 |

Fifteen tests exist that hosted CI has never run. Six of them are load-bearing:

| Test | What it establishes |
|---|---|
| `cuda.texture_device_parity` | CPU/CUDA bit-identity for the texture operations |
| `cuda.backward_device_parity` | 33 comparisons over 13 operations; 26 held by `memcmp` |
| `cuda.plugin.real_device_probe` | the plugin negotiates with a real driver |
| `cuda.plugin.default_discovery` | the shipped discovery path finds the real provider |
| `cuda.binary_plugin_isolation` | the built plugin binary keeps its symbol boundary |
| `cuda.operation_dispatch` | absent operations are null rather than stubs |

## Why the trigger is manual

`workflow_dispatch` only. No `pull_request`, no `schedule`, and the contract test in
`tests/packaging/test_ci_contract.py` fails if either appears.

A self-hosted runner attached to a **public** repository is the configuration GitHub explicitly
warns against, because a workflow triggered by a pull request executes the *fork's* code on the
host. Anyone with a GitHub account can open a pull request. `workflow_dispatch` cannot be reached
without write access to this repository, so the trigger itself is the security boundary rather
than a setting layered on top of one.

The schedule exclusion is not security. The reference host is a laptop, so a nightly cron would
mostly fire while it is asleep, and a lane whose red means "the machine was off" teaches you to
stop reading red. A manual lane you actually run beats an automatic one you learn to ignore.

## Why it does not trust its own exit status

Every device-gated fixture in GFFX declines when no usable device is present. So a lane that lost
its GPU — a driver update, a runner relabelled onto the wrong machine, a build configured without
`GFFX_ENABLE_CUDA` — would report a *complete pass* over the tests that remained. That is not a
hypothetical: the PyTorch CUDA fixtures in this project were skipping unnoticed for weeks because
the installed PyTorch was a CPU build. Nothing failed, so nothing was investigated.

`tools/verify_cuda_lane.py` therefore parses the CTest JUnit result and asserts the device
fixtures are present *by name*, plus a floor on the total inventory so tests cannot quietly
disappear. Fed a run of 36 passing tests with the device fixtures absent, it exits 1.

It also records `nvidia-smi` and `nvcc --version` into the retained report, because bitwise
agreement is a claim about one driver on one architecture. `log2f` is not required by IEEE 754 to
be correctly rounded, so cross-backend agreement on it is measured, not guaranteed, and the
measurement has to name what produced it.

## What it adds beyond the fixtures

`compute-sanitizer --tool memcheck` over both parity executables. This is the class of defect a
CPU-side CUDA emulator would have been useful for — out-of-bounds access, uninitialised reads, bad
launch configuration — checked on the hardware instead, where the floating-point behaviour stays
real. See [emulation](#on-emulating-a-gpu) for why the emulator is not a substitute.

## Registering the runner

The runner registration token is scoped to the repository and is yours to create; nothing in this
repository stores one.

1. **Create the protected environment first.** Settings → Environments → New environment →
   `cuda-hardware` → Required reviewers → yourself. The job names this environment, so it will not
   start until you approve it. While the trigger stays manual this is redundant; it stops being
   redundant the moment anyone widens the trigger, which is the point.
2. **Register the runner.** Settings → Actions → Runners → New self-hosted runner → Windows x64.
   Follow the download commands it prints, then configure with the `cuda` label added:

   ```powershell
   ./config.cmd --url https://github.com/dependanz/gffx --labels cuda
   ```

   `self-hosted`, `windows`, and `x64` are applied automatically. The job selects on
   `[self-hosted, windows, x64, cuda]`, so a second GPU host can join by carrying the same labels
   and nothing in the workflow names a machine.
3. **Do not install it as a service.** Start `./run.cmd` when you want the lane, dispatch the
   workflow, and stop it with Ctrl-C when the run finishes. The host is then exposed only during
   the window you are deliberately using it, which for a single-maintainer laptop is a stronger
   and simpler mitigation than any amount of sandboxing.
4. **Run it as a dedicated local account**, not your own. A self-hosted runner job has that
   account's filesystem, environment, and credentials. It should not be able to reach your SSH
   keys, your browser profile, or the other repositories on this machine.

## Prerequisites on the host

CMake and a Visual Studio C++ toolchain on `PATH`, a CUDA toolkit including `nvcc` and
`compute-sanitizer`, an NVIDIA driver, and Python for the verifier. The reference host is:

```text
NVIDIA GeForce RTX 5090 Laptop GPU, driver 595.79, compute capability 12.0, 24463 MiB
Cuda compilation tools, release 12.9, V12.9.41
```

The lane deliberately builds `Release`. The bit-identity discipline depends on `-fmad=false` and
on kernels mirroring the host operation for operation; whether that survives the optimiser is
exactly the thing worth checking, and a `Debug`-only result would not have checked it.

## On emulating a GPU

The idea is sound and the history is real. NVIDIA shipped `nvcc -deviceemu` until CUDA 3.0 and
removed it; GPU Ocelot (Georgia Tech) was a working PTX emulator with an LLVM CPU backend, dead at
compute capability 2.0; CuPBoP compiles CUDA to CPU through LLVM as research code; HIP-CPU
implements AMD's HIP runtime on the CPU. ZLUDA and SCALE are sometimes offered as answers but are
not — they translate CUDA onto *other GPUs*, so they still need hardware.

None targets a 2025 PTX ISA, and GFFX embeds `sm_120` PTX loaded through the CUDA driver API.

The deeper problem is that emulation cannot carry this project's claim even if it existed. What
the parity fixtures assert is bit-identity between the CPU host and the GPU, and that is a
statement about *the hardware's* floating-point behaviour: fused-multiply-add contraction, the
special-function unit's `log2f`, denormal handling, atomic accumulation order. An emulator
executes PTX on the host FPU, so it would agree with the CPU reference trivially and prove
nothing.

It would be worse than uninformative. The signed-zero divergence found in the
`closest_point_on_mesh` backward — host `+0.0`, device `-0.0`, equal under `==` and visible only
to `memcmp` — is precisely a divergence an emulator cannot produce. It would have returned green
on the bug.

Emulation would still be genuinely useful for a different tier: memory safety, indexing errors,
race conditions under a scheduler that can explore interleavings real hardware will not, and
running the shape of a kernel where no GPU exists at all. That tier is worth having. It is also
what `compute-sanitizer` already does, on hardware, today — which is why this lane runs it rather
than building an emulator.
