# Development wheels

M1 is complete for the adopted development-readiness scope. A development wheel
is a source/build snapshot with an explicit profile; it does not establish the
whole platform or backend support matrix.

The source version is `0.2.0.dev0`. The existing PyPI project still lists the
older `0.1.4` release as of 2026-10-15. This preparation does not upload a release
or replace that package. A local wheel is pip-installable without PyPI.

The first fresh preparation target is Windows x64 CPU, CPython 3.10 stable ABI,
with the PyTorch adapter enabled and the gffx CUDA provider disabled. The exact
validated framework is PyTorch `2.10.0+cu126`, selected separately; the wheel
does not bundle it. The adapter built against that runtime links `torch_cuda.dll`
as well as `torch_cpu.dll`, even though the tested gffx operations run on CPU.
This artifact is not a standalone CPU-only PyTorch distribution. Compatibility
with a CPU-only PyTorch installation needs its own build and verification.
Use the wheel's exact hash and accompanying validation report to identify a build.
Build and installation must be verified before that candidate is called usable.

After installing the matching PyTorch runtime, a local installation is:

```powershell
python -m pip install --no-deps C:/path/to/gffx-0.2.0.dev0-cp310-abi3-win_amd64.whl
```

The intended library boundary is generic graphics in gffx, general fitting and
tracking in gffit, and face animation/models such as FLAME in gfface. Existing
consumer examples remain available while that model-adapter migration is planned.

The independent SH136 filtered-renderer reference campaign remains unqualified
and paused indefinitely. Development-wheel preparation does not resume it. The
B1 photograph's unacceptable likeness remains a documented consumer limitation.

See [building](BUILDING.md), [support status](SUPPORT_STATUS.md),
[M1 fitting profiles](M1_FITTING_PROFILES.md), and
[M1 closeout](M1_DEVELOPMENT_READINESS_CLOSEOUT.md).
Licensed assets, private photographs, generated outputs and build products belong
outside the source repository.

The fresh 2026-10-15 local candidate installs in a new environment and passes
233 selected CPU fitting/camera/image/SH tests plus nine subtests. Ten CUDA or
independent opaque checks are outside that selection. The broader run is retained:
CUDA tests fail without the optional gffx provider, and one 4x4 opaque-mask exact
equality check differs by 1.1102230246251565e-16 (the associated secondary
gradient is 0.24999999999999994 rather than 0.25). That result remains a known
independent-profile limitation; no assertion or renderer was changed to hide it.
The source snapshot and local artifact hashes are recorded in the publication
preparation evidence packet under the device's external devspace.
