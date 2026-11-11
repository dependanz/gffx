# GFFX 0.2.2 protected NVIDIA runner setup

These are the GitHub and hardware settings still needed before the production
release-candidate workflow can pass. This file does not register runners.

## Required self-hosted runner labels

- Windows: `self-hosted`, `windows`, `x64`, `cuda`
- Linux: `self-hosted`, `linux`, `x64`, `cuda`

The CUDA build and test jobs use the protected `cuda-hardware` environment.
The release is blocked if either runner is offline or incomplete. A single
host cannot be both Windows and Linux. A Linux VM with NVIDIA passthrough
may be useful for testing, but WSL2-only results do not establish bare-metal
Linux fleet support; record the tested configuration accurately.

## Host preparation and preflight

Both systems require a driver-visible NVIDIA GPU, NVCC from CUDA Toolkit 12,
`compute-sanitizer`, CPython 3.10–3.14, CMake, Git and suitable compilers.
Windows additionally needs MSVC and the Windows SDK; Linux needs GCC/Clang.
From a trusted source checkout, run:

Windows PowerShell:

```powershell
python tools/release_runner_preflight.py --target windows --json
```

Linux shell:

```bash
python tools/release_runner_preflight.py --target linux --json
```

Expected status is `pass`. The script does not download software, modify
configuration, contact GitHub, or read runner credentials. It checks hardware
and tool availability only; actual GPU operation tests are a separate gate.

## GitHub registration and security

1. Go to dependanz/gffx → Settings → Actions → Runners → New self-hosted runner.
   Follow GitHub's platform-specific setup and label the runners as above.
   Do NOT paste registration tokens into chat, source, or CI logs.
2. Confirm both runners are online/idle. Prefer dedicated, isolated hosts
   without private data, personal credentials, or persistent secrets.
3. Under Settings → Environments, protect `cuda-hardware` with required
   reviewer approval and appropriate trusted ref restrictions.
4. Never allow untrusted `pull_request` or scheduled events to execute
   on the protected hardware runners. Protect default branch and release tags.
5. Separately protect the `pypi` environment and configure a PyPI Trusted
   Publisher for the new `publish-multiplatform.yml` workflow.

Runner registration, environment approvals and PyPI publisher authorization
are account-level actions that cannot be completed by a source-code PR.

## Running the release qualification

Dispatch `package-foundation-rc` with an EXACT 40-character commit SHA.
Do not dispatch it with a moving branch. Windows and Linux GPU jobs must
pass their native no-skips CTest device inventory and compute-sanitizer checks.
The Linux runner also installs the exact manylinux candidate wheel and proves
its bundled `libgffx_cuda12.so` loads on the real NVIDIA device.

The final release additionally requires the complete 100-environment
installed-wheel matrix, matching cross-platform binaries and sdist, SHA-256
receipt, qualified driver/device support, and human publication approval.
Approval cannot be inferred from builds, statuses or missing hardware.
