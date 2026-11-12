# GFFX 0.2.2 protected NVIDIA runner setup

These are the GitHub and hardware settings still needed before the production
release-candidate workflow can pass. This file does not register runners.

## Required self-hosted runner labels

- Windows: `self-hosted`, `windows`, `x64`, `cuda`
- Linux: `self-hosted`, `linux`, `x64`, `cuda`

The CUDA build and test jobs use the protected `cuda-hardware` environment.
The release is blocked if either runner is offline or incomplete. A single
host cannot be both Windows and Linux. A Linux VM with independently verified NVIDIA passthrough may be useful
for supplemental testing, but WSL2-only results do not establish native Linux
fleet support. The production Linux RC gate uses `--require-native-linux`
and rejects WSL2 kernels.

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
python tools/release_runner_preflight.py --target linux --require-native-linux --json
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
   reviewer approval and appropriate trusted ref restrictions. Protect
   `release-candidate-build` as well; three hosted platform builders use it.
   Make sure the reviewed RC branch is permitted by deployment policies.
4. Never allow untrusted `pull_request` or scheduled events to execute
   on the protected hardware runners. Protect default branch and release tags.
5. Separately protect the `pypi` environment and configure a PyPI Trusted
   Publisher for the new `publish-multiplatform.yml` workflow.

Runner registration, environment approvals and PyPI publisher authorization
are account-level actions that cannot be completed by a source-code PR.

## Running the release qualification

Dispatch `package-foundation-rc` as owner `dependanz` from the reviewed
feature branch tip, supplying that **same** exact 40-character SHA as `revision`.
The workflow requires `GITHUB_SHA == inputs.revision` before any build.
Do not dispatch from a moving tag or a branch at a different SHA. Windows and Linux GPU jobs must
pass their native no-skips CTest device inventory and compute-sanitizer checks.
The Linux runner also installs the exact manylinux candidate wheel and proves
its bundled `libgffx_cuda12.so` loads on the real NVIDIA device.

The final release additionally requires the complete 100-environment
installed-wheel matrix, matching cross-platform binaries and sdist, SHA-256
receipt, qualified driver/device support, and human publication approval.
Approval cannot be inferred from builds, statuses or missing hardware.

## Protected 0.2.2 RC admission and dispatch

Inspect the live GitHub repository settings through an authenticated `gh api`
before dispatch. `cuda-hardware` and `release-candidate-build` must have
effective required-reviewer protection and branch policies that permit the
reviewed source branch. Environment *names alone* are not protection.
Keep `pypi` separately protected for later publication; never publish as
part of RC qualification.

Require online, trusted `self-hosted,windows,x64,cuda` and
`self-hosted,linux,x64,cuda` runners. The latter must run native Linux,
not WSL2; do not run untrusted PR jobs on these runners.

Since the RC workflow exists on `main` but its candidate implementation is
on the feature branch, dispatch **that feature branch** explicitly:

```powershell
$repo = 'dependanz/gffx'
$branch = 'chore/multiplatform-release-gate-20261111'
$sha = (gh api "repos/$repo/pulls/8" --jq '.head.sha').Trim()
if ($LASTEXITCODE -ne 0 -or $sha -notmatch '^[0-9a-f]{40}$') { throw 'Missing exact SHA' }
gh workflow run package-foundation-rc.yml --repo $repo --ref $branch -f "revision=$sha"
if ($LASTEXITCODE -ne 0) { throw 'RC dispatch failed' }
```

This is an owner action to perform **only after** verifying both runner
registrations, effective environment protection and current CI. Re-read
the branch SHA immediately before dispatch; subsequent fixes change the
candidate revision. The RC qualifies but never publishes. It must pass
all GPU, artifact provenance, and 100-environment gates and retain an
exact SHA/run-bound receipt before a separate tag and publication review.
