# GFFX 0.2.2 protected Windows NVIDIA runner and future assurance

These are the GitHub and hardware settings for the protected nonpublishing
0.2.2 release-candidate workflow. The required Windows GPU lane is separate
from optional future independent native-Linux GPU and native-Mac assurance.
This file does not register runners.

## Required self-hosted runner labels

- Windows (required for 0.2.2): `self-hosted`, `windows`, `x64`, `cuda`
- Linux (future optional assurance): `self-hosted`, `linux`, `x64`, `cuda`

The required Windows CUDA wheel builder uses the protected `cuda-hardware`
environment. The RC cannot finish without this runner and its real NVIDIA tests.
No independent native-Linux NVIDIA runner is required for 0.2.2; WSL2 CUDA
execution evidence is retained as supplemental hardware qualification, not
as native-Linux certification. GitHub-hosted macOS ARM64 wheel tests remain
required, but independent native-Mac host testing is deferred.

## Host preparation and preflight

The required Windows builder needs a driver-visible NVIDIA GPU, NVCC from
CUDA Toolkit 12, `compute-sanitizer`, CPython 3.10–3.14, CMake, Git, MSVC
and the Windows SDK. The Linux toolchain and hardware preflight is retained
for optional future assurance.
From a trusted source checkout, run:

Windows PowerShell:

```powershell
python tools/release_runner_preflight.py --target windows --json
```

Future native Linux hardware audit only:

```bash
python tools/release_runner_preflight.py --target linux --require-native-linux --json
```

Expected status is `pass` on the respective intended hardware. The script does not download software, modify
configuration, contact GitHub, or read runner credentials. It checks hardware
and tool availability only; actual GPU operation tests are a separate gate.

## GitHub registration and security

1. Go to dependanz/gffx → Settings → Actions → Runners → New self-hosted runner.
   Follow GitHub's platform-specific setup and label the runners as above.
   Do NOT paste registration tokens into chat, source, or CI logs.
2. Confirm the **Windows** runner is online/idle. A native Linux runner is
   optional future assurance. Prefer a dedicated, isolated host without
   private data, personal credentials, or persistent secrets.
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
Do not dispatch from a moving tag or a branch at a different SHA.
Windows GPU hardware checks must pass native no-skip CTest device inventory,
Compute Sanitizer and exact candidate-wheel CUDA provider loading. The Linux
x64 candidate wheel is built in the pinned manylinux CUDA image and must
contain the CUDA provider; hosted installation/CPU/adapter smoke tests remain
mandatory. Previous WSL2 GPU runs are supplementary release evidence.

The final release additionally requires the complete 100-environment
installed-wheel matrix, matching cross-platform binaries and sdist, SHA-256
receipt, Windows GPU evidence and human publication approval. Independent
native Linux GPU and dedicated native Mac tests will be planned separately.
Approval cannot be inferred from builds, statuses or missing hardware.

## Protected 0.2.2 RC admission and dispatch

Inspect the live GitHub repository settings through an authenticated `gh api`
before dispatch. `cuda-hardware` and `release-candidate-build` must have
effective required-reviewer protection and branch policies that permit the
reviewed source branch. Environment *names alone* are not protection.
Keep `pypi` separately protected for later publication; never publish as
part of RC qualification.

Require an online, trusted **Windows** runner labeled
`self-hosted,windows,x64,cuda`. Independent Linux NVIDIA hardware
(`self-hosted,linux,x64,cuda`) is future optional assurance, **not** a
0.2.2 release dependency. Do not run untrusted PR jobs on the Windows host.

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

This is an owner action to perform **only after** verifying the Windows
runner registration, effective environment protection and current CI. Re-read
the branch SHA immediately before dispatch; subsequent fixes change the
candidate revision. The RC qualifies but never publishes. It must pass
all GPU, artifact provenance, and 100-environment gates and retain an
exact SHA/run-bound receipt before a separate tag and publication review.
