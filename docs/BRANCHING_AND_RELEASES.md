# GFFX development and release workflow

Status: adopted branching policy; repository protections and future release automation must be verified separately.

## Branch roles

- **`main`**: stable, released code. Keep `main` as the default GitHub branch. Update it only using a reviewed release PR from `dev` (or an approved `hotfix/*`), with release gates satisfied.
- **`dev`**: integration branch. It starts at the latest stable commit and receives reviewed feature/fix/chore PRs. `dev` is allowed to move ahead of `main`.
- **`feature/<topic>`**: one scoped enhancement based on current `dev`. Target pull requests to `dev`, not `main`.
- **`fix/<topic>`** and **`chore/<topic>`**: repairs and maintenance based on `dev`, also targeting `dev`.
- **`hotfix/<topic>`**: exceptional production repairs based on `main`; after publishing, carry the fix into `dev` as well.
- **`release/<version>`** (optional): stabilization branch created from `dev` when a release needs extra validation.

## Development loop

1. `git fetch origin --prune`, then create a feature branch or worktree from `origin/dev`. Never switch a dirty working tree to `dev` to force synchronization.
2. Commit coherent changes and their tests together. Keep reproducible test runners in `tools/` and regressions in `tests/`; use short PowerShell commands for one-off local diagnostics.
3. Open a PR with **base `dev`**. Describe scope, expected observable behavior, tests (CPU/CUDA where relevant), API compatibility, and evidence.
4. Run the PR CI matrix and any domain-specific tests; acknowledge limitations. Review before merging into `dev`.
5. Keep prerelease tags and release-candidate artifacts distinct from production releases; a push to `dev` must never publish to production PyPI.

## Stable release loop

1. Decide the next version explicitly using PEP 440 (for example `0.3.0`, `0.3.0.dev1`, or `0.3.0rc1`); do not reuse a previously published version.
2. Update `pyproject.toml` and relevant release notes, build from an exact release-candidate commit, and run the project's packaging, installed-wheel, D01/D02, CPU, CUDA, and platform-specific acceptance gates applicable to the release.
3. Open a PR **from `dev` to `main`** (or an audited release branch/hotfix). Require human release sign-off. Do not merge based solely on a green packaging-foundation check.
4. After the reviewed release commit lands on `main`, use an annotated `v<version>` tag and a GitHub Release with the validated artifacts and provenance. Publish using the approved GitHub Actions/PyPI trusted-publishing release path.
5. Confirm the uploaded version, artifact digest, installation and smoke test. Bring hotfixes back into `dev`.

**Publishing safety:** The current `.github/workflows/publish-alpha.yml` has explicitly gated historical versions (`v0.2.0a2` and `v0.2.1`) and is triggered by published GitHub Releases, not branch pushes. A future stable version must update/review its release conditions and version-specific gates before publication. This policy does not itself activate an automated future PyPI release or modify the PyPI trusted publisher.

## Branch protections and checks

In repository Settings > Rules > Rulesets (or Branches), configure protections for `main` and `dev`: require pull requests, prevent force pushes and deletions, restrict bypasses, and require **actual observed successful** CI checks before making them mandatory. On `main`, require release review. On `dev`, require the PR foundation and relevant functional checks. GitHub settings protections are not enforced by the files in this PR alone.

The lightweight `branch-policy` workflow checks PR target conventions and development-source CI contracts. It is not a replacement for tests or repository rulesets. Keep the existing foundation, Windows functional, nightly, RC and publishing workflows; review their status independently.

## Current Studio D02 integration

The D01/D02 candidate at `C:\dev\gffx-d02-integration-v17` is an **uncommitted worktree based on historical `m1-camera`**, not a feature branch cut from the refreshed `dev`. Preserve it and its dirty source checkout. Do not force-merge its broad staged snapshot into `dev`. First finish the v19 scoped regression and source-inventory review; then build reviewed integration commits with explicit dependencies on a separate branch based on `origin/dev`.

## Useful local commands (PowerShell)

```powershell
git -C C:\dev\gffx fetch origin --prune
git -C C:\dev\gffx branch -avv
git -C C:\dev\gffx worktree list
git -C C:\dev\gffx status --short --branch
```

For a **new clean feature worktree** (replace the name and path):

```powershell
git -C C:\dev\gffx fetch origin --prune
git -C C:\dev\gffx worktree add -b feature/my-change C:\dev\gffx-my-change origin/dev
```

Create PRs using `gh pr create --base dev --head feature/my-change` after pushing that feature branch. Do not run `git reset --hard`, `git clean -fd`, force push, or merge from the uncommitted D02 snapshot.
