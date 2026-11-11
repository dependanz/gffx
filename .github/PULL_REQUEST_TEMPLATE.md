## Target branch

- [ ] This feature, fix, or maintenance PR targets `dev`.
- [ ] OR this is an explicitly approved `dev`/release/hotfix promotion into `main`.

## Scope and compatibility

Summarize the change, public API impact, and files or prior milestones it depends on.

## Validation

- [ ] Relevant source and packaging tests passed (include commands and results).
- [ ] Relevant CPU and CUDA checks were run, or their omission is explained.
- [ ] Existing behavior and accepted baselines were checked for regressions.
- [ ] Any experimental limitations and performance tradeoffs are documented.
- [ ] Existing worktrees and unrelated uncommitted changes were not overwritten.

## Release-specific (only for PRs into `main`)

- [ ] Approved target version and changelog.
- [ ] Validated built artifacts and digest/provenance.
- [ ] Required CI, platform, and release gates complete.
- [ ] Explicit maintainer sign-off and planned tagged GitHub Release / PyPI action.

Reference: [GFFX branching and releases](../docs/BRANCHING_AND_RELEASES.md).

**A merged PR does not automatically authorize a PyPI publish.**
