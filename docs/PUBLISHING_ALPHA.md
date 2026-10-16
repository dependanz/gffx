# Published alpha history and publishing configuration

The original a2 instructions below describe the frozen alpha transaction. The
normal 0.2.0 release uses a separate exact-artifact gate in `tools/stable_release.py`:
fresh complete concrete checks, no open blockers, source/metadata integrity and
the actual owner-published normal-release event at the reviewed commit. The user
replaced the initial 0.2.0 calendar wait with explicit release sign-off. `--check`
reports readiness without approval; the publisher accepts neither that mode nor
`--candidate`. It tests the exact attached release wheel on five CPU profiles
before OIDC upload. Source/default-main integration and functional CI have passed;
the new v0.2.0 tag policy/private-reporting/release/upload actions await final
approval. See [the 0.2.0 policy](RELEASE_POLICY_0_2_0.md).


The publisher uploads the already tested wheel. It does not rebuild a candidate,
run numerical work, or publish an sdist. The verifier runs with Python 3.11+ as a packaging tool; it is not imported by
the Python 3.10+ runtime package. The checked-in release manifest fixes
the artifact filename, length and SHA-256. `tools/verify_release.py` checks that
manifest, source version, tag, exact upload-directory contents and wheel metadata.

## Existing publishing configuration

The existing PyPI `gffx` project already has the following GitHub Trusted
Publisher, proven by the successful a1 publication. Reuse it:

| Setting | Value |
|---|---|
| Owner | `dependanz` |
| Repository | `gffx` |
| Workflow filename | `publish-alpha.yml` |
| Environment | `pypi` |

The existing Trusted Publisher and GitHub `pypi` environment are reused. Their
repository/workflow/environment identity remains unchanged. Final approval must
include adding only the `v0.2.0a2` deployment-tag policy while retaining the a1
policy. No new PyPI token or security-setting submission is needed. The publish
job retains only `contents: read` and `id-token: write` with pinned actions/OIDC.
The workflow at the new release commit is restricted to the a2 prerelease;
the old a1 tag retains its original frozen workflow and manifest.

## Approved release transaction

1. Review and approve the exact source diff, Git identity, commit message and
   existing `m1-camera` destination before committing/pushing the preparation.
2. Recheck the manifest's exact wheel hash and local validation receipts.
3. Create a **draft prerelease** `v0.2.0a2` at the approved commit and attach the
   exact tested wheel and reviewed release notes.
4. Verify the draft, target commit and remote asset digest before publishing it.
5. Publish the draft; GitHub creates a previously absent tag at this point.
   Immediately verify that `v0.2.0a2` resolves to the approved commit. The `release: published` event uses the tagged commit and
   triggers `.github/workflows/publish-alpha.yml`, including when a draft becomes
   a prerelease. The workflow downloads the attached wheel, verifies its approved
   bytes and metadata, then uploads it to PyPI using the Trusted Publisher.
6. Verify the workflow outcome and PyPI's final filename, size, SHA-256 and
   prerelease version. Record actual outcomes; never infer success from dispatch.

Draft staging ensures the asset exists before the publication event. This route
does not require merging the development snapshot into the default branch.
There is no push/PR trigger, manual dispatch, scheduled release, automatic rebuild,
skip-existing option, or general-purpose version publisher. A different version
requires a reviewed new manifest, workflow scope and publication transaction.

See [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [GitHub release events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release).
