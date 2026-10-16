# Publishing the tested 0.2.0a1 alpha

The publisher uploads the already tested wheel. It does not rebuild a candidate,
run numerical work, or publish an sdist. The verifier runs with Python 3.11+ as a packaging tool; it is not imported by
the Python 3.10+ runtime package. The checked-in release manifest fixes
the artifact filename, length and SHA-256. `tools/verify_release.py` checks that
manifest, source version, tag, exact upload-directory contents and wheel metadata.

## One-time configuration

After explicit publication approval, configure the existing PyPI `gffx` project
with a GitHub Trusted Publisher:

| Setting | Value |
|---|---|
| Owner | `dependanz` |
| Repository | `gffx` |
| Workflow filename | `publish-alpha.yml` |
| Environment | `pypi` |

Create the GitHub `pypi` environment with a custom deployment tag policy limited
to `v0.2.0a1`. The workflow grants the publish job only `contents: read` and
`id-token: write`, uses pinned actions and OIDC, and stores no PyPI API token.

## Approved release transaction

1. Review and approve the exact source diff, Git identity, commit message and
   existing `m1-camera` destination before committing/pushing the preparation.
2. Recheck the manifest's exact wheel hash and local validation receipts.
3. Create a **draft prerelease** `v0.2.0a1` at the approved commit and attach the
   exact tested wheel and reviewed release notes.
4. Verify the draft, target commit and remote asset digest before publishing it.
5. Publish the draft. The `release: published` event uses the tagged commit and
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
