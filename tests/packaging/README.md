# Packaging tests

`test_metadata_contract.py` pins the static package identity, dependency groups, and truthful
documentation boundaries established in Phase 1 Step 10.

`test_foundation_verifier.py` tests the dependency-free Step 11 verifier itself. Its synthetic bad
inputs prove that undeclared dependencies, bundled optional providers, generated files, prototype
namespaces, and credential-like paths/content are rejected without echoing secret values.

The end-to-end gate is:

```powershell
python tools/verify_foundation.py `
  --work-dir build/foundation-verification `
  --adapter-python C:/path/to/pytorch-2.10-plus-environment/python.exe
```

The work directory must not already exist. It contains the clean source snapshot, native build,
wheel, sdist, clean-install environment, expected-failure builds, and machine-readable report.
