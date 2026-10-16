# Security policy

The Windows 0.2.1 release scope and maintained combinations are listed in
[the current policy](docs/RELEASE_POLICY_0_2_1.md). Exact release gates determine
admission; experimental combinations retain their stated limitations.

Fresh installations select Torch 2.14.1. The retained eager 2.10 compatibility
profile is upstream-unmaintained and has the TorchScript advisory
[GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47). GFFX does
not invoke TorchScript; compilation/export is excluded from this admission.
Retained compatibility is not an upstream security-maintenance promise.

Report suspected vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/dependanz/gffx/security/advisories/new).
Do not post exploit details or credentials in public issues. Private reporting
must be enabled before stable publication; the release gate checks it.

Include the package version, wheel/platform/Python/framework identity, minimal
reproduction, expected impact and relevant diagnostics without private data.
Scope includes native memory safety, malformed tensor/file inputs, loader and
adapter boundaries, release-artifact integrity and publishing infrastructure.

We target acknowledgement within7days, initial assessment within14days and
updates at least every14days while unresolved. These are best-effort targets,
not a commercial SLA. Validated serious fixes receive a new release/advisory;
published files are never silently replaced. Supported API removals normally
receive two minor releases and six months of notice; platform/framework removals
receive two minor releases and12months, subject to documented security/legal
exceptions. Experimental paths have no compatibility or release-timing promise.
