# Security policy

The initial Windows0.2 release scope and maintained combinations are listed in
[the release policy](docs/RELEASE_POLICY_0_2_0.md). Candidate support is pending
stable admission; experimental combinations carry their stated limitations.

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
