# Migrating to gffx0.2

The0.1.x public packages are an inherited prototype with different APIs.0.2 is
the native C11/optional framework implementation; upgrading the package does not
rewrite old application code. Review the current operation modules and shapes,
explicit int32 topology, float32/64 types, same-device tensor contract, local
gradient conventions and error behavior before changing a production project.

Base `import gffx` stays dependency-free and lazy. Activate `gffx.torch` explicitly
after selecting Torch2.10.0. Use explicit CUDA inputs for GPU work. Alpha1 shipped
CPU support only; alpha2 and the0.2.0 candidate include the optional CUDA provider.
The stable admission table is narrower than all available experimental modules.

Start in a new environment, install the selected framework and pinned wheel,
run representative application tests, then update the project's lockfile.
The release validation checks package-file upgrade/uninstall cleanup from0.1.4,
a1 and a2; it does not claim source/API compatibility with0.1.x or validate every
consumer pipeline. Published alpha artifacts and saved gffit baselines remain intact.
