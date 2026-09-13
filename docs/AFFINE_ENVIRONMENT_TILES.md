# Affine environment image-tile qualification

This is development verification, not a new runtime API or a full-image claim.
The prior sixteen-point reference remains frozen. The new `tools.affine_tiles`
harness owns a copy of fixed-material reference setup once per worker, while
every pixel still receives independent angular convergence and two actual public
material/light integrations. Tests require identical complete histories at all
sixteen previously qualified points and isolation from mutable caller/output data.

The prospective profile selects row-major visible-pixel tiles 0, 26 and 51 for
each material: 264 pixels each, 528 comparisons. Six serial CPU workers have a
900-second aggregate launch allowance, with the existing 120-second cooperative,
180-second watchdog and 1 GiB sampled working-set ceilings. Candidate chunks
remain 1024 light samples. No tolerance, resource table or installed wheel changed.
Moment work is counted once per principled worker, not once per embedded history.

The saved cohort is under `build/affine-tiles-20260908/cohort`. Hash-bound replay
checks exact sources, wheel, immutable geometry, pixel ownership, histories,
both estimates, work counts and actual-worker supervision. Devbrain's shading
contract section 8.27 and numbers ledger own the measured results and hashes.
Never overwrite the cohort or interpret a partial result as a completed image.

Testing pages:

- `shading/affine-tiles`: setup, safety, corruption and saved-evidence tests.
- `shading/affine-tiles-demo`: finite selected-pixel diagram and measured costs.

Green means measured in this qualification; gray means not computed here,
including background. This is not a shaded-head image. The time plot separates
fixed setup, per-pixel reference, actual public sampling and total worker time.
These are laptop measurements, not throughput guarantees. Remaining full images
need their own prospective budget. Arbitrary HDR maps, varying materials, image
gradients, CUDA/export and M1 acceptance remain separate work.
