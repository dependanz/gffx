# Fixed coloured-environment image verification

This is developer acceptance evidence for two fixed CPU-float64 images, not an
arbitrary environment-map renderer or a real-time performance claim.

The 128×128 immutable synthetic fixture has 6,536 covered pixels. Every pixel is
evaluated independently for classic and principled materials under the fixed
world-space affine RGB field. The material colour and other parameters stay fixed.
The reference retains its independent convergence history; the public candidate
integrates the complete material with two angular estimates. Backgrounds must
match exactly; linear RGB uses elementwise absolute tolerance 1e-10 and relative
tolerance 1e-8. No image interpolation substitutes for a missing comparison.

`tools.affine_images` completes only the 98 tiles absent from the earlier six-tile
qualification. Seventeen prospectively bounded batches preserve immutable
checkpoints. The final replay verifies their hashes, exact pixel ownership,
convergence histories, numerical comparisons and reconstructed image bytes.
Never overwrite a report or restart a completed tile to reset its clock.

The completed cohort is `build/affine-full-20260908/cohort`, SHA-256
`e029d65fcd9bcec43c6c92e8c4f0ae9fa82e930b04239aca4c0d7c2f3267c58d`.
Its original isolated wheel is `1c97f8df…`; newer CUDA artifacts do not replace that
identity. Exact wheel/source hashes and cumulative measurements are in devbrain's
`projects/gffx/NUMBERS.md`. All 104 parent/new tiles passed; full replay and
presentation tests passed before publishing the complete comparison.

The finite dashboard page is `shading/affine-images-demo`; its companion tests
are under `shading/affine-images`. Earlier progress snapshots remain history.
Gray in a partial snapshot means uncomputed, not a measured black pixel.

Other albedo fields, image-level derivatives, general environment maps and broader
device/export profiles still require their own evidence. The synthetic head is a
numerical fixture, not anatomical quality, Cycles parity or gffit acceptance.
