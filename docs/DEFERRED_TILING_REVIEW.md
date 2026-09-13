# Deferred storage and tiling review

Status: design only, 2026-09-09. No tiling implementation or larger-render
admission follows from this note. Keep the 64 MiB renderer limit and the existing
whole-process supervision limits. Preserve the reference system's exactness gate.

## Distinguish three kinds of number

1. The current planner charges `256 * element_bytes + 32` per buffer pixel.
   This is a conservative accounting coefficient, not a measured allocation and
   not a proof that every charged byte must exist simultaneously.
2. Actual tensor storage can be inventoried by unique storage identity, with a
   separate account of autograd-saved storage and Python/native transients. Views
   must not count as copies; sequential scratch must not count as retained data.
3. Requested output storage is an unavoidable lower bound for the existing
   full-image return API. Float64 1920x1080 RGB plus mask alone is 66,355,200 bytes.
   This leaves only 753,664 bytes of 64 MiB for all inputs, resources and work.
   More output channels require more bytes. This is arithmetic, not a runtime peak.

## Current representation inventory

`_buffers` computes the following even when only colour and mask are requested:

- Packed geometry projection: clip `[V,4]`, NDC `[V,3]`, vertex validity, filtered
  faces and original-face mapping. These scale with geometry, not image pixels.
- Native raster products: fragment index int32, affine barycentrics `[B,H,W,1,3]`,
  depth and signed squared distance. K is one. Native backward may retain these
  and geometry; inspect its saved-input contract rather than assuming release.
- Perspective interpolation: mapped original face index, active mask, corner-w
  gather, numerator and denominator, corrected barycentrics. Public barycentrics
  and internal weights are views of the same corrected storage.
- World position, normalized surface normal, coverage, background-filled depth,
  material ID, UV/vertex-colour values and their presence masks. Some are eagerly
  zero-filled even when absent or not requested. Public face index and internal
  face index are views, not two independent arrays.
- Shading selection: each material/frame keeps a full boolean selection, gathers
  selected positions/normals/view vectors, interpolates attribute sources, creates
  light packets and material results, and scatters into a full RGB tensor.
  Principled's inner shader recomputes in four-point blocks, but full-image source,
  light integration and selection graphs can remain live until backward.
- Soft or supersampled geometry output may require a second hard centre raster.
  Colour/mask-only does not. Supersample reconstruction retains the explicitly
  ordered box reduction and must not filter discrete IDs or depth.

Removing unused public UV/colour/depth buffers and releasing unnecessary forward
temporaries are candidates, not yet proven optimizations. Every removal needs a
small red/green test, saved-storage accounting, numerical regression and a fresh
bounded peak measurement before reducing its planner allowance.

## Proposed next design checkpoint

1. Instrument only the already admitted tiny CPU profiles: unique output/internal
   storages and saved tensor storage identities, without retaining extra graphs.
   Compare forward-only and first-order backward. Keep process peaks separate.
2. Derive output-demand dependencies and eliminate genuinely unused arrays while
   preserving every requested channel and material source. Do not remove native
   raster products needed for backward simply because they are not public outputs.
3. For image tiling, review a native pixel-window raster entry point operating in
   the original global pixel/NDC coordinate system. Shifting camera intrinsics and
   rerasterizing a smaller image can change floating-point arithmetic and edge
   inclusion; it cannot be assumed bit-identical to the full raster.
4. Keep a fixed row-major tile order, original face IDs and tie-breaking, and the
   exact supersample box grouping. Soft coverage needs the same global signed
   distance and truncation as the reference. No unreviewed tile-edge approximation.
5. Full-image autograd concatenation alone does not bound retained graphs. Review
   tile-level first-order recomputation with complete original input/version
   tracking and independent derivative controls, including shared geometry and
   camera/light/material parameters across tiles.
6. Separately decide an explicit streamed-output interface for targets whose
   requested output storage cannot fit the inclusive envelope. A sink must have
   bounded ownership/backpressure and must not silently detach differentiable
   outputs. Keep ordinary `render(scene,camera)` semantics unchanged. Training may
   instead need an explicitly reviewed tile-loss reduction interface; an arbitrary
   user callback cannot be certified by checking only its returned shape.

No new public sink or loss callback is adopted by this document. Root integrates
the reviewed contract and consumer requirements before implementation. A safely
refused 1080p render is evidence of a safeguard, not completion of the M1 profile.
