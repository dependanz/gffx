# Independent fixed-head silhouette gate

Prospective DF04 edge fixture,2026-09-13, fixed before oracle execution. Reuse only
the original synthetic_face_m1 geometry and declared pinhole inputs. Do not reuse
native fragments, coverage, IDs or interpolated attributes as the reference.

Use80-digit Decimal arithmetic to intersect pixel-centre camera rays with each
front-facing triangle plane, then classify the hit using three oriented 3D
half-spaces. Restrict to the finite128-square,1986-vertex/3968-triangle fixture and
its original near/far range; no clipping or arbitrary-corpus claim. Screen bounds
only skip impossible candidates; all acceptance calculations stay in world space.
The oracle produces coverage only, not shading, depth or a universal rasterizer.

Require the entire16384-pixel hard mask to match exactly on CPU and CUDA. This also
fixes every silhouette-edge pixel without inventing an image tolerance. No ambiguous
pixels are silently excluded. A closed analytic square, reversed winding and known
one-column translation qualify the oracle first; a one-pixel altered mask must fail.
Keep this separate from the existing independent scalar material/shared-raster
image comparisons, soft-edge finite differences, and visibility semantics.

Execute serially with the existing1GiB CPU /2GiB CUDA host,64MiB GPU,120s cooperative/
180s watchdog limits. A90s oracle-local checkpoint stops before later native work.
No reference criterion may be weakened after seeing a comparison failure.
