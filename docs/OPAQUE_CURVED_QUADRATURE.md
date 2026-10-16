# Bounded native curved quadrature

The opaque quadrature primitive accepts optional `curves[F,6]`, containing
homogeneous barycentric quadratic coefficients in the order
`q00,q11,q22,2q01,2q02,2q12`. The polynomial is
`q00*b0^2 + q11*b1^2 + q22*b2^2 + 2q01*b0*b1 + 2q02*b0*b2 + 2q12*b1*b2`.
A zero row explicitly disables that face's curved cut. It is padding, not a
claim that an identically grazing physical horizon has a defined derivative.

The existing preallocated C ABI functions also accept cuts `[F,1,3*K+6]`:
K linear corner columns flattened in corner-major order, then the six conic
coefficients. K must be at most15, keeping the total cut allowance16.
The old `[F,3,K]` layout and ordered arithmetic are unchanged.
Backward returns the same coefficient shape. The thin Torch adapter calls the
same C functions; CUDA executes the shared scalar-dual implementation on-device.
No Torch-only integration, host geometry fallback, I/O or persistent cache is used.

After ordinary visibility and linear cuts, each fan triangle is tested using
outward binary64 interval arithmetic. One-sign Bernstein controls exclude a
crossing. A crossing needs one strictly isolated-sign vertex, a strictly
one-sign opposite edge and a strictly monotone radial derivative. Guards certify
the represented polynomial on the computed triangle; they do not prove exact
real-arithmetic clipping or the quadrature error. A numerically shifted decimal
clip vertex is not an exact vertex-event fixture. Exact dyadic event controls
exercise refusal. Tangencies, uncertain topology and capacity overflow refuse.

From the isolated vertex A toward the opposite edge B--C, the conic root is
`r(s)=-2*a/(b(s)+sqrt(b(s)^2-4*a*c(s)))`. The two maps are
`t=u*r(s)` and `t=r(s)+u*(1-r(s))`. The Jacobian is
`abs(det(B-A,C-A))*t*dt/du`. Scalar duals differentiate the root, node coordinates,
geometry and Jacobian. No boundary-motion contribution is dropped or zeroed.

Each side uses Gauss12 angularly. A geometry-only count pass chooses Gauss12
radially when every curved piece fits the existing 800 slots; otherwise it uses
Gauss5 radially. After preserving that curved choice, remaining sample capacity
selects one inner Gauss order5through8 for every ordinary triangle, with outer
Gauss5 retained (25through40nodes per flat fan). Each conic still charges five
of the 32 logical blocks, and actual sample writes separately enforce 800 slots.
The count and forward emission share the existing work guard. Backward reuses
the choice from unseeded geometry for all scalar-dual derivative seeds.

Radial samples concentrate at the horizon: the inner side uses
`u=1-(1-t)^2`, the outer side `u=t^2`. Their Jacobians `2*(1-t)` and `2*t`
multiply the full integration measure. There is no face/material special case
or new allocation. Sparse pixels can use 288 samples per conic; crowded pixels
retain 120. This finite rule still needs independent image and gradient bounds;
its capacity choice is not an integration-error certificate. Only first
derivatives are supported; transitions between discrete topology/capacity
choices are not independently qualified by fixed-topology controls.

Scene view horizons carry the actual quadratic from perspective homogeneous
normal/view fields. Material and directional-light cuts remain linear. Geometry
accounting keeps576 scalar slots per face for unknown/attribute/texture-cut
profiles. Of these,192 reserve four three-corner copies of16 cut columns. For
constant materials,the maximum is known from camera batches and directional
sample counts: one view column plus one per light sample per batch. Remove only
impossible columns,keeping the other384 slots. The original single-camera,
single-light profile therefore reserves408 slots/face,32640000 bytes for10000
faces. The active64MiB allowance,backward planner and input/output charges remain.

Qualification records live outside the source tree under the danziepc resource
mapping. Opaque62 preserves failed one-strip complete-pixel and two-strip2x2
frame gradient tests. Finite control materials use roughness=F0=1,albedo=0;
they do not qualify the original mesh's full F0=.04/RGB material or the final
SIGGRAPH-quality showcase. The complete-pixel oracle independently composes
corner normalization using90-digit arithmetic, then bounds its explicit rounded
packed-input model. Existing normalization acceptance and native tolerance remain
separate from the oracle's1e-12 interval radius.

Opaque63 evaluates the actual original material on the complete pixel(101,78),
including all visible pieces,through the public installed CPU/CUDA path. Its
colour,mask and constant-albedo Jacobian agree byte-for-byte. This is execution
and device evidence,not yet independent full-material accuracy. The new direct
square reference passes closed-integral and horizon-loss controls,but the full
original material exhausts256 cells on its first visible piece. Preserve the
original memory/capacity refusals and every failed reference attempt. Complete
original frames and the final showcase remain unqualified. The fixed-geometry
reference does not certify moving geometry/view/normal or angular jump transport.


## Original material accuracy checkpoint — 24 September 2026

Opaque120 completed all 13 independent original-piece certificates. Opaque121
localized the remaining native discrepancy to radial integration. Opaque122
qualifies a fresh wheel against both endpoints of the complete original
pixel colour/constant-albedo reference on CPU and CUDA, with exact device bytes.
The five-node-only coordinate change failed an existing equal-normal horizon
motion control. The capacity-counted12/5rule passes that regression and the
original accuracy checks together. Failed candidates and the build exceeding
1GiB remain preserved. Sparse/dense area, boundary motion, all-input gradients,
float32/64device bytes and strict capacity refusals are checked separately.
This closes selected fixed-geometry pixel accuracy, not moving original-material
fields, complete original frames or the final SIGGRAPH-quality showcase.


## Flat shading refinement candidate — 30 September 2026

Opaque152 independently bounds all three packed-input motion directions of the
three visible pieces on face5740. The installed Gauss5x5normal gradient fails
the unchanged tolerance, while research-only Gauss5x8flat resampling passes
normal,view and attached-screen comparisons. A weaker5x6rule fails the same
normal comparison; higher order is not a general nonlinear-error guarantee.
Opaque153 implements the capacity-counted shared native candidate. Original
fixed-pixel and complete-face CPU/CUDA acceptance, existing curved/horizon
regressions and exact first-VJP device bytes remain required. Fixed-topology
controls do not qualify transitions between discrete capacity choices or full
frames. No acceptance or resource limit is relaxed.
