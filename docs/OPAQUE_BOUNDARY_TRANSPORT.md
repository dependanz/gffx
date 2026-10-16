# Opaque principal-point transport candidate

For a fixed scene, changing camera cx/cy translates its screen radiance field.
The first derivative of a unit-box integral is the integral along its entering
edge minus its leaving edge (left-right for cx; top-bottom for cy). This includes
visibility and material horizons. It does not replace arbitrary geometry VJPs.

A new C entry `gffx_render_opaque_boundary_quadrature` has the quadrature inputs
plus edge=0(left),1(right),2(top),3(bottom). Outputs retain owners[P,800] and
samples[P,800,4]; the fourth channel is nonnegative edge-length weight. Padding
is -1/zero. It reuses `gffx_render_opaque_quadrature_workspace`. The caller owns
all storage; metadata, finite-domain, alias and device checks remain mandatory.
At most32 AABB candidates,128 temporary visible intervals,32 final edge segments,
16 combined material/conic cuts,800 slots and2000000 interval operations. Each
segment uses Gauss12. Refuse coincident silhouette edges and uncertain/tangent
conic events. Smaller depth wins, original face order breaks identical planes.

Shared C11 arithmetic executes on CPU or the optional CUDA provider, whose new
operation is an appended, size-gated tail. Python exposes the operation through
both facades and rejects differentiable geometry/cuts: these nodes supply a
first camera VJP, not higher derivatives. No new shader or smoothing is added.

Production composition detaches only cx/cy in projection, evaluates boundary
radiance before retaining the ordinary volume graph, and attaches the edge
flux as the cx/cy first VJP. Other parameter graphs remain connected. Forward
values, material resources, accumulation policy and acceptance stay unchanged.
Boundary buffers reuse the800-sample workload sequentially; retained flux and
backward products must be explicitly charged in resident and streaming budgets.

Gauss12 is not a universal integration-error certificate. Admission requires
independent polynomial/visibility controls, original-material endpoint bounds,
unchanged forward values, explicit resource refusal, installed-wheel tests and
separate exact CPU/CUDA checks. General moving geometry remains unfinished.
