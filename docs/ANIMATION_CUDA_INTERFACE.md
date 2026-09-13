# Animation CUDA interface extension

This is an implementation design, not a CUDA qualification result.

The private plugin handshake stays at v1.0 and its outer size stays unchanged. The
animation extension appends ten function pointers after the existing operations
table's reserved tail. No existing field offset changes. Operation IDs 14 through
18 identify blend shapes, Rodrigues, forward kinematics, LBS, and barycentric
evaluation respectively. Old operation IDs retain their meaning.

Every new core call checks that the provider's struct_size covers the specific
function-pointer field before reading it, and then checks that pointer for null.
A shorter or incomplete provider returns UNSUPPORTED. The loader's historical
minimum size remains unchanged; unrelated operations in an older provider must
remain loadable. No old reserved slot is reinterpreted.

Dedicated animation kernels and plugin glue use the CUDA Driver API already used
by gffx. They introduce no additional library dependency. Tensor values are never
copied to the CPU to perform geometry or differentiation. Metadata validation must
precede launches; finite-value, topology and hierarchy validation run on the
caller's stream before output writes. Only small validation status words may be
read back. Empty views, byte offsets, output aliasing, scratch size and integer
overflow retain the public C ABI rules. Device-wide synchronization is forbidden.

Reference reductions have fixed gather order, without floating-point atomics.
CPU/CUDA forward and gradient results are compared byte for byte using inputs and
cotangents created once on the CPU and transferred unchanged. Independently
recomputing cotangents on each device would confound operation parity with fixture
construction differences. Rodrigues transcendental parity is an explicit measured
gate: host and CUDA sin/cos are not assumed identical, and tolerance relaxation or
a hidden host numerical fallback cannot close that gate.

Tests cover absent/short/null providers, malformed descriptors, current-stream
execution, all floating input gradients, both dtypes, empties and scratch errors.
Qualification retains the 2 GiB CUDA worker host limit, 64 MiB GPU allocation cap,
120-second cooperative deadline and 180-second watchdog. Large all-input float64
blend-shape gradients that exceed the GPU cap remain inadmissible, not skipped
successes. CPU limits remain unchanged.
