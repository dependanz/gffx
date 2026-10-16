# Global pixel-window rasterization

The forward window is an additive native entry. It receives original image
height/width and integer row/column offsets separately from window height/width.
The existing CPU implementation and CUDA kernel retain their original edge,
barycentric,depth,distance,coverage and face-insertion expressions. Only pixel
iteration/output indexing use window dimensions. Vertex-to-pixel conversion uses
original dimensions and each sample is (global_column+.5,global_row+.5).

The ordinary raster entry passes its original dimensions and zero offsets into
the same arithmetic. The CUDA operation table adds a size-checked tail field;
older providers remain usable for ordinary rasterization and report unsupported
for a nontrivial window. There is no implicit device copy or callback.

The initial window entry is forward-only. The private deferred buffer path
refuses gradient-bearing projected geometry before dispatch. This supports the
approved fixed-scene constant-albedo streamed qualification,whose material VJP
is downstream of rasterization. It does not claim window geometry derivatives.
The complete original renderer retains its existing geometry backward.

Reject nonpositive or out-of-frame windows before output allocation. Outputs
contain only the window. Workspace uses the existing raster query. Test every
output byte against full-frame slices on CPU/CUDA,float32/64,odd dimensions,
offsets,short windows,culling,soft edges and equal-depth face ties. Then verify
the deferred interpolation and colour,not only raw fragment indices.


## Streamed measurement storage and ownership

`tools/deferred_stream.py` is the separately approved caller-owned measurement
profile,not a public arbitrary-loss renderer. It selects the largest row window
that fits64 MiB while preserving the calibrated four-point shading partition.
The ordinary complete-frame renderer and its inclusive output bound are unchanged.

The existing shape planner is evaluated for the window extent only;projection
uses the original camera and full raster dimensions. Every original scene/resource
storage remains in the input charge. Packing retains the existing96-component
vertex and face/attribute reserve. The window buffer charge remains256 scalar
components plus32 bytes per pixel;lighting keeps128 forward or512 backward scalar
components per pixel/light. No such term is reduced for streaming.

During forward,one window's buffers and colour coexist with the caller destination
until the disjoint CPU copy completes. They are then released. During backward,
one recomputed window is replaced by compact normal/world/mask records (seven
float64 components per covered pixel),before material work. The original buffer
and lighting reserves cover that overlap and the per-window selections. Native
material scratch retains the GROUPED_STORAGE_BOUND.md bound and the selected
window's admitted cohort count. A backward cohort cannot exceed either that count
or the selected window pixel extent,even when groups span two windows.

An additional32768-byte reserve is removed before planner admission. A queued
cohort holds at most128 cloned seven-component records (7168 bytes),an incomplete
group holds at most three (168 bytes),and the scalar albedo gradient/cotangent
are fixed size. Queue preparation/cotangent/partial point storage is already in
the native192-per-point/light backward reserve. The single scalar prepared curve
uses the existing angular margin,as documented for shared38. No tile graph list,
full device output,concatenated complete output,memmap or persistent curve cache
is retained. Window metadata/counts and Python object overhead are process costs,
not device tensor payload;the complete supervised process retains its1/2 GiB cap.

The CPU caller allocates exactly one complete RGB and mask destination,32 bytes
per pixel. Both count against the complete process peak. Output tensor identities
and versions,original input identities and versions,and camera extents/pose are
checked between windows and before success. Interrupted outputs are invalid.
The initial contract qualifies constant RGB albedo gradients only. Geometry
backward through native windows is unsupported and broader M1 gradients remain
separate. Timings include the complete output allocation,copies and second pass;
a fast tile is never reported as a completed full frame.

Small window40 evidence:18 raw/deferred checks and12 streaming controls pass;
six postwarm CUDA calls retain12089856 bytes each,with5110272 incremental peak
against27880312 planned bytes. These small measurements do not by themselves
qualify large jobs or prove bounds for unsupported scene types.

## Fixed-group calibration profile

The optional measurement-only `group_point_limit=128` constrains `plan_windows`
and `stream_into` to the already qualified 32 groups of four points. The probe
selects it with `--group-point-limit 128` and records it in the profile. The
default stays unchanged. A requested explicit window that cannot retain this
capacity fails before rendering or writing destinations. Automatic selection
finds the largest window that satisfies both memory and fixed-capacity constraints.

Choosing the largest window without this constraint can squeeze native material
scratch into a smaller group capacity. For the fixed 64-square backward fixture,
the old full-height plan allows 100 points; the pinned profile selects 46 rows
and retains 128. Calibration and target must use the same declared group capacity;
the forecast must not discard that identity check to reuse mismatched timings.

This changes scheduling only. Global pixel coordinates, original four-point
partials and reverse accumulation remain. Window ownership, input mutation checks,
32 KiB carry reserve, 64 MiB renderer allowance and process/deadline limits remain.
No general tile API, arbitrary loss or geometry-window VJP is qualified by it.
See tests/python/test_deferred_stream_fixed_groups.py for maximal-window refusal,
exact CPU/CUDA pixels and gradients, invalid profiles and untouched destinations.
