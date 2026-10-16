# Bounded raster block rejection

Design and implemented candidate,2026-09-14;approved M1 scheduling exception. Preserve original
floating-point operations for every evaluated pixel and ascending face order.
The frozen window40 scan is the regression reference. No new public operation,
heap storage,workspace size,cache,material limit or output convention is introduced.

CPU traversal initializes outputs once,then scans disjoint32-by32 pixel blocks.
For each block it scans faces in their original ascending order,computing projected
triangle coordinates once per face/block. The original per-pixel edge,area,
barycentric,distance,depth and insertion statements are retained. Different pixels
never share an accumulation;face reordering within a pixel is forbidden.

A conservative rejection requires BOTH (a) guaranteed outside barycentric coverage
and (b) guaranteed boundary distance greater than the original rounded blur square.
Only finite projected coordinates and pixel coordinates with magnitude<=2^20
qualify. All other inputs follow the original scan,including overflow/extreme cases.

For (a),fix the rounded projected vertex coordinates. The exact-real edge
expression is affine in pixel coordinates,so extrema on a rectangle occur at its
four corners. Let M>=1 bound all vertex and rectangle coordinate magnitudes.
The original double evaluation has four subtractions,two products and a final
subtraction. With unit roundoff2^-53,coordinate differences<=2M and products<=4M^2,
its absolute error is below64*2^-53*M^2,including a conservative subnormal allowance.
Use E=2^-44*M^2 (512 unit roundoffs),and require the computed corner minimum to
exceed2E or maximum to be below-2E. This covers error at both corners and any
interior pixel. If one edge is uniformly positive and another uniformly negative,
the original finite area division cannot give three nonnegative barycentrics.
The strict edge margin prevents a negative quotient rounding to signed zero.
Otherwise no block rejection is permitted. No replacement constant area is used.

For (b),the segment routine clamps its parameter to[0,1]. With finite coordinates
bounded by2^20,its computed closest-point coordinates stay within the vertex AABB
plus far less than2^-20 pixels,including rounded subtraction,multiply,add and
pixel-minus-closest-point error. Use PAD=2^-20 pixels as a conservative combined
reserve. Find a rectangle/vertex-AABB gap on either axis. Only if gap>PAD and
fl((gap-PAD)^2)>the original blur_squared may distance be excluded. Rounded square
and nonnegative distance summation are monotone;the reserve covers bound arithmetic.
The same distance check can skip expensive segment work AFTER the original inside
predicate is false at one pixel,on CPU/CUDA. Never use an AABB alone to infer the
floating-point inside predicate. Nonfinite or unbounded cases fall back unchanged.

All helpers have fixed-size automatic scalars,under1 KiB CPU frame allowance;
no candidate list or per-face array is retained. Existing zero CPU raster scratch
and output ownership remain. CUDA retains its existing per-pixel face traversal
and only gains the proven outside-distance shortcut in this first stage.

Qualification: capture frozen original raw bytes for deterministic odd grids,
windows,culling,ties,degeneracy,large/small coordinates and both precisions/devices.
Take a missing-helper rejection red,check the exclusion proof against independent
high-precision edge arithmetic and the original scan,then compare the rebuilt
installed wheel with frozen outputs. Retain exact gradients,resource and ABI gates.
A small source proof or kernel timing alone never qualifies a large full render.

Cohort42 keeps the raster41 native wheel and only changes the measurement helper.
It assembles the already admitted cohort once for pointwise light preparation and
integration. Each original four-point albedo leaf still passes through the same
material layout,grouped native VJP and adjacent-pair broadcast reduction. The
returned partials are accumulated in the original reverse global group order.
No simultaneous point/pair limit grows. The additional packed normal,position and mask copies plus concatenated RGB
BRDF occupy at most128*(3+3+1+3)*8=10240 bytes,within the existing32768-byte
carry reserve. Other per-point temporaries replace the original separate-group
arrays rather than increasing their combined extent;the native
material and full outputs remain charged separately. The repeated-allocation test
continues to measure the complete live window,not just the new helper arrays.


## CPU face-major traversal

Stage48 prepares the same pixel-space vertex coordinates and conservative box
once per face per window, then visits that face's existing 32-by-32 rectangles.
For any fixed output pixel, faces still arrive in ascending order. Fragment
initialization, floating-point edge/distance expressions, strict depth insertion,
tie-breaking and global window coordinates are unchanged. Pixels own disjoint
fragment lists, so changing the order in which different pixels are visited has
no effect on a valid call's values. CUDA traversal is unchanged.

Only automatic scalar constants survive between rectangles; the existing sub-1-KiB
CPU frame bound and zero caller raster workspace remain. There is no retained
geometry cache, allocation, face list, new bound, new predicate or gradient rule.
Frozen original scan fixtures precede this optimization: run their exact-byte
CPU/CUDA float32/float64, packed/empty-mesh, tie, blur and offset-window checks
against the freshly built installed wheel, followed by stream/VJP/resource controls.
Stage-cost profiling is diagnostic only and cannot replace matching acceptance
anchors. Cold autograd imports and previous slow timings remain in the evidence.


## Hierarchical CPU rectangles (scale50)

The same conservative predicate accepts arbitrary closed pixel-centre rectangles,
not just32x32 leaves. Start from the whole current window. If the predicate cannot
exclude it, bisect its larger integer extent until both extents are at most32.
The original pixel equations run in each surviving leaf. The tree partitions the
window exactly once, with no overlapping/gapped leaves. Faces still advance in
ascending order for each disjoint pixel list. All failed bounds still fall back
to evaluating the original pixels. No new geometric predicate or tolerance exists.

Traversal uses a fixed128x4 int64 stack (4096 B), no heap or retained cache.
Each positive signed-int64 extent needs at most58 halvings to reach32, hence
depth at most116 and depth-first pending storage at most117 entries. Midpoints
use left+(right-left)/2, avoiding addition overflow. The previous sub1-KiB frame
plus this stack fits the existing32768 B streamed carry reserve and is included
in the supervised process peak; tensor scratch and caller workspace stay unchanged.
Empty windows exit through the existing precondition path. CUDA code is unchanged.
Frozen original broad fixtures and newly captured thin/wide1080p windows must
match byte-for-byte on the new installed wheel before scale admission.

## CUDA cooperative rectangle rejection (opaque202)

Batch-one calls prepare 256 consecutive faces per existing 256-thread pixel block.
Their exact projected coordinates and the existing conservative exclusion test are
shared across that block. A stable ascending compaction retains every uncertain
face. Surviving pixels execute the unchanged edge, area, division, distance, depth
and insertion statements. Rectangles include all active pixel centres, including
blocks crossing rows and globally offset windows. Packed batches retain the old
kernel. Padded inactive tail lanes reach every barrier; no output is written for them.

Static shared hardware storage is 256*9 doubles, two 256-int arrays and one count:
20484 bytes per hardware block. No global allocation or caller workspace changes.
No new candidate cap is imposed: all faces are processed in consecutive chunks.
The existing rectangle proof applies unchanged; nonfinite/extreme/uncertain bounds
retain the original calculations. Frozen pre-change byte fixtures and complete
installed-wheel CPU/CUDA qualification are required before performance admission.

### Spatial cooperative blocks,2October2026

The batch-one CUDA block now covers a clipped16x16 pixel rectangle. It retains
256lanes,20484bytes static shared memory,ascending face compaction and unchanged
per-pixel arithmetic. Partial rows and columns have an explicit active predicate;
every lane still reaches every barrier. Dimensions are at mostINT32_MAX and the
product of ceil-divided tile counts is checked againstINT32_MAX before multiplying
by256. Unsupported extents and packed batches retain the original kernel. The
same conservative rectangle exclusion applies to the tighter spatial rectangle;
no additional face is excluded without that predicate proving it irrelevant.
