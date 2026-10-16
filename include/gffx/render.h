#ifndef GFFX_RENDER_H
#define GFFX_RENDER_H

#include <gffx/execution.h>
#include <gffx/tensor.h>

/*
 * render.rasterize - triangle rasterization to per-pixel fragments.
 *
 * Row zero is the top of the image and sampling is at pixel centres, so pixel (r, c) samples the
 * continuous point (c + 0.5, r + 0.5). NDC maps to pixels as
 * pixel_x = (ndc_x + 1) * W / 2 and pixel_y = (1 - ndc_y) * H / 2. All coverage, distance, and
 * barycentric arithmetic happens in pixel space, because signed_distance is contractually in
 * squared pixel units and a non-square image scales the axes differently.
 *
 * Coverage uses the edge functions e0, e1, e2 whose sum is the signed doubled area; the pixel is
 * inside when all three barycentric weights w_i = e_i / area2 are nonnegative. A face is
 * rasterized only when the absolute area exceeds eps in pixel units. Culling is decided from the
 * NDC orientation, which is the negation of the pixel-space area because the y-flip reverses
 * handedness: a face is front-facing when its NDC signed area is positive.
 *
 * A face is a candidate for a pixel when it is inside, or when it is outside and the distance to
 * the nearest triangle boundary is at most blur_radius_px. Candidates are ordered by
 * (depth, global_face_index) ascending with depth = w0*z0 + w1*z1 + w2*z2, so the nearest
 * fragment occupies slot zero and an exact depth tie resolves to the lower face index. Unfilled
 * slots carry face_index -1, zero barycentrics, +inf depth, and +inf signed distance.
 *
 * signed_distance is the squared pixel distance to the nearest triangle boundary, negated when
 * the pixel is inside, so it still varies inside a large triangle and can drive an alpha ramp.
 *
 * Backward differentiates barycentric, depth, and signed_distance with respect to ndc_vertices,
 * holding the selected face, its coverage state, and its nearest-edge region fixed. Face
 * selection, visibility and coverage changes, the inside sign, the nearest-edge identity, K,
 * image_size, blur_radius_px, and cull_mode are all nondifferentiable. Workspace is zero bytes
 * for both directions.
 *
 * render.interpolate - barycentric interpolation of per-face-corner attributes.
 *
 * attributes[b][h][w][k][:] = sum over i of barycentric[b][h][w][k][i] * face_attributes[f][i][:]
 * for the fragment's face f, and exactly zero for a background fragment, which is the same
 * face_index == -1 condition the rasterizer writes. The operation is bilinear in its two
 * differentiable inputs, so its backward is exact: the attribute gradient is a barycentric
 * weighted scatter and the barycentric gradient is a contraction against the attributes.
 * Workspace is zero bytes for both directions.
 */

#define GFFX_CULL_NONE UINT32_C(1)
#define GFFX_CULL_BACK UINT32_C(2)
#define GFFX_CULL_FRONT UINT32_C(3)


/*
 * render.texture_pyramid and render.texture. Semantics and the sixteen acceptance fixtures are
 * fixed by TEXTURE_ACCEPTANCE_V0_1.md; API_CONTRACT_V0_1.md section 4.7 owns the public schema.
 *
 * Coordinates are normalised to [0,1] with (0,0) at the first texel of the first row and v
 * increasing with row index, so the output of gffx_render_interpolate is a valid input unchanged.
 * At most one of derivatives or lod may be supplied; with neither, level 0 is read. The operation
 * never derives its own screen-space derivatives, because that would make each sample depend on
 * its neighbours and break both per-element independence and unstructured point sampling.
 *
 * Every output element depends only on its own coordinate and the texture, so both operations are
 * order-independent and CPU/CUDA results are bit-identical. The pyramid reduction order is fixed
 * rather than left to a parallel schedule for that reason.
 */

#define GFFX_FILTER_NEAREST UINT32_C(1)
#define GFFX_FILTER_BILINEAR UINT32_C(2)

#define GFFX_MIP_NEAREST UINT32_C(1)
#define GFFX_MIP_LINEAR UINT32_C(2)

#define GFFX_WRAP_REPEAT UINT32_C(1)
#define GFFX_WRAP_CLAMP UINT32_C(2)
#define GFFX_WRAP_MIRROR UINT32_C(3)
#define GFFX_WRAP_BORDER UINT32_C(4)

GFFX_EXTERN_C_BEGIN

/* Experimental CPU opaque alpha: area of the union of pixel-space triangles.
 * x right, y down, integer pixel corners; z is finite but does not affect alpha.
 * No culling or depth clipping here. Caller supplies projected accepted geometry.
 * Unit box footprint, optional global pixel origin, ordered first VJP. At branch
 * events derivatives follow the selected clipping topology (no smoothness claim).
 * At most32 conservative triangle-AABB candidates per pixel;64 polygon vertices,
 *128 visible pieces and2000000 clip-edge visits per pixel evaluation. Overflow
 * fails explicitly, with output contents unspecified. No hidden allocation.
 * CPU float32/float64; coordinates within +/-1048576; image axes <=1048576.
 * Not a colour renderer, supersampling scheme, CUDA or scale-qualified operation.
 */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_coverage_workspace(
    gffx_dtype dtype,const gffx_execution_context *context,uint64_t *required_bytes,
    uint64_t *required_alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_coverage(
    const gffx_tensor_view *pixel_vertices,const gffx_tensor_view *faces,
    int64_t height,int64_t width,double origin_x,double origin_y,
    const gffx_execution_context *context,gffx_tensor_view *coverage,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_coverage_backward(
    const gffx_tensor_view *pixel_vertices,const gffx_tensor_view *faces,
    int64_t height,int64_t width,double origin_x,double origin_y,
    const gffx_tensor_view *grad_coverage,const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);


/* Depth-ordered box integration of screen-affine corner radiance.
 * Vertices[V,3] in pixel coordinates with affine depth (smaller wins), faces[F,3],
 * corner_radiance[F,3,3]; output[H,W,4] is premultiplied RGB and alpha.
 * Original face order wins identical depth planes. Selected topology first VJP;
 * no derivative claim at depth ties or other topology events. CPU f32/f64 only.
 * Same coordinate/candidate/polygon/work bounds as opaque_coverage. All scratch
 * belongs to the caller; errors leave outputs unspecified. No hidden allocation.
 * This integrates the declared affine field, not a nonlinear material shader.
 */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_integrate_workspace(
    gffx_dtype dtype,const gffx_execution_context *context,uint64_t *required_bytes,
    uint64_t *required_alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_integrate(
    const gffx_tensor_view *pixel_vertices,const gffx_tensor_view *faces,const gffx_tensor_view *corner_radiance,
    int64_t height,int64_t width,double origin_x,double origin_y,
    const gffx_execution_context *context,gffx_tensor_view *rgba,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_integrate_backward(
    const gffx_tensor_view *pixel_vertices,const gffx_tensor_view *faces,const gffx_tensor_view *corner_radiance,
    int64_t height,int64_t width,double origin_x,double origin_y,
    const gffx_tensor_view *grad_rgba,const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,gffx_tensor_view *grad_radiance,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);


/* Visible-region bounded tensor quadrature: owners[H*W,800] int32 (-1 padding),
 * samples[H*W,800,4] contain three screen barycentrics and area weight.
 * cuts[F,3,K],K<=16, are affine corner numerators of material/texture isolines;
 * for perspective fields use (attribute-threshold)*inverse_clip_w per corner.
 * CPU float32/64, bounded32 candidate faces and32 final triangles/pixel, original
 * depth tie order, global pixel origin, selected-topology first VJP. No retained
 * state, allocations, shader or general quadrature accuracy claim. C caller owns
 * scratch and preallocated outputs. All capacity errors invalidate outputs.
 */
/* Curved extension: cuts may be [F,1,3*K+6], K<=15. Each row packs K
 * linear corner columns (corner-major), then q00,q11,q22,2q01,2q02,2q12.
 * Zero conic coefficients explicitly disable that face's curve. Strict radial
 * topology certification may refuse; curved sides use288or120nodes together,
 * charged as five of32logical blocks. Preserve curved12/5 priority, then choose
 * flat outerGauss5/innerGauss5..8 from remaining800slots. Choices are unseeded
 * and frozen for all first-VJP seeds; no general integration-error certificate.
 * Backward writes the same packed shape. Existing [F,3,K] behavior is unchanged. */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_quadrature_workspace(
    gffx_dtype dtype,const gffx_execution_context *ctx,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_quadrature(
    const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
    int64_t height,int64_t width,double origin_x,double origin_y,const gffx_execution_context *ctx,
    gffx_tensor_view *owners,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic);
/* Forward-only pixel-edge quadrature; edge0/1/2/3=left/right/top/bottom.
 * Same input/output layouts and workspace query as opaque_quadrature; weights
 * are edge lengths, not areas. Gauss12 per segment,32 segments,800 slots.
 * Coincident silhouette or uncertain conic events refuse. No general nonlinear
 * error certificate; caller qualifies its shader. No boundary-node VJP. */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_boundary_quadrature(
    const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
    int64_t height,int64_t width,double origin_x,double origin_y,int32_t edge,const gffx_execution_context *ctx,
    gffx_tensor_view *owners,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_quadrature_backward(
    const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
    int64_t height,int64_t width,double origin_x,double origin_y,const gffx_tensor_view *gradient,const gffx_execution_context *ctx,
    gffx_tensor_view *grad_vertices,gffx_tensor_view *grad_cuts,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic);
/* Explicit material-integral first-VJP profile. Same800slots/32blocks and
 * workspace query. When44slots per flat fan fit after curved priority, with no linear cuts, use
 * Gauss5x7 volume plus Gauss3 on each edge. Barycentrics are held material-fixed;
 * weights carry source-area divergence and relative boundary flux. Zero-weight
 * edge nodes must be shaded. This is a first-order weighted-integral rule, NOT
 * differentiation of exposed node coordinates. Explicit linear shading cuts
 * retain raw interior quadrature because point values cannot supply both
 * one-sided traces. Capacity fallback also retains the raw rule.
 * Shader integration accuracy and discrete profile boundaries require qualification. */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_quadrature_workspace(
    gffx_dtype dtype, const gffx_execution_context *context, uint64_t *bytes,
    uint64_t *alignment, gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_quadrature(
    const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
    int64_t height,int64_t width,double origin_x,double origin_y,const gffx_execution_context *ctx,
    gffx_tensor_view *owners,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_quadrature_backward(
    const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
    int64_t height,int64_t width,double origin_x,double origin_y,const gffx_tensor_view *gradient,const gffx_execution_context *ctx,
    gffx_tensor_view *grad_vertices,gffx_tensor_view *grad_cuts,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic);

/* One caller-owned material-integration batch for one pixel. CPU/CUDA.
 * cursor[4] (HOST int32): next-piece, batch-index, prefix-slots, prefix-blocks;
 * start with four zeros. receipt[8] (HOST int32): next-piece, complete, local
 * slots/blocks, total slots/blocks, emitted pieces, clipping work incl. replay.
 * Continue with {receipt[0], batch+1, receipt[4], receipt[5]}. At most16batches,
 * each800slots/32blocks; aggregate12800slots/512blocks. Candidate cap32 remains.
 * Same tensor layouts and workspace query as one-pixel material quadrature.
 * Fixed curved288 and flat44(no linear cuts)/40 rules; zero-weight nodes MUST
 * be shaded. Errors invalidate ALL outputs. Caller binds unchanged inputs,
 * consumes receipts exactly once, shares its deadline/work budget, and exposes
 * no pixel until complete. Cursor counts are not an authentication mechanism.
 * Cursor/receipt host arrays must not alias tensors, scratch, or each other.
 * No allocation, shader, second VJP, or general quadrature-error guarantee. */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_workspace(
    gffx_dtype dtype, const gffx_execution_context *context, uint64_t *bytes,
    uint64_t *alignment, gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch(
 const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
 double origin_x,double origin_y,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,
 gffx_tensor_view *owners,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_backward(
 const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
 double origin_x,double origin_y,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *gradient,const gffx_execution_context *ctx,
 gffx_tensor_view *grad_vertices,gffx_tensor_view *grad_cuts,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic);

/* Explicit shader-qualified rule. STREAM52 changes only continuation capacity:
 * 52 calls,1664 blocks,41600 slots; STREAM64 permits64/2048/51200; each call retains32 blocks/800 slots.
 * DEFAULT and SH_BILINEAR25 retain16 calls,512 blocks,12800 slots. SH_BILINEAR25 refuses curved cuts.
 * The caller must certify shader error and upstream derivatives separately. */
typedef enum gffx_opaque_batch_rule { GFFX_OPAQUE_BATCH_DEFAULT=0, GFFX_OPAQUE_BATCH_SH_BILINEAR25=1, GFFX_OPAQUE_BATCH_SH_BILINEAR25_STREAM52=2, GFFX_OPAQUE_BATCH_SH_BILINEAR25_STREAM64=3, GFFX_OPAQUE_BATCH_SH_BILINEAR25_STREAM80=4 } gffx_opaque_batch_rule;
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_rule(
 const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
 double origin_x,double origin_y,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,
 gffx_tensor_view *owners,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic,int32_t rule);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_rule_backward(
 const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *cuts,
 double origin_x,double origin_y,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *gradient,const gffx_execution_context *ctx,
 gffx_tensor_view *grad_vertices,gffx_tensor_view *grad_cuts,const gffx_buffer *scratch,gffx_diagnostic_buffer *diagnostic,int32_t rule,int32_t derivatives);


GFFX_API gffx_status GFFX_CALL gffx_render_rasterize_workspace(
    int64_t vertex_count,
    int64_t face_count,
    int64_t image_height,
    int64_t image_width,
    int64_t faces_per_pixel,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_rasterize(
    const gffx_tensor_view *ndc_vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    int64_t image_height,
    int64_t image_width,
    int64_t faces_per_pixel,
    double blur_radius_px,
    uint32_t cull_mode,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *face_index,
    gffx_tensor_view *barycentric,
    gffx_tensor_view *depth,
    gffx_tensor_view *signed_distance,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

/* Forward-only window in original full-frame pixel coordinates. Outputs are
 * [B,image_height,image_width,K]; use the existing raster workspace query.
 * No window geometry backward is advertised by this entry. */
GFFX_API gffx_status GFFX_CALL gffx_render_rasterize_window(
    const gffx_tensor_view *ndc_vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    int64_t global_height,
    int64_t global_width,
    int64_t row_offset,
    int64_t column_offset,
    int64_t image_height,
    int64_t image_width,
    int64_t faces_per_pixel,
    double blur_radius_px,
    uint32_t cull_mode,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *face_index,
    gffx_tensor_view *barycentric,
    gffx_tensor_view *depth,
    gffx_tensor_view *signed_distance,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_rasterize_backward(
    const gffx_tensor_view *ndc_vertices,
    const gffx_tensor_view *faces,
    int64_t image_height,
    int64_t image_width,
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *grad_barycentric,
    const gffx_tensor_view *grad_depth,
    const gffx_tensor_view *grad_signed_distance,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_ndc_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_interpolate_workspace(
    int64_t fragment_count,
    int64_t channel_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_interpolate(
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,
    const gffx_tensor_view *face_attributes,
    const gffx_execution_context *context,
    gffx_tensor_view *attributes,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_interpolate_backward(
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,
    const gffx_tensor_view *face_attributes,
    const gffx_tensor_view *grad_attributes,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_barycentric,
    gffx_tensor_view *grad_face_attributes,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);


GFFX_API gffx_status GFFX_CALL gffx_render_texture_pyramid_workspace(
    int64_t texture_height,
    int64_t texture_width,
    int64_t channel_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_texture_pyramid(
    const gffx_tensor_view *texture,
    int64_t levels,
    const gffx_execution_context *context,
    gffx_tensor_view *pyramid,
    gffx_tensor_view *level_offsets,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_texture_pyramid_backward(
    const gffx_tensor_view *level_offsets,
    int64_t texture_height,
    int64_t texture_width,
    int64_t channel_count,
    const gffx_tensor_view *grad_pyramid,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_texture,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_texture_workspace(
    int64_t sample_count,
    int64_t channel_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_texture(
    const gffx_tensor_view *pyramid,
    const gffx_tensor_view *level_offsets,
    int64_t texture_height,
    int64_t texture_width,
    const gffx_tensor_view *coordinates,
    const gffx_tensor_view *derivatives,
    const gffx_tensor_view *lod,
    uint32_t filter,
    uint32_t mip_filter,
    uint32_t wrap_u,
    uint32_t wrap_v,
    const gffx_tensor_view *border,
    const gffx_execution_context *context,
    gffx_tensor_view *samples,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_render_texture_backward(
    const gffx_tensor_view *pyramid,
    const gffx_tensor_view *level_offsets,
    int64_t texture_height,
    int64_t texture_width,
    const gffx_tensor_view *coordinates,
    const gffx_tensor_view *derivatives,
    const gffx_tensor_view *lod,
    uint32_t filter,
    uint32_t mip_filter,
    uint32_t wrap_u,
    uint32_t wrap_v,
    const gffx_tensor_view *border,
    const gffx_tensor_view *grad_samples,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_pyramid,
    gffx_tensor_view *grad_coordinates,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

/* Experimental CPU float64 SH52 partition. Forward-only; caller must bind immutable inputs.
 * rows[1664,17] stores face, rounded corners, barycentric fields and determinant.
 * No sample order or clipping-work exemption; unsupported devices refuse. */
/* Qualification prefix belongs to exact immutable inputs and derivative mask.
 * Caller authenticates continuation. On failure no prefix/output is accepted. */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_prefix_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t prefix[2]);
/* Opt-in SH80:80 batches/2560 blocks/64000 slots,40 linear cuts; canonical shared edges for all candidate counts. */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_prefix_backward80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t prefix[2]);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_prepare80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *rows,int32_t result[2],const gffx_buffer *scratch,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit_workspace80(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit80(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_boundary_batch80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,int32_t edge,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_prefix_backward64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t prefix[2]);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_prepare(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *rows,int32_t result[2],const gffx_buffer *scratch,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_prepare64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *rows,int32_t result[2],const gffx_buffer *scratch,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit_workspace(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit_workspace64(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,gffx_diagnostic_buffer *d);
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit64(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,gffx_diagnostic_buffer *d);

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_boundary_batch64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,int32_t edge,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d);

GFFX_EXTERN_C_END

#endif /* GFFX_RENDER_H */

/* Explicit-rule backward derivatives: bit1 vertices, bit2 cuts; other bits invalid.
 * Unrequested outputs are zero; layout and requested arithmetic are unchanged. */
