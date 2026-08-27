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

GFFX_EXTERN_C_BEGIN

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

GFFX_EXTERN_C_END

#endif /* GFFX_RENDER_H */
