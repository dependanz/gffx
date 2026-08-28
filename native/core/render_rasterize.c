/*
 * render.rasterize and render.interpolate - Phase 2 CPU reference kernels.
 *
 * Coverage, distance, and barycentric arithmetic run in pixel space: signed_distance is
 * contractually in squared pixel units, and a non-square image scales the axes differently, so
 * NDC-space distances would be anisotropic. Barycentric values are unaffected by the change of
 * space because both the edge numerators and their sum negate under the y-flip.
 *
 * The per-pixel fragment list is maintained by insertion directly in the output tensors, ordered
 * by (depth, global_face_index); scanning faces in ascending order with strict improvement makes
 * an exact depth tie resolve to the lower face index. Both operations need zero workspace.
 */

#include <gffx/execution.h>
#include <gffx/render.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "cuda_loader.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

/* Squared distance in pixel units from p to the segment uv, plus the parameter t of the closest
 * point. Within a fixed clamping region the closest point is (1-t)u + t*v with t held at its
 * optimum, which is what makes the envelope-theorem gradient in the backward pass exact. */
static double gffx_segment_distance_squared(
    double px, double py, double ux, double uy, double vx, double vy, double *out_t
) {
    double ex = vx - ux;
    double ey = vy - uy;
    double length_squared = ex * ex + ey * ey;
    double t = 0.0;
    double cx;
    double cy;
    double dx;
    double dy;
    if (length_squared > 0.0) {
        t = ((px - ux) * ex + (py - uy) * ey) / length_squared;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
    }
    cx = ux + t * ex;
    cy = uy + t * ey;
    dx = px - cx;
    dy = py - cy;
    *out_t = t;
    return dx * dx + dy * dy;
}

/* Nearest boundary distance squared, plus which edge won and its parameter. Edges are indexed
 * 0 = (a,b), 1 = (b,c), 2 = (c,a) and scanned in that fixed order with strict improvement, so
 * the selection is deterministic. */
static double gffx_boundary_distance_squared(
    double px, double py,
    double ax, double ay, double bx, double by, double cx, double cy,
    int *out_edge, double *out_t
) {
    double best_t;
    double best = gffx_segment_distance_squared(px, py, ax, ay, bx, by, &best_t);
    int best_edge = 0;
    double t;
    double candidate = gffx_segment_distance_squared(px, py, bx, by, cx, cy, &t);
    if (candidate < best) { best = candidate; best_edge = 1; best_t = t; }
    candidate = gffx_segment_distance_squared(px, py, cx, cy, ax, ay, &t);
    if (candidate < best) { best = candidate; best_edge = 2; best_t = t; }
    *out_edge = best_edge;
    *out_t = best_t;
    return best;
}

static gffx_status gffx_render_zero_workspace(
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (required_bytes == NULL || required_alignment == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "workspace query result pointers must not be null"
        );
    }
    if (dtype != GFFX_DTYPE_FLOAT32 && dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render operations support the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render operations implement only the CPU backend in this phase"
        );
    }
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

static gffx_status gffx_render_check_offsets(
    const gffx_tensor_view *offsets,
    int64_t total_count,
    int64_t batch_count,
    const char *role_message,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    const int32_t *data;
    int64_t index;
    if (offsets == NULL || offsets->rank != 1u || offsets->shape == NULL ||
        offsets->shape[0] != batch_count + INT64_C(1)) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
    }
    status = gffx_validate_tensor_view(offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "offsets must use the int32 dtype"
        );
    }
    data = (const int32_t *)gffx_mesh_elements_const(offsets);
    if (data[0] != INT32_C(0)) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
    }
    for (index = 0; index < batch_count; ++index) {
        if (data[index + 1] < data[index]) {
            return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
        }
    }
    if ((int64_t)data[batch_count] != total_count) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
    }
    return GFFX_STATUS_OK;
}

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
) {
    /* The device answer is the plugin's: a CUDA implementation may need scratch
     * where the scalar CPU reference needs none, so the query routes too. */
    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        const gffx_cuda_operations *operations = gffx_cuda_loader_operations();
        if (operations == NULL || operations->workspace_query == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "no CUDA provider is available to report a device workspace "
                "requirement");
        }
        return operations->workspace_query(
            GFFX_CUDA_OP_RENDER_RASTERIZE, NULL, 0u, dtype, context, required_bytes,
            required_alignment, diagnostic);
    }
    if (vertex_count < INT64_C(0) || face_count < INT64_C(0) || image_height < INT64_C(0) ||
        image_width < INT64_C(0) || faces_per_pixel < INT64_C(0)) {
        gffx_status prepared = gffx_internal_prepare_diagnostic(diagnostic);
        if (prepared != GFFX_STATUS_OK) return prepared;
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "counts and image dimensions must be nonnegative"
        );
    }
    return gffx_render_zero_workspace(dtype, context, required_bytes, required_alignment,
                                      diagnostic);
}

GFFX_API gffx_status GFFX_CALL gffx_render_interpolate_workspace(
    int64_t fragment_count,
    int64_t channel_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    /* The device answer is the plugin's: a CUDA implementation may need scratch
     * where the scalar CPU reference needs none, so the query routes too. */
    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        const gffx_cuda_operations *operations = gffx_cuda_loader_operations();
        if (operations == NULL || operations->workspace_query == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "no CUDA provider is available to report a device workspace "
                "requirement");
        }
        return operations->workspace_query(
            GFFX_CUDA_OP_RENDER_INTERPOLATE, NULL, 0u, dtype, context, required_bytes,
            required_alignment, diagnostic);
    }
    if (fragment_count < INT64_C(0) || channel_count < INT64_C(0)) {
        gffx_status prepared = gffx_internal_prepare_diagnostic(diagnostic);
        if (prepared != GFFX_STATUS_OK) return prepared;
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "counts must be nonnegative"
        );
    }
    return gffx_render_zero_workspace(dtype, context, required_bytes, required_alignment,
                                      diagnostic);
}

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
) {
    /*
     * Device dispatch before any CPU validation. The shared validators dereference tensor data,
     * which must not happen for device memory, so the forward has to precede them rather than
     * follow. A backend that publishes no such operation returns UNSUPPORTED rather than falling
     * back to the CPU, keeping a missing kernel visible instead of an unannounced copy.
     */
    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        const gffx_cuda_operations *operations = gffx_cuda_loader_operations();
        if (operations == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "no CUDA provider is available; install the gffx CUDA plugin or "
                "run on the CPU");
        }
        if (operations->render_rasterize == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "the CUDA provider does not implement this operation");
        }
        return operations->render_rasterize(
            ndc_vertices, faces, vertex_offsets, face_offsets, image_height, image_width, faces_per_pixel, blur_radius_px, cull_mode, eps, context, face_index, barycentric, depth, signed_distance, workspace, diagnostic);
    }
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t vertex_count;
    int64_t face_count;
    int64_t batch_count;
    int64_t batch;
    int is_double;
    if (status != GFFX_STATUS_OK) return status;
    if (faces_per_pixel <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "faces per pixel must be positive"
        );
    }
    if (image_height <= INT64_C(0) || image_width <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "image dimensions must be positive"
        );
    }
    if (isnan(blur_radius_px) || isinf(blur_radius_px) || blur_radius_px < 0.0) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the blur radius must be finite and nonnegative"
        );
    }
    if (cull_mode != GFFX_CULL_NONE && cull_mode != GFFX_CULL_BACK &&
        cull_mode != GFFX_CULL_FRONT) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "cull mode must be none, back, or front"
        );
    }
    if (isnan(eps) || isinf(eps) || eps < 0.0) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "eps must be finite and nonnegative"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.rasterize implements only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(ndc_vertices, "ndc vertices must be a [V,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (ndc_vertices->dtype != GFFX_DTYPE_FLOAT32 &&
        ndc_vertices->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "ndc vertices must use the float32 or float64 dtype"
        );
    }
    status = gffx_mesh_check_view(faces, "faces must be a [F,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (faces->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "faces must use the int32 dtype"
        );
    }
    vertex_count = ndc_vertices->shape[0];
    face_count = faces->shape[0];
    is_double = ndc_vertices->dtype == GFFX_DTYPE_FLOAT64;

    if (vertex_offsets == NULL || vertex_offsets->rank != 1u ||
        vertex_offsets->shape == NULL || vertex_offsets->shape[0] < INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertex offsets must be a [B+1] tensor view"
        );
    }
    batch_count = vertex_offsets->shape[0] - INT64_C(1);
    status = gffx_render_check_offsets(vertex_offsets, vertex_count, batch_count,
                                       "vertex offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_render_check_offsets(face_offsets, face_count, batch_count,
                                       "face offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    /* Fragment outputs share the [B,H,W,K] shape; barycentrics add a trailing 3. */
    if (face_index == NULL || face_index->rank != 4u || face_index->shape == NULL ||
        face_index->shape[0] != batch_count || face_index->shape[1] != image_height ||
        face_index->shape[2] != image_width || face_index->shape[3] != faces_per_pixel) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must be a [B,H,W,K] output view"
        );
    }
    status = gffx_validate_tensor_view(face_index, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (face_index->dtype != GFFX_DTYPE_INT32 ||
        (face_index->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must be an int32 output view"
        );
    }
    if (depth == NULL || depth->rank != 4u || depth->shape == NULL ||
        depth->shape[0] != batch_count || depth->shape[1] != image_height ||
        depth->shape[2] != image_width || depth->shape[3] != faces_per_pixel ||
        depth->dtype != ndc_vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "depth must be a [B,H,W,K] output view matching the vertex dtype"
        );
    }
    status = gffx_validate_tensor_view(depth, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (signed_distance == NULL || signed_distance->rank != 4u ||
        signed_distance->shape == NULL || signed_distance->shape[0] != batch_count ||
        signed_distance->shape[1] != image_height ||
        signed_distance->shape[2] != image_width ||
        signed_distance->shape[3] != faces_per_pixel ||
        signed_distance->dtype != ndc_vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "signed distance must be a [B,H,W,K] output view matching the vertex dtype"
        );
    }
    status = gffx_validate_tensor_view(signed_distance, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (barycentric == NULL || barycentric->rank != 5u || barycentric->shape == NULL ||
        barycentric->shape[0] != batch_count || barycentric->shape[1] != image_height ||
        barycentric->shape[2] != image_width || barycentric->shape[3] != faces_per_pixel ||
        barycentric->shape[4] != INT64_C(3) || barycentric->dtype != ndc_vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "barycentrics must be a [B,H,W,K,3] output view matching the vertex dtype"
        );
    }
    status = gffx_validate_tensor_view(barycentric, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (gffx_mesh_views_overlap(face_index, faces) ||
        gffx_mesh_views_overlap(barycentric, ndc_vertices) ||
        gffx_mesh_views_overlap(depth, ndc_vertices) ||
        gffx_mesh_views_overlap(signed_distance, ndc_vertices) ||
        gffx_mesh_views_overlap(barycentric, depth) ||
        gffx_mesh_views_overlap(depth, signed_distance)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    {
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const int32_t *vertex_bounds =
            (const int32_t *)gffx_mesh_elements_const(vertex_offsets);
        const int32_t *face_bounds = (const int32_t *)gffx_mesh_elements_const(face_offsets);
        const double *vertex_d =
            is_double ? (const double *)gffx_mesh_elements_const(ndc_vertices) : NULL;
        const float *vertex_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(ndc_vertices);
        int32_t *index_data = (int32_t *)gffx_mesh_elements(face_index);
        double *depth_d = is_double ? (double *)gffx_mesh_elements(depth) : NULL;
        float *depth_f = is_double ? NULL : (float *)gffx_mesh_elements(depth);
        double *distance_d = is_double ? (double *)gffx_mesh_elements(signed_distance) : NULL;
        float *distance_f = is_double ? NULL : (float *)gffx_mesh_elements(signed_distance);
        double *bary_d = is_double ? (double *)gffx_mesh_elements(barycentric) : NULL;
        float *bary_f = is_double ? NULL : (float *)gffx_mesh_elements(barycentric);
        double half_width = (double)image_width * 0.5;
        double half_height = (double)image_height * 0.5;
        double blur_squared = blur_radius_px * blur_radius_px;
        int64_t row;
        int64_t column;
        int64_t slot;

        /* Every face must reference vertices of its own batch element. */
        for (batch = 0; batch < batch_count; ++batch) {
            int64_t first_vertex = (int64_t)vertex_bounds[batch];
            int64_t last_vertex = (int64_t)vertex_bounds[batch + 1];
            int64_t face;
            for (face = (int64_t)face_bounds[batch]; face < (int64_t)face_bounds[batch + 1];
                 ++face) {
                int corner;
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = (int64_t)face_data[face * 3 + corner];
                    if (vertex < first_vertex || vertex >= last_vertex) {
                        return gffx_internal_fail(
                            diagnostic,
                            GFFX_STATUS_INVALID_ARGUMENT,
                            "each face must reference vertices of its own batch element"
                        );
                    }
                }
            }
        }

        for (batch = 0; batch < batch_count; ++batch) {
            int64_t first_face = (int64_t)face_bounds[batch];
            int64_t last_face = (int64_t)face_bounds[batch + 1];
            for (row = 0; row < image_height; ++row) {
                for (column = 0; column < image_width; ++column) {
                    int64_t pixel_base =
                        ((batch * image_height + row) * image_width + column) * faces_per_pixel;
                    double px = (double)column + 0.5;
                    double py = (double)row + 0.5;
                    int64_t face;
                    /* Initialize the whole fragment list to background. */
                    for (slot = 0; slot < faces_per_pixel; ++slot) {
                        int64_t entry = pixel_base + slot;
                        index_data[entry] = -1;
                        if (is_double) {
                            depth_d[entry] = (double)INFINITY;
                            distance_d[entry] = (double)INFINITY;
                            bary_d[entry * 3 + 0] = 0.0;
                            bary_d[entry * 3 + 1] = 0.0;
                            bary_d[entry * 3 + 2] = 0.0;
                        } else {
                            depth_f[entry] = (float)INFINITY;
                            distance_f[entry] = (float)INFINITY;
                            bary_f[entry * 3 + 0] = 0.0f;
                            bary_f[entry * 3 + 1] = 0.0f;
                            bary_f[entry * 3 + 2] = 0.0f;
                        }
                    }
                    for (face = first_face; face < last_face; ++face) {
                        int64_t i0 = (int64_t)face_data[face * 3 + 0];
                        int64_t i1 = (int64_t)face_data[face * 3 + 1];
                        int64_t i2 = (int64_t)face_data[face * 3 + 2];
                        double nx0 = is_double ? vertex_d[i0 * 3 + 0]
                                               : (double)vertex_f[i0 * 3 + 0];
                        double ny0 = is_double ? vertex_d[i0 * 3 + 1]
                                               : (double)vertex_f[i0 * 3 + 1];
                        double nz0 = is_double ? vertex_d[i0 * 3 + 2]
                                               : (double)vertex_f[i0 * 3 + 2];
                        double nx1 = is_double ? vertex_d[i1 * 3 + 0]
                                               : (double)vertex_f[i1 * 3 + 0];
                        double ny1 = is_double ? vertex_d[i1 * 3 + 1]
                                               : (double)vertex_f[i1 * 3 + 1];
                        double nz1 = is_double ? vertex_d[i1 * 3 + 2]
                                               : (double)vertex_f[i1 * 3 + 2];
                        double nx2 = is_double ? vertex_d[i2 * 3 + 0]
                                               : (double)vertex_f[i2 * 3 + 0];
                        double ny2 = is_double ? vertex_d[i2 * 3 + 1]
                                               : (double)vertex_f[i2 * 3 + 1];
                        double nz2 = is_double ? vertex_d[i2 * 3 + 2]
                                               : (double)vertex_f[i2 * 3 + 2];
                        double ax = (nx0 + 1.0) * half_width;
                        double ay = (1.0 - ny0) * half_height;
                        double bx = (nx1 + 1.0) * half_width;
                        double by = (1.0 - ny1) * half_height;
                        double cx = (nx2 + 1.0) * half_width;
                        double cy = (1.0 - ny2) * half_height;
                        double e0 = (bx - px) * (cy - py) - (by - py) * (cx - px);
                        double e1 = (cx - px) * (ay - py) - (cy - py) * (ax - px);
                        double e2 = (ax - px) * (by - py) - (ay - py) * (bx - px);
                        double area2 = e0 + e1 + e2;
                        double w0;
                        double w1;
                        double w2;
                        double fragment_depth;
                        double distance_squared;
                        double signed_value;
                        double edge_t;
                        int edge;
                        int inside;
                        int64_t position;
                        if (!(fabs(area2) > eps)) continue;
                        /* Culling reads the NDC orientation, the negation of the pixel-space
                         * area because the y-flip reverses handedness. */
                        if (cull_mode == GFFX_CULL_BACK && -area2 <= 0.0) continue;
                        if (cull_mode == GFFX_CULL_FRONT && -area2 > 0.0) continue;
                        w0 = e0 / area2;
                        w1 = e1 / area2;
                        w2 = e2 / area2;
                        inside = (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0);
                        distance_squared = gffx_boundary_distance_squared(
                            px, py, ax, ay, bx, by, cx, cy, &edge, &edge_t);
                        if (!inside && distance_squared > blur_squared) continue;
                        fragment_depth = w0 * nz0 + w1 * nz1 + w2 * nz2;
                        signed_value = inside ? -distance_squared : distance_squared;

                        /* Insert by (depth, face index); ascending scan plus strict
                         * improvement keeps the lower face index on an exact tie. */
                        position = faces_per_pixel;
                        while (position > 0) {
                            int64_t above = pixel_base + position - 1;
                            double previous = is_double ? depth_d[above] : (double)depth_f[above];
                            if (index_data[above] < 0 || fragment_depth < previous) {
                                --position;
                            } else {
                                break;
                            }
                        }
                        if (position >= faces_per_pixel) continue;
                        for (slot = faces_per_pixel - 1; slot > position; --slot) {
                            int64_t to = pixel_base + slot;
                            int64_t from = pixel_base + slot - 1;
                            index_data[to] = index_data[from];
                            if (is_double) {
                                depth_d[to] = depth_d[from];
                                distance_d[to] = distance_d[from];
                                bary_d[to * 3 + 0] = bary_d[from * 3 + 0];
                                bary_d[to * 3 + 1] = bary_d[from * 3 + 1];
                                bary_d[to * 3 + 2] = bary_d[from * 3 + 2];
                            } else {
                                depth_f[to] = depth_f[from];
                                distance_f[to] = distance_f[from];
                                bary_f[to * 3 + 0] = bary_f[from * 3 + 0];
                                bary_f[to * 3 + 1] = bary_f[from * 3 + 1];
                                bary_f[to * 3 + 2] = bary_f[from * 3 + 2];
                            }
                        }
                        {
                            int64_t entry = pixel_base + position;
                            index_data[entry] = (int32_t)face;
                            if (is_double) {
                                depth_d[entry] = fragment_depth;
                                distance_d[entry] = signed_value;
                                bary_d[entry * 3 + 0] = w0;
                                bary_d[entry * 3 + 1] = w1;
                                bary_d[entry * 3 + 2] = w2;
                            } else {
                                depth_f[entry] = (float)fragment_depth;
                                distance_f[entry] = (float)signed_value;
                                bary_f[entry * 3 + 0] = (float)w0;
                                bary_f[entry * 3 + 1] = (float)w1;
                                bary_f[entry * 3 + 2] = (float)w2;
                            }
                        }
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

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
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t vertex_count;
    int64_t face_count;
    int64_t batch_count;
    int64_t faces_per_pixel;
    int64_t index;
    int64_t batch;
    int is_double;
    if (status != GFFX_STATUS_OK) return status;
    if (image_height <= INT64_C(0) || image_width <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "image dimensions must be positive"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.rasterize implements only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(ndc_vertices, "ndc vertices must be a [V,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (ndc_vertices->dtype != GFFX_DTYPE_FLOAT32 &&
        ndc_vertices->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "ndc vertices must use the float32 or float64 dtype"
        );
    }
    status = gffx_mesh_check_view(faces, "faces must be a [F,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    vertex_count = ndc_vertices->shape[0];
    face_count = faces->shape[0];
    is_double = ndc_vertices->dtype == GFFX_DTYPE_FLOAT64;

    if (face_index == NULL || face_index->rank != 4u || face_index->shape == NULL ||
        face_index->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must be a [B,H,W,K] int32 tensor view"
        );
    }
    status = gffx_validate_tensor_view(face_index, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    batch_count = face_index->shape[0];
    faces_per_pixel = face_index->shape[3];
    if (face_index->shape[1] != image_height || face_index->shape[2] != image_width) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must agree with the image dimensions"
        );
    }
    if (grad_barycentric == NULL && grad_depth == NULL && grad_signed_distance == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "at least one cotangent is required"
        );
    }
    status = gffx_mesh_check_view(grad_ndc_vertices,
                                  "vertex gradients must be a [V,3] output view",
                                  2u, vertex_count, INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_ndc_vertices->dtype != ndc_vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertex gradients must match the vertex dtype"
        );
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    {
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const int32_t *index_data = (const int32_t *)gffx_mesh_elements_const(face_index);
        const double *vertex_d =
            is_double ? (const double *)gffx_mesh_elements_const(ndc_vertices) : NULL;
        const float *vertex_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(ndc_vertices);
        const double *gb_d =
            (is_double && grad_barycentric != NULL)
                ? (const double *)gffx_mesh_elements_const(grad_barycentric) : NULL;
        const float *gb_f =
            (!is_double && grad_barycentric != NULL)
                ? (const float *)gffx_mesh_elements_const(grad_barycentric) : NULL;
        const double *gd_d =
            (is_double && grad_depth != NULL)
                ? (const double *)gffx_mesh_elements_const(grad_depth) : NULL;
        const float *gd_f =
            (!is_double && grad_depth != NULL)
                ? (const float *)gffx_mesh_elements_const(grad_depth) : NULL;
        const double *gs_d =
            (is_double && grad_signed_distance != NULL)
                ? (const double *)gffx_mesh_elements_const(grad_signed_distance) : NULL;
        const float *gs_f =
            (!is_double && grad_signed_distance != NULL)
                ? (const float *)gffx_mesh_elements_const(grad_signed_distance) : NULL;
        double *gradient_d = is_double ? (double *)gffx_mesh_elements(grad_ndc_vertices) : NULL;
        float *gradient_f = is_double ? NULL : (float *)gffx_mesh_elements(grad_ndc_vertices);
        double half_width = (double)image_width * 0.5;
        double half_height = (double)image_height * 0.5;
        int64_t row;
        int64_t column;
        int64_t slot;

        if (is_double) {
            for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient_d[index] = 0.0;
        } else {
            for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient_f[index] = 0.0f;
        }

        for (batch = 0; batch < batch_count; ++batch) {
            for (row = 0; row < image_height; ++row) {
                for (column = 0; column < image_width; ++column) {
                    for (slot = 0; slot < faces_per_pixel; ++slot) {
                        int64_t entry =
                            ((batch * image_height + row) * image_width + column) *
                                faces_per_pixel + slot;
                        int64_t face = (int64_t)index_data[entry];
                        int64_t i0;
                        int64_t i1;
                        int64_t i2;
                        double px = (double)column + 0.5;
                        double py = (double)row + 0.5;
                        double corner_x[3];
                        double corner_y[3];
                        double corner_z[3];
                        double pixel_grad_x[3];
                        double pixel_grad_y[3];
                        double e[3];
                        double area2;
                        double w[3];
                        double de_dx[3][3];
                        double de_dy[3][3];
                        double darea_dx[3];
                        double darea_dy[3];
                        double gz[3];
                        int corner;
                        int component;
                        if (face < 0) continue;
                        if (face >= face_count) {
                            return gffx_internal_fail(
                                diagnostic,
                                GFFX_STATUS_INVALID_ARGUMENT,
                                "a fragment names a face outside the face range"
                            );
                        }
                        i0 = (int64_t)face_data[face * 3 + 0];
                        i1 = (int64_t)face_data[face * 3 + 1];
                        i2 = (int64_t)face_data[face * 3 + 2];
                        if (i0 < 0 || i0 >= vertex_count || i1 < 0 || i1 >= vertex_count ||
                            i2 < 0 || i2 >= vertex_count) {
                            return gffx_internal_fail(
                                diagnostic,
                                GFFX_STATUS_INVALID_ARGUMENT,
                                "a face references a vertex outside the vertex range"
                            );
                        }
                        {
                            int64_t ids[3];
                            ids[0] = i0; ids[1] = i1; ids[2] = i2;
                            for (corner = 0; corner < 3; ++corner) {
                                double nx = is_double ? vertex_d[ids[corner] * 3 + 0]
                                                      : (double)vertex_f[ids[corner] * 3 + 0];
                                double ny = is_double ? vertex_d[ids[corner] * 3 + 1]
                                                      : (double)vertex_f[ids[corner] * 3 + 1];
                                double nz = is_double ? vertex_d[ids[corner] * 3 + 2]
                                                      : (double)vertex_f[ids[corner] * 3 + 2];
                                corner_x[corner] = (nx + 1.0) * half_width;
                                corner_y[corner] = (1.0 - ny) * half_height;
                                corner_z[corner] = nz;
                                pixel_grad_x[corner] = 0.0;
                                pixel_grad_y[corner] = 0.0;
                                gz[corner] = 0.0;
                            }
                        }
                        e[0] = (corner_x[1] - px) * (corner_y[2] - py) -
                               (corner_y[1] - py) * (corner_x[2] - px);
                        e[1] = (corner_x[2] - px) * (corner_y[0] - py) -
                               (corner_y[2] - py) * (corner_x[0] - px);
                        e[2] = (corner_x[0] - px) * (corner_y[1] - py) -
                               (corner_y[0] - py) * (corner_x[1] - px);
                        area2 = e[0] + e[1] + e[2];
                        if (area2 == 0.0) continue;
                        w[0] = e[0] / area2;
                        w[1] = e[1] / area2;
                        w[2] = e[2] / area2;

                        /* Partial derivatives of the edge numerators with respect to each
                         * pixel-space corner coordinate. */
                        de_dx[0][0] = 0.0;                     de_dy[0][0] = 0.0;
                        de_dx[0][1] = corner_y[2] - py;        de_dy[0][1] = -(corner_x[2] - px);
                        de_dx[0][2] = -(corner_y[1] - py);     de_dy[0][2] = corner_x[1] - px;
                        de_dx[1][0] = -(corner_y[2] - py);     de_dy[1][0] = corner_x[2] - px;
                        de_dx[1][1] = 0.0;                     de_dy[1][1] = 0.0;
                        de_dx[1][2] = corner_y[0] - py;        de_dy[1][2] = -(corner_x[0] - px);
                        de_dx[2][0] = corner_y[1] - py;        de_dy[2][0] = -(corner_x[1] - px);
                        de_dx[2][1] = -(corner_y[0] - py);     de_dy[2][1] = corner_x[0] - px;
                        de_dx[2][2] = 0.0;                     de_dy[2][2] = 0.0;
                        for (corner = 0; corner < 3; ++corner) {
                            darea_dx[corner] =
                                de_dx[0][corner] + de_dx[1][corner] + de_dx[2][corner];
                            darea_dy[corner] =
                                de_dy[0][corner] + de_dy[1][corner] + de_dy[2][corner];
                        }

                        /* Barycentric and depth cotangents share the same weight derivatives. */
                        for (component = 0; component < 3; ++component) {
                            double weight_cotangent = 0.0;
                            if (gb_d != NULL) weight_cotangent += gb_d[entry * 3 + component];
                            if (gb_f != NULL) {
                                weight_cotangent += (double)gb_f[entry * 3 + component];
                            }
                            if (gd_d != NULL) weight_cotangent += gd_d[entry] * corner_z[component];
                            if (gd_f != NULL) {
                                weight_cotangent += (double)gd_f[entry] * corner_z[component];
                            }
                            if (weight_cotangent == 0.0) continue;
                            for (corner = 0; corner < 3; ++corner) {
                                double dw_dx = (de_dx[component][corner] -
                                                w[component] * darea_dx[corner]) / area2;
                                double dw_dy = (de_dy[component][corner] -
                                                w[component] * darea_dy[corner]) / area2;
                                pixel_grad_x[corner] += weight_cotangent * dw_dx;
                                pixel_grad_y[corner] += weight_cotangent * dw_dy;
                            }
                        }
                        /* Depth also differentiates directly through each vertex z. */
                        if (gd_d != NULL || gd_f != NULL) {
                            double depth_cotangent =
                                gd_d != NULL ? gd_d[entry] : (double)gd_f[entry];
                            for (corner = 0; corner < 3; ++corner) {
                                gz[corner] += depth_cotangent * w[corner];
                            }
                        }
                        /* The signed distance uses the envelope argument: within a fixed
                         * nearest-edge region the closest boundary point is (1-t)u + t*v with
                         * t at its optimum, so only the endpoint weights contribute. */
                        if (gs_d != NULL || gs_f != NULL) {
                            double distance_cotangent =
                                gs_d != NULL ? gs_d[entry] : (double)gs_f[entry];
                            if (distance_cotangent != 0.0) {
                                int edge;
                                double edge_t;
                                int inside = (w[0] >= 0.0 && w[1] >= 0.0 && w[2] >= 0.0);
                                double sign = inside ? -1.0 : 1.0;
                                int first;
                                int second;
                                double ux;
                                double uy;
                                double vx;
                                double vy;
                                double cx;
                                double cy;
                                double residual_x;
                                double residual_y;
                                (void)gffx_boundary_distance_squared(
                                    px, py, corner_x[0], corner_y[0], corner_x[1], corner_y[1],
                                    corner_x[2], corner_y[2], &edge, &edge_t);
                                first = edge;
                                second = (edge + 1) % 3;
                                ux = corner_x[first]; uy = corner_y[first];
                                vx = corner_x[second]; vy = corner_y[second];
                                cx = ux + edge_t * (vx - ux);
                                cy = uy + edge_t * (vy - uy);
                                residual_x = -2.0 * (px - cx) * sign * distance_cotangent;
                                residual_y = -2.0 * (py - cy) * sign * distance_cotangent;
                                pixel_grad_x[first] += residual_x * (1.0 - edge_t);
                                pixel_grad_y[first] += residual_y * (1.0 - edge_t);
                                pixel_grad_x[second] += residual_x * edge_t;
                                pixel_grad_y[second] += residual_y * edge_t;
                            }
                        }

                        /* Convert pixel-space gradients back to NDC by the mapping of the
                         * contract: x scales by W/2 and y by -H/2. */
                        {
                            int64_t ids[3];
                            ids[0] = i0; ids[1] = i1; ids[2] = i2;
                            for (corner = 0; corner < 3; ++corner) {
                                double gx = pixel_grad_x[corner] * half_width;
                                double gy = pixel_grad_y[corner] * (-half_height);
                                if (is_double) {
                                    gradient_d[ids[corner] * 3 + 0] += gx;
                                    gradient_d[ids[corner] * 3 + 1] += gy;
                                    gradient_d[ids[corner] * 3 + 2] += gz[corner];
                                } else {
                                    gradient_f[ids[corner] * 3 + 0] += (float)gx;
                                    gradient_f[ids[corner] * 3 + 1] += (float)gy;
                                    gradient_f[ids[corner] * 3 + 2] += (float)gz[corner];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

/* ------------------------------------------------------------------ render.interpolate */

/* Validates the shared fragment/attribute shape family for both interpolate directions. */
static gffx_status gffx_interpolate_check(
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,
    const gffx_tensor_view *face_attributes,
    const gffx_execution_context *context,
    int64_t *fragment_count,
    int64_t *face_count,
    int64_t *channel_count,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_validate_execution_context(context, diagnostic);
    int64_t total = 1;
    uint32_t axis;
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "render.interpolate implements only the CPU backend in this phase"
        );
    }
    if (face_index == NULL || face_index->rank != 4u || face_index->shape == NULL ||
        face_index->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must be a [B,H,W,K] int32 tensor view"
        );
    }
    status = gffx_validate_tensor_view(face_index, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    for (axis = 0u; axis < 4u; ++axis) total *= face_index->shape[axis];
    if (barycentric == NULL || barycentric->rank != 5u || barycentric->shape == NULL ||
        barycentric->shape[4] != INT64_C(3)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "barycentrics must be a [B,H,W,K,3] tensor view"
        );
    }
    for (axis = 0u; axis < 4u; ++axis) {
        if (barycentric->shape[axis] != face_index->shape[axis]) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "barycentrics must agree with the fragment shape"
            );
        }
    }
    status = gffx_validate_tensor_view(barycentric, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (barycentric->dtype != GFFX_DTYPE_FLOAT32 && barycentric->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "barycentrics must use the float32 or float64 dtype"
        );
    }
    if (face_attributes == NULL || face_attributes->rank != 3u ||
        face_attributes->shape == NULL || face_attributes->shape[1] != INT64_C(3) ||
        face_attributes->dtype != barycentric->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face attributes must be a [F,3,C] view matching the barycentric dtype"
        );
    }
    status = gffx_validate_tensor_view(face_attributes, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    *fragment_count = total;
    *face_count = face_attributes->shape[0];
    *channel_count = face_attributes->shape[2];
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_render_interpolate(
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,
    const gffx_tensor_view *face_attributes,
    const gffx_execution_context *context,
    gffx_tensor_view *attributes,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    /*
     * Device dispatch before any CPU validation. The shared validators dereference tensor data,
     * which must not happen for device memory, so the forward has to precede them rather than
     * follow. A backend that publishes no such operation returns UNSUPPORTED rather than falling
     * back to the CPU, keeping a missing kernel visible instead of an unannounced copy.
     */
    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        const gffx_cuda_operations *operations = gffx_cuda_loader_operations();
        if (operations == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "no CUDA provider is available; install the gffx CUDA plugin or "
                "run on the CPU");
        }
        if (operations->render_interpolate == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "the CUDA provider does not implement this operation");
        }
        return operations->render_interpolate(
            face_index, barycentric, face_attributes, context, attributes, workspace, diagnostic);
    }
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t fragment_count = 0;
    int64_t face_count = 0;
    int64_t channel_count = 0;
    int64_t fragment;
    int is_double;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_interpolate_check(face_index, barycentric, face_attributes, context,
                                    &fragment_count, &face_count, &channel_count, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (attributes == NULL || attributes->rank != 5u || attributes->shape == NULL ||
        attributes->shape[4] != channel_count ||
        attributes->dtype != barycentric->dtype ||
        (attributes->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "attributes must be a [B,H,W,K,C] output view matching the dtype"
        );
    }
    status = gffx_validate_tensor_view(attributes, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (gffx_mesh_views_overlap(attributes, barycentric) ||
        gffx_mesh_views_overlap(attributes, face_attributes)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    is_double = barycentric->dtype == GFFX_DTYPE_FLOAT64;

    {
        const int32_t *index_data = (const int32_t *)gffx_mesh_elements_const(face_index);
        const double *bary_d =
            is_double ? (const double *)gffx_mesh_elements_const(barycentric) : NULL;
        const float *bary_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(barycentric);
        const double *attribute_d =
            is_double ? (const double *)gffx_mesh_elements_const(face_attributes) : NULL;
        const float *attribute_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(face_attributes);
        double *output_d = is_double ? (double *)gffx_mesh_elements(attributes) : NULL;
        float *output_f = is_double ? NULL : (float *)gffx_mesh_elements(attributes);

        for (fragment = 0; fragment < fragment_count; ++fragment) {
            int64_t face = (int64_t)index_data[fragment];
            int64_t channel;
            int corner;
            /* Background fragments are exactly zero and never read the attributes. */
            if (face < 0) {
                for (channel = 0; channel < channel_count; ++channel) {
                    if (is_double) output_d[fragment * channel_count + channel] = 0.0;
                    else output_f[fragment * channel_count + channel] = 0.0f;
                }
                continue;
            }
            if (face >= face_count) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "a fragment names a face outside the attribute range"
                );
            }
            for (channel = 0; channel < channel_count; ++channel) {
                double total = 0.0;
                for (corner = 0; corner < 3; ++corner) {
                    double weight = is_double ? bary_d[fragment * 3 + corner]
                                              : (double)bary_f[fragment * 3 + corner];
                    double value =
                        is_double
                            ? attribute_d[(face * 3 + corner) * channel_count + channel]
                            : (double)attribute_f[(face * 3 + corner) * channel_count + channel];
                    total += weight * value;
                }
                if (is_double) output_d[fragment * channel_count + channel] = total;
                else output_f[fragment * channel_count + channel] = (float)total;
            }
        }
    }
    return GFFX_STATUS_OK;
}

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
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t fragment_count = 0;
    int64_t face_count = 0;
    int64_t channel_count = 0;
    int64_t fragment;
    int64_t index;
    int is_double;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_interpolate_check(face_index, barycentric, face_attributes, context,
                                    &fragment_count, &face_count, &channel_count, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_barycentric == NULL && grad_face_attributes == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "at least one gradient output is required"
        );
    }
    if (grad_attributes == NULL || grad_attributes->rank != 5u ||
        grad_attributes->shape == NULL || grad_attributes->shape[4] != channel_count ||
        grad_attributes->dtype != barycentric->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "attribute cotangents must be a [B,H,W,K,C] view matching the dtype"
        );
    }
    status = gffx_validate_tensor_view(grad_attributes, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_barycentric != NULL) {
        if (grad_barycentric->rank != 5u || grad_barycentric->shape == NULL ||
            grad_barycentric->shape[4] != INT64_C(3) ||
            grad_barycentric->dtype != barycentric->dtype ||
            (grad_barycentric->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "barycentric gradients must be a [B,H,W,K,3] output view matching the dtype"
            );
        }
        status = gffx_validate_tensor_view(grad_barycentric, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    if (grad_face_attributes != NULL) {
        if (grad_face_attributes->rank != 3u || grad_face_attributes->shape == NULL ||
            grad_face_attributes->shape[0] != face_count ||
            grad_face_attributes->shape[1] != INT64_C(3) ||
            grad_face_attributes->shape[2] != channel_count ||
            grad_face_attributes->dtype != barycentric->dtype ||
            (grad_face_attributes->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "attribute gradients must be a [F,3,C] output view matching the dtype"
            );
        }
        status = gffx_validate_tensor_view(grad_face_attributes, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    is_double = barycentric->dtype == GFFX_DTYPE_FLOAT64;

    {
        const int32_t *index_data = (const int32_t *)gffx_mesh_elements_const(face_index);
        const double *bary_d =
            is_double ? (const double *)gffx_mesh_elements_const(barycentric) : NULL;
        const float *bary_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(barycentric);
        const double *attribute_d =
            is_double ? (const double *)gffx_mesh_elements_const(face_attributes) : NULL;
        const float *attribute_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(face_attributes);
        const double *cotangent_d =
            is_double ? (const double *)gffx_mesh_elements_const(grad_attributes) : NULL;
        const float *cotangent_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(grad_attributes);
        double *grad_bary_d =
            (is_double && grad_barycentric != NULL)
                ? (double *)gffx_mesh_elements(grad_barycentric) : NULL;
        float *grad_bary_f =
            (!is_double && grad_barycentric != NULL)
                ? (float *)gffx_mesh_elements(grad_barycentric) : NULL;
        double *grad_attribute_d =
            (is_double && grad_face_attributes != NULL)
                ? (double *)gffx_mesh_elements(grad_face_attributes) : NULL;
        float *grad_attribute_f =
            (!is_double && grad_face_attributes != NULL)
                ? (float *)gffx_mesh_elements(grad_face_attributes) : NULL;

        if (grad_bary_d != NULL) {
            for (index = 0; index < fragment_count * INT64_C(3); ++index) grad_bary_d[index] = 0.0;
        }
        if (grad_bary_f != NULL) {
            for (index = 0; index < fragment_count * INT64_C(3); ++index) grad_bary_f[index] = 0.0f;
        }
        if (grad_attribute_d != NULL) {
            for (index = 0; index < face_count * INT64_C(3) * channel_count; ++index) {
                grad_attribute_d[index] = 0.0;
            }
        }
        if (grad_attribute_f != NULL) {
            for (index = 0; index < face_count * INT64_C(3) * channel_count; ++index) {
                grad_attribute_f[index] = 0.0f;
            }
        }

        for (fragment = 0; fragment < fragment_count; ++fragment) {
            int64_t face = (int64_t)index_data[fragment];
            int64_t channel;
            int corner;
            if (face < 0) continue;
            if (face >= face_count) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "a fragment names a face outside the attribute range"
                );
            }
            for (corner = 0; corner < 3; ++corner) {
                double weight = is_double ? bary_d[fragment * 3 + corner]
                                          : (double)bary_f[fragment * 3 + corner];
                double weight_gradient = 0.0;
                for (channel = 0; channel < channel_count; ++channel) {
                    int64_t attribute_entry = (face * 3 + corner) * channel_count + channel;
                    double cotangent = is_double
                                           ? cotangent_d[fragment * channel_count + channel]
                                           : (double)cotangent_f[fragment * channel_count +
                                                                 channel];
                    double value = is_double ? attribute_d[attribute_entry]
                                             : (double)attribute_f[attribute_entry];
                    weight_gradient += value * cotangent;
                    if (grad_attribute_d != NULL) {
                        grad_attribute_d[attribute_entry] += weight * cotangent;
                    }
                    if (grad_attribute_f != NULL) {
                        grad_attribute_f[attribute_entry] += (float)(weight * cotangent);
                    }
                }
                if (grad_bary_d != NULL) grad_bary_d[fragment * 3 + corner] += weight_gradient;
                if (grad_bary_f != NULL) {
                    grad_bary_f[fragment * 3 + corner] += (float)weight_gradient;
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}
