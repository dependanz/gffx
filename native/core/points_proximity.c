/*
 * points.knn and points.closest_point_on_mesh - Phase 2 CPU reference kernels.
 *
 * Both are exhaustive searches: v0.1 ships no spatial acceleration structure, so the complexity
 * bounds O(P*R) and O(P*F) are contractual rather than incidental. Selection orders by
 * (distance_squared, index) so exact ties resolve to the lower global index, and unfilled or
 * impossible results carry the +inf / -1 / false sentinels.
 */

#include <gffx/execution.h>
#include <gffx/points.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

static gffx_status gffx_points_zero_workspace(
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
            "proximity operations support the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "proximity operations implement only the CPU backend in this phase"
        );
    }
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

/* Validates one packed-offset array against a total count and batch count. */
static gffx_status gffx_points_check_offsets(
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
    if ((offsets->flags & GFFX_TENSOR_OUTPUT) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation inputs may not carry the output flag"
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

/* ------------------------------------------------------------------------- points.knn */

GFFX_API gffx_status GFFX_CALL gffx_points_knn_workspace(
    int64_t query_count,
    int64_t reference_count,
    int64_t neighbor_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    if (query_count < INT64_C(0) || reference_count < INT64_C(0)) {
        gffx_status prepared = gffx_internal_prepare_diagnostic(diagnostic);
        if (prepared != GFFX_STATUS_OK) return prepared;
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "counts must be nonnegative"
        );
    }
    if (neighbor_count <= INT64_C(0)) {
        gffx_status prepared = gffx_internal_prepare_diagnostic(diagnostic);
        if (prepared != GFFX_STATUS_OK) return prepared;
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the neighbor count must be positive"
        );
    }
    return gffx_points_zero_workspace(dtype, context, required_bytes, required_alignment,
                                      diagnostic);
}

/* Shared query/reference/offset validation for both knn directions. */
static gffx_status gffx_knn_check_inputs(
    const gffx_tensor_view *query,
    const gffx_tensor_view *reference,
    int64_t neighbor_count,
    const gffx_execution_context *context,
    int64_t *query_count,
    int64_t *reference_count,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    if (neighbor_count <= INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the neighbor count must be positive"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "points.knn implements only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(query, "query must be a [P,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (query->dtype != GFFX_DTYPE_FLOAT32 && query->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "query points must use the float32 or float64 dtype"
        );
    }
    status = gffx_mesh_check_view(reference, "reference must be a [R,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (reference->dtype != query->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "reference points must match the query dtype"
        );
    }
    *query_count = query->shape[0];
    *reference_count = reference->shape[0];
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_points_knn(
    const gffx_tensor_view *query,
    const gffx_tensor_view *reference,
    const gffx_tensor_view *query_offsets,
    const gffx_tensor_view *reference_offsets,
    int64_t neighbor_count,
    const gffx_execution_context *context,
    gffx_tensor_view *distance_squared,
    gffx_tensor_view *reference_index,
    gffx_tensor_view *valid,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t query_count = 0;
    int64_t reference_count = 0;
    int64_t batch_count;
    int64_t batch;
    int64_t point;
    int64_t slot;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_knn_check_inputs(query, reference, neighbor_count, context, &query_count,
                                   &reference_count, diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    if (query_offsets == NULL || query_offsets->rank != 1u || query_offsets->shape == NULL ||
        query_offsets->shape[0] < INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "query offsets must be a [B+1] tensor view"
        );
    }
    batch_count = query_offsets->shape[0] - INT64_C(1);
    status = gffx_points_check_offsets(query_offsets, query_count, batch_count,
                                       "query offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_points_check_offsets(reference_offsets, reference_count, batch_count,
                                       "reference offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    status = gffx_mesh_check_view(distance_squared,
                                  "distances must be a [P,K] output view",
                                  2u, query_count, neighbor_count, 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (distance_squared->dtype != query->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "distances must match the query dtype"
        );
    }
    status = gffx_mesh_check_view(reference_index, "indices must be a [P,K] output view",
                                  2u, query_count, neighbor_count, 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (reference_index->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "indices must use the int32 dtype"
        );
    }
    status = gffx_mesh_check_view(valid, "valid must be a [P,K] output view",
                                  2u, query_count, neighbor_count, 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (valid->dtype != GFFX_DTYPE_BOOL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "valid must use the bool dtype"
        );
    }
    if (gffx_mesh_views_overlap(distance_squared, query) ||
        gffx_mesh_views_overlap(distance_squared, reference) ||
        gffx_mesh_views_overlap(reference_index, query) ||
        gffx_mesh_views_overlap(reference_index, reference) ||
        gffx_mesh_views_overlap(valid, query) || gffx_mesh_views_overlap(valid, reference) ||
        gffx_mesh_views_overlap(distance_squared, reference_index) ||
        gffx_mesh_views_overlap(distance_squared, valid) ||
        gffx_mesh_views_overlap(reference_index, valid)) {
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
    if (query_count == INT64_C(0)) return GFFX_STATUS_OK;

    {
        const int32_t *query_bounds = (const int32_t *)gffx_mesh_elements_const(query_offsets);
        const int32_t *reference_bounds =
            (const int32_t *)gffx_mesh_elements_const(reference_offsets);
        int32_t *index_data = (int32_t *)gffx_mesh_elements(reference_index);
        uint8_t *valid_data = (uint8_t *)gffx_mesh_elements(valid);

#define GFFX_KNN_SELECT(scalar_type, distance_ptr, query_ptr, reference_ptr, infinity_value)   \
        do {                                                                                   \
            for (batch = 0; batch < batch_count; ++batch) {                                    \
                int64_t first_reference = (int64_t)reference_bounds[batch];                    \
                int64_t last_reference = (int64_t)reference_bounds[batch + 1];                 \
                for (point = (int64_t)query_bounds[batch];                                     \
                     point < (int64_t)query_bounds[batch + 1]; ++point) {                      \
                    scalar_type qx = (query_ptr)[point * 3 + 0];                               \
                    scalar_type qy = (query_ptr)[point * 3 + 1];                               \
                    scalar_type qz = (query_ptr)[point * 3 + 2];                               \
                    int64_t candidate;                                                         \
                    for (slot = 0; slot < neighbor_count; ++slot) {                            \
                        (distance_ptr)[point * neighbor_count + slot] = (infinity_value);      \
                        index_data[point * neighbor_count + slot] = -1;                        \
                        valid_data[point * neighbor_count + slot] = 0u;                        \
                    }                                                                          \
                    for (candidate = first_reference; candidate < last_reference; ++candidate) {\
                        scalar_type dx = qx - (reference_ptr)[candidate * 3 + 0];               \
                        scalar_type dy = qy - (reference_ptr)[candidate * 3 + 1];               \
                        scalar_type dz = qz - (reference_ptr)[candidate * 3 + 2];               \
                        scalar_type value = dx * dx + dy * dy + dz * dz;                        \
                        int64_t position = neighbor_count;                                      \
                        /* Insertion keeps the list ordered by (distance, index); scanning     \
                         * references in ascending order makes an exact tie keep the earlier,  \
                         * lower-index entry ahead of the newcomer. */                          \
                        while (position > 0) {                                                  \
                            int64_t above = position - 1;                                        \
                            int is_padding =                                                     \
                                valid_data[point * neighbor_count + above] == 0u;                \
                            if (is_padding ||                                                    \
                                value < (distance_ptr)[point * neighbor_count + above]) {        \
                                --position;                                                      \
                            } else {                                                             \
                                break;                                                           \
                            }                                                                    \
                        }                                                                        \
                        if (position >= neighbor_count) continue;                                \
                        for (slot = neighbor_count - 1; slot > position; --slot) {               \
                            (distance_ptr)[point * neighbor_count + slot] =                      \
                                (distance_ptr)[point * neighbor_count + slot - 1];               \
                            index_data[point * neighbor_count + slot] =                          \
                                index_data[point * neighbor_count + slot - 1];                   \
                            valid_data[point * neighbor_count + slot] =                          \
                                valid_data[point * neighbor_count + slot - 1];                   \
                        }                                                                        \
                        (distance_ptr)[point * neighbor_count + position] = value;               \
                        index_data[point * neighbor_count + position] = (int32_t)candidate;      \
                        valid_data[point * neighbor_count + position] = 1u;                      \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        } while (0)

        if (query->dtype == GFFX_DTYPE_FLOAT64) {
            const double *query_data = (const double *)gffx_mesh_elements_const(query);
            const double *reference_data = (const double *)gffx_mesh_elements_const(reference);
            double *distance_data = (double *)gffx_mesh_elements(distance_squared);
            GFFX_KNN_SELECT(double, distance_data, query_data, reference_data, (double)INFINITY);
        } else {
            const float *query_data = (const float *)gffx_mesh_elements_const(query);
            const float *reference_data = (const float *)gffx_mesh_elements_const(reference);
            float *distance_data = (float *)gffx_mesh_elements(distance_squared);
            GFFX_KNN_SELECT(float, distance_data, query_data, reference_data, (float)INFINITY);
        }
#undef GFFX_KNN_SELECT
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_points_knn_backward(
    const gffx_tensor_view *query,
    const gffx_tensor_view *reference,
    const gffx_tensor_view *reference_index,
    const gffx_tensor_view *valid,
    const gffx_tensor_view *grad_distance_squared,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_query,
    gffx_tensor_view *grad_reference,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t query_count = 0;
    int64_t reference_count = 0;
    int64_t neighbor_count;
    int64_t index;
    int64_t point;
    int64_t slot;
    if (status != GFFX_STATUS_OK) return status;
    if (reference_index == NULL || reference_index->rank != 2u ||
        reference_index->shape == NULL || reference_index->shape[1] < INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "indices must be a [P,K] tensor view"
        );
    }
    neighbor_count = reference_index->shape[1];
    status = gffx_knn_check_inputs(query, reference, neighbor_count, context, &query_count,
                                   &reference_count, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_query == NULL && grad_reference == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "at least one gradient output is required"
        );
    }
    status = gffx_mesh_check_view(reference_index, "indices must be a [P,K] tensor view",
                                  2u, query_count, neighbor_count, 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (reference_index->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "indices must use the int32 dtype"
        );
    }
    status = gffx_mesh_check_view(valid, "valid must be a [P,K] tensor view",
                                  2u, query_count, neighbor_count, 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (valid->dtype != GFFX_DTYPE_BOOL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "valid must use the bool dtype"
        );
    }
    status = gffx_mesh_check_view(grad_distance_squared,
                                  "distance cotangents must be a [P,K] tensor view",
                                  2u, query_count, neighbor_count, 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_distance_squared->dtype != query->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "cotangents must match the query dtype"
        );
    }
    if (grad_query != NULL) {
        status = gffx_mesh_check_view(grad_query, "query gradients must be a [P,3] output view",
                                      2u, query_count, INT64_C(3), 1, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (grad_query->dtype != query->dtype) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "query gradients must match the query dtype"
            );
        }
    }
    if (grad_reference != NULL) {
        status = gffx_mesh_check_view(grad_reference,
                                      "reference gradients must be a [R,3] output view",
                                      2u, reference_count, INT64_C(3), 1, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (grad_reference->dtype != query->dtype) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "reference gradients must match the query dtype"
            );
        }
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    {
        const int32_t *index_data = (const int32_t *)gffx_mesh_elements_const(reference_index);
        const uint8_t *valid_data = (const uint8_t *)gffx_mesh_elements_const(valid);
        if (query->dtype == GFFX_DTYPE_FLOAT64) {
            const double *query_data = (const double *)gffx_mesh_elements_const(query);
            const double *reference_data = (const double *)gffx_mesh_elements_const(reference);
            const double *cotangent =
                (const double *)gffx_mesh_elements_const(grad_distance_squared);
            double *query_gradient =
                grad_query != NULL ? (double *)gffx_mesh_elements(grad_query) : NULL;
            double *reference_gradient =
                grad_reference != NULL ? (double *)gffx_mesh_elements(grad_reference) : NULL;
            if (query_gradient != NULL) {
                for (index = 0; index < query_count * INT64_C(3); ++index) {
                    query_gradient[index] = 0.0;
                }
            }
            if (reference_gradient != NULL) {
                for (index = 0; index < reference_count * INT64_C(3); ++index) {
                    reference_gradient[index] = 0.0;
                }
            }
            for (point = 0; point < query_count; ++point) {
                for (slot = 0; slot < neighbor_count; ++slot) {
                    int64_t entry = point * neighbor_count + slot;
                    int64_t neighbor;
                    double scale;
                    int axis;
                    if (valid_data[entry] == 0u) continue;
                    neighbor = (int64_t)index_data[entry];
                    if (neighbor < 0 || neighbor >= reference_count) {
                        return gffx_internal_fail(
                            diagnostic,
                            GFFX_STATUS_INVALID_ARGUMENT,
                            "a valid neighbor index lies outside the reference range"
                        );
                    }
                    scale = 2.0 * cotangent[entry];
                    for (axis = 0; axis < 3; ++axis) {
                        double delta = query_data[point * 3 + axis] -
                                       reference_data[neighbor * 3 + axis];
                        if (query_gradient != NULL) {
                            query_gradient[point * 3 + axis] += scale * delta;
                        }
                        if (reference_gradient != NULL) {
                            reference_gradient[neighbor * 3 + axis] -= scale * delta;
                        }
                    }
                }
            }
        } else {
            const float *query_data = (const float *)gffx_mesh_elements_const(query);
            const float *reference_data = (const float *)gffx_mesh_elements_const(reference);
            const float *cotangent =
                (const float *)gffx_mesh_elements_const(grad_distance_squared);
            float *query_gradient =
                grad_query != NULL ? (float *)gffx_mesh_elements(grad_query) : NULL;
            float *reference_gradient =
                grad_reference != NULL ? (float *)gffx_mesh_elements(grad_reference) : NULL;
            if (query_gradient != NULL) {
                for (index = 0; index < query_count * INT64_C(3); ++index) {
                    query_gradient[index] = 0.0f;
                }
            }
            if (reference_gradient != NULL) {
                for (index = 0; index < reference_count * INT64_C(3); ++index) {
                    reference_gradient[index] = 0.0f;
                }
            }
            for (point = 0; point < query_count; ++point) {
                for (slot = 0; slot < neighbor_count; ++slot) {
                    int64_t entry = point * neighbor_count + slot;
                    int64_t neighbor;
                    float scale;
                    int axis;
                    if (valid_data[entry] == 0u) continue;
                    neighbor = (int64_t)index_data[entry];
                    if (neighbor < 0 || neighbor >= reference_count) {
                        return gffx_internal_fail(
                            diagnostic,
                            GFFX_STATUS_INVALID_ARGUMENT,
                            "a valid neighbor index lies outside the reference range"
                        );
                    }
                    scale = 2.0f * cotangent[entry];
                    for (axis = 0; axis < 3; ++axis) {
                        float delta = query_data[point * 3 + axis] -
                                      reference_data[neighbor * 3 + axis];
                        if (query_gradient != NULL) {
                            query_gradient[point * 3 + axis] += scale * delta;
                        }
                        if (reference_gradient != NULL) {
                            reference_gradient[neighbor * 3 + axis] -= scale * delta;
                        }
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

/* --------------------------------------------------- points.closest_point_on_mesh */

/* Closest point on a triangle, by the standard region decomposition evaluated in a fixed
 * order. Returns the barycentric weights; the caller reconstructs the position. */
static void gffx_closest_barycentric(
    double px, double py, double pz,
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double *b0, double *b1, double *b2
) {
    double abx = bx - ax, aby = by - ay, abz = bz - az;
    double acx = cx - ax, acy = cy - ay, acz = cz - az;
    double apx = px - ax, apy = py - ay, apz = pz - az;
    double d1 = abx * apx + aby * apy + abz * apz;
    double d2 = acx * apx + acy * apy + acz * apz;
    double bpx, bpy, bpz, d3, d4, vc;
    double cpx, cpy, cpz, d5, d6, vb, va;
    double denom, v, w;

    if (d1 <= 0.0 && d2 <= 0.0) { *b0 = 1.0; *b1 = 0.0; *b2 = 0.0; return; }

    bpx = px - bx; bpy = py - by; bpz = pz - bz;
    d3 = abx * bpx + aby * bpy + abz * bpz;
    d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0.0 && d4 <= d3) { *b0 = 0.0; *b1 = 1.0; *b2 = 0.0; return; }

    vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        v = d1 / (d1 - d3);
        *b0 = 1.0 - v; *b1 = v; *b2 = 0.0;
        return;
    }

    cpx = px - cx; cpy = py - cy; cpz = pz - cz;
    d5 = abx * cpx + aby * cpy + abz * cpz;
    d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0.0 && d5 <= d6) { *b0 = 0.0; *b1 = 0.0; *b2 = 1.0; return; }

    vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        w = d2 / (d2 - d6);
        *b0 = 1.0 - w; *b1 = 0.0; *b2 = w;
        return;
    }

    va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *b0 = 0.0; *b1 = 1.0 - w; *b2 = w;
        return;
    }

    denom = 1.0 / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    *b0 = 1.0 - v - w; *b1 = v; *b2 = w;
}

GFFX_API gffx_status GFFX_CALL gffx_points_closest_point_on_mesh_workspace(
    int64_t point_count,
    int64_t vertex_count,
    int64_t face_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    if (point_count < INT64_C(0) || vertex_count < INT64_C(0) || face_count < INT64_C(0)) {
        gffx_status prepared = gffx_internal_prepare_diagnostic(diagnostic);
        if (prepared != GFFX_STATUS_OK) return prepared;
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "counts must be nonnegative"
        );
    }
    return gffx_points_zero_workspace(dtype, context, required_bytes, required_alignment,
                                      diagnostic);
}

GFFX_API gffx_status GFFX_CALL gffx_points_closest_point_on_mesh(
    const gffx_tensor_view *points,
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *point_offsets,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *distance_squared,
    gffx_tensor_view *face_index,
    gffx_tensor_view *barycentric,
    gffx_tensor_view *closest,
    gffx_tensor_view *valid,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t point_count;
    int64_t vertex_count;
    int64_t face_count;
    int64_t batch_count;
    int64_t batch;
    int64_t point;
    if (status != GFFX_STATUS_OK) return status;
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
            "points.closest_point_on_mesh implements only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(points, "points must be a [P,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (points->dtype != GFFX_DTYPE_FLOAT32 && points->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "points must use the float32 or float64 dtype"
        );
    }
    status = gffx_mesh_check_view(vertices, "vertices must be a [V,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (vertices->dtype != points->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertices must match the points dtype"
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
    point_count = points->shape[0];
    vertex_count = vertices->shape[0];
    face_count = faces->shape[0];

    if (point_offsets == NULL || point_offsets->rank != 1u || point_offsets->shape == NULL ||
        point_offsets->shape[0] < INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "point offsets must be a [B+1] tensor view"
        );
    }
    batch_count = point_offsets->shape[0] - INT64_C(1);
    status = gffx_points_check_offsets(point_offsets, point_count, batch_count,
                                       "point offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_points_check_offsets(vertex_offsets, vertex_count, batch_count,
                                       "vertex offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_points_check_offsets(face_offsets, face_count, batch_count,
                                       "face offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    status = gffx_mesh_check_view(distance_squared, "distances must be a [P] output view",
                                  1u, point_count, INT64_C(0), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (distance_squared->dtype != points->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "distances must match the points dtype"
        );
    }
    status = gffx_mesh_check_view(face_index, "face indices must be a [P] output view",
                                  1u, point_count, INT64_C(0), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (face_index->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must use the int32 dtype"
        );
    }
    status = gffx_mesh_check_view(barycentric, "barycentrics must be a [P,3] output view",
                                  2u, point_count, INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (barycentric->dtype != points->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "barycentrics must match the points dtype"
        );
    }
    status = gffx_mesh_check_view(closest, "closest points must be a [P,3] output view",
                                  2u, point_count, INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (closest->dtype != points->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "closest points must match the points dtype"
        );
    }
    status = gffx_mesh_check_view(valid, "valid must be a [P] output view",
                                  1u, point_count, INT64_C(0), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (valid->dtype != GFFX_DTYPE_BOOL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "valid must use the bool dtype"
        );
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    /* Every face must reference vertices of its own batch element. */
    if (face_count > INT64_C(0)) {
        const int32_t *face_data = (const int32_t *)gffx_mesh_elements_const(faces);
        const int32_t *vertex_bounds =
            (const int32_t *)gffx_mesh_elements_const(vertex_offsets);
        const int32_t *face_bounds = (const int32_t *)gffx_mesh_elements_const(face_offsets);
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
    }
    if (point_count == INT64_C(0)) return GFFX_STATUS_OK;

    {
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const int32_t *point_bounds = (const int32_t *)gffx_mesh_elements_const(point_offsets);
        const int32_t *face_bounds = (const int32_t *)gffx_mesh_elements_const(face_offsets);
        int32_t *face_index_data = (int32_t *)gffx_mesh_elements(face_index);
        uint8_t *valid_data = (uint8_t *)gffx_mesh_elements(valid);
        int is_double = points->dtype == GFFX_DTYPE_FLOAT64;
        const double *point_d = is_double ? (const double *)gffx_mesh_elements_const(points) : NULL;
        const float *point_f = is_double ? NULL : (const float *)gffx_mesh_elements_const(points);
        const double *vertex_d =
            is_double ? (const double *)gffx_mesh_elements_const(vertices) : NULL;
        const float *vertex_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(vertices);
        double *distance_d = is_double ? (double *)gffx_mesh_elements(distance_squared) : NULL;
        float *distance_f = is_double ? NULL : (float *)gffx_mesh_elements(distance_squared);
        double *bary_d = is_double ? (double *)gffx_mesh_elements(barycentric) : NULL;
        float *bary_f = is_double ? NULL : (float *)gffx_mesh_elements(barycentric);
        double *closest_d = is_double ? (double *)gffx_mesh_elements(closest) : NULL;
        float *closest_f = is_double ? NULL : (float *)gffx_mesh_elements(closest);

        for (batch = 0; batch < batch_count; ++batch) {
            int64_t first_face = (int64_t)face_bounds[batch];
            int64_t last_face = (int64_t)face_bounds[batch + 1];
            for (point = (int64_t)point_bounds[batch];
                 point < (int64_t)point_bounds[batch + 1]; ++point) {
                double px = is_double ? point_d[point * 3 + 0] : (double)point_f[point * 3 + 0];
                double py = is_double ? point_d[point * 3 + 1] : (double)point_f[point * 3 + 1];
                double pz = is_double ? point_d[point * 3 + 2] : (double)point_f[point * 3 + 2];
                double best_distance = (double)INFINITY;
                double best_b0 = 0.0, best_b1 = 0.0, best_b2 = 0.0;
                double best_cx = 0.0, best_cy = 0.0, best_cz = 0.0;
                int64_t best_face = -1;
                int64_t face;
                for (face = first_face; face < last_face; ++face) {
                    int64_t i0 = (int64_t)face_data[face * 3 + 0];
                    int64_t i1 = (int64_t)face_data[face * 3 + 1];
                    int64_t i2 = (int64_t)face_data[face * 3 + 2];
                    double ax = is_double ? vertex_d[i0 * 3 + 0] : (double)vertex_f[i0 * 3 + 0];
                    double ay = is_double ? vertex_d[i0 * 3 + 1] : (double)vertex_f[i0 * 3 + 1];
                    double az = is_double ? vertex_d[i0 * 3 + 2] : (double)vertex_f[i0 * 3 + 2];
                    double bx = is_double ? vertex_d[i1 * 3 + 0] : (double)vertex_f[i1 * 3 + 0];
                    double by = is_double ? vertex_d[i1 * 3 + 1] : (double)vertex_f[i1 * 3 + 1];
                    double bz = is_double ? vertex_d[i1 * 3 + 2] : (double)vertex_f[i1 * 3 + 2];
                    double cx = is_double ? vertex_d[i2 * 3 + 0] : (double)vertex_f[i2 * 3 + 0];
                    double cy = is_double ? vertex_d[i2 * 3 + 1] : (double)vertex_f[i2 * 3 + 1];
                    double cz = is_double ? vertex_d[i2 * 3 + 2] : (double)vertex_f[i2 * 3 + 2];
                    double e1x = bx - ax, e1y = by - ay, e1z = bz - az;
                    double e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
                    double nx = e1y * e2z - e1z * e2y;
                    double ny = e1z * e2x - e1x * e2z;
                    double nz = e1x * e2y - e1y * e2x;
                    double doubled = sqrt(nx * nx + ny * ny + nz * nz);
                    double b0, b1, b2, qx, qy, qz, dx, dy, dz, candidate;
                    /* Degenerate triangles are skipped, matching mesh.face_geometry validity. */
                    if (!(doubled > eps)) continue;
                    gffx_closest_barycentric(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz,
                                             &b0, &b1, &b2);
                    qx = b0 * ax + b1 * bx + b2 * cx;
                    qy = b0 * ay + b1 * by + b2 * cy;
                    qz = b0 * az + b1 * bz + b2 * cz;
                    dx = px - qx; dy = py - qy; dz = pz - qz;
                    candidate = dx * dx + dy * dy + dz * dz;
                    /* Strict improvement keeps the lower face index on an exact tie. */
                    if (candidate < best_distance) {
                        best_distance = candidate;
                        best_face = face;
                        best_b0 = b0; best_b1 = b1; best_b2 = b2;
                        best_cx = qx; best_cy = qy; best_cz = qz;
                    }
                }
                if (best_face < 0) {
                    if (is_double) {
                        distance_d[point] = (double)INFINITY;
                        bary_d[point * 3 + 0] = 0.0;
                        bary_d[point * 3 + 1] = 0.0;
                        bary_d[point * 3 + 2] = 0.0;
                        closest_d[point * 3 + 0] = 0.0;
                        closest_d[point * 3 + 1] = 0.0;
                        closest_d[point * 3 + 2] = 0.0;
                    } else {
                        distance_f[point] = (float)INFINITY;
                        bary_f[point * 3 + 0] = 0.0f;
                        bary_f[point * 3 + 1] = 0.0f;
                        bary_f[point * 3 + 2] = 0.0f;
                        closest_f[point * 3 + 0] = 0.0f;
                        closest_f[point * 3 + 1] = 0.0f;
                        closest_f[point * 3 + 2] = 0.0f;
                    }
                    face_index_data[point] = -1;
                    valid_data[point] = 0u;
                } else {
                    if (is_double) {
                        distance_d[point] = best_distance;
                        bary_d[point * 3 + 0] = best_b0;
                        bary_d[point * 3 + 1] = best_b1;
                        bary_d[point * 3 + 2] = best_b2;
                        closest_d[point * 3 + 0] = best_cx;
                        closest_d[point * 3 + 1] = best_cy;
                        closest_d[point * 3 + 2] = best_cz;
                    } else {
                        distance_f[point] = (float)best_distance;
                        bary_f[point * 3 + 0] = (float)best_b0;
                        bary_f[point * 3 + 1] = (float)best_b1;
                        bary_f[point * 3 + 2] = (float)best_b2;
                        closest_f[point * 3 + 0] = (float)best_cx;
                        closest_f[point * 3 + 1] = (float)best_cy;
                        closest_f[point * 3 + 2] = (float)best_cz;
                    }
                    face_index_data[point] = (int32_t)best_face;
                    valid_data[point] = 1u;
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_points_closest_point_on_mesh_backward(
    const gffx_tensor_view *points,
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,
    const gffx_tensor_view *closest,
    const gffx_tensor_view *valid,
    const gffx_tensor_view *grad_distance_squared,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_points,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t point_count;
    int64_t vertex_count;
    int64_t face_count;
    int64_t index;
    int64_t point;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "points.closest_point_on_mesh implements only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(points, "points must be a [P,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (points->dtype != GFFX_DTYPE_FLOAT32 && points->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "points must use the float32 or float64 dtype"
        );
    }
    status = gffx_mesh_check_view(vertices, "vertices must be a [V,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
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
    point_count = points->shape[0];
    vertex_count = vertices->shape[0];
    face_count = faces->shape[0];

    if (grad_points == NULL && grad_vertices == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "at least one gradient output is required"
        );
    }
    status = gffx_mesh_check_view(face_index, "face indices must be a [P] tensor view",
                                  1u, point_count, INT64_C(0), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_view(barycentric, "barycentrics must be a [P,3] tensor view",
                                  2u, point_count, INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_view(closest, "closest points must be a [P,3] tensor view",
                                  2u, point_count, INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_view(valid, "valid must be a [P] tensor view",
                                  1u, point_count, INT64_C(0), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_view(grad_distance_squared,
                                  "distance cotangents must be a [P] tensor view",
                                  1u, point_count, INT64_C(0), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_points != NULL) {
        status = gffx_mesh_check_view(grad_points, "point gradients must be a [P,3] output view",
                                      2u, point_count, INT64_C(3), 1, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    if (grad_vertices != NULL) {
        status = gffx_mesh_check_view(grad_vertices,
                                      "vertex gradients must be a [V,3] output view",
                                      2u, vertex_count, INT64_C(3), 1, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    {
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const int32_t *face_index_data = (const int32_t *)gffx_mesh_elements_const(face_index);
        const uint8_t *valid_data = (const uint8_t *)gffx_mesh_elements_const(valid);
        int is_double = points->dtype == GFFX_DTYPE_FLOAT64;
        const double *point_d = is_double ? (const double *)gffx_mesh_elements_const(points) : NULL;
        const float *point_f = is_double ? NULL : (const float *)gffx_mesh_elements_const(points);
        const double *bary_d =
            is_double ? (const double *)gffx_mesh_elements_const(barycentric) : NULL;
        const float *bary_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(barycentric);
        const double *closest_d =
            is_double ? (const double *)gffx_mesh_elements_const(closest) : NULL;
        const float *closest_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(closest);
        const double *cotangent_d =
            is_double ? (const double *)gffx_mesh_elements_const(grad_distance_squared) : NULL;
        const float *cotangent_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(grad_distance_squared);
        double *point_gradient_d =
            (is_double && grad_points != NULL) ? (double *)gffx_mesh_elements(grad_points) : NULL;
        float *point_gradient_f =
            (!is_double && grad_points != NULL) ? (float *)gffx_mesh_elements(grad_points) : NULL;
        double *vertex_gradient_d =
            (is_double && grad_vertices != NULL) ? (double *)gffx_mesh_elements(grad_vertices)
                                                 : NULL;
        float *vertex_gradient_f =
            (!is_double && grad_vertices != NULL) ? (float *)gffx_mesh_elements(grad_vertices)
                                                  : NULL;

        if (point_gradient_d != NULL) {
            for (index = 0; index < point_count * INT64_C(3); ++index) point_gradient_d[index] = 0.0;
        }
        if (point_gradient_f != NULL) {
            for (index = 0; index < point_count * INT64_C(3); ++index) point_gradient_f[index] = 0.0f;
        }
        if (vertex_gradient_d != NULL) {
            for (index = 0; index < vertex_count * INT64_C(3); ++index) {
                vertex_gradient_d[index] = 0.0;
            }
        }
        if (vertex_gradient_f != NULL) {
            for (index = 0; index < vertex_count * INT64_C(3); ++index) {
                vertex_gradient_f[index] = 0.0f;
            }
        }

        for (point = 0; point < point_count; ++point) {
            int64_t face;
            double g;
            double weights[3];
            int64_t corner_index[3];
            int axis;
            if (valid_data[point] == 0u) continue;
            face = (int64_t)face_index_data[point];
            if (face < 0 || face >= face_count) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "a valid face index lies outside the face range"
                );
            }
            g = is_double ? cotangent_d[point] : (double)cotangent_f[point];
            weights[0] = is_double ? bary_d[point * 3 + 0] : (double)bary_f[point * 3 + 0];
            weights[1] = is_double ? bary_d[point * 3 + 1] : (double)bary_f[point * 3 + 1];
            weights[2] = is_double ? bary_d[point * 3 + 2] : (double)bary_f[point * 3 + 2];
            corner_index[0] = (int64_t)face_data[face * 3 + 0];
            corner_index[1] = (int64_t)face_data[face * 3 + 1];
            corner_index[2] = (int64_t)face_data[face * 3 + 2];
            for (axis = 0; axis < 3; ++axis) {
                double px = is_double ? point_d[point * 3 + axis]
                                      : (double)point_f[point * 3 + axis];
                double qx = is_double ? closest_d[point * 3 + axis]
                                      : (double)closest_f[point * 3 + axis];
                /* Envelope theorem: at the optimum the residual is orthogonal to every
                 * direction the closest feature allows, so the barycentric weights are held
                 * fixed and only the residual contributes. */
                double residual = 2.0 * (px - qx) * g;
                int corner;
                if (point_gradient_d != NULL) point_gradient_d[point * 3 + axis] += residual;
                if (point_gradient_f != NULL) {
                    point_gradient_f[point * 3 + axis] += (float)residual;
                }
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = corner_index[corner];
                    if (vertex < 0 || vertex >= vertex_count) {
                        return gffx_internal_fail(
                            diagnostic,
                            GFFX_STATUS_INVALID_ARGUMENT,
                            "a face references a vertex outside the vertex range"
                        );
                    }
                    if (vertex_gradient_d != NULL) {
                        vertex_gradient_d[vertex * 3 + axis] -= weights[corner] * residual;
                    }
                    if (vertex_gradient_f != NULL) {
                        vertex_gradient_f[vertex * 3 + axis] -= (float)(weights[corner] * residual);
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}
