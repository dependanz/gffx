/*
 * mesh.sample_surface - Phase 2 CPU reference kernels.
 *
 * Randomness is Philox4x32-10, counter-based and stateless. The 128-bit counter for sample
 * (b, s) embeds the batch and sample indices, so a sample's value never depends on how many
 * samples preceded it and the operation stays reproducible under any future parallelization.
 * Face selection is area-weighted through a cumulative table accumulated in double precision
 * regardless of the operand dtype, because a float32 running sum loses monotonicity over many
 * faces and would bias the search.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

#define GFFX_PHILOX_M0 UINT32_C(0xD2511F53)
#define GFFX_PHILOX_M1 UINT32_C(0xCD9E8D57)
#define GFFX_PHILOX_W0 UINT32_C(0x9E3779B9)
#define GFFX_PHILOX_W1 UINT32_C(0xBB67AE85)
#define GFFX_PHILOX_ROUNDS 10

/* Philox4x32-10. Ten rounds over a four-word counter with a two-word key bumped between
 * rounds; the multiply-high halves supply the diffusion. */
static void gffx_philox4x32_10(const uint32_t counter[4], const uint32_t key[2],
                               uint32_t output[4]) {
    uint32_t state[4];
    uint32_t local_key[2];
    int round;
    state[0] = counter[0];
    state[1] = counter[1];
    state[2] = counter[2];
    state[3] = counter[3];
    local_key[0] = key[0];
    local_key[1] = key[1];
    for (round = 0; round < GFFX_PHILOX_ROUNDS; ++round) {
        uint64_t product0;
        uint64_t product1;
        uint32_t high0;
        uint32_t low0;
        uint32_t high1;
        uint32_t low1;
        if (round > 0) {
            local_key[0] += GFFX_PHILOX_W0;
            local_key[1] += GFFX_PHILOX_W1;
        }
        product0 = (uint64_t)GFFX_PHILOX_M0 * (uint64_t)state[0];
        product1 = (uint64_t)GFFX_PHILOX_M1 * (uint64_t)state[2];
        high0 = (uint32_t)(product0 >> 32);
        low0 = (uint32_t)product0;
        high1 = (uint32_t)(product1 >> 32);
        low1 = (uint32_t)product1;
        {
            uint32_t next0 = high1 ^ state[1] ^ local_key[0];
            uint32_t next1 = low1;
            uint32_t next2 = high0 ^ state[3] ^ local_key[1];
            uint32_t next3 = low0;
            state[0] = next0;
            state[1] = next1;
            state[2] = next2;
            state[3] = next3;
        }
    }
    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

static double gffx_uniform_from_word(uint32_t word) {
    /* 2^-32 exactly; the result lies in [0, 1). */
    return (double)word * 2.3283064365386963e-10;
}

static gffx_status gffx_sample_check_rng_view(
    const gffx_tensor_view *view,
    const char *role_message,
    int is_output,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_mesh_check_view(view, role_message, 1u, INT64_C(2), INT64_C(0),
                                              is_output, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (view->dtype != GFFX_DTYPE_UINT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "generator key and counter must use the uint32 dtype"
        );
    }
    return GFFX_STATUS_OK;
}

static gffx_status gffx_sample_check_offsets(
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

GFFX_API gffx_status GFFX_CALL gffx_mesh_sample_surface_workspace(
    int64_t vertex_count,
    int64_t face_count,
    int64_t sample_count,
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
    if (vertex_count < INT64_C(0) || face_count < INT64_C(0) || sample_count < INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "counts must be nonnegative"
        );
    }
    if (dtype != GFFX_DTYPE_FLOAT32 && dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.sample_surface supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.sample_surface implements only the CPU backend in this phase"
        );
    }
    if ((uint64_t)face_count > UINT64_MAX / (uint64_t)sizeof(double)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_OVERFLOW,
            "workspace byte requirement overflows 64-bit capacity"
        );
    }
    /* The cumulative area table is always double, so the requirement is dtype-independent. */
    *required_bytes = (uint64_t)face_count * (uint64_t)sizeof(double);
    *required_alignment = (uint64_t)sizeof(double);
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_sample_surface(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    int64_t sample_count,
    const gffx_tensor_view *rng_key,
    const gffx_tensor_view *rng_counter,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *points,
    gffx_tensor_view *face_index,
    gffx_tensor_view *barycentric,
    gffx_tensor_view *next_counter,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t vertex_count;
    int64_t face_count;
    int64_t batch_count;
    int64_t batch;
    uint64_t required_bytes;
    if (status != GFFX_STATUS_OK) return status;
    if (sample_count < INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the sample count must be nonnegative"
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
            "mesh.sample_surface implements only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(vertices, "vertices must be a [V,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (vertices->dtype != GFFX_DTYPE_FLOAT32 && vertices->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "vertices must use the float32 or float64 dtype"
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
    vertex_count = vertices->shape[0];
    face_count = faces->shape[0];

    status = gffx_sample_check_rng_view(rng_key, "the generator key must be a [2] uint32 view",
                                        0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_sample_check_rng_view(rng_counter,
                                        "the generator counter must be a [2] uint32 view",
                                        0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_sample_check_rng_view(next_counter,
                                        "the next counter must be a [2] uint32 output view",
                                        1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    if (next_counter == NULL || vertex_offsets == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "required arguments must not be null"
        );
    }
    if (vertex_offsets->rank != 1u || vertex_offsets->shape == NULL ||
        vertex_offsets->shape[0] < INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertex offsets must be a [B+1] tensor view"
        );
    }
    batch_count = vertex_offsets->shape[0] - INT64_C(1);
    status = gffx_sample_check_offsets(vertex_offsets, vertex_count, batch_count,
                                       "vertex offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_sample_check_offsets(face_offsets, face_count, batch_count,
                                       "face offsets must satisfy the packed-offset rules",
                                       diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    /* The counter advance is reported even when no sample is requested. */
    {
        const uint32_t *counter_data = (const uint32_t *)gffx_mesh_elements_const(rng_counter);
        uint32_t *next_data = (uint32_t *)gffx_mesh_elements(next_counter);
        uint32_t low = counter_data[0] + 1u;
        uint32_t high = counter_data[1] + (low == 0u ? 1u : 0u);
        next_data[0] = low;
        next_data[1] = high;
    }
    if (sample_count == INT64_C(0) || batch_count == INT64_C(0)) return GFFX_STATUS_OK;

    status = gffx_mesh_check_view(points, "points must be a [B,S,3] output view",
                                  3u, batch_count, INT64_C(0), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (points->shape[1] != sample_count || points->shape[2] != INT64_C(3)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "points must be a [B,S,3] output view"
        );
    }
    if (points->dtype != vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "points must match the vertices dtype"
        );
    }
    status = gffx_mesh_check_view(barycentric, "barycentrics must be a [B,S,3] output view",
                                  3u, batch_count, INT64_C(0), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (barycentric->shape[1] != sample_count || barycentric->shape[2] != INT64_C(3)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "barycentrics must be a [B,S,3] output view"
        );
    }
    if (barycentric->dtype != vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "barycentrics must match the vertices dtype"
        );
    }
    status = gffx_mesh_check_view(face_index, "face indices must be a [B,S] output view",
                                  2u, batch_count, sample_count, 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (face_index->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must use the int32 dtype"
        );
    }
    if (gffx_mesh_views_overlap(points, vertices) || gffx_mesh_views_overlap(points, faces) ||
        gffx_mesh_views_overlap(barycentric, vertices) ||
        gffx_mesh_views_overlap(points, barycentric) ||
        gffx_mesh_views_overlap(points, face_index)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }

    required_bytes = (uint64_t)face_count * (uint64_t)sizeof(double);
    if (workspace == NULL || workspace->capacity_bytes < required_bytes) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INSUFFICIENT_WORKSPACE,
            "the forward pass requires the workspace capacity reported by the query"
        );
    }
    status = gffx_validate_buffer(workspace, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (((uintptr_t)workspace->data % (uintptr_t)sizeof(double)) != 0u) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the workspace data pointer must be aligned to double"
        );
    }

    /* Every face must reference vertices of its own batch element. */
    {
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

    {
        const int32_t *face_data = (const int32_t *)gffx_mesh_elements_const(faces);
        const int32_t *face_bounds = (const int32_t *)gffx_mesh_elements_const(face_offsets);
        const uint32_t *key_data = (const uint32_t *)gffx_mesh_elements_const(rng_key);
        const uint32_t *counter_data = (const uint32_t *)gffx_mesh_elements_const(rng_counter);
        double *cumulative = (double *)workspace->data;
        int32_t *index_data = (int32_t *)gffx_mesh_elements(face_index);
        int is_double = vertices->dtype == GFFX_DTYPE_FLOAT64;
        const double *vertex_d =
            is_double ? (const double *)gffx_mesh_elements_const(vertices) : NULL;
        const float *vertex_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(vertices);
        double *point_d = is_double ? (double *)gffx_mesh_elements(points) : NULL;
        float *point_f = is_double ? NULL : (float *)gffx_mesh_elements(points);
        double *bary_d = is_double ? (double *)gffx_mesh_elements(barycentric) : NULL;
        float *bary_f = is_double ? NULL : (float *)gffx_mesh_elements(barycentric);

        for (batch = 0; batch < batch_count; ++batch) {
            int64_t first_face = (int64_t)face_bounds[batch];
            int64_t last_face = (int64_t)face_bounds[batch + 1];
            int64_t eligible = 0;
            double running = 0.0;
            int64_t face;
            int64_t sample;
            /* Cumulative area table over eligible faces, always in double so the running sum
             * stays monotone and the binary search cannot land on a collapsed interval. */
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
                if (doubled > eps) {
                    running += doubled * 0.5;
                    ++eligible;
                }
                cumulative[face] = running;
            }
            if (eligible == 0) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "cannot sample a batch element with no positive-area face"
                );
            }
            for (sample = 0; sample < sample_count; ++sample) {
                uint32_t counter_words[4];
                uint32_t key_words[2];
                uint32_t words[4];
                double target;
                double b0, b1, b2, su;
                int64_t low = first_face;
                int64_t high = last_face - 1;
                int64_t chosen;
                int64_t i0, i1, i2;
                int64_t out_base = (batch * sample_count + sample) * INT64_C(3);
                counter_words[0] = counter_data[0];
                counter_words[1] = counter_data[1];
                counter_words[2] = (uint32_t)batch;
                counter_words[3] = (uint32_t)sample;
                key_words[0] = key_data[0];
                key_words[1] = key_data[1];
                gffx_philox4x32_10(counter_words, key_words, words);

                target = gffx_uniform_from_word(words[0]) * running;
                /* First face whose cumulative area strictly exceeds the target; ineligible
                 * faces leave the running sum unchanged and so are never chosen. */
                while (low < high) {
                    int64_t middle = low + (high - low) / 2;
                    if (cumulative[middle] > target) {
                        high = middle;
                    } else {
                        low = middle + 1;
                    }
                }
                chosen = low;
                while (chosen > first_face && cumulative[chosen - 1] >= cumulative[chosen]) {
                    /* Skip back over ineligible faces, which share their predecessor's sum. */
                    --chosen;
                }
                if (chosen < first_face) chosen = first_face;

                su = sqrt(gffx_uniform_from_word(words[1]));
                b1 = su * (1.0 - gffx_uniform_from_word(words[2]));
                b2 = su * gffx_uniform_from_word(words[2]);
                b0 = 1.0 - su;

                i0 = (int64_t)face_data[chosen * 3 + 0];
                i1 = (int64_t)face_data[chosen * 3 + 1];
                i2 = (int64_t)face_data[chosen * 3 + 2];
                index_data[batch * sample_count + sample] = (int32_t)chosen;
                if (is_double) {
                    int axis;
                    bary_d[out_base + 0] = b0;
                    bary_d[out_base + 1] = b1;
                    bary_d[out_base + 2] = b2;
                    for (axis = 0; axis < 3; ++axis) {
                        point_d[out_base + axis] = b0 * vertex_d[i0 * 3 + axis] +
                                                   b1 * vertex_d[i1 * 3 + axis] +
                                                   b2 * vertex_d[i2 * 3 + axis];
                    }
                } else {
                    int axis;
                    bary_f[out_base + 0] = (float)b0;
                    bary_f[out_base + 1] = (float)b1;
                    bary_f[out_base + 2] = (float)b2;
                    for (axis = 0; axis < 3; ++axis) {
                        point_f[out_base + axis] = (float)b0 * vertex_f[i0 * 3 + axis] +
                                                   (float)b1 * vertex_f[i1 * 3 + axis] +
                                                   (float)b2 * vertex_f[i2 * 3 + axis];
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_sample_surface_backward(
    const gffx_tensor_view *faces,
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,
    const gffx_tensor_view *grad_points,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t face_count;
    int64_t batch_count;
    int64_t sample_count;
    int64_t vertex_count;
    int64_t index;
    int64_t batch;
    int64_t sample;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.sample_surface implements only the CPU backend in this phase"
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
    face_count = faces->shape[0];
    if (face_index == NULL || face_index->rank != 2u || face_index->shape == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must be a [B,S] tensor view"
        );
    }
    batch_count = face_index->shape[0];
    sample_count = face_index->shape[1];
    status = gffx_mesh_check_view(face_index, "face indices must be a [B,S] tensor view",
                                  2u, batch_count, sample_count, 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (face_index->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face indices must use the int32 dtype"
        );
    }
    if (barycentric == NULL || barycentric->rank != 3u || barycentric->shape == NULL ||
        barycentric->shape[0] != batch_count || barycentric->shape[1] != sample_count ||
        barycentric->shape[2] != INT64_C(3)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "barycentrics must be a [B,S,3] tensor view"
        );
    }
    status = gffx_validate_tensor_view(barycentric, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_points == NULL || grad_points->rank != 3u || grad_points->shape == NULL ||
        grad_points->shape[0] != batch_count || grad_points->shape[1] != sample_count ||
        grad_points->shape[2] != INT64_C(3)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "point cotangents must be a [B,S,3] tensor view"
        );
    }
    status = gffx_validate_tensor_view(grad_points, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_points->dtype != barycentric->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "cotangents must match the barycentric dtype"
        );
    }
    status = gffx_mesh_check_view(grad_vertices, "vertex gradients must be a [V,3] output view",
                                  2u, INT64_C(-1), INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_vertices->dtype != barycentric->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertex gradients must match the barycentric dtype"
        );
    }
    vertex_count = grad_vertices->shape[0];
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }

    {
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const int32_t *index_data = (const int32_t *)gffx_mesh_elements_const(face_index);
        int is_double = barycentric->dtype == GFFX_DTYPE_FLOAT64;
        const double *bary_d =
            is_double ? (const double *)gffx_mesh_elements_const(barycentric) : NULL;
        const float *bary_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(barycentric);
        const double *cotangent_d =
            is_double ? (const double *)gffx_mesh_elements_const(grad_points) : NULL;
        const float *cotangent_f =
            is_double ? NULL : (const float *)gffx_mesh_elements_const(grad_points);
        double *gradient_d = is_double ? (double *)gffx_mesh_elements(grad_vertices) : NULL;
        float *gradient_f = is_double ? NULL : (float *)gffx_mesh_elements(grad_vertices);

        if (is_double) {
            for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient_d[index] = 0.0;
        } else {
            for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient_f[index] = 0.0f;
        }
        for (batch = 0; batch < batch_count; ++batch) {
            for (sample = 0; sample < sample_count; ++sample) {
                int64_t entry = batch * sample_count + sample;
                int64_t base = entry * INT64_C(3);
                int64_t face = (int64_t)index_data[entry];
                int corner;
                if (face < 0 || face >= face_count) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_INVALID_ARGUMENT,
                        "a sampled face index lies outside the face range"
                    );
                }
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = (int64_t)face_data[face * 3 + corner];
                    int axis;
                    if (vertex < 0 || vertex >= vertex_count) {
                        return gffx_internal_fail(
                            diagnostic,
                            GFFX_STATUS_INVALID_ARGUMENT,
                            "a face references a vertex outside the gradient range"
                        );
                    }
                    for (axis = 0; axis < 3; ++axis) {
                        if (is_double) {
                            gradient_d[vertex * 3 + axis] +=
                                bary_d[base + corner] * cotangent_d[base + axis];
                        } else {
                            gradient_f[vertex * 3 + axis] +=
                                bary_f[base + corner] * cotangent_f[base + axis];
                        }
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}
