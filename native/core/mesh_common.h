#ifndef GFFX_CORE_MESH_COMMON_H
#define GFFX_CORE_MESH_COMMON_H

/*
 * Shared validation and addressing helpers for the Phase 2 mesh operations. Every mesh kernel
 * validates through these so role errors, status codes, and index prevalidation stay identical
 * across operations. Header-only static inline functions keep the scaffold dependency-free.
 */

#include <gffx/execution.h>
#include <gffx/tensor.h>

#include "internal.h"

#include <math.h>
#include <stdint.h>

static inline uint64_t gffx_mesh_dtype_size(gffx_dtype dtype) {
    switch (dtype) {
        case GFFX_DTYPE_FLOAT32:
        case GFFX_DTYPE_INT32:
        case GFFX_DTYPE_UINT32:
            return UINT64_C(4);
        case GFFX_DTYPE_FLOAT64:
            return UINT64_C(8);
        case GFFX_DTYPE_BOOL:
            return UINT64_C(1);
        default:
            return UINT64_C(0);
    }
}

static inline const void *gffx_mesh_elements_const(const gffx_tensor_view *view) {
    return (const void *)((const char *)view->data + (uintptr_t)view->byte_offset);
}

static inline void *gffx_mesh_elements(const gffx_tensor_view *view) {
    return (void *)((char *)view->data + (uintptr_t)view->byte_offset);
}

static inline uint64_t gffx_mesh_element_count(const gffx_tensor_view *view) {
    uint64_t count = UINT64_C(1);
    uint32_t index;
    if (view->rank == UINT32_C(0)) return UINT64_C(1);
    for (index = 0u; index < view->rank; ++index) {
        count *= (uint64_t)view->shape[index];
    }
    return count;
}

static inline int gffx_mesh_views_overlap(const gffx_tensor_view *a, const gffx_tensor_view *b) {
    uint64_t a_bytes = gffx_mesh_element_count(a) * gffx_mesh_dtype_size(a->dtype);
    uint64_t b_bytes = gffx_mesh_element_count(b) * gffx_mesh_dtype_size(b->dtype);
    uintptr_t a_start;
    uintptr_t b_start;
    if (a->data == NULL || b->data == NULL) return 0;
    if (a_bytes == UINT64_C(0) || b_bytes == UINT64_C(0)) return 0;
    a_start = (uintptr_t)a->data + (uintptr_t)a->byte_offset;
    b_start = (uintptr_t)b->data + (uintptr_t)b->byte_offset;
    return a_start < b_start + (uintptr_t)b_bytes && b_start < a_start + (uintptr_t)a_bytes;
}

static inline int gffx_mesh_range_overlaps_view(
    const void *range_data, uint64_t range_bytes, const gffx_tensor_view *view
) {
    uint64_t view_bytes = gffx_mesh_element_count(view) * gffx_mesh_dtype_size(view->dtype);
    uintptr_t range_start;
    uintptr_t view_start;
    if (range_data == NULL || view->data == NULL) return 0;
    if (range_bytes == UINT64_C(0) || view_bytes == UINT64_C(0)) return 0;
    range_start = (uintptr_t)range_data;
    view_start = (uintptr_t)view->data + (uintptr_t)view->byte_offset;
    return range_start < view_start + (uintptr_t)view_bytes &&
           view_start < range_start + (uintptr_t)range_bytes;
}

/* Role-level shape and flag checks. Shape checks run before the full view validation so a
 * wrong-shaped view is reported as an invalid argument rather than as an incidental stride
 * finding, matching the acceptance fixtures. */
static inline gffx_status gffx_mesh_check_view(
    const gffx_tensor_view *view,
    const char *role_message,
    uint32_t expected_rank,
    int64_t expected_rows,
    int64_t expected_cols,
    int is_output,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    if (view == NULL) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
    }
    if (view->rank != expected_rank || view->shape == NULL) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
    }
    if (expected_rows >= INT64_C(0) && view->shape[0] != expected_rows) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
    }
    if (expected_rank == UINT32_C(2) && view->shape[1] != expected_cols) {
        return gffx_internal_fail(diagnostic, GFFX_STATUS_INVALID_ARGUMENT, role_message);
    }
    status = gffx_validate_tensor_view(view, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (view->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh operations implement only the CPU backend in this phase"
        );
    }
    if (is_output) {
        if ((view->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "operation outputs must carry the output flag"
            );
        }
    } else if ((view->flags & GFFX_TENSOR_OUTPUT) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation inputs may not carry the output flag"
        );
    }
    return GFFX_STATUS_OK;
}

/* Shared vertices/faces/eps/context/workspace validation plus index prevalidation: every face
 * index is range-checked before any vertex data dereference. */
static inline gffx_status gffx_mesh_check_common(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    const gffx_execution_context *context,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
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
            "mesh operations implement only the CPU backend in this phase"
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
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (workspace->device_type != GFFX_DEVICE_CPU) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_UNSUPPORTED,
                "mesh operations accept only CPU workspace storage"
            );
        }
    }
    {
        int64_t face_count = faces->shape[0];
        int64_t vertex_count = vertices->shape[0];
        const int32_t *face_data;
        int64_t index;
        if (face_count > INT64_C(0)) {
            face_data = (const int32_t *)gffx_mesh_elements_const(faces);
            for (index = 0; index < face_count * INT64_C(3); ++index) {
                if (face_data[index] < INT32_C(0) ||
                    (int64_t)face_data[index] >= vertex_count) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_INVALID_ARGUMENT,
                        "face indices must lie in [0, V)"
                    );
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

#endif /* GFFX_CORE_MESH_COMMON_H */
