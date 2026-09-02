/*
 * mesh.gather_faces - Phase 2 CPU reference kernels.
 *
 * A pure gather: face_vertices[k][j][:] = vertices[faces[k][j]][:], preserving face and corner
 * order and copying values bit-for-bit. Backward scatter-adds in ascending face then corner
 * order. Both directions require zero workspace bytes.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "cuda_loader.h"
#include "mesh_common.h"

#include <stdint.h>

/* The gather needs vertices, faces, context, and workspace validated, but not eps; pass the
 * contract's zero, which the shared checker accepts. */
#define GFFX_GATHER_NO_EPS 0.0

GFFX_API gffx_status GFFX_CALL gffx_mesh_gather_faces_workspace(
    int64_t vertex_count,
    int64_t face_count,
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
            GFFX_CUDA_OP_MESH_GATHER_FACES, NULL, 0u, dtype, context, required_bytes,
            required_alignment, diagnostic);
    }
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (required_bytes == NULL || required_alignment == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "workspace query result pointers must not be null"
        );
    }
    if (vertex_count < INT64_C(0) || face_count < INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertex and face counts must be nonnegative"
        );
    }
    if (dtype != GFFX_DTYPE_FLOAT32 && dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.gather_faces supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.gather_faces implements only the CPU backend in this phase"
        );
    }
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_gather_faces(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_execution_context *context,
    gffx_tensor_view *face_vertices,
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
        if (operations->mesh_gather_faces == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "the CUDA provider does not implement this operation");
        }
        return operations->mesh_gather_faces(
            vertices, faces, context, face_vertices, workspace, diagnostic);
    }
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t face_count;
    int64_t face;
    int64_t corner;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_common(vertices, faces, GFFX_GATHER_NO_EPS, context, workspace,
                                    diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    face_count = faces->shape[0];

    if (face_vertices == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face vertices must be a [F,3,3] output view"
        );
    }
    if (face_vertices->rank != 3u || face_vertices->shape == NULL ||
        face_vertices->shape[0] != face_count || face_vertices->shape[1] != INT64_C(3) ||
        face_vertices->shape[2] != INT64_C(3)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face vertices must be a [F,3,3] output view"
        );
    }
    status = gffx_validate_tensor_view(face_vertices, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (face_vertices->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.gather_faces implements only the CPU backend in this phase"
        );
    }
    if ((face_vertices->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation outputs must carry the output flag"
        );
    }
    if (face_vertices->dtype != vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face vertices must match the vertices dtype"
        );
    }
    if (gffx_mesh_views_overlap(face_vertices, vertices) ||
        gffx_mesh_views_overlap(face_vertices, faces)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (face_count == INT64_C(0)) return GFFX_STATUS_OK;

    {
        const int32_t *face_data = (const int32_t *)gffx_mesh_elements_const(faces);
        if (vertices->dtype == GFFX_DTYPE_FLOAT64) {
            const double *source = (const double *)gffx_mesh_elements_const(vertices);
            double *target = (double *)gffx_mesh_elements(face_vertices);
            for (face = 0; face < face_count; ++face) {
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = (int64_t)face_data[face * 3 + corner];
                    target[face * 9 + corner * 3 + 0] = source[vertex * 3 + 0];
                    target[face * 9 + corner * 3 + 1] = source[vertex * 3 + 1];
                    target[face * 9 + corner * 3 + 2] = source[vertex * 3 + 2];
                }
            }
        } else {
            const float *source = (const float *)gffx_mesh_elements_const(vertices);
            float *target = (float *)gffx_mesh_elements(face_vertices);
            for (face = 0; face < face_count; ++face) {
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = (int64_t)face_data[face * 3 + corner];
                    target[face * 9 + corner * 3 + 0] = source[vertex * 3 + 0];
                    target[face * 9 + corner * 3 + 1] = source[vertex * 3 + 1];
                    target[face * 9 + corner * 3 + 2] = source[vertex * 3 + 2];
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_gather_faces_backward(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *grad_face_vertices,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    /* Device dispatch before validation, as in the forward. */
    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        const gffx_cuda_operations *operations = gffx_cuda_loader_operations();
        if (operations == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "no CUDA provider is available; install the gffx CUDA plugin or run on the CPU");
        }
        if (operations->mesh_gather_faces_backward == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "the CUDA provider does not implement this operation");
        }
        return operations->mesh_gather_faces_backward(
            vertices, faces, grad_face_vertices, context, grad_vertices, workspace, diagnostic);
    }
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t face_count;
    int64_t vertex_count;
    int64_t index;
    int64_t face;
    int64_t corner;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_common(vertices, faces, GFFX_GATHER_NO_EPS, context, workspace,
                                    diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    face_count = faces->shape[0];
    vertex_count = vertices->shape[0];

    if (grad_face_vertices == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face-vertex cotangents must be a [F,3,3] tensor view"
        );
    }
    if (grad_face_vertices->rank != 3u || grad_face_vertices->shape == NULL ||
        grad_face_vertices->shape[0] != face_count ||
        grad_face_vertices->shape[1] != INT64_C(3) ||
        grad_face_vertices->shape[2] != INT64_C(3)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face-vertex cotangents must be a [F,3,3] tensor view"
        );
    }
    status = gffx_validate_tensor_view(grad_face_vertices, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_face_vertices->dtype != vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "cotangents must match the vertices dtype"
        );
    }
    status = gffx_mesh_check_view(grad_vertices, "vertex gradients must be a [V,3] output view",
                                  2u, vertex_count, INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_vertices->dtype != vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertex gradients must match the vertices dtype"
        );
    }
    if (gffx_mesh_views_overlap(grad_vertices, vertices) ||
        gffx_mesh_views_overlap(grad_vertices, faces) ||
        gffx_mesh_views_overlap(grad_vertices, grad_face_vertices)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (vertex_count == INT64_C(0)) return GFFX_STATUS_OK;

    {
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        if (vertices->dtype == GFFX_DTYPE_FLOAT64) {
            const double *cotangent = (const double *)gffx_mesh_elements_const(grad_face_vertices);
            double *gradient = (double *)gffx_mesh_elements(grad_vertices);
            for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient[index] = 0.0;
            for (face = 0; face < face_count; ++face) {
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = (int64_t)face_data[face * 3 + corner];
                    gradient[vertex * 3 + 0] += cotangent[face * 9 + corner * 3 + 0];
                    gradient[vertex * 3 + 1] += cotangent[face * 9 + corner * 3 + 1];
                    gradient[vertex * 3 + 2] += cotangent[face * 9 + corner * 3 + 2];
                }
            }
        } else {
            const float *cotangent = (const float *)gffx_mesh_elements_const(grad_face_vertices);
            float *gradient = (float *)gffx_mesh_elements(grad_vertices);
            for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient[index] = 0.0f;
            for (face = 0; face < face_count; ++face) {
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = (int64_t)face_data[face * 3 + corner];
                    gradient[vertex * 3 + 0] += cotangent[face * 9 + corner * 3 + 0];
                    gradient[vertex * 3 + 1] += cotangent[face * 9 + corner * 3 + 1];
                    gradient[vertex * 3 + 2] += cotangent[face * 9 + corner * 3 + 2];
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}
