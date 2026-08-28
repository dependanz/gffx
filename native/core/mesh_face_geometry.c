/*
 * mesh.face_geometry - Phase 2 CPU reference kernels.
 *
 * Semantics follow the public contract in <gffx/mesh.h>: per-face cross product, strict
 * d > eps validity evaluated in double precision, zero sentinels with exactly zero gradients
 * through the invalid branch, ascending-face deterministic accumulation in the backward pass,
 * and computation carried out in the vertices dtype.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "cuda_loader.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

static void gffx_mesh_face_geometry_compute_f32(
    const float *vertices, const int32_t *faces, int64_t face_count, double eps,
    float *unit_normals, float *areas, uint8_t *valid
) {
    int64_t face;
    for (face = 0; face < face_count; ++face) {
        const float *a = vertices + (int64_t)faces[face * 3 + 0] * 3;
        const float *b = vertices + (int64_t)faces[face * 3 + 1] * 3;
        const float *c = vertices + (int64_t)faces[face * 3 + 2] * 3;
        float e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
        float e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
        float cx = e1y * e2z - e1z * e2y;
        float cy = e1z * e2x - e1x * e2z;
        float cz = e1x * e2y - e1y * e2x;
        float doubled = sqrtf(cx * cx + cy * cy + cz * cz);
        if ((double)doubled > eps) {
            unit_normals[face * 3 + 0] = cx / doubled;
            unit_normals[face * 3 + 1] = cy / doubled;
            unit_normals[face * 3 + 2] = cz / doubled;
            areas[face] = doubled * 0.5f;
            valid[face] = 1u;
        } else {
            unit_normals[face * 3 + 0] = 0.0f;
            unit_normals[face * 3 + 1] = 0.0f;
            unit_normals[face * 3 + 2] = 0.0f;
            areas[face] = 0.0f;
            valid[face] = 0u;
        }
    }
}

static void gffx_mesh_face_geometry_compute_f64(
    const double *vertices, const int32_t *faces, int64_t face_count, double eps,
    double *unit_normals, double *areas, uint8_t *valid
) {
    int64_t face;
    for (face = 0; face < face_count; ++face) {
        const double *a = vertices + (int64_t)faces[face * 3 + 0] * 3;
        const double *b = vertices + (int64_t)faces[face * 3 + 1] * 3;
        const double *c = vertices + (int64_t)faces[face * 3 + 2] * 3;
        double e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
        double e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
        double cx = e1y * e2z - e1z * e2y;
        double cy = e1z * e2x - e1x * e2z;
        double cz = e1x * e2y - e1y * e2x;
        double doubled = sqrt(cx * cx + cy * cy + cz * cz);
        if (doubled > eps) {
            unit_normals[face * 3 + 0] = cx / doubled;
            unit_normals[face * 3 + 1] = cy / doubled;
            unit_normals[face * 3 + 2] = cz / doubled;
            areas[face] = doubled * 0.5;
            valid[face] = 1u;
        } else {
            unit_normals[face * 3 + 0] = 0.0;
            unit_normals[face * 3 + 1] = 0.0;
            unit_normals[face * 3 + 2] = 0.0;
            areas[face] = 0.0;
            valid[face] = 0u;
        }
    }
}

/* Backward: gc = (gn - (gn.n)n)/d + ga*n/2; dL/de1 = e2 x gc; dL/de2 = gc x e1;
 * dL/dv0 = -(dL/de1 + dL/de2). Invalid faces contribute exactly zero. */
static void gffx_mesh_face_geometry_backward_f32(
    const float *vertices, const int32_t *faces, int64_t face_count, double eps,
    const float *grad_unit_normals, const float *grad_areas, float *grad_vertices
) {
    int64_t face;
    for (face = 0; face < face_count; ++face) {
        int64_t i0 = (int64_t)faces[face * 3 + 0];
        int64_t i1 = (int64_t)faces[face * 3 + 1];
        int64_t i2 = (int64_t)faces[face * 3 + 2];
        const float *a = vertices + i0 * 3;
        const float *b = vertices + i1 * 3;
        const float *c = vertices + i2 * 3;
        float e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
        float e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
        float cx = e1y * e2z - e1z * e2y;
        float cy = e1z * e2x - e1x * e2z;
        float cz = e1x * e2y - e1y * e2x;
        float doubled = sqrtf(cx * cx + cy * cy + cz * cz);
        float nx, ny, nz, gnx, gny, gnz, ga, dot, gcx, gcy, gcz;
        float g1x, g1y, g1z, g2x, g2y, g2z;
        if (!((double)doubled > eps)) continue;
        nx = cx / doubled; ny = cy / doubled; nz = cz / doubled;
        gnx = grad_unit_normals != NULL ? grad_unit_normals[face * 3 + 0] : 0.0f;
        gny = grad_unit_normals != NULL ? grad_unit_normals[face * 3 + 1] : 0.0f;
        gnz = grad_unit_normals != NULL ? grad_unit_normals[face * 3 + 2] : 0.0f;
        ga = grad_areas != NULL ? grad_areas[face] : 0.0f;
        dot = gnx * nx + gny * ny + gnz * nz;
        gcx = (gnx - dot * nx) / doubled + 0.5f * ga * nx;
        gcy = (gny - dot * ny) / doubled + 0.5f * ga * ny;
        gcz = (gnz - dot * nz) / doubled + 0.5f * ga * nz;
        g1x = e2y * gcz - e2z * gcy;
        g1y = e2z * gcx - e2x * gcz;
        g1z = e2x * gcy - e2y * gcx;
        g2x = gcy * e1z - gcz * e1y;
        g2y = gcz * e1x - gcx * e1z;
        g2z = gcx * e1y - gcy * e1x;
        grad_vertices[i1 * 3 + 0] += g1x;
        grad_vertices[i1 * 3 + 1] += g1y;
        grad_vertices[i1 * 3 + 2] += g1z;
        grad_vertices[i2 * 3 + 0] += g2x;
        grad_vertices[i2 * 3 + 1] += g2y;
        grad_vertices[i2 * 3 + 2] += g2z;
        grad_vertices[i0 * 3 + 0] -= g1x + g2x;
        grad_vertices[i0 * 3 + 1] -= g1y + g2y;
        grad_vertices[i0 * 3 + 2] -= g1z + g2z;
    }
}

static void gffx_mesh_face_geometry_backward_f64(
    const double *vertices, const int32_t *faces, int64_t face_count, double eps,
    const double *grad_unit_normals, const double *grad_areas, double *grad_vertices
) {
    int64_t face;
    for (face = 0; face < face_count; ++face) {
        int64_t i0 = (int64_t)faces[face * 3 + 0];
        int64_t i1 = (int64_t)faces[face * 3 + 1];
        int64_t i2 = (int64_t)faces[face * 3 + 2];
        const double *a = vertices + i0 * 3;
        const double *b = vertices + i1 * 3;
        const double *c = vertices + i2 * 3;
        double e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
        double e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
        double cx = e1y * e2z - e1z * e2y;
        double cy = e1z * e2x - e1x * e2z;
        double cz = e1x * e2y - e1y * e2x;
        double doubled = sqrt(cx * cx + cy * cy + cz * cz);
        double nx, ny, nz, gnx, gny, gnz, ga, dot, gcx, gcy, gcz;
        double g1x, g1y, g1z, g2x, g2y, g2z;
        if (!(doubled > eps)) continue;
        nx = cx / doubled; ny = cy / doubled; nz = cz / doubled;
        gnx = grad_unit_normals != NULL ? grad_unit_normals[face * 3 + 0] : 0.0;
        gny = grad_unit_normals != NULL ? grad_unit_normals[face * 3 + 1] : 0.0;
        gnz = grad_unit_normals != NULL ? grad_unit_normals[face * 3 + 2] : 0.0;
        ga = grad_areas != NULL ? grad_areas[face] : 0.0;
        dot = gnx * nx + gny * ny + gnz * nz;
        gcx = (gnx - dot * nx) / doubled + 0.5 * ga * nx;
        gcy = (gny - dot * ny) / doubled + 0.5 * ga * ny;
        gcz = (gnz - dot * nz) / doubled + 0.5 * ga * nz;
        g1x = e2y * gcz - e2z * gcy;
        g1y = e2z * gcx - e2x * gcz;
        g1z = e2x * gcy - e2y * gcx;
        g2x = gcy * e1z - gcz * e1y;
        g2y = gcz * e1x - gcx * e1z;
        g2z = gcx * e1y - gcy * e1x;
        grad_vertices[i1 * 3 + 0] += g1x;
        grad_vertices[i1 * 3 + 1] += g1y;
        grad_vertices[i1 * 3 + 2] += g1z;
        grad_vertices[i2 * 3 + 0] += g2x;
        grad_vertices[i2 * 3 + 1] += g2y;
        grad_vertices[i2 * 3 + 2] += g2z;
        grad_vertices[i0 * 3 + 0] -= g1x + g2x;
        grad_vertices[i0 * 3 + 1] -= g1y + g2y;
        grad_vertices[i0 * 3 + 2] -= g1z + g2z;
    }
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_face_geometry_workspace(
    int64_t vertex_count,
    int64_t face_count,
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
            "mesh.face_geometry supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type == GFFX_DEVICE_CUDA) {
        /* The device answer is the plugin's, never the host's: the CUDA path needs scratch for
         * the index-validation status word where the scalar CPU reference needs none. */
        const gffx_cuda_operations *operations = gffx_cuda_loader_operations();
        if (operations == NULL || operations->workspace_query == NULL) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_UNSUPPORTED,
                "no CUDA provider is available to report a device workspace requirement"
            );
        }
        return operations->workspace_query(
            GFFX_CUDA_OP_MESH_FACE_GEOMETRY, NULL, 0u, dtype, context, required_bytes,
            required_alignment, diagnostic);
    }
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.face_geometry implements only the CPU and CUDA backends"
        );
    }
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_face_geometry(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *unit_normals,
    gffx_tensor_view *areas,
    gffx_tensor_view *valid,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    /*
     * Device dispatch happens before any CPU validation, not after it.
     *
     * The shared validator dereferences the face tensor to range-check indices, which is exactly
     * what must not happen when that tensor lives in device memory; running it first would read a
     * device pointer from host code. The CUDA backend performs the same check on the device, so
     * the guarantee is kept rather than skipped, and the caller sees one contract regardless of
     * where the work runs.
     *
     * A backend that publishes no such operation returns UNSUPPORTED rather than silently falling
     * back to the CPU, which would turn a missing kernel into an unannounced device-to-host copy.
     */
    if (context != NULL && context->struct_size >= sizeof(*context) &&
        context->device_type == GFFX_DEVICE_CUDA) {
        const gffx_cuda_operations *operations = gffx_cuda_loader_operations();
        if (operations == NULL) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_UNSUPPORTED,
                "no CUDA provider is available; install the gffx CUDA plugin or run on the CPU"
            );
        }
        if (operations->mesh_face_geometry == NULL) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_UNSUPPORTED,
                "the CUDA provider does not implement mesh.face_geometry"
            );
        }
        return operations->mesh_face_geometry(
            vertices, faces, eps, context, unit_normals, areas, valid, workspace, diagnostic);
    }

    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t face_count;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_common(vertices, faces, eps, context, workspace, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    face_count = faces->shape[0];

    status = gffx_mesh_check_view(unit_normals, "unit normals must be a [F,3] output view",
                                  2u, face_count, INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_view(areas, "areas must be a [F] output view",
                                  1u, face_count, INT64_C(0), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_view(valid, "valid must be a [F] output view",
                                  1u, face_count, INT64_C(0), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (unit_normals->dtype != vertices->dtype || areas->dtype != vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "floating outputs must match the vertices dtype"
        );
    }
    if (valid->dtype != GFFX_DTYPE_BOOL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "valid must use the bool dtype"
        );
    }
    {
        const gffx_tensor_view *views[5];
        int first;
        int second;
        views[0] = unit_normals; views[1] = areas; views[2] = valid;
        views[3] = vertices; views[4] = faces;
        for (first = 0; first < 3; ++first) {
            for (second = first + 1; second < 5; ++second) {
                if (gffx_mesh_views_overlap(views[first], views[second])) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_INVALID_ARGUMENT,
                        "outputs may not alias an input or another output"
                    );
                }
            }
        }
    }
    if (face_count == INT64_C(0)) return GFFX_STATUS_OK;
    if (vertices->dtype == GFFX_DTYPE_FLOAT64) {
        gffx_mesh_face_geometry_compute_f64(
            (const double *)gffx_mesh_elements_const(vertices),
            (const int32_t *)gffx_mesh_elements_const(faces),
            face_count, eps,
            (double *)gffx_mesh_elements(unit_normals),
            (double *)gffx_mesh_elements(areas),
            (uint8_t *)gffx_mesh_elements(valid));
    } else {
        gffx_mesh_face_geometry_compute_f32(
            (const float *)gffx_mesh_elements_const(vertices),
            (const int32_t *)gffx_mesh_elements_const(faces),
            face_count, eps,
            (float *)gffx_mesh_elements(unit_normals),
            (float *)gffx_mesh_elements(areas),
            (uint8_t *)gffx_mesh_elements(valid));
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_face_geometry_backward(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    const gffx_tensor_view *grad_unit_normals,
    const gffx_tensor_view *grad_areas,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t face_count;
    int64_t vertex_count;
    int64_t index;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_common(vertices, faces, eps, context, workspace, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    face_count = faces->shape[0];
    vertex_count = vertices->shape[0];

    if (grad_unit_normals == NULL && grad_areas == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "at least one cotangent view is required"
        );
    }
    if (grad_unit_normals != NULL) {
        status = gffx_mesh_check_view(grad_unit_normals,
                                      "normal cotangents must be a [F,3] tensor view",
                                      2u, face_count, INT64_C(3), 0, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (grad_unit_normals->dtype != vertices->dtype) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "cotangents must match the vertices dtype"
            );
        }
    }
    if (grad_areas != NULL) {
        status = gffx_mesh_check_view(grad_areas, "area cotangents must be a [F] tensor view",
                                      1u, face_count, INT64_C(0), 0, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (grad_areas->dtype != vertices->dtype) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "cotangents must match the vertices dtype"
            );
        }
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
    {
        const gffx_tensor_view *others[4];
        int second;
        others[0] = vertices; others[1] = faces;
        others[2] = grad_unit_normals; others[3] = grad_areas;
        for (second = 0; second < 4; ++second) {
            if (others[second] != NULL &&
                gffx_mesh_views_overlap(grad_vertices, others[second])) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "outputs may not alias an input or another output"
                );
            }
        }
    }
    if (vertex_count == INT64_C(0)) return GFFX_STATUS_OK;
    if (vertices->dtype == GFFX_DTYPE_FLOAT64) {
        double *gradient = (double *)gffx_mesh_elements(grad_vertices);
        for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient[index] = 0.0;
        gffx_mesh_face_geometry_backward_f64(
            (const double *)gffx_mesh_elements_const(vertices),
            (const int32_t *)gffx_mesh_elements_const(faces),
            face_count, eps,
            grad_unit_normals != NULL
                ? (const double *)gffx_mesh_elements_const(grad_unit_normals) : NULL,
            grad_areas != NULL
                ? (const double *)gffx_mesh_elements_const(grad_areas) : NULL,
            gradient);
    } else {
        float *gradient = (float *)gffx_mesh_elements(grad_vertices);
        for (index = 0; index < vertex_count * INT64_C(3); ++index) gradient[index] = 0.0f;
        gffx_mesh_face_geometry_backward_f32(
            (const float *)gffx_mesh_elements_const(vertices),
            (const int32_t *)gffx_mesh_elements_const(faces),
            face_count, eps,
            grad_unit_normals != NULL
                ? (const float *)gffx_mesh_elements_const(grad_unit_normals) : NULL,
            grad_areas != NULL
                ? (const float *)gffx_mesh_elements_const(grad_areas) : NULL,
            gradient);
    }
    return GFFX_STATUS_OK;
}
