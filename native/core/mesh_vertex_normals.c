/*
 * mesh.vertex_normals - Phase 2 CPU reference kernels.
 *
 * Semantics follow <gffx/mesh.h> and the project acceptance record: valid faces contribute
 * c/2 (area mode) or c/||c|| (uniform mode) to their three vertices; sums are normalized when
 * ||s|| > eps in double precision and are otherwise exact zeros. Accumulation is per-face
 * ascending, normalization per-vertex ascending. The backward pass recomputes the sums into the
 * caller workspace, converts them in place to dL/ds, and distributes per-face contributions in
 * ascending order.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

static gffx_status gffx_vertex_normals_check_weighting(
    uint32_t weighting, gffx_diagnostic_buffer *diagnostic
) {
    if (weighting != GFFX_MESH_WEIGHTING_AREA && weighting != GFFX_MESH_WEIGHTING_UNIFORM) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "weighting must be GFFX_MESH_WEIGHTING_AREA or GFFX_MESH_WEIGHTING_UNIFORM"
        );
    }
    return GFFX_STATUS_OK;
}

static void gffx_vertex_normals_accumulate_f32(
    const float *vertices, const int32_t *faces, int64_t face_count,
    double eps, uint32_t weighting, float *sums
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
        float wx, wy, wz;
        if (!((double)doubled > eps)) continue;
        if (weighting == GFFX_MESH_WEIGHTING_AREA) {
            wx = cx * 0.5f; wy = cy * 0.5f; wz = cz * 0.5f;
        } else {
            wx = cx / doubled; wy = cy / doubled; wz = cz / doubled;
        }
        sums[i0 * 3 + 0] += wx; sums[i0 * 3 + 1] += wy; sums[i0 * 3 + 2] += wz;
        sums[i1 * 3 + 0] += wx; sums[i1 * 3 + 1] += wy; sums[i1 * 3 + 2] += wz;
        sums[i2 * 3 + 0] += wx; sums[i2 * 3 + 1] += wy; sums[i2 * 3 + 2] += wz;
    }
}

static void gffx_vertex_normals_accumulate_f64(
    const double *vertices, const int32_t *faces, int64_t face_count,
    double eps, uint32_t weighting, double *sums
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
        double wx, wy, wz;
        if (!(doubled > eps)) continue;
        if (weighting == GFFX_MESH_WEIGHTING_AREA) {
            wx = cx * 0.5; wy = cy * 0.5; wz = cz * 0.5;
        } else {
            wx = cx / doubled; wy = cy / doubled; wz = cz / doubled;
        }
        sums[i0 * 3 + 0] += wx; sums[i0 * 3 + 1] += wy; sums[i0 * 3 + 2] += wz;
        sums[i1 * 3 + 0] += wx; sums[i1 * 3 + 1] += wy; sums[i1 * 3 + 2] += wz;
        sums[i2 * 3 + 0] += wx; sums[i2 * 3 + 1] += wy; sums[i2 * 3 + 2] += wz;
    }
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_vertex_normals_workspace(
    int64_t vertex_count,
    int64_t face_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    uint64_t element_size;
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
            "mesh.vertex_normals supports the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.vertex_normals implements only the CPU backend in this phase"
        );
    }
    element_size = gffx_mesh_dtype_size(dtype);
    if ((uint64_t)vertex_count > UINT64_MAX / (UINT64_C(3) * element_size)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_OVERFLOW,
            "workspace byte requirement overflows 64-bit capacity"
        );
    }
    *required_bytes = (uint64_t)vertex_count * UINT64_C(3) * element_size;
    *required_alignment = element_size;
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_vertex_normals(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    uint32_t weighting,
    const gffx_execution_context *context,
    gffx_tensor_view *unit_normals,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t vertex_count;
    int64_t face_count;
    int64_t vertex;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_vertex_normals_check_weighting(weighting, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_common(vertices, faces, eps, context, workspace, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    vertex_count = vertices->shape[0];
    face_count = faces->shape[0];

    status = gffx_mesh_check_view(unit_normals, "unit normals must be a [V,3] output view",
                                  2u, vertex_count, INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (unit_normals->dtype != vertices->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "unit normals must match the vertices dtype"
        );
    }
    if (gffx_mesh_views_overlap(unit_normals, vertices) ||
        gffx_mesh_views_overlap(unit_normals, faces)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (vertex_count == INT64_C(0)) return GFFX_STATUS_OK;
    if (vertices->dtype == GFFX_DTYPE_FLOAT64) {
        double *normals = (double *)gffx_mesh_elements(unit_normals);
        for (vertex = 0; vertex < vertex_count * INT64_C(3); ++vertex) normals[vertex] = 0.0;
        if (face_count > INT64_C(0)) {
            gffx_vertex_normals_accumulate_f64(
                (const double *)gffx_mesh_elements_const(vertices),
                (const int32_t *)gffx_mesh_elements_const(faces),
                face_count, eps, weighting, normals);
        }
        for (vertex = 0; vertex < vertex_count; ++vertex) {
            double sx = normals[vertex * 3 + 0];
            double sy = normals[vertex * 3 + 1];
            double sz = normals[vertex * 3 + 2];
            double magnitude = sqrt(sx * sx + sy * sy + sz * sz);
            if (magnitude > eps) {
                normals[vertex * 3 + 0] = sx / magnitude;
                normals[vertex * 3 + 1] = sy / magnitude;
                normals[vertex * 3 + 2] = sz / magnitude;
            } else {
                normals[vertex * 3 + 0] = 0.0;
                normals[vertex * 3 + 1] = 0.0;
                normals[vertex * 3 + 2] = 0.0;
            }
        }
    } else {
        float *normals = (float *)gffx_mesh_elements(unit_normals);
        for (vertex = 0; vertex < vertex_count * INT64_C(3); ++vertex) normals[vertex] = 0.0f;
        if (face_count > INT64_C(0)) {
            gffx_vertex_normals_accumulate_f32(
                (const float *)gffx_mesh_elements_const(vertices),
                (const int32_t *)gffx_mesh_elements_const(faces),
                face_count, eps, weighting, normals);
        }
        for (vertex = 0; vertex < vertex_count; ++vertex) {
            float sx = normals[vertex * 3 + 0];
            float sy = normals[vertex * 3 + 1];
            float sz = normals[vertex * 3 + 2];
            float magnitude = sqrtf(sx * sx + sy * sy + sz * sz);
            if ((double)magnitude > eps) {
                normals[vertex * 3 + 0] = sx / magnitude;
                normals[vertex * 3 + 1] = sy / magnitude;
                normals[vertex * 3 + 2] = sz / magnitude;
            } else {
                normals[vertex * 3 + 0] = 0.0f;
                normals[vertex * 3 + 1] = 0.0f;
                normals[vertex * 3 + 2] = 0.0f;
            }
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_vertex_normals_backward(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    uint32_t weighting,
    const gffx_tensor_view *grad_unit_normals,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t vertex_count;
    int64_t face_count;
    int64_t vertex;
    uint64_t element_size;
    uint64_t required_bytes;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_vertex_normals_check_weighting(weighting, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_mesh_check_common(vertices, faces, eps, context, workspace, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    vertex_count = vertices->shape[0];
    face_count = faces->shape[0];
    element_size = gffx_mesh_dtype_size(vertices->dtype);
    required_bytes = (uint64_t)vertex_count * UINT64_C(3) * element_size;

    status = gffx_mesh_check_view(grad_unit_normals,
                                  "normal cotangents must be a [V,3] tensor view",
                                  2u, vertex_count, INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_unit_normals->dtype != vertices->dtype) {
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
        gffx_mesh_views_overlap(grad_vertices, grad_unit_normals)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (vertex_count == INT64_C(0)) return GFFX_STATUS_OK;
    if (workspace == NULL || workspace->capacity_bytes < required_bytes) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INSUFFICIENT_WORKSPACE,
            "the backward pass requires the workspace capacity reported by the query"
        );
    }
    if (((uintptr_t)workspace->data % (uintptr_t)element_size) != 0u) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the workspace data pointer must be aligned to the dtype"
        );
    }
    if (gffx_mesh_range_overlaps_view(workspace->data, required_bytes, vertices) ||
        gffx_mesh_range_overlaps_view(workspace->data, required_bytes, faces) ||
        gffx_mesh_range_overlaps_view(workspace->data, required_bytes, grad_unit_normals) ||
        gffx_mesh_range_overlaps_view(workspace->data, required_bytes, grad_vertices)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the workspace may not alias an operand"
        );
    }

    if (vertices->dtype == GFFX_DTYPE_FLOAT64) {
        const double *vertex_data = (const double *)gffx_mesh_elements_const(vertices);
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const double *cotangent = (const double *)gffx_mesh_elements_const(grad_unit_normals);
        double *gradient = (double *)gffx_mesh_elements(grad_vertices);
        double *sums = (double *)workspace->data;
        int64_t face;
        for (vertex = 0; vertex < vertex_count * INT64_C(3); ++vertex) {
            sums[vertex] = 0.0;
            gradient[vertex] = 0.0;
        }
        if (face_count > INT64_C(0)) {
            gffx_vertex_normals_accumulate_f64(vertex_data, face_data, face_count, eps,
                                               weighting, sums);
        }
        /* Convert the sums in place to q = dL/ds. */
        for (vertex = 0; vertex < vertex_count; ++vertex) {
            double sx = sums[vertex * 3 + 0];
            double sy = sums[vertex * 3 + 1];
            double sz = sums[vertex * 3 + 2];
            double magnitude = sqrt(sx * sx + sy * sy + sz * sz);
            if (magnitude > eps) {
                double nx = sx / magnitude;
                double ny = sy / magnitude;
                double nz = sz / magnitude;
                double gx = cotangent[vertex * 3 + 0];
                double gy = cotangent[vertex * 3 + 1];
                double gz = cotangent[vertex * 3 + 2];
                double dot = gx * nx + gy * ny + gz * nz;
                sums[vertex * 3 + 0] = (gx - dot * nx) / magnitude;
                sums[vertex * 3 + 1] = (gy - dot * ny) / magnitude;
                sums[vertex * 3 + 2] = (gz - dot * nz) / magnitude;
            } else {
                sums[vertex * 3 + 0] = 0.0;
                sums[vertex * 3 + 1] = 0.0;
                sums[vertex * 3 + 2] = 0.0;
            }
        }
        for (face = 0; face < face_count; ++face) {
            int64_t i0 = (int64_t)face_data[face * 3 + 0];
            int64_t i1 = (int64_t)face_data[face * 3 + 1];
            int64_t i2 = (int64_t)face_data[face * 3 + 2];
            const double *a = vertex_data + i0 * 3;
            const double *b = vertex_data + i1 * 3;
            const double *c = vertex_data + i2 * 3;
            double e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
            double e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
            double cx = e1y * e2z - e1z * e2y;
            double cy = e1z * e2x - e1x * e2z;
            double cz = e1x * e2y - e1y * e2x;
            double doubled = sqrt(cx * cx + cy * cy + cz * cz);
            double qx, qy, qz, gcx, gcy, gcz;
            double g1x, g1y, g1z, g2x, g2y, g2z;
            if (!(doubled > eps)) continue;
            qx = sums[i0 * 3 + 0] + sums[i1 * 3 + 0] + sums[i2 * 3 + 0];
            qy = sums[i0 * 3 + 1] + sums[i1 * 3 + 1] + sums[i2 * 3 + 1];
            qz = sums[i0 * 3 + 2] + sums[i1 * 3 + 2] + sums[i2 * 3 + 2];
            if (weighting == GFFX_MESH_WEIGHTING_AREA) {
                gcx = 0.5 * qx; gcy = 0.5 * qy; gcz = 0.5 * qz;
            } else {
                double nx = cx / doubled;
                double ny = cy / doubled;
                double nz = cz / doubled;
                double dot = qx * nx + qy * ny + qz * nz;
                gcx = (qx - dot * nx) / doubled;
                gcy = (qy - dot * ny) / doubled;
                gcz = (qz - dot * nz) / doubled;
            }
            g1x = e2y * gcz - e2z * gcy;
            g1y = e2z * gcx - e2x * gcz;
            g1z = e2x * gcy - e2y * gcx;
            g2x = gcy * e1z - gcz * e1y;
            g2y = gcz * e1x - gcx * e1z;
            g2z = gcx * e1y - gcy * e1x;
            gradient[i1 * 3 + 0] += g1x;
            gradient[i1 * 3 + 1] += g1y;
            gradient[i1 * 3 + 2] += g1z;
            gradient[i2 * 3 + 0] += g2x;
            gradient[i2 * 3 + 1] += g2y;
            gradient[i2 * 3 + 2] += g2z;
            gradient[i0 * 3 + 0] -= g1x + g2x;
            gradient[i0 * 3 + 1] -= g1y + g2y;
            gradient[i0 * 3 + 2] -= g1z + g2z;
        }
    } else {
        const float *vertex_data = (const float *)gffx_mesh_elements_const(vertices);
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const float *cotangent = (const float *)gffx_mesh_elements_const(grad_unit_normals);
        float *gradient = (float *)gffx_mesh_elements(grad_vertices);
        float *sums = (float *)workspace->data;
        int64_t face;
        for (vertex = 0; vertex < vertex_count * INT64_C(3); ++vertex) {
            sums[vertex] = 0.0f;
            gradient[vertex] = 0.0f;
        }
        if (face_count > INT64_C(0)) {
            gffx_vertex_normals_accumulate_f32(vertex_data, face_data, face_count, eps,
                                               weighting, sums);
        }
        for (vertex = 0; vertex < vertex_count; ++vertex) {
            float sx = sums[vertex * 3 + 0];
            float sy = sums[vertex * 3 + 1];
            float sz = sums[vertex * 3 + 2];
            float magnitude = sqrtf(sx * sx + sy * sy + sz * sz);
            if ((double)magnitude > eps) {
                float nx = sx / magnitude;
                float ny = sy / magnitude;
                float nz = sz / magnitude;
                float gx = cotangent[vertex * 3 + 0];
                float gy = cotangent[vertex * 3 + 1];
                float gz = cotangent[vertex * 3 + 2];
                float dot = gx * nx + gy * ny + gz * nz;
                sums[vertex * 3 + 0] = (gx - dot * nx) / magnitude;
                sums[vertex * 3 + 1] = (gy - dot * ny) / magnitude;
                sums[vertex * 3 + 2] = (gz - dot * nz) / magnitude;
            } else {
                sums[vertex * 3 + 0] = 0.0f;
                sums[vertex * 3 + 1] = 0.0f;
                sums[vertex * 3 + 2] = 0.0f;
            }
        }
        for (face = 0; face < face_count; ++face) {
            int64_t i0 = (int64_t)face_data[face * 3 + 0];
            int64_t i1 = (int64_t)face_data[face * 3 + 1];
            int64_t i2 = (int64_t)face_data[face * 3 + 2];
            const float *a = vertex_data + i0 * 3;
            const float *b = vertex_data + i1 * 3;
            const float *c = vertex_data + i2 * 3;
            float e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
            float e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
            float cx = e1y * e2z - e1z * e2y;
            float cy = e1z * e2x - e1x * e2z;
            float cz = e1x * e2y - e1y * e2x;
            float doubled = sqrtf(cx * cx + cy * cy + cz * cz);
            float qx, qy, qz, gcx, gcy, gcz;
            float g1x, g1y, g1z, g2x, g2y, g2z;
            if (!((double)doubled > eps)) continue;
            qx = sums[i0 * 3 + 0] + sums[i1 * 3 + 0] + sums[i2 * 3 + 0];
            qy = sums[i0 * 3 + 1] + sums[i1 * 3 + 1] + sums[i2 * 3 + 1];
            qz = sums[i0 * 3 + 2] + sums[i1 * 3 + 2] + sums[i2 * 3 + 2];
            if (weighting == GFFX_MESH_WEIGHTING_AREA) {
                gcx = 0.5f * qx; gcy = 0.5f * qy; gcz = 0.5f * qz;
            } else {
                float nx = cx / doubled;
                float ny = cy / doubled;
                float nz = cz / doubled;
                float dot = qx * nx + qy * ny + qz * nz;
                gcx = (qx - dot * nx) / doubled;
                gcy = (qy - dot * ny) / doubled;
                gcz = (qz - dot * nz) / doubled;
            }
            g1x = e2y * gcz - e2z * gcy;
            g1y = e2z * gcx - e2x * gcz;
            g1z = e2x * gcy - e2y * gcx;
            g2x = gcy * e1z - gcz * e1y;
            g2y = gcz * e1x - gcx * e1z;
            g2z = gcx * e1y - gcy * e1x;
            gradient[i1 * 3 + 0] += g1x;
            gradient[i1 * 3 + 1] += g1y;
            gradient[i1 * 3 + 2] += g1z;
            gradient[i2 * 3 + 0] += g2x;
            gradient[i2 * 3 + 1] += g2y;
            gradient[i2 * 3 + 2] += g2z;
            gradient[i0 * 3 + 0] -= g1x + g2x;
            gradient[i0 * 3 + 1] -= g1y + g2y;
            gradient[i0 * 3 + 2] -= g1z + g2z;
        }
    }
    return GFFX_STATUS_OK;
}
