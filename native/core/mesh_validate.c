/*
 * mesh.validate - Phase 2 eager survey utility.
 *
 * This reports rather than gates. The operation kernels reject the first problem they find and
 * return INVALID_ARGUMENT because they must not dereference bad memory; this utility surveys the
 * whole mesh and returns OK with a populated report, so a caller sees every finding at once.
 *
 * Findings are established structurally before geometrically, and the survey stops short of work
 * that would be unsafe given what it already found: malformed offsets end the survey because no
 * element range can then be trusted, and an out-of-range or cross-element index ends it because
 * vertex lookup would be unsafe. Counts left unreached keep their initial values.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

GFFX_API gffx_status GFFX_CALL gffx_mesh_validate_workspace(
    int64_t vertex_count,
    int64_t face_count,
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
            "counts must be nonnegative"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.validate implements only the CPU backend in this phase"
        );
    }
    /* The unreferenced-vertex survey marks vertices by walking faces twice rather than holding a
     * mark array, so the utility needs no scratch storage. */
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

/* Checks one offsets array without failing the call: returns the first malformed batch element,
 * or -1 when the array is well formed. */
static int64_t gffx_validate_scan_offsets(
    const int32_t *offsets, int64_t total_count, int64_t batch_count
) {
    int64_t index;
    if (offsets[0] != INT32_C(0)) return 0;
    for (index = 0; index < batch_count; ++index) {
        if (offsets[index + 1] < offsets[index]) return index;
    }
    if ((int64_t)offsets[batch_count] != total_count) return batch_count - 1;
    return -1;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_validate(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    double eps,
    uint32_t flags,
    const gffx_execution_context *context,
    gffx_mesh_validation_report *report,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t vertex_count;
    int64_t face_count;
    int64_t batch_count;
    int64_t batch;
    int is_double;
    if (status != GFFX_STATUS_OK) return status;
    if (report == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the validation report pointer must not be null"
        );
    }
    if (report->struct_size < (uint32_t)sizeof(gffx_mesh_validation_report)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_ABI_MISMATCH,
            "the validation report is smaller than this ABI version requires"
        );
    }
    if ((flags & ~GFFX_MESH_VALIDATE_GEOMETRY) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "unknown validation flags"
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
            "mesh.validate implements only the CPU backend in this phase"
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
    if (vertex_offsets == NULL || vertex_offsets->rank != 1u ||
        vertex_offsets->shape == NULL || vertex_offsets->shape[0] < INT64_C(1) ||
        vertex_offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "vertex offsets must be a [B+1] int32 tensor view"
        );
    }
    status = gffx_validate_tensor_view(vertex_offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    batch_count = vertex_offsets->shape[0] - INT64_C(1);
    if (face_offsets == NULL || face_offsets->rank != 1u || face_offsets->shape == NULL ||
        face_offsets->shape[0] != batch_count + INT64_C(1) ||
        face_offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face offsets must be a [B+1] int32 tensor view"
        );
    }
    status = gffx_validate_tensor_view(face_offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    vertex_count = vertices->shape[0];
    face_count = faces->shape[0];
    is_double = vertices->dtype == GFFX_DTYPE_FLOAT64;

    /* A clean report is the starting point; the geometry count begins at -1 so "not checked"
     * stays distinguishable from "checked and clean". */
    report->findings = UINT32_C(0);
    report->first_bad_face = -1;
    report->first_bad_offset_batch = -1;
    report->degenerate_face_count = 0;
    report->nonfinite_vertex_count =
        (flags & GFFX_MESH_VALIDATE_GEOMETRY) != UINT32_C(0) ? 0 : -1;
    report->unreferenced_vertex_count = 0;

    {
        const int32_t *vertex_bounds =
            (const int32_t *)gffx_mesh_elements_const(vertex_offsets);
        const int32_t *face_bounds = (const int32_t *)gffx_mesh_elements_const(face_offsets);
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const double *vertex_d =
            (is_double && vertex_count > INT64_C(0))
                ? (const double *)gffx_mesh_elements_const(vertices) : NULL;
        const float *vertex_f =
            (!is_double && vertex_count > INT64_C(0))
                ? (const float *)gffx_mesh_elements_const(vertices) : NULL;
        int64_t bad_batch;

        /* 1. Offsets first: without them, no element range can be trusted. */
        bad_batch = gffx_validate_scan_offsets(vertex_bounds, vertex_count, batch_count);
        if (bad_batch < 0) {
            bad_batch = gffx_validate_scan_offsets(face_bounds, face_count, batch_count);
        }
        if (bad_batch >= 0) {
            report->findings |= GFFX_MESH_FINDING_OFFSETS;
            report->first_bad_offset_batch = bad_batch;
            return GFFX_STATUS_OK;
        }

        /* 2. Face indices: range first, then batch containment. */
        for (batch = 0; batch < batch_count; ++batch) {
            int64_t first_vertex = (int64_t)vertex_bounds[batch];
            int64_t last_vertex = (int64_t)vertex_bounds[batch + 1];
            int64_t face;
            for (face = (int64_t)face_bounds[batch]; face < (int64_t)face_bounds[batch + 1];
                 ++face) {
                int corner;
                for (corner = 0; corner < 3; ++corner) {
                    int64_t vertex = (int64_t)face_data[face * 3 + corner];
                    if (vertex < 0 || vertex >= vertex_count) {
                        report->findings |= GFFX_MESH_FINDING_FACE_INDEX_RANGE;
                    } else if (vertex < first_vertex || vertex >= last_vertex) {
                        report->findings |= GFFX_MESH_FINDING_FACE_INDEX_BATCH;
                    } else {
                        continue;
                    }
                    if (report->first_bad_face < 0) report->first_bad_face = face;
                }
            }
        }
        if ((report->findings & (GFFX_MESH_FINDING_FACE_INDEX_RANGE |
                                 GFFX_MESH_FINDING_FACE_INDEX_BATCH)) != UINT32_C(0)) {
            /* Vertex lookup is unsafe, so the geometric surveys are skipped. The optional
             * geometry pass touches no face data, so it may still run. */
            if ((flags & GFFX_MESH_VALIDATE_GEOMETRY) != UINT32_C(0)) {
                int64_t component;
                for (component = 0; component < vertex_count * INT64_C(3); component += 3) {
                    double x = is_double ? vertex_d[component] : (double)vertex_f[component];
                    double y = is_double ? vertex_d[component + 1]
                                         : (double)vertex_f[component + 1];
                    double z = is_double ? vertex_d[component + 2]
                                         : (double)vertex_f[component + 2];
                    if (isnan(x) || isinf(x) || isnan(y) || isinf(y) || isnan(z) || isinf(z)) {
                        report->findings |= GFFX_MESH_FINDING_NONFINITE_GEOMETRY;
                        report->nonfinite_vertex_count += 1;
                    }
                }
            }
            return GFFX_STATUS_OK;
        }

        /* 3. Geometry: degenerate faces, then vertices no face in their element references. */
        for (batch = 0; batch < batch_count; ++batch) {
            int64_t face;
            for (face = (int64_t)face_bounds[batch]; face < (int64_t)face_bounds[batch + 1];
                 ++face) {
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
                if (!(doubled > eps)) {
                    report->findings |= GFFX_MESH_FINDING_DEGENERATE_FACE;
                    report->degenerate_face_count += 1;
                }
            }
        }
        for (batch = 0; batch < batch_count; ++batch) {
            int64_t first_vertex = (int64_t)vertex_bounds[batch];
            int64_t last_vertex = (int64_t)vertex_bounds[batch + 1];
            int64_t vertex;
            for (vertex = first_vertex; vertex < last_vertex; ++vertex) {
                int referenced = 0;
                int64_t face;
                for (face = (int64_t)face_bounds[batch];
                     face < (int64_t)face_bounds[batch + 1] && !referenced; ++face) {
                    int corner;
                    for (corner = 0; corner < 3; ++corner) {
                        if ((int64_t)face_data[face * 3 + corner] == vertex) {
                            referenced = 1;
                            break;
                        }
                    }
                }
                if (!referenced) {
                    report->findings |= GFFX_MESH_FINDING_UNREFERENCED_VERTEX;
                    report->unreferenced_vertex_count += 1;
                }
            }
        }
        if ((flags & GFFX_MESH_VALIDATE_GEOMETRY) != UINT32_C(0)) {
            int64_t component;
            for (component = 0; component < vertex_count * INT64_C(3); component += 3) {
                double x = is_double ? vertex_d[component] : (double)vertex_f[component];
                double y = is_double ? vertex_d[component + 1]
                                     : (double)vertex_f[component + 1];
                double z = is_double ? vertex_d[component + 2]
                                     : (double)vertex_f[component + 2];
                if (isnan(x) || isinf(x) || isnan(y) || isinf(y) || isnan(z) || isinf(z)) {
                    report->findings |= GFFX_MESH_FINDING_NONFINITE_GEOMETRY;
                    report->nonfinite_vertex_count += 1;
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}
