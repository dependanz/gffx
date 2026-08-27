#ifndef GFFX_MESH_H
#define GFFX_MESH_H

#include <gffx/execution.h>
#include <gffx/tensor.h>

/*
 * mesh.face_geometry - per-face unit normals, areas, and validity.
 *
 * For face k with vertex indices (i0, i1, i2):
 *   e1 = vertices[i1] - vertices[i0]
 *   e2 = vertices[i2] - vertices[i0]
 *   c  = cross(e1, e2)              right-handed cross product
 *   d  = ||c||                      doubled triangle area
 *   valid[k] = (d > eps)            strict, evaluated in double precision
 *   valid:   unit_normals[k] = c / d,  areas[k] = d / 2
 *   invalid: unit_normals[k] = 0,      areas[k] = 0
 *
 * Contract summary (the project acceptance record is normative):
 * - vertices is [V,3] FLOAT32 or FLOAT64 and selects the computation dtype; faces is [F,3]
 *   INT32 with every index in [0, V). All views are dense, C-contiguous, CPU, and validated
 *   before any data dereference.
 * - unit_normals [F,3] and areas [F] use the vertices dtype; valid [F] is GFFX_DTYPE_BOOL with
 *   one byte per element. Outputs carry GFFX_TENSOR_OUTPUT and may not alias any input or each
 *   other.
 * - eps must be finite and >= 0. A NaN doubled area compares as not-valid and takes the
 *   invalid branch; the valid branch performs the stated arithmetic under IEEE semantics with
 *   no clamping or sanitization.
 * - Faces are processed independently in ascending index order; repeated calls on identical
 *   inputs are bitwise identical for the same build and configuration.
 * - The workspace pointer may be NULL when the workspace query reports zero required bytes.
 *   The scalar CPU reference requires zero bytes; callers must still use the query.
 *
 * Backward: accumulates d(sum(grad_unit_normals . unit_normals + grad_areas . areas))/d(vertices)
 * into grad_vertices [V,3], overwriting it. Either cotangent view pointer may be NULL, which is
 * treated as a zero cotangent, but not both. Contributions are accumulated per face in ascending
 * face order (deterministic). Invalid faces contribute exactly zero gradient.
 */

GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_mesh_face_geometry_workspace(
    int64_t vertex_count,
    int64_t face_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

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
);

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
);

GFFX_EXTERN_C_END

#endif /* GFFX_MESH_H */
