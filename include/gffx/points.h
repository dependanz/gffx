#ifndef GFFX_POINTS_H
#define GFFX_POINTS_H

#include <gffx/execution.h>
#include <gffx/tensor.h>

/*
 * points.knn - K nearest reference points per query, within a batch element.
 *
 * Selection uses squared Euclidean distance, dot(q - r, q - r), with no square root, so the
 * output carries no rounding from a root. Results are ordered by
 * (distance_squared, reference_index) ascending, so an exact distance tie selects the lower
 * global index and the ordering is fully determined. reference_index carries global packed
 * indices. When a batch element holds fewer than K reference points, trailing entries are +inf,
 * -1, and false. K is a static positive argument. Complexity is O(P*R) per batch element; v0.1
 * ships no spatial acceleration structure.
 *
 * Backward consumes the forward reference_index and valid outputs rather than recomputing the
 * selection, keeping the gradient consistent with the results the caller received:
 *
 *   grad_query[p]     += 2 * (query[p] - reference[r]) * g
 *   grad_reference[r] -= 2 * (query[p] - reference[r]) * g
 *
 * Both outputs are overwritten, then accumulated in ascending (p, k) order; invalid entries
 * contribute exactly zero. Either gradient output may be null to skip it, but not both. The
 * selection is piecewise constant and the contract makes no claim at ties or Voronoi boundaries.
 * Workspace is zero bytes for both directions.
 *
 * points.closest_point_on_mesh - closest point on any valid triangle of the matching mesh.
 *
 * A triangle is valid when its doubled area exceeds eps, the same rule mesh.face_geometry uses;
 * degenerate triangles are skipped rather than contributing a degenerate closest point. The
 * winning face is the one with the smallest distance_squared, and an exact tie selects the lower
 * global face index. barycentric reproduces closest exactly as b0*v0 + b1*v1 + b2*v2 and sums to
 * one for any valid result. An empty mesh, or one whose faces are all degenerate, returns +inf,
 * -1, zero barycentrics, zero closest point, and false. Faces must reference vertices inside
 * their own batch element. Complexity is O(P*F) per batch element.
 *
 * Backward propagates the distance_squared cotangent only; the signature accepts no cotangent
 * for closest or barycentric, so that v0.1 limit is visible in the ABI rather than buried in
 * prose. Within a fixed closest-feature region the envelope theorem gives
 *
 *   grad_points[p]     += 2 * (p - c) * g
 *   grad_vertices[v_i] -= 2 * b_i * (p - c) * g
 *
 * for the three vertices of the winning face, because at the optimum the residual p - c is
 * orthogonal to every direction the feature allows the closest point to move. Invalid results
 * contribute exactly zero. Workspace is zero bytes for both directions.
 */

GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_points_knn_workspace(
    int64_t query_count,
    int64_t reference_count,
    int64_t neighbor_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

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
);

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
);

GFFX_API gffx_status GFFX_CALL gffx_points_closest_point_on_mesh_workspace(
    int64_t point_count,
    int64_t vertex_count,
    int64_t face_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

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
);

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
);

GFFX_EXTERN_C_END

#endif /* GFFX_POINTS_H */
