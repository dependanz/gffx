#ifndef GFFX_TRANSFORMS_H
#define GFFX_TRANSFORMS_H

#include <gffx/execution.h>
#include <gffx/tensor.h>

/*
 * transforms.transform_points - packed-batch homogeneous point transform.
 *
 *   homogeneous[p][i] = M[i][0]*x + M[i][1]*y + M[i][2]*z + M[i][3]
 *
 * where M = matrices[b] is the row-major [4,4] matrix of the batch element containing point p,
 * and the point is the column vector [x, y, z, 1]. No perspective divide is performed and no
 * structure is assumed of M, so affine, projective, and singular matrices all execute.
 * Accumulation order within a row is fixed (x, then y, then z, then the translation column) so
 * repeated calls are bitwise identical.
 *
 * point_offsets is int32 [B+1] with offsets[0] == 0, nondecreasing entries, and
 * offsets[B] == P; point p belongs to batch element b when offsets[b] <= p < offsets[b+1].
 * Empty batch elements are legal. The whole array is validated before any dereference.
 *
 * Backward computes grad_points [P,3] and grad_matrices [B,4,4]; either output view may be null
 * to skip it, but not both. Both are overwritten, and grad_matrices accumulates over points in
 * ascending point order. point_offsets is nondifferentiable. Workspace is zero bytes.
 *
 * transforms.perspective_divide - guarded homogeneous divide.
 *
 *   w = homogeneous[p][3]; valid[p] = (|w| > eps), compared in double precision
 *   valid:   ndc[p][j] = homogeneous[p][j] / w
 *   invalid: ndc[p][:] = 0
 *
 * eps must be finite and >= 0. A NaN w compares false and takes the invalid branch. Backward
 * gives grad_homogeneous[p][j] = grad_ndc[p][j] / w and
 * grad_homogeneous[p][3] = -(grad_ndc[p] . ndc[p]) / w for valid points, and exactly zero in all
 * four components for invalid ones. Workspace is zero bytes.
 */

GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_transforms_transform_points_workspace(
    int64_t point_count,
    int64_t batch_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_transforms_transform_points(
    const gffx_tensor_view *points,
    const gffx_tensor_view *matrices,
    const gffx_tensor_view *point_offsets,
    const gffx_execution_context *context,
    gffx_tensor_view *homogeneous,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_transforms_transform_points_backward(
    const gffx_tensor_view *points,
    const gffx_tensor_view *matrices,
    const gffx_tensor_view *point_offsets,
    const gffx_tensor_view *grad_homogeneous,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_points,
    gffx_tensor_view *grad_matrices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_transforms_perspective_divide_workspace(
    int64_t point_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_transforms_perspective_divide(
    const gffx_tensor_view *homogeneous,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *ndc,
    gffx_tensor_view *valid,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_transforms_perspective_divide_backward(
    const gffx_tensor_view *homogeneous,
    double eps,
    const gffx_tensor_view *grad_ndc,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_homogeneous,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif /* GFFX_TRANSFORMS_H */
