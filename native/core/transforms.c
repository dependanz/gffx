/*
 * transforms.transform_points and transforms.perspective_divide - Phase 2 CPU reference kernels.
 *
 * Semantics follow <gffx/transforms.h> and the project acceptance record: packed batch offsets
 * select one row-major [4,4] matrix per point, the point is the column vector [x,y,z,1], no
 * structure is assumed of the matrix, and accumulation order within a row is fixed so repeated
 * calls are bitwise identical. The divide applies a strict double-precision |w| > eps guard and
 * gives exactly zero gradients through the invalid branch.
 */

#include <gffx/execution.h>
#include <gffx/tensor.h>
#include <gffx/transforms.h>

#include "internal.h"
#include "cuda_loader.h"
#include "mesh_common.h"

#include <math.h>
#include <stdint.h>

/* Validates the offsets array in full before any point data is touched. */
static gffx_status gffx_transforms_check_offsets(
    const gffx_tensor_view *point_offsets,
    int64_t point_count,
    int64_t batch_count,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status;
    const int32_t *offsets;
    int64_t index;
    if (point_offsets == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "point offsets must be a [B+1] tensor view"
        );
    }
    if (point_offsets->rank != 1u || point_offsets->shape == NULL ||
        point_offsets->shape[0] != batch_count + INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "point offsets must have extent B+1"
        );
    }
    status = gffx_validate_tensor_view(point_offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (point_offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "point offsets must use the int32 dtype"
        );
    }
    if (point_offsets->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "transforms implement only the CPU backend in this phase"
        );
    }
    if ((point_offsets->flags & GFFX_TENSOR_OUTPUT) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation inputs may not carry the output flag"
        );
    }
    offsets = (const int32_t *)gffx_mesh_elements_const(point_offsets);
    if (offsets[0] != INT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the first point offset must be zero"
        );
    }
    for (index = 0; index < batch_count; ++index) {
        if (offsets[index + 1] < offsets[index]) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "point offsets must be nondecreasing"
            );
        }
    }
    if ((int64_t)offsets[batch_count] != point_count) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the final point offset must equal the point count"
        );
    }
    return GFFX_STATUS_OK;
}

/* Shared points/matrices/offsets/context/workspace validation for both directions. */
static gffx_status gffx_transforms_check_common(
    const gffx_tensor_view *points,
    const gffx_tensor_view *matrices,
    const gffx_tensor_view *point_offsets,
    const gffx_execution_context *context,
    const gffx_buffer *workspace,
    int64_t *point_count,
    int64_t *batch_count,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "transforms implement only the CPU backend in this phase"
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
    if (matrices == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "matrices must be a [B,4,4] tensor view"
        );
    }
    if (matrices->rank != 3u || matrices->shape == NULL ||
        matrices->shape[1] != INT64_C(4) || matrices->shape[2] != INT64_C(4)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "matrices must be a [B,4,4] tensor view"
        );
    }
    status = gffx_validate_tensor_view(matrices, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (matrices->dtype != points->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "matrices must match the points dtype"
        );
    }
    if (matrices->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "transforms implement only the CPU backend in this phase"
        );
    }
    if ((matrices->flags & GFFX_TENSOR_OUTPUT) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation inputs may not carry the output flag"
        );
    }
    *point_count = points->shape[0];
    *batch_count = matrices->shape[0];
    status = gffx_transforms_check_offsets(point_offsets, *point_count, *batch_count,
                                           diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    return GFFX_STATUS_OK;
}

static gffx_status gffx_transforms_zero_workspace(
    int64_t first_count,
    int64_t second_count,
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
    if (first_count < INT64_C(0) || second_count < INT64_C(0)) {
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
            "transforms support the float32 and float64 dtypes"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "transforms implement only the CPU backend in this phase"
        );
    }
    *required_bytes = UINT64_C(0);
    *required_alignment = UINT64_C(1);
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_transforms_transform_points_workspace(
    int64_t point_count,
    int64_t batch_count,
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
            GFFX_CUDA_OP_TRANSFORMS_TRANSFORM_POINTS, NULL, 0u, dtype, context, required_bytes,
            required_alignment, diagnostic);
    }
    return gffx_transforms_zero_workspace(point_count, batch_count, dtype, context,
                                          required_bytes, required_alignment, diagnostic);
}

GFFX_API gffx_status GFFX_CALL gffx_transforms_perspective_divide_workspace(
    int64_t point_count,
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
            GFFX_CUDA_OP_TRANSFORMS_PERSPECTIVE_DIVIDE, NULL, 0u, dtype, context, required_bytes,
            required_alignment, diagnostic);
    }
    return gffx_transforms_zero_workspace(point_count, INT64_C(0), dtype, context,
                                          required_bytes, required_alignment, diagnostic);
}

GFFX_API gffx_status GFFX_CALL gffx_transforms_transform_points(
    const gffx_tensor_view *points,
    const gffx_tensor_view *matrices,
    const gffx_tensor_view *point_offsets,
    const gffx_execution_context *context,
    gffx_tensor_view *homogeneous,
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
        if (operations->transforms_transform_points == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "the CUDA provider does not implement this operation");
        }
        return operations->transforms_transform_points(
            points, matrices, point_offsets, context, homogeneous, workspace, diagnostic);
    }
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t point_count = 0;
    int64_t batch_count = 0;
    int64_t batch;
    int64_t point;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_transforms_check_common(points, matrices, point_offsets, context, workspace,
                                          &point_count, &batch_count, diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    if (homogeneous == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "homogeneous output must be a [P,4] view"
        );
    }
    if (homogeneous->rank != 2u || homogeneous->shape == NULL ||
        homogeneous->shape[0] != point_count || homogeneous->shape[1] != INT64_C(4)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "homogeneous output must be a [P,4] view"
        );
    }
    status = gffx_validate_tensor_view(homogeneous, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if ((homogeneous->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation outputs must carry the output flag"
        );
    }
    if (homogeneous->dtype != points->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "the homogeneous output must match the points dtype"
        );
    }
    if (gffx_mesh_views_overlap(homogeneous, points) ||
        gffx_mesh_views_overlap(homogeneous, matrices) ||
        gffx_mesh_views_overlap(homogeneous, point_offsets)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (point_count == INT64_C(0)) return GFFX_STATUS_OK;

    {
        const int32_t *offsets = (const int32_t *)gffx_mesh_elements_const(point_offsets);
        if (points->dtype == GFFX_DTYPE_FLOAT64) {
            const double *source = (const double *)gffx_mesh_elements_const(points);
            const double *matrix_data = (const double *)gffx_mesh_elements_const(matrices);
            double *target = (double *)gffx_mesh_elements(homogeneous);
            for (batch = 0; batch < batch_count; ++batch) {
                const double *m = matrix_data + batch * 16;
                for (point = (int64_t)offsets[batch]; point < (int64_t)offsets[batch + 1];
                     ++point) {
                    double x = source[point * 3 + 0];
                    double y = source[point * 3 + 1];
                    double z = source[point * 3 + 2];
                    int row;
                    for (row = 0; row < 4; ++row) {
                        target[point * 4 + row] =
                            m[row * 4 + 0] * x + m[row * 4 + 1] * y + m[row * 4 + 2] * z +
                            m[row * 4 + 3];
                    }
                }
            }
        } else {
            const float *source = (const float *)gffx_mesh_elements_const(points);
            const float *matrix_data = (const float *)gffx_mesh_elements_const(matrices);
            float *target = (float *)gffx_mesh_elements(homogeneous);
            for (batch = 0; batch < batch_count; ++batch) {
                const float *m = matrix_data + batch * 16;
                for (point = (int64_t)offsets[batch]; point < (int64_t)offsets[batch + 1];
                     ++point) {
                    float x = source[point * 3 + 0];
                    float y = source[point * 3 + 1];
                    float z = source[point * 3 + 2];
                    int row;
                    for (row = 0; row < 4; ++row) {
                        target[point * 4 + row] =
                            m[row * 4 + 0] * x + m[row * 4 + 1] * y + m[row * 4 + 2] * z +
                            m[row * 4 + 3];
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

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
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t point_count = 0;
    int64_t batch_count = 0;
    int64_t batch;
    int64_t point;
    int64_t index;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_transforms_check_common(points, matrices, point_offsets, context, workspace,
                                          &point_count, &batch_count, diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    if (grad_points == NULL && grad_matrices == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "at least one gradient output is required"
        );
    }
    if (grad_homogeneous == NULL || grad_homogeneous->rank != 2u ||
        grad_homogeneous->shape == NULL || grad_homogeneous->shape[0] != point_count ||
        grad_homogeneous->shape[1] != INT64_C(4)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "homogeneous cotangents must be a [P,4] tensor view"
        );
    }
    status = gffx_validate_tensor_view(grad_homogeneous, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_homogeneous->dtype != points->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "cotangents must match the points dtype"
        );
    }
    if (grad_points != NULL) {
        status = gffx_mesh_check_view(grad_points, "point gradients must be a [P,3] output view",
                                      2u, point_count, INT64_C(3), 1, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (grad_points->dtype != points->dtype) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "point gradients must match the points dtype"
            );
        }
        if (gffx_mesh_views_overlap(grad_points, points) ||
            gffx_mesh_views_overlap(grad_points, matrices) ||
            gffx_mesh_views_overlap(grad_points, grad_homogeneous)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "outputs may not alias an input or another output"
            );
        }
    }
    if (grad_matrices != NULL) {
        if (grad_matrices->rank != 3u || grad_matrices->shape == NULL ||
            grad_matrices->shape[0] != batch_count ||
            grad_matrices->shape[1] != INT64_C(4) || grad_matrices->shape[2] != INT64_C(4)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "matrix gradients must be a [B,4,4] output view"
            );
        }
        status = gffx_validate_tensor_view(grad_matrices, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if ((grad_matrices->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "operation outputs must carry the output flag"
            );
        }
        if (grad_matrices->dtype != points->dtype) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "matrix gradients must match the points dtype"
            );
        }
        if (gffx_mesh_views_overlap(grad_matrices, points) ||
            gffx_mesh_views_overlap(grad_matrices, matrices) ||
            gffx_mesh_views_overlap(grad_matrices, grad_homogeneous) ||
            (grad_points != NULL && gffx_mesh_views_overlap(grad_matrices, grad_points))) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "outputs may not alias an input or another output"
            );
        }
    }

    if (points->dtype == GFFX_DTYPE_FLOAT64) {
        const double *source = (const double *)gffx_mesh_elements_const(points);
        const double *matrix_data = (const double *)gffx_mesh_elements_const(matrices);
        const double *cotangent = (const double *)gffx_mesh_elements_const(grad_homogeneous);
        const int32_t *offsets = (const int32_t *)gffx_mesh_elements_const(point_offsets);
        double *point_gradient =
            grad_points != NULL ? (double *)gffx_mesh_elements(grad_points) : NULL;
        double *matrix_gradient =
            grad_matrices != NULL ? (double *)gffx_mesh_elements(grad_matrices) : NULL;
        if (point_gradient != NULL) {
            for (index = 0; index < point_count * INT64_C(3); ++index) point_gradient[index] = 0.0;
        }
        if (matrix_gradient != NULL) {
            for (index = 0; index < batch_count * INT64_C(16); ++index) {
                matrix_gradient[index] = 0.0;
            }
        }
        for (batch = 0; batch < batch_count; ++batch) {
            const double *m = matrix_data + batch * 16;
            double *gm = matrix_gradient != NULL ? matrix_gradient + batch * 16 : NULL;
            for (point = (int64_t)offsets[batch]; point < (int64_t)offsets[batch + 1]; ++point) {
                const double *g = cotangent + point * 4;
                int row;
                if (point_gradient != NULL) {
                    int column;
                    for (column = 0; column < 3; ++column) {
                        point_gradient[point * 3 + column] =
                            g[0] * m[0 * 4 + column] + g[1] * m[1 * 4 + column] +
                            g[2] * m[2 * 4 + column] + g[3] * m[3 * 4 + column];
                    }
                }
                if (gm != NULL) {
                    double x = source[point * 3 + 0];
                    double y = source[point * 3 + 1];
                    double z = source[point * 3 + 2];
                    for (row = 0; row < 4; ++row) {
                        gm[row * 4 + 0] += g[row] * x;
                        gm[row * 4 + 1] += g[row] * y;
                        gm[row * 4 + 2] += g[row] * z;
                        gm[row * 4 + 3] += g[row];
                    }
                }
            }
        }
    } else {
        const float *source = (const float *)gffx_mesh_elements_const(points);
        const float *matrix_data = (const float *)gffx_mesh_elements_const(matrices);
        const float *cotangent = (const float *)gffx_mesh_elements_const(grad_homogeneous);
        const int32_t *offsets = (const int32_t *)gffx_mesh_elements_const(point_offsets);
        float *point_gradient =
            grad_points != NULL ? (float *)gffx_mesh_elements(grad_points) : NULL;
        float *matrix_gradient =
            grad_matrices != NULL ? (float *)gffx_mesh_elements(grad_matrices) : NULL;
        if (point_gradient != NULL) {
            for (index = 0; index < point_count * INT64_C(3); ++index) point_gradient[index] = 0.0f;
        }
        if (matrix_gradient != NULL) {
            for (index = 0; index < batch_count * INT64_C(16); ++index) {
                matrix_gradient[index] = 0.0f;
            }
        }
        for (batch = 0; batch < batch_count; ++batch) {
            const float *m = matrix_data + batch * 16;
            float *gm = matrix_gradient != NULL ? matrix_gradient + batch * 16 : NULL;
            for (point = (int64_t)offsets[batch]; point < (int64_t)offsets[batch + 1]; ++point) {
                const float *g = cotangent + point * 4;
                int row;
                if (point_gradient != NULL) {
                    int column;
                    for (column = 0; column < 3; ++column) {
                        point_gradient[point * 3 + column] =
                            g[0] * m[0 * 4 + column] + g[1] * m[1 * 4 + column] +
                            g[2] * m[2 * 4 + column] + g[3] * m[3 * 4 + column];
                    }
                }
                if (gm != NULL) {
                    float x = source[point * 3 + 0];
                    float y = source[point * 3 + 1];
                    float z = source[point * 3 + 2];
                    for (row = 0; row < 4; ++row) {
                        gm[row * 4 + 0] += g[row] * x;
                        gm[row * 4 + 1] += g[row] * y;
                        gm[row * 4 + 2] += g[row] * z;
                        gm[row * 4 + 3] += g[row];
                    }
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

/* Shared validation for both divide directions. */
static gffx_status gffx_divide_check_common(
    const gffx_tensor_view *homogeneous,
    double eps,
    const gffx_execution_context *context,
    const gffx_buffer *workspace,
    int64_t *point_count,
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
            "transforms implement only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(homogeneous, "homogeneous input must be a [P,4] tensor view",
                                  2u, INT64_C(-1), INT64_C(4), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (homogeneous->dtype != GFFX_DTYPE_FLOAT32 && homogeneous->dtype != GFFX_DTYPE_FLOAT64) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "the homogeneous input must use the float32 or float64 dtype"
        );
    }
    if (workspace != NULL) {
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
    }
    *point_count = homogeneous->shape[0];
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_transforms_perspective_divide(
    const gffx_tensor_view *homogeneous,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *ndc,
    gffx_tensor_view *valid,
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
        if (operations->transforms_perspective_divide == NULL) {
            return gffx_internal_fail(
                diagnostic, GFFX_STATUS_UNSUPPORTED,
                "the CUDA provider does not implement this operation");
        }
        return operations->transforms_perspective_divide(
            homogeneous, eps, context, ndc, valid, workspace, diagnostic);
    }
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t point_count = 0;
    int64_t point;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_divide_check_common(homogeneous, eps, context, workspace, &point_count,
                                      diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    status = gffx_mesh_check_view(ndc, "ndc must be a [P,3] output view",
                                  2u, point_count, INT64_C(3), 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (ndc->dtype != homogeneous->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "ndc must match the homogeneous dtype"
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
    if (gffx_mesh_views_overlap(ndc, homogeneous) || gffx_mesh_views_overlap(valid, homogeneous) ||
        gffx_mesh_views_overlap(ndc, valid)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (point_count == INT64_C(0)) return GFFX_STATUS_OK;

    {
        uint8_t *valid_data = (uint8_t *)gffx_mesh_elements(valid);
        if (homogeneous->dtype == GFFX_DTYPE_FLOAT64) {
            const double *source = (const double *)gffx_mesh_elements_const(homogeneous);
            double *target = (double *)gffx_mesh_elements(ndc);
            for (point = 0; point < point_count; ++point) {
                double w = source[point * 4 + 3];
                if (fabs(w) > eps) {
                    target[point * 3 + 0] = source[point * 4 + 0] / w;
                    target[point * 3 + 1] = source[point * 4 + 1] / w;
                    target[point * 3 + 2] = source[point * 4 + 2] / w;
                    valid_data[point] = 1u;
                } else {
                    target[point * 3 + 0] = 0.0;
                    target[point * 3 + 1] = 0.0;
                    target[point * 3 + 2] = 0.0;
                    valid_data[point] = 0u;
                }
            }
        } else {
            const float *source = (const float *)gffx_mesh_elements_const(homogeneous);
            float *target = (float *)gffx_mesh_elements(ndc);
            for (point = 0; point < point_count; ++point) {
                float w = source[point * 4 + 3];
                if ((double)fabsf(w) > eps) {
                    target[point * 3 + 0] = source[point * 4 + 0] / w;
                    target[point * 3 + 1] = source[point * 4 + 1] / w;
                    target[point * 3 + 2] = source[point * 4 + 2] / w;
                    valid_data[point] = 1u;
                } else {
                    target[point * 3 + 0] = 0.0f;
                    target[point * 3 + 1] = 0.0f;
                    target[point * 3 + 2] = 0.0f;
                    valid_data[point] = 0u;
                }
            }
        }
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_transforms_perspective_divide_backward(
    const gffx_tensor_view *homogeneous,
    double eps,
    const gffx_tensor_view *grad_ndc,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_homogeneous,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t point_count = 0;
    int64_t point;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_divide_check_common(homogeneous, eps, context, workspace, &point_count,
                                      diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    status = gffx_mesh_check_view(grad_ndc, "ndc cotangents must be a [P,3] tensor view",
                                  2u, point_count, INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (grad_ndc->dtype != homogeneous->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "cotangents must match the homogeneous dtype"
        );
    }
    if (grad_homogeneous == NULL || grad_homogeneous->rank != 2u ||
        grad_homogeneous->shape == NULL || grad_homogeneous->shape[0] != point_count ||
        grad_homogeneous->shape[1] != INT64_C(4)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "homogeneous gradients must be a [P,4] output view"
        );
    }
    status = gffx_validate_tensor_view(grad_homogeneous, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if ((grad_homogeneous->flags & GFFX_TENSOR_OUTPUT) == UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation outputs must carry the output flag"
        );
    }
    if (grad_homogeneous->dtype != homogeneous->dtype) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "homogeneous gradients must match the homogeneous dtype"
        );
    }
    if (gffx_mesh_views_overlap(grad_homogeneous, homogeneous) ||
        gffx_mesh_views_overlap(grad_homogeneous, grad_ndc)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "outputs may not alias an input or another output"
        );
    }
    if (point_count == INT64_C(0)) return GFFX_STATUS_OK;

    if (homogeneous->dtype == GFFX_DTYPE_FLOAT64) {
        const double *source = (const double *)gffx_mesh_elements_const(homogeneous);
        const double *cotangent = (const double *)gffx_mesh_elements_const(grad_ndc);
        double *gradient = (double *)gffx_mesh_elements(grad_homogeneous);
        for (point = 0; point < point_count; ++point) {
            double w = source[point * 4 + 3];
            if (fabs(w) > eps) {
                double gx = cotangent[point * 3 + 0];
                double gy = cotangent[point * 3 + 1];
                double gz = cotangent[point * 3 + 2];
                double nx = source[point * 4 + 0] / w;
                double ny = source[point * 4 + 1] / w;
                double nz = source[point * 4 + 2] / w;
                gradient[point * 4 + 0] = gx / w;
                gradient[point * 4 + 1] = gy / w;
                gradient[point * 4 + 2] = gz / w;
                gradient[point * 4 + 3] = -(gx * nx + gy * ny + gz * nz) / w;
            } else {
                gradient[point * 4 + 0] = 0.0;
                gradient[point * 4 + 1] = 0.0;
                gradient[point * 4 + 2] = 0.0;
                gradient[point * 4 + 3] = 0.0;
            }
        }
    } else {
        const float *source = (const float *)gffx_mesh_elements_const(homogeneous);
        const float *cotangent = (const float *)gffx_mesh_elements_const(grad_ndc);
        float *gradient = (float *)gffx_mesh_elements(grad_homogeneous);
        for (point = 0; point < point_count; ++point) {
            float w = source[point * 4 + 3];
            if ((double)fabsf(w) > eps) {
                float gx = cotangent[point * 3 + 0];
                float gy = cotangent[point * 3 + 1];
                float gz = cotangent[point * 3 + 2];
                float nx = source[point * 4 + 0] / w;
                float ny = source[point * 4 + 1] / w;
                float nz = source[point * 4 + 2] / w;
                gradient[point * 4 + 0] = gx / w;
                gradient[point * 4 + 1] = gy / w;
                gradient[point * 4 + 2] = gz / w;
                gradient[point * 4 + 3] = -(gx * nx + gy * ny + gz * nz) / w;
            } else {
                gradient[point * 4 + 0] = 0.0f;
                gradient[point * 4 + 1] = 0.0f;
                gradient[point * 4 + 2] = 0.0f;
                gradient[point * 4 + 3] = 0.0f;
            }
        }
    }
    return GFFX_STATUS_OK;
}
