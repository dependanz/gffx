/*
 * Phase 2 acceptance fixtures for transforms.transform_points (TP-01..TP-12),
 * transforms.perspective_divide (PD-01..PD-08), and the camera integration fixtures
 * (CAM-01..CAM-07, CAM-09) from the project camera contract.
 *
 * CAM-08 covers eager-layer parameter rejection and has no native surface to test, so it is
 * deferred to the Phase 3 adapter rather than asserted here.
 */

#include <gffx/execution.h>
#include <gffx/status.h>
#include <gffx/tensor.h>
#include <gffx/transforms.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define TF_EPS_DEFAULT 9.5367431640625e-7 /* 2^-20 */

static const int64_t pair_strides[2] = {3, 1};
static const int64_t quad_strides[2] = {4, 1};
static const int64_t matrix_strides[3] = {16, 4, 1};
static const int64_t scalar_strides[1] = {1};

static gffx_execution_context cpu_context(void) {
    gffx_execution_context context = {0};
    context.struct_size = (uint32_t)sizeof(context);
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    context.device_index = 0;
    return context;
}

static gffx_tensor_view make_view(
    void *data, gffx_dtype dtype, uint32_t rank,
    const int64_t *shape, const int64_t *strides, uint32_t flags
) {
    gffx_tensor_view view = {0};
    view.struct_size = (uint32_t)sizeof(view);
    view.abi_version = GFFX_ABI_VERSION;
    view.data = data;
    view.rank = rank;
    view.shape = shape;
    view.strides = strides;
    view.dtype = dtype;
    view.device_type = GFFX_DEVICE_CPU;
    view.device_index = 0;
    view.flags = flags;
    return view;
}

static void set_component(void *data, gffx_dtype dtype, int64_t index, double value) {
    if (dtype == GFFX_DTYPE_FLOAT64) ((double *)data)[index] = value;
    else ((float *)data)[index] = (float)value;
}

static double get_component(const void *data, gffx_dtype dtype, int64_t index) {
    if (dtype == GFFX_DTYPE_FLOAT64) return ((const double *)data)[index];
    return (double)((const float *)data)[index];
}

static void fill_components(void *data, gffx_dtype dtype, const double *values, int64_t count) {
    int64_t index;
    for (index = 0; index < count; ++index) set_component(data, dtype, index, values[index]);
}

static int relative_close(double actual, double expected, double tolerance) {
    double magnitude = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
    return fabs(actual - expected) <= tolerance * magnitude;
}

static gffx_status run_transform(
    const void *points, int64_t point_count,
    const void *matrices, int64_t batch_count,
    const int32_t *offsets, gffx_dtype dtype, void *homogeneous
) {
    int64_t point_shape[2];
    int64_t matrix_shape[3];
    int64_t offset_shape[1];
    int64_t out_shape[2];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view points_view;
    gffx_tensor_view matrices_view;
    gffx_tensor_view offsets_view;
    gffx_tensor_view out_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    point_shape[0] = point_count; point_shape[1] = 3;
    matrix_shape[0] = batch_count; matrix_shape[1] = 4; matrix_shape[2] = 4;
    offset_shape[0] = batch_count + 1;
    out_shape[0] = point_count; out_shape[1] = 4;

    points_view = make_view((void *)points, dtype, 2u, point_shape, pair_strides,
                            GFFX_TENSOR_READ_ONLY);
    matrices_view = make_view((void *)matrices, dtype, 3u, matrix_shape, matrix_strides,
                              GFFX_TENSOR_READ_ONLY);
    offsets_view = make_view((void *)offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                             GFFX_TENSOR_READ_ONLY);
    out_view = make_view(homogeneous, dtype, 2u, out_shape, quad_strides, GFFX_TENSOR_OUTPUT);
    return gffx_transforms_transform_points(&points_view, &matrices_view, &offsets_view,
                                            &context, &out_view, NULL, &diagnostic);
}

static gffx_status run_transform_backward(
    const void *points, int64_t point_count,
    const void *matrices, int64_t batch_count,
    const int32_t *offsets, gffx_dtype dtype,
    const void *grad_homogeneous, void *grad_points, void *grad_matrices
) {
    int64_t point_shape[2];
    int64_t matrix_shape[3];
    int64_t offset_shape[1];
    int64_t out_shape[2];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view points_view;
    gffx_tensor_view matrices_view;
    gffx_tensor_view offsets_view;
    gffx_tensor_view cotangent_view;
    gffx_tensor_view grad_points_view;
    gffx_tensor_view grad_matrices_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    point_shape[0] = point_count; point_shape[1] = 3;
    matrix_shape[0] = batch_count; matrix_shape[1] = 4; matrix_shape[2] = 4;
    offset_shape[0] = batch_count + 1;
    out_shape[0] = point_count; out_shape[1] = 4;

    points_view = make_view((void *)points, dtype, 2u, point_shape, pair_strides,
                            GFFX_TENSOR_READ_ONLY);
    matrices_view = make_view((void *)matrices, dtype, 3u, matrix_shape, matrix_strides,
                              GFFX_TENSOR_READ_ONLY);
    offsets_view = make_view((void *)offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                             GFFX_TENSOR_READ_ONLY);
    cotangent_view = make_view((void *)grad_homogeneous, dtype, 2u, out_shape, quad_strides,
                               GFFX_TENSOR_READ_ONLY);
    grad_points_view = make_view(grad_points, dtype, 2u, point_shape, pair_strides,
                                 GFFX_TENSOR_OUTPUT);
    grad_matrices_view = make_view(grad_matrices, dtype, 3u, matrix_shape, matrix_strides,
                                   GFFX_TENSOR_OUTPUT);
    return gffx_transforms_transform_points_backward(
        &points_view, &matrices_view, &offsets_view, &cotangent_view, &context,
        grad_points != NULL ? &grad_points_view : NULL,
        grad_matrices != NULL ? &grad_matrices_view : NULL, NULL, &diagnostic);
}

static gffx_status run_divide(
    const void *homogeneous, int64_t point_count, gffx_dtype dtype, double eps,
    void *ndc, uint8_t *valid
) {
    int64_t in_shape[2];
    int64_t ndc_shape[2];
    int64_t valid_shape[1];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view in_view;
    gffx_tensor_view ndc_view;
    gffx_tensor_view valid_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    in_shape[0] = point_count; in_shape[1] = 4;
    ndc_shape[0] = point_count; ndc_shape[1] = 3;
    valid_shape[0] = point_count;

    in_view = make_view((void *)homogeneous, dtype, 2u, in_shape, quad_strides,
                        GFFX_TENSOR_READ_ONLY);
    ndc_view = make_view(ndc, dtype, 2u, ndc_shape, pair_strides, GFFX_TENSOR_OUTPUT);
    valid_view = make_view(valid, GFFX_DTYPE_BOOL, 1u, valid_shape, scalar_strides,
                           GFFX_TENSOR_OUTPUT);
    return gffx_transforms_perspective_divide(&in_view, eps, &context, &ndc_view, &valid_view,
                                              NULL, &diagnostic);
}

static gffx_status run_divide_backward(
    const void *homogeneous, int64_t point_count, gffx_dtype dtype, double eps,
    const void *grad_ndc, void *grad_homogeneous
) {
    int64_t in_shape[2];
    int64_t ndc_shape[2];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view in_view;
    gffx_tensor_view cotangent_view;
    gffx_tensor_view grad_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    in_shape[0] = point_count; in_shape[1] = 4;
    ndc_shape[0] = point_count; ndc_shape[1] = 3;

    in_view = make_view((void *)homogeneous, dtype, 2u, in_shape, quad_strides,
                        GFFX_TENSOR_READ_ONLY);
    cotangent_view = make_view((void *)grad_ndc, dtype, 2u, ndc_shape, pair_strides,
                               GFFX_TENSOR_READ_ONLY);
    grad_view = make_view(grad_homogeneous, dtype, 2u, in_shape, quad_strides,
                          GFFX_TENSOR_OUTPUT);
    return gffx_transforms_perspective_divide_backward(&in_view, eps, &cotangent_view, &context,
                                                       &grad_view, NULL, &diagnostic);
}

static void identity_matrix(double *m) {
    int64_t index;
    for (index = 0; index < 16; ++index) m[index] = 0.0;
    m[0] = 1.0; m[5] = 1.0; m[10] = 1.0; m[15] = 1.0;
}

/* Row-major projection matrix from pinhole intrinsics, per the camera contract section 3. */
static void projection_matrix(
    double fx, double fy, double cx, double cy,
    double width, double height, double near_plane, double far_plane, double *m
) {
    int64_t index;
    for (index = 0; index < 16; ++index) m[index] = 0.0;
    m[0] = 2.0 * fx / width;
    m[2] = 1.0 - 2.0 * cx / width;
    m[5] = 2.0 * fy / height;
    m[6] = 2.0 * cy / height - 1.0;
    m[10] = (far_plane + near_plane) / (near_plane - far_plane);
    m[11] = (2.0 * far_plane * near_plane) / (near_plane - far_plane);
    m[14] = -1.0;
}

static const double three_points[9] = {
    0.25, -0.5, 2.0,   1.5, 0.75, -3.0,   -2.0, 0.125, 0.5
};
static const int32_t offsets_single[2] = {0, 3};

static int test_tp01_tp03_basic(gffx_dtype dtype) {
    double matrix[16];
    double hd[12]; float hf[12];
    double md[16]; float mf[16];
    double pd[9]; float pf[9];
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *m = dtype == GFFX_DTYPE_FLOAT64 ? (void *)md : (void *)mf;
    void *h = dtype == GFFX_DTYPE_FLOAT64 ? (void *)hd : (void *)hf;
    int64_t point;

    fill_components(p, dtype, three_points, 9);

    /* TP-01 identity. */
    identity_matrix(matrix);
    fill_components(m, dtype, matrix, 16);
    CHECK(run_transform(p, 3, m, 1, offsets_single, dtype, h) == GFFX_STATUS_OK);
    for (point = 0; point < 3; ++point) {
        CHECK(get_component(h, dtype, point * 4 + 0) == three_points[point * 3 + 0]);
        CHECK(get_component(h, dtype, point * 4 + 1) == three_points[point * 3 + 1]);
        CHECK(get_component(h, dtype, point * 4 + 2) == three_points[point * 3 + 2]);
        CHECK(get_component(h, dtype, point * 4 + 3) == 1.0);
    }

    /* TP-02 translation. */
    identity_matrix(matrix);
    matrix[3] = 2.0; matrix[7] = -3.0; matrix[11] = 0.5;
    fill_components(m, dtype, matrix, 16);
    CHECK(run_transform(p, 3, m, 1, offsets_single, dtype, h) == GFFX_STATUS_OK);
    for (point = 0; point < 3; ++point) {
        CHECK(get_component(h, dtype, point * 4 + 0) == three_points[point * 3 + 0] + 2.0);
        CHECK(get_component(h, dtype, point * 4 + 1) == three_points[point * 3 + 1] - 3.0);
        CHECK(get_component(h, dtype, point * 4 + 2) == three_points[point * 3 + 2] + 0.5);
        CHECK(get_component(h, dtype, point * 4 + 3) == 1.0);
    }

    /* TP-03 exact Rz(90deg): (x, y, z) -> (-y, x, z). */
    identity_matrix(matrix);
    matrix[0] = 0.0; matrix[1] = -1.0;
    matrix[4] = 1.0; matrix[5] = 0.0;
    fill_components(m, dtype, matrix, 16);
    CHECK(run_transform(p, 3, m, 1, offsets_single, dtype, h) == GFFX_STATUS_OK);
    for (point = 0; point < 3; ++point) {
        CHECK(get_component(h, dtype, point * 4 + 0) == -three_points[point * 3 + 1]);
        CHECK(get_component(h, dtype, point * 4 + 1) == three_points[point * 3 + 0]);
        CHECK(get_component(h, dtype, point * 4 + 2) == three_points[point * 3 + 2]);
    }
    return 0;
}

static int test_tp04_scale_and_projective(gffx_dtype dtype) {
    double matrix[16];
    double hd[12]; float hf[12];
    double md[16]; float mf[16];
    double pd[9]; float pf[9];
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *m = dtype == GFFX_DTYPE_FLOAT64 ? (void *)md : (void *)mf;
    void *h = dtype == GFFX_DTYPE_FLOAT64 ? (void *)hd : (void *)hf;
    int64_t point;

    fill_components(p, dtype, three_points, 9);
    identity_matrix(matrix);
    matrix[0] = 2.0; matrix[5] = 2.0; matrix[10] = 2.0;
    fill_components(m, dtype, matrix, 16);
    CHECK(run_transform(p, 3, m, 1, offsets_single, dtype, h) == GFFX_STATUS_OK);
    for (point = 0; point < 3; ++point) {
        CHECK(get_component(h, dtype, point * 4 + 0) == three_points[point * 3 + 0] * 2.0);
        CHECK(get_component(h, dtype, point * 4 + 3) == 1.0);
    }

    /* A projective bottom row produces w != 1. */
    identity_matrix(matrix);
    matrix[14] = -1.0; matrix[15] = 0.0;
    fill_components(m, dtype, matrix, 16);
    CHECK(run_transform(p, 3, m, 1, offsets_single, dtype, h) == GFFX_STATUS_OK);
    for (point = 0; point < 3; ++point) {
        CHECK(get_component(h, dtype, point * 4 + 3) == -three_points[point * 3 + 2]);
    }
    return 0;
}

static int test_tp05_tp07_batching(gffx_dtype dtype) {
    static const int32_t offsets_two[3] = {0, 1, 3};
    static const int32_t offsets_empty_middle[4] = {0, 1, 1, 3};
    double matrices[32];
    double single[16];
    double hd[12]; float hf[12];
    double h_ref_d[12]; float h_ref_f[12];
    double md[32]; float mf[32];
    double ms_d[16]; float ms_f[16];
    double pd[9]; float pf[9];
    double three_matrices[48];
    double m3_d[48]; float m3_f[48];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    int64_t index;
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *m = dtype == GFFX_DTYPE_FLOAT64 ? (void *)md : (void *)mf;
    void *ms = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ms_d : (void *)ms_f;
    void *m3 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)m3_d : (void *)m3_f;
    void *h = dtype == GFFX_DTYPE_FLOAT64 ? (void *)hd : (void *)hf;
    void *h_ref = dtype == GFFX_DTYPE_FLOAT64 ? (void *)h_ref_d : (void *)h_ref_f;

    fill_components(p, dtype, three_points, 9);

    /* TP-05: element 0 scales, element 1 translates. */
    identity_matrix(matrices);
    matrices[0] = 3.0;
    identity_matrix(matrices + 16);
    matrices[16 + 3] = 5.0;
    fill_components(m, dtype, matrices, 32);
    CHECK(run_transform(p, 3, m, 2, offsets_two, dtype, h) == GFFX_STATUS_OK);

    identity_matrix(single);
    single[0] = 3.0;
    fill_components(ms, dtype, single, 16);
    {
        static const int32_t offsets_one_point[2] = {0, 1};
        CHECK(run_transform(p, 1, ms, 1, offsets_one_point, dtype, h_ref) == GFFX_STATUS_OK);
        CHECK(memcmp(h, h_ref, 4u * element) == 0);
    }
    identity_matrix(single);
    single[3] = 5.0;
    fill_components(ms, dtype, single, 16);
    {
        static const int32_t offsets_two_points[2] = {0, 2};
        double tail_d[6]; float tail_f[6];
        void *tail = dtype == GFFX_DTYPE_FLOAT64 ? (void *)tail_d : (void *)tail_f;
        for (index = 0; index < 6; ++index) {
            set_component(tail, dtype, index, three_points[3 + index]);
        }
        CHECK(run_transform(tail, 2, ms, 1, offsets_two_points, dtype, h_ref) == GFFX_STATUS_OK);
        CHECK(memcmp((const char *)h + 4u * element, h_ref, 8u * element) == 0);
    }

    /* TP-06: an empty middle batch element leaves the others untouched. */
    identity_matrix(three_matrices);
    three_matrices[0] = 3.0;
    identity_matrix(three_matrices + 16);
    identity_matrix(three_matrices + 32);
    three_matrices[32 + 3] = 5.0;
    fill_components(m3, dtype, three_matrices, 48);
    CHECK(run_transform(p, 3, m3, 3, offsets_empty_middle, dtype, h_ref) == GFFX_STATUS_OK);
    CHECK(memcmp(h, h_ref, 12u * element) == 0);

    /* TP-07: empty inputs. */
    {
        static const int32_t offsets_zero_points[2] = {0, 0};
        static const int32_t offsets_zero_batch[1] = {0};
        CHECK(run_transform(NULL, 0, ms, 1, offsets_zero_points, dtype, NULL)
              == GFFX_STATUS_OK);
        CHECK(run_transform(NULL, 0, NULL, 0, offsets_zero_batch, dtype, NULL)
              == GFFX_STATUS_OK);
    }
    return 0;
}

static int test_tp08_offset_validation(void) {
    double matrix[16];
    double homogeneous[12];
    int32_t offsets[2];
    int64_t point_shape[2] = {3, 3};
    int64_t matrix_shape[3] = {1, 4, 4};
    int64_t offset_shape[1] = {2};
    static const int64_t wrong_offset_shape[1] = {3};
    int64_t out_shape[2] = {3, 4};
    double points[9];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view points_view;
    gffx_tensor_view matrices_view;
    gffx_tensor_view offsets_view;
    gffx_tensor_view out_view;
    gffx_tensor_view broken;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    identity_matrix(matrix);
    memcpy(points, three_points, sizeof(points));
    points_view = make_view(points, GFFX_DTYPE_FLOAT64, 2u, point_shape, pair_strides,
                            GFFX_TENSOR_READ_ONLY);
    matrices_view = make_view(matrix, GFFX_DTYPE_FLOAT64, 3u, matrix_shape, matrix_strides,
                              GFFX_TENSOR_READ_ONLY);
    offsets_view = make_view(offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                             GFFX_TENSOR_READ_ONLY);
    out_view = make_view(homogeneous, GFFX_DTYPE_FLOAT64, 2u, out_shape, quad_strides,
                         GFFX_TENSOR_OUTPUT);

    /* offsets[0] must be zero. */
    offsets[0] = 1; offsets[1] = 3;
    CHECK(gffx_transforms_transform_points(&points_view, &matrices_view, &offsets_view, &context,
                                           &out_view, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Final offset must equal P. */
    offsets[0] = 0; offsets[1] = 2;
    CHECK(gffx_transforms_transform_points(&points_view, &matrices_view, &offsets_view, &context,
                                           &out_view, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Negative entry. */
    offsets[0] = 0; offsets[1] = -1;
    CHECK(gffx_transforms_transform_points(&points_view, &matrices_view, &offsets_view, &context,
                                           &out_view, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Extent must be B + 1. */
    offsets[0] = 0; offsets[1] = 3;
    broken = offsets_view;
    broken.shape = wrong_offset_shape;
    CHECK(gffx_transforms_transform_points(&points_view, &matrices_view, &broken, &context,
                                           &out_view, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Wrong dtype. */
    broken = offsets_view;
    broken.dtype = GFFX_DTYPE_FLOAT32;
    CHECK(gffx_transforms_transform_points(&points_view, &matrices_view, &broken, &context,
                                           &out_view, NULL, &diagnostic)
          == GFFX_STATUS_UNSUPPORTED);
    /* Output aliasing an input. */
    broken = out_view;
    broken.data = points;
    CHECK(gffx_transforms_transform_points(&points_view, &matrices_view, &offsets_view, &context,
                                           &broken, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

/* Decreasing offsets need three batch elements to express. */
static int test_tp08_decreasing_offsets(void) {
    static const int32_t decreasing[4] = {0, 2, 1, 3};
    double matrices[48];
    double homogeneous[12];
    double points[9];
    identity_matrix(matrices);
    identity_matrix(matrices + 16);
    identity_matrix(matrices + 32);
    memcpy(points, three_points, sizeof(points));
    CHECK(run_transform(points, 3, matrices, 3, decreasing, GFFX_DTYPE_FLOAT64, homogeneous)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static double transform_objective(
    const double *points, const double *matrices, const int32_t *offsets,
    int64_t point_count, int64_t batch_count, const double *weights
) {
    double homogeneous[16];
    double total = 0.0;
    int64_t index;
    if (run_transform(points, point_count, matrices, batch_count, offsets, GFFX_DTYPE_FLOAT64,
                      homogeneous) != GFFX_STATUS_OK) {
        return (double)NAN;
    }
    for (index = 0; index < point_count * 4; ++index) total += weights[index] * homogeneous[index];
    return total;
}

static int test_tp09_tp11_gradients(void) {
    static const int32_t offsets_two[3] = {0, 1, 3};
    static const double weights[12] = {
        0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 0.625, -0.75, 0.375, 0.875, -0.125, 0.5
    };
    const double tolerance = 1e-6;
    double matrices[32];
    double points[9];
    double grad_points[9];
    double grad_matrices[32];
    double perturbed_points[9];
    double perturbed_matrices[32];
    int64_t index;

    memcpy(points, three_points, sizeof(points));
    identity_matrix(matrices);
    matrices[0] = 1.5; matrices[1] = 0.25; matrices[3] = -0.5;
    identity_matrix(matrices + 16);
    matrices[16 + 5] = 2.0; matrices[16 + 7] = 1.25; matrices[16 + 14] = -1.0;

    CHECK(run_transform_backward(points, 3, matrices, 2, offsets_two, GFFX_DTYPE_FLOAT64,
                                 weights, grad_points, grad_matrices) == GFFX_STATUS_OK);

    /* TP-09: gradients with respect to points. */
    for (index = 0; index < 9; ++index) {
        double step = 1e-6 * (fabs(points[index]) > 1.0 ? fabs(points[index]) : 1.0);
        double forward_value;
        double backward_value;
        memcpy(perturbed_points, points, sizeof(points));
        perturbed_points[index] = points[index] + step;
        forward_value = transform_objective(perturbed_points, matrices, offsets_two, 3, 2,
                                            weights);
        perturbed_points[index] = points[index] - step;
        backward_value = transform_objective(perturbed_points, matrices, offsets_two, 3, 2,
                                             weights);
        CHECK(relative_close(grad_points[index],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }

    /* TP-10: gradients with respect to matrices, including the translation column. */
    for (index = 0; index < 32; ++index) {
        double step = 1e-6 * (fabs(matrices[index]) > 1.0 ? fabs(matrices[index]) : 1.0);
        double forward_value;
        double backward_value;
        memcpy(perturbed_matrices, matrices, sizeof(matrices));
        perturbed_matrices[index] = matrices[index] + step;
        forward_value = transform_objective(points, perturbed_matrices, offsets_two, 3, 2,
                                            weights);
        perturbed_matrices[index] = matrices[index] - step;
        backward_value = transform_objective(points, perturbed_matrices, offsets_two, 3, 2,
                                             weights);
        CHECK(relative_close(grad_matrices[index],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }

    /* TP-11: one-sided requests, and both-null rejection. */
    {
        double only_points[9];
        double only_matrices[32];
        CHECK(run_transform_backward(points, 3, matrices, 2, offsets_two, GFFX_DTYPE_FLOAT64,
                                     weights, only_points, NULL) == GFFX_STATUS_OK);
        CHECK(memcmp(only_points, grad_points, sizeof(only_points)) == 0);
        CHECK(run_transform_backward(points, 3, matrices, 2, offsets_two, GFFX_DTYPE_FLOAT64,
                                     weights, NULL, only_matrices) == GFFX_STATUS_OK);
        CHECK(memcmp(only_matrices, grad_matrices, sizeof(only_matrices)) == 0);
        CHECK(run_transform_backward(points, 3, matrices, 2, offsets_two, GFFX_DTYPE_FLOAT64,
                                     weights, NULL, NULL) == GFFX_STATUS_INVALID_ARGUMENT);
    }
    return 0;
}

static int test_tp12_determinism(gffx_dtype dtype) {
    double matrix[16];
    double h1_d[12]; float h1_f[12];
    double h2_d[12]; float h2_f[12];
    double md[16]; float mf[16];
    double pd[9]; float pf[9];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *m = dtype == GFFX_DTYPE_FLOAT64 ? (void *)md : (void *)mf;
    void *h1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)h1_d : (void *)h1_f;
    void *h2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)h2_d : (void *)h2_f;

    fill_components(p, dtype, three_points, 9);
    identity_matrix(matrix);
    matrix[0] = 1.25; matrix[6] = -0.5; matrix[11] = 3.0;
    fill_components(m, dtype, matrix, 16);
    CHECK(run_transform(p, 3, m, 1, offsets_single, dtype, h1) == GFFX_STATUS_OK);
    CHECK(run_transform(p, 3, m, 1, offsets_single, dtype, h2) == GFFX_STATUS_OK);
    CHECK(memcmp(h1, h2, 12u * element) == 0);
    return 0;
}

static int test_pd01_pd05_forward(gffx_dtype dtype) {
    static const double cases[24] = {
        1.0, 2.0, 3.0, 1.0,          /* w = 1 passthrough */
        2.0, 4.0, 8.0, 2.0,          /* w = 2 exact divide */
        2.0, 4.0, 8.0, -2.0,         /* negative w */
        1.0, 1.0, 1.0, 0.0,          /* w = 0 */
        1.0, 1.0, 1.0, 0.5,          /* valid, exactly representable */
        1.0, 1.0, 1.0, 0.25          /* valid */
    };
    double hd[24]; float hf[24];
    double nd[18]; float nf[18];
    uint8_t valid[6];
    void *h = dtype == GFFX_DTYPE_FLOAT64 ? (void *)hd : (void *)hf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;

    fill_components(h, dtype, cases, 24);
    CHECK(run_divide(h, 6, dtype, TF_EPS_DEFAULT, n, valid) == GFFX_STATUS_OK);

    CHECK(valid[0] == 1u);
    CHECK(get_component(n, dtype, 0) == 1.0);
    CHECK(get_component(n, dtype, 1) == 2.0);
    CHECK(get_component(n, dtype, 2) == 3.0);

    CHECK(valid[1] == 1u);
    CHECK(get_component(n, dtype, 3) == 1.0);
    CHECK(get_component(n, dtype, 4) == 2.0);
    CHECK(get_component(n, dtype, 5) == 4.0);

    CHECK(valid[2] == 1u);
    CHECK(get_component(n, dtype, 6) == -1.0);
    CHECK(get_component(n, dtype, 7) == -2.0);
    CHECK(get_component(n, dtype, 8) == -4.0);

    CHECK(valid[3] == 0u);
    CHECK(get_component(n, dtype, 9) == 0.0);
    CHECK(get_component(n, dtype, 10) == 0.0);
    CHECK(get_component(n, dtype, 11) == 0.0);

    CHECK(valid[4] == 1u);
    CHECK(get_component(n, dtype, 12) == 2.0);
    return 0;
}

static int test_pd03_pd04_boundary(void) {
    double homogeneous[12];
    double ndc[9];
    uint8_t valid[3];
    int64_t index;

    for (index = 0; index < 3; ++index) {
        homogeneous[index * 4 + 0] = 1.0;
        homogeneous[index * 4 + 1] = 1.0;
        homogeneous[index * 4 + 2] = 1.0;
    }
    homogeneous[3] = TF_EPS_DEFAULT;          /* exactly eps: strict > fails */
    homogeneous[7] = TF_EPS_DEFAULT * 2.0;    /* above eps: valid */
    homogeneous[11] = (double)NAN;            /* NaN: invalid branch */

    CHECK(run_divide(homogeneous, 3, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 0u);
    CHECK(ndc[0] == 0.0 && ndc[1] == 0.0 && ndc[2] == 0.0);
    CHECK(valid[1] == 1u);
    CHECK(valid[2] == 0u);
    CHECK(ndc[6] == 0.0 && ndc[7] == 0.0 && ndc[8] == 0.0);

    /* An infinite component with finite w yields the IEEE result. */
    homogeneous[0] = (double)INFINITY;
    homogeneous[3] = 1.0;
    CHECK(run_divide(homogeneous, 1, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u);
    CHECK(isinf(ndc[0]) && ndc[0] > 0.0);
    return 0;
}

static double divide_objective(const double *homogeneous, int64_t point_count,
                               const double *weights) {
    double ndc[9];
    uint8_t valid[3];
    double total = 0.0;
    int64_t index;
    if (run_divide(homogeneous, point_count, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
        != GFFX_STATUS_OK) {
        return (double)NAN;
    }
    for (index = 0; index < point_count * 3; ++index) total += weights[index] * ndc[index];
    return total;
}

static int test_pd06_gradients(void) {
    static const double homogeneous[8] = {
        0.5, -1.5, 2.25, 3.0,
        1.0, 1.0, 1.0, 0.0        /* invalid: exactly zero gradient expected */
    };
    static const double weights[6] = {0.75, -0.5, 0.25, 1.0, 1.0, 1.0};
    const double tolerance = 1e-6;
    double gradient[8];
    double perturbed[8];
    int64_t index;

    CHECK(run_divide_backward(homogeneous, 2, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, weights,
                              gradient) == GFFX_STATUS_OK);
    for (index = 0; index < 4; ++index) {
        double step = 1e-6 * (fabs(homogeneous[index]) > 1.0 ? fabs(homogeneous[index]) : 1.0);
        double forward_value;
        double backward_value;
        memcpy(perturbed, homogeneous, sizeof(perturbed));
        perturbed[index] = homogeneous[index] + step;
        forward_value = divide_objective(perturbed, 2, weights);
        perturbed[index] = homogeneous[index] - step;
        backward_value = divide_objective(perturbed, 2, weights);
        CHECK(relative_close(gradient[index],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }
    for (index = 4; index < 8; ++index) CHECK(gradient[index] == 0.0);
    return 0;
}

static int test_pd07_pd08_empty_and_determinism(gffx_dtype dtype) {
    double hd[12]; float hf[12];
    double n1_d[9]; float n1_f[9];
    double n2_d[9]; float n2_f[9];
    static const double cases[12] = {
        1.0, 2.0, 3.0, 4.0, 0.5, -0.25, 0.75, 2.0, -1.0, 1.5, 2.5, -0.5
    };
    uint8_t valid1[3];
    uint8_t valid2[3];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    void *h = dtype == GFFX_DTYPE_FLOAT64 ? (void *)hd : (void *)hf;
    void *n1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n1_d : (void *)n1_f;
    void *n2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n2_d : (void *)n2_f;

    CHECK(run_divide(NULL, 0, dtype, TF_EPS_DEFAULT, NULL, NULL) == GFFX_STATUS_OK);
    fill_components(h, dtype, cases, 12);
    CHECK(run_divide(h, 3, dtype, TF_EPS_DEFAULT, n1, valid1) == GFFX_STATUS_OK);
    CHECK(run_divide(h, 3, dtype, TF_EPS_DEFAULT, n2, valid2) == GFFX_STATUS_OK);
    CHECK(memcmp(n1, n2, 9u * element) == 0);
    CHECK(memcmp(valid1, valid2, sizeof(valid1)) == 0);

    /* Negative and non-finite eps are rejected. */
    CHECK(run_divide(h, 3, dtype, -1.0, n1, valid1) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_divide(h, 3, dtype, (double)NAN, n1, valid1) == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

/* CAM-01..CAM-07 and CAM-09 compose both operations against the camera contract. */
static int test_camera_integration(void) {
    const double fx = 2377.489709675;
    const double cx = 400.0;
    const double width = 800.0;
    const double near_plane = 0.01;
    const double far_plane = 3.0;
    const double tolerance = 1e-12;
    double matrix[16];
    double points[9];
    double homogeneous[12];
    double ndc[9];
    uint8_t valid[3];
    static const int32_t offsets[2] = {0, 3};

    projection_matrix(fx, fx, cx, cx, width, width, near_plane, far_plane, matrix);

    /* CAM-09: the tabulated entries. */
    CHECK(relative_close(matrix[0], 5.9437242741875, tolerance));
    CHECK(relative_close(matrix[5], 5.9437242741875, tolerance));
    CHECK(matrix[2] == 0.0);
    CHECK(matrix[6] == 0.0);
    CHECK(relative_close(matrix[10], -1.0066889632107024, tolerance));
    CHECK(relative_close(matrix[11], -0.020066889632107024, tolerance));
    CHECK(matrix[14] == -1.0);

    /* CAM-01: on-axis points at the near and far planes map to NDC z = -1 and +1.
     * CAM-03: a point with positive view-space y lands in the upper image half. */
    points[0] = 0.0; points[1] = 0.0; points[2] = -near_plane;
    points[3] = 0.0; points[4] = 0.0; points[5] = -far_plane;
    points[6] = 0.0; points[7] = 0.5; points[8] = -1.0;
    CHECK(run_transform(points, 3, matrix, 1, offsets, GFFX_DTYPE_FLOAT64, homogeneous)
          == GFFX_STATUS_OK);
    CHECK(run_divide(homogeneous, 3, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u && valid[1] == 1u && valid[2] == 1u);
    CHECK(relative_close(ndc[0], 0.0, 1e-9) && relative_close(ndc[1], 0.0, 1e-9));
    CHECK(relative_close(ndc[2], -1.0, 1e-9));
    CHECK(relative_close(ndc[5], 1.0, 1e-9));
    CHECK(ndc[7] > 0.0);

    /* CAM-02: one pixel right of centre at unit depth recovers pixel column cx + 1. */
    points[0] = 1.0 / fx; points[1] = 0.0; points[2] = -1.0;
    CHECK(run_transform(points, 1, matrix, 1, (const int32_t[2]){0, 1}, GFFX_DTYPE_FLOAT64,
                        homogeneous) == GFFX_STATUS_OK);
    CHECK(run_divide(homogeneous, 1, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
          == GFFX_STATUS_OK);
    CHECK(relative_close((ndc[0] + 1.0) * 0.5 * width, cx + 1.0, 1e-9));

    /* CAM-04: an off-centre principal point moves the optical axis away from the origin. */
    {
        double off_axis[16];
        projection_matrix(fx, fx, 0.25 * width, 0.75 * width, width, width, near_plane,
                          far_plane, off_axis);
        points[0] = 0.0; points[1] = 0.0; points[2] = -1.0;
        CHECK(run_transform(points, 1, off_axis, 1, (const int32_t[2]){0, 1},
                            GFFX_DTYPE_FLOAT64, homogeneous) == GFFX_STATUS_OK);
        CHECK(run_divide(homogeneous, 1, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
              == GFFX_STATUS_OK);
        CHECK(relative_close(ndc[0], -0.5, 1e-9));
        CHECK(relative_close(ndc[1], -0.5, 1e-9));
    }

    /* CAM-05: a point at the camera plane divides invalid. */
    points[0] = 0.1; points[1] = 0.1; points[2] = 0.0;
    CHECK(run_transform(points, 1, matrix, 1, (const int32_t[2]){0, 1}, GFFX_DTYPE_FLOAT64,
                        homogeneous) == GFFX_STATUS_OK);
    CHECK(run_divide(homogeneous, 1, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 0u);
    CHECK(ndc[0] == 0.0 && ndc[1] == 0.0 && ndc[2] == 0.0);

    /* CAM-06: round trip against the direct pinhole formula. */
    {
        static const double samples[9] = {
            0.3, 0.2, -1.5,   -0.4, 0.6, -2.0,   0.05, -0.35, -0.75
        };
        int64_t index;
        memcpy(points, samples, sizeof(points));
        CHECK(run_transform(points, 3, matrix, 1, offsets, GFFX_DTYPE_FLOAT64, homogeneous)
              == GFFX_STATUS_OK);
        CHECK(run_divide(homogeneous, 3, GFFX_DTYPE_FLOAT64, TF_EPS_DEFAULT, ndc, valid)
              == GFFX_STATUS_OK);
        for (index = 0; index < 3; ++index) {
            double depth = -samples[index * 3 + 2];
            double expected_col = fx * samples[index * 3 + 0] / depth + cx;
            double expected_row = cx - fx * samples[index * 3 + 1] / depth;
            double actual_col = (ndc[index * 3 + 0] + 1.0) * 0.5 * width;
            double actual_row = (1.0 - ndc[index * 3 + 1]) * 0.5 * width;
            CHECK(valid[index] == 1u);
            CHECK(relative_close(actual_col, expected_col, 1e-9));
            CHECK(relative_close(actual_row, expected_row, 1e-9));
        }
    }

    /* CAM-07: two cameras batched equal two single-camera runs. */
    {
        double batched[32];
        double narrow[16];
        double single_out[8];
        double batched_out[8];
        static const int32_t two_offsets[3] = {0, 1, 2};
        double pair[6] = {0.2, 0.1, -1.0, 0.2, 0.1, -1.0};
        projection_matrix(fx * 0.5, fx * 0.5, cx, cx, width, width, near_plane, far_plane,
                          narrow);
        memcpy(batched, matrix, sizeof(matrix));
        memcpy(batched + 16, narrow, sizeof(narrow));
        CHECK(run_transform(pair, 2, batched, 2, two_offsets, GFFX_DTYPE_FLOAT64, batched_out)
              == GFFX_STATUS_OK);
        CHECK(run_transform(pair, 1, matrix, 1, (const int32_t[2]){0, 1}, GFFX_DTYPE_FLOAT64,
                            single_out) == GFFX_STATUS_OK);
        CHECK(memcmp(batched_out, single_out, 4 * sizeof(double)) == 0);
        CHECK(run_transform(pair + 3, 1, narrow, 1, (const int32_t[2]){0, 1},
                            GFFX_DTYPE_FLOAT64, single_out) == GFFX_STATUS_OK);
        CHECK(memcmp(batched_out + 4, single_out, 4 * sizeof(double)) == 0);
    }
    return 0;
}

static int test_workspace_queries(void) {
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;

    CHECK(gffx_transforms_transform_points_workspace(3, 1, GFFX_DTYPE_FLOAT64, &context,
                                                     &required_bytes, &required_alignment,
                                                     &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    CHECK(gffx_transforms_transform_points_workspace(3, 1, GFFX_DTYPE_BOOL, &context,
                                                     &required_bytes, &required_alignment,
                                                     &diagnostic) == GFFX_STATUS_UNSUPPORTED);
    CHECK(gffx_transforms_perspective_divide_workspace(3, GFFX_DTYPE_FLOAT32, &context,
                                                       &required_bytes, &required_alignment,
                                                       &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    CHECK(gffx_transforms_perspective_divide_workspace(-1, GFFX_DTYPE_FLOAT32, &context,
                                                       &required_bytes, &required_alignment,
                                                       &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    size_t index;

    for (index = 0u; index < 2u; ++index) {
        gffx_dtype dtype = dtypes[index];
        result = test_tp01_tp03_basic(dtype); if (result != 0) return result;
        result = test_tp04_scale_and_projective(dtype); if (result != 0) return result;
        result = test_tp05_tp07_batching(dtype); if (result != 0) return result;
        result = test_tp12_determinism(dtype); if (result != 0) return result;
        result = test_pd01_pd05_forward(dtype); if (result != 0) return result;
        result = test_pd07_pd08_empty_and_determinism(dtype); if (result != 0) return result;
    }
    result = test_tp08_offset_validation(); if (result != 0) return result;
    result = test_tp08_decreasing_offsets(); if (result != 0) return result;
    result = test_tp09_tp11_gradients(); if (result != 0) return result;
    result = test_pd03_pd04_boundary(); if (result != 0) return result;
    result = test_pd06_gradients(); if (result != 0) return result;
    result = test_camera_integration(); if (result != 0) return result;
    result = test_workspace_queries(); if (result != 0) return result;
    return 0;
}
