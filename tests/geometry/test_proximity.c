/*
 * Phase 2 acceptance fixtures KN-01..KN-09 and CP-01..CP-10 for points.knn and
 * points.closest_point_on_mesh. Fixture numbers match the project acceptance record.
 */

#include <gffx/execution.h>
#include <gffx/points.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define PX_EPS_DEFAULT 9.5367431640625e-7 /* 2^-20 */

static const int64_t pair_strides[2] = {3, 1};
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

/* ----------------------------------------------------------------- points.knn helpers */

static gffx_status run_knn(
    const void *query, int64_t query_count,
    const void *reference, int64_t reference_count,
    const int32_t *query_offsets, const int32_t *reference_offsets, int64_t batch_count,
    int64_t neighbor_count, gffx_dtype dtype,
    void *distance_squared, int32_t *reference_index, uint8_t *valid
) {
    int64_t query_shape[2];
    int64_t reference_shape[2];
    int64_t offset_shape[1];
    int64_t out_shape[2];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view query_view;
    gffx_tensor_view reference_view;
    gffx_tensor_view query_offsets_view;
    gffx_tensor_view reference_offsets_view;
    gffx_tensor_view distance_view;
    gffx_tensor_view index_view;
    gffx_tensor_view valid_view;
    int64_t out_strides[2];

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    query_shape[0] = query_count; query_shape[1] = 3;
    reference_shape[0] = reference_count; reference_shape[1] = 3;
    offset_shape[0] = batch_count + 1;
    out_shape[0] = query_count; out_shape[1] = neighbor_count;
    out_strides[0] = neighbor_count; out_strides[1] = 1;

    query_view = make_view((void *)query, dtype, 2u, query_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    reference_view = make_view((void *)reference, dtype, 2u, reference_shape, pair_strides,
                               GFFX_TENSOR_READ_ONLY);
    query_offsets_view = make_view((void *)query_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                   scalar_strides, GFFX_TENSOR_READ_ONLY);
    reference_offsets_view = make_view((void *)reference_offsets, GFFX_DTYPE_INT32, 1u,
                                       offset_shape, scalar_strides, GFFX_TENSOR_READ_ONLY);
    distance_view = make_view(distance_squared, dtype, 2u, out_shape, out_strides,
                              GFFX_TENSOR_OUTPUT);
    index_view = make_view(reference_index, GFFX_DTYPE_INT32, 2u, out_shape, out_strides,
                           GFFX_TENSOR_OUTPUT);
    valid_view = make_view(valid, GFFX_DTYPE_BOOL, 2u, out_shape, out_strides,
                           GFFX_TENSOR_OUTPUT);
    return gffx_points_knn(&query_view, &reference_view, &query_offsets_view,
                           &reference_offsets_view, neighbor_count, &context, &distance_view,
                           &index_view, &valid_view, NULL, &diagnostic);
}

static gffx_status run_knn_backward(
    const void *query, int64_t query_count,
    const void *reference, int64_t reference_count,
    const int32_t *reference_index, const uint8_t *valid,
    int64_t neighbor_count, gffx_dtype dtype,
    const void *grad_distance_squared, void *grad_query, void *grad_reference
) {
    int64_t query_shape[2];
    int64_t reference_shape[2];
    int64_t out_shape[2];
    int64_t out_strides[2];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view query_view;
    gffx_tensor_view reference_view;
    gffx_tensor_view index_view;
    gffx_tensor_view valid_view;
    gffx_tensor_view cotangent_view;
    gffx_tensor_view grad_query_view;
    gffx_tensor_view grad_reference_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    query_shape[0] = query_count; query_shape[1] = 3;
    reference_shape[0] = reference_count; reference_shape[1] = 3;
    out_shape[0] = query_count; out_shape[1] = neighbor_count;
    out_strides[0] = neighbor_count; out_strides[1] = 1;

    query_view = make_view((void *)query, dtype, 2u, query_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    reference_view = make_view((void *)reference, dtype, 2u, reference_shape, pair_strides,
                               GFFX_TENSOR_READ_ONLY);
    index_view = make_view((void *)reference_index, GFFX_DTYPE_INT32, 2u, out_shape,
                           out_strides, GFFX_TENSOR_READ_ONLY);
    valid_view = make_view((void *)valid, GFFX_DTYPE_BOOL, 2u, out_shape, out_strides,
                           GFFX_TENSOR_READ_ONLY);
    cotangent_view = make_view((void *)grad_distance_squared, dtype, 2u, out_shape, out_strides,
                               GFFX_TENSOR_READ_ONLY);
    grad_query_view = make_view(grad_query, dtype, 2u, query_shape, pair_strides,
                                GFFX_TENSOR_OUTPUT);
    grad_reference_view = make_view(grad_reference, dtype, 2u, reference_shape, pair_strides,
                                    GFFX_TENSOR_OUTPUT);
    return gffx_points_knn_backward(&query_view, &reference_view, &index_view, &valid_view,
                                    &cotangent_view, &context,
                                    grad_query != NULL ? &grad_query_view : NULL,
                                    grad_reference != NULL ? &grad_reference_view : NULL,
                                    NULL, &diagnostic);
}

/* ------------------------------------------------------- closest_point_on_mesh helpers */

static gffx_status run_closest(
    const void *points, int64_t point_count,
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    const int32_t *point_offsets, const int32_t *vertex_offsets, const int32_t *face_offsets,
    int64_t batch_count, double eps, gffx_dtype dtype,
    void *distance_squared, int32_t *face_index, void *barycentric, void *closest, uint8_t *valid
) {
    int64_t point_shape[2];
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t offset_shape[1];
    int64_t scalar_shape[1];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view points_view;
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view point_offsets_view;
    gffx_tensor_view vertex_offsets_view;
    gffx_tensor_view face_offsets_view;
    gffx_tensor_view distance_view;
    gffx_tensor_view face_index_view;
    gffx_tensor_view barycentric_view;
    gffx_tensor_view closest_view;
    gffx_tensor_view valid_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    point_shape[0] = point_count; point_shape[1] = 3;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    offset_shape[0] = batch_count + 1;
    scalar_shape[0] = point_count;

    points_view = make_view((void *)points, dtype, 2u, point_shape, pair_strides,
                            GFFX_TENSOR_READ_ONLY);
    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    point_offsets_view = make_view((void *)point_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                   scalar_strides, GFFX_TENSOR_READ_ONLY);
    vertex_offsets_view = make_view((void *)vertex_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                    scalar_strides, GFFX_TENSOR_READ_ONLY);
    face_offsets_view = make_view((void *)face_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);
    distance_view = make_view(distance_squared, dtype, 1u, scalar_shape, scalar_strides,
                              GFFX_TENSOR_OUTPUT);
    face_index_view = make_view(face_index, GFFX_DTYPE_INT32, 1u, scalar_shape, scalar_strides,
                                GFFX_TENSOR_OUTPUT);
    barycentric_view = make_view(barycentric, dtype, 2u, point_shape, pair_strides,
                                 GFFX_TENSOR_OUTPUT);
    closest_view = make_view(closest, dtype, 2u, point_shape, pair_strides, GFFX_TENSOR_OUTPUT);
    valid_view = make_view(valid, GFFX_DTYPE_BOOL, 1u, scalar_shape, scalar_strides,
                           GFFX_TENSOR_OUTPUT);
    return gffx_points_closest_point_on_mesh(
        &points_view, &vertices_view, &faces_view, &point_offsets_view, &vertex_offsets_view,
        &face_offsets_view, eps, &context, &distance_view, &face_index_view, &barycentric_view,
        &closest_view, &valid_view, NULL, &diagnostic);
}

static gffx_status run_closest_backward(
    const void *points, int64_t point_count,
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    const int32_t *face_index, const void *barycentric, const void *closest,
    const uint8_t *valid, const void *grad_distance_squared, gffx_dtype dtype,
    void *grad_points, void *grad_vertices
) {
    int64_t point_shape[2];
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t scalar_shape[1];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view points_view;
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view face_index_view;
    gffx_tensor_view barycentric_view;
    gffx_tensor_view closest_view;
    gffx_tensor_view valid_view;
    gffx_tensor_view cotangent_view;
    gffx_tensor_view grad_points_view;
    gffx_tensor_view grad_vertices_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    point_shape[0] = point_count; point_shape[1] = 3;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    scalar_shape[0] = point_count;

    points_view = make_view((void *)points, dtype, 2u, point_shape, pair_strides,
                            GFFX_TENSOR_READ_ONLY);
    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    face_index_view = make_view((void *)face_index, GFFX_DTYPE_INT32, 1u, scalar_shape,
                                scalar_strides, GFFX_TENSOR_READ_ONLY);
    barycentric_view = make_view((void *)barycentric, dtype, 2u, point_shape, pair_strides,
                                 GFFX_TENSOR_READ_ONLY);
    closest_view = make_view((void *)closest, dtype, 2u, point_shape, pair_strides,
                             GFFX_TENSOR_READ_ONLY);
    valid_view = make_view((void *)valid, GFFX_DTYPE_BOOL, 1u, scalar_shape, scalar_strides,
                           GFFX_TENSOR_READ_ONLY);
    cotangent_view = make_view((void *)grad_distance_squared, dtype, 1u, scalar_shape,
                               scalar_strides, GFFX_TENSOR_READ_ONLY);
    grad_points_view = make_view(grad_points, dtype, 2u, point_shape, pair_strides,
                                 GFFX_TENSOR_OUTPUT);
    grad_vertices_view = make_view(grad_vertices, dtype, 2u, vertex_shape, pair_strides,
                                   GFFX_TENSOR_OUTPUT);
    return gffx_points_closest_point_on_mesh_backward(
        &points_view, &vertices_view, &faces_view, &face_index_view, &barycentric_view,
        &closest_view, &valid_view, &cotangent_view, &context,
        grad_points != NULL ? &grad_points_view : NULL,
        grad_vertices != NULL ? &grad_vertices_view : NULL, NULL, &diagnostic);
}

/* --------------------------------------------------------------------- knn fixtures */

static int test_kn01_kn02_basic_and_ties(gffx_dtype dtype) {
    /* References at squared distances 1, 4, 9 along +x from the query at the origin. */
    static const double query[3] = {0.0, 0.0, 0.0};
    static const double reference[9] = {2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    static const int32_t query_offsets[2] = {0, 1};
    static const int32_t reference_offsets[2] = {0, 3};
    /* Two references exactly equidistant: indices 0 and 1 both at squared distance 1. */
    static const double tie_reference[6] = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
    static const int32_t tie_offsets[2] = {0, 2};
    double dd[2]; float df[2];
    double qd[3]; float qf[3];
    double rd[9]; float rf[9];
    int32_t index[2];
    uint8_t valid[2];
    void *q = dtype == GFFX_DTYPE_FLOAT64 ? (void *)qd : (void *)qf;
    void *r = dtype == GFFX_DTYPE_FLOAT64 ? (void *)rd : (void *)rf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;

    fill_components(q, dtype, query, 3);
    fill_components(r, dtype, reference, 9);
    CHECK(run_knn(q, 1, r, 3, query_offsets, reference_offsets, 1, 2, dtype, d, index, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u && valid[1] == 1u);
    CHECK(get_component(d, dtype, 0) == 1.0);
    CHECK(index[0] == 1);
    CHECK(get_component(d, dtype, 1) == 4.0);
    CHECK(index[1] == 0);

    fill_components(r, dtype, tie_reference, 6);
    CHECK(run_knn(q, 1, r, 2, query_offsets, tie_offsets, 1, 2, dtype, d, index, valid)
          == GFFX_STATUS_OK);
    CHECK(get_component(d, dtype, 0) == 1.0 && get_component(d, dtype, 1) == 1.0);
    CHECK(index[0] == 0 && index[1] == 1);
    return 0;
}

static int test_kn03_kn05_padding(gffx_dtype dtype) {
    static const double query[3] = {0.0, 0.0, 0.0};
    static const double reference[3] = {1.0, 0.0, 0.0};
    static const int32_t query_offsets[2] = {0, 1};
    static const int32_t reference_offsets[2] = {0, 1};
    static const int32_t empty_reference_offsets[2] = {0, 0};
    double dd[3]; float df[3];
    double qd[3]; float qf[3];
    double rd[3]; float rf[3];
    int32_t index[3];
    uint8_t valid[3];
    void *q = dtype == GFFX_DTYPE_FLOAT64 ? (void *)qd : (void *)qf;
    void *r = dtype == GFFX_DTYPE_FLOAT64 ? (void *)rd : (void *)rf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;

    fill_components(q, dtype, query, 3);
    fill_components(r, dtype, reference, 3);
    CHECK(run_knn(q, 1, r, 1, query_offsets, reference_offsets, 1, 3, dtype, d, index, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u && valid[1] == 0u && valid[2] == 0u);
    CHECK(index[0] == 0 && index[1] == -1 && index[2] == -1);
    CHECK(isinf(get_component(d, dtype, 1)) && get_component(d, dtype, 1) > 0.0);
    CHECK(isinf(get_component(d, dtype, 2)) && get_component(d, dtype, 2) > 0.0);

    /* An element with zero references pads every entry. */
    CHECK(run_knn(q, 1, NULL, 0, query_offsets, empty_reference_offsets, 1, 3, dtype, d, index,
                  valid) == GFFX_STATUS_OK);
    CHECK(valid[0] == 0u && index[0] == -1);
    return 0;
}

static int test_kn04_batching(gffx_dtype dtype) {
    /* Element 0: query at origin, references at x = 1, 5. Element 1: query at x = 10,
     * references at x = 9, 20. Global reference indices must appear in element 1. */
    static const double query[6] = {0.0, 0.0, 0.0, 10.0, 0.0, 0.0};
    static const double reference[12] = {
        1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 9.0, 0.0, 0.0, 20.0, 0.0, 0.0
    };
    static const int32_t query_offsets[3] = {0, 1, 2};
    static const int32_t reference_offsets[3] = {0, 2, 4};
    double dd[2]; float df[2];
    double qd[6]; float qf[6];
    double rd[12]; float rf[12];
    int32_t index[2];
    uint8_t valid[2];
    void *q = dtype == GFFX_DTYPE_FLOAT64 ? (void *)qd : (void *)qf;
    void *r = dtype == GFFX_DTYPE_FLOAT64 ? (void *)rd : (void *)rf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;

    fill_components(q, dtype, query, 6);
    fill_components(r, dtype, reference, 12);
    CHECK(run_knn(q, 2, r, 4, query_offsets, reference_offsets, 2, 1, dtype, d, index, valid)
          == GFFX_STATUS_OK);
    CHECK(index[0] == 0);
    CHECK(get_component(d, dtype, 0) == 1.0);
    /* Element 1 selects global index 2, not the element-local 0. */
    CHECK(index[1] == 2);
    CHECK(get_component(d, dtype, 1) == 1.0);
    return 0;
}

static double knn_objective(const double *query, const double *reference, const double *weights) {
    static const int32_t query_offsets[2] = {0, 2};
    static const int32_t reference_offsets[2] = {0, 3};
    double distance[4];
    int32_t index[4];
    uint8_t valid[4];
    double total = 0.0;
    int64_t entry;
    if (run_knn(query, 2, reference, 3, query_offsets, reference_offsets, 1, 2,
                GFFX_DTYPE_FLOAT64, distance, index, valid) != GFFX_STATUS_OK) {
        return (double)NAN;
    }
    for (entry = 0; entry < 4; ++entry) total += weights[entry] * distance[entry];
    return total;
}

static int test_kn06_gradients(void) {
    static const double query[6] = {0.1, 0.2, 0.3, 1.4, -0.5, 0.6};
    static const double reference[9] = {
        0.7, 0.1, -0.2, 2.0, 0.4, 0.9, -1.1, 0.3, 0.5
    };
    static const double weights[4] = {0.5, -0.25, 0.75, 1.25};
    static const int32_t query_offsets[2] = {0, 2};
    static const int32_t reference_offsets[2] = {0, 3};
    const double tolerance = 1e-6;
    double distance[4];
    int32_t index[4];
    uint8_t valid[4];
    double grad_query[6];
    double grad_reference[9];
    double perturbed_query[6];
    double perturbed_reference[9];
    int64_t coordinate;

    CHECK(run_knn(query, 2, reference, 3, query_offsets, reference_offsets, 1, 2,
                  GFFX_DTYPE_FLOAT64, distance, index, valid) == GFFX_STATUS_OK);
    CHECK(run_knn_backward(query, 2, reference, 3, index, valid, 2, GFFX_DTYPE_FLOAT64,
                           weights, grad_query, grad_reference) == GFFX_STATUS_OK);

    for (coordinate = 0; coordinate < 6; ++coordinate) {
        double step = 1e-6;
        double forward_value;
        double backward_value;
        memcpy(perturbed_query, query, sizeof(perturbed_query));
        perturbed_query[coordinate] = query[coordinate] + step;
        forward_value = knn_objective(perturbed_query, reference, weights);
        perturbed_query[coordinate] = query[coordinate] - step;
        backward_value = knn_objective(perturbed_query, reference, weights);
        CHECK(relative_close(grad_query[coordinate],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }
    for (coordinate = 0; coordinate < 9; ++coordinate) {
        double step = 1e-6;
        double forward_value;
        double backward_value;
        memcpy(perturbed_reference, reference, sizeof(perturbed_reference));
        perturbed_reference[coordinate] = reference[coordinate] + step;
        forward_value = knn_objective(query, perturbed_reference, weights);
        perturbed_reference[coordinate] = reference[coordinate] - step;
        backward_value = knn_objective(query, perturbed_reference, weights);
        CHECK(relative_close(grad_reference[coordinate],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }

    /* Padding entries contribute exactly zero. */
    {
        static const int32_t one_reference_offsets[2] = {0, 1};
        static const double pad_weights[4] = {1.0, 1.0, 1.0, 1.0};
        double pad_distance[4];
        int32_t pad_index[4];
        uint8_t pad_valid[4];
        double pad_grad_query[6];
        double pad_grad_reference[3];
        CHECK(run_knn(query, 2, reference, 1, query_offsets, one_reference_offsets, 1, 2,
                      GFFX_DTYPE_FLOAT64, pad_distance, pad_index, pad_valid)
              == GFFX_STATUS_OK);
        CHECK(run_knn_backward(query, 2, reference, 1, pad_index, pad_valid, 2,
                               GFFX_DTYPE_FLOAT64, pad_weights, pad_grad_query,
                               pad_grad_reference) == GFFX_STATUS_OK);
        /* Only the single valid neighbour per query contributes. */
        CHECK(pad_valid[1] == 0u && pad_valid[3] == 0u);
    }
    return 0;
}

static int test_kn07_kn09_determinism_and_workspace(gffx_dtype dtype) {
    static const double query[3] = {0.25, 0.5, -0.75};
    static const double reference[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    static const int32_t query_offsets[2] = {0, 1};
    static const int32_t reference_offsets[2] = {0, 3};
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    double d1_d[2]; float d1_f[2];
    double d2_d[2]; float d2_f[2];
    double qd[3]; float qf[3];
    double rd[9]; float rf[9];
    int32_t index1[2];
    int32_t index2[2];
    uint8_t valid1[2];
    uint8_t valid2[2];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    void *q = dtype == GFFX_DTYPE_FLOAT64 ? (void *)qd : (void *)qf;
    void *r = dtype == GFFX_DTYPE_FLOAT64 ? (void *)rd : (void *)rf;
    void *d1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)d1_d : (void *)d1_f;
    void *d2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)d2_d : (void *)d2_f;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    fill_components(q, dtype, query, 3);
    fill_components(r, dtype, reference, 9);
    CHECK(run_knn(q, 1, r, 3, query_offsets, reference_offsets, 1, 2, dtype, d1, index1, valid1)
          == GFFX_STATUS_OK);
    CHECK(run_knn(q, 1, r, 3, query_offsets, reference_offsets, 1, 2, dtype, d2, index2, valid2)
          == GFFX_STATUS_OK);
    CHECK(memcmp(d1, d2, 2u * element) == 0);
    CHECK(memcmp(index1, index2, sizeof(index1)) == 0);
    CHECK(memcmp(valid1, valid2, sizeof(valid1)) == 0);

    CHECK(gffx_points_knn_workspace(1, 3, 2, dtype, &context, &required_bytes,
                                    &required_alignment, &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    return 0;
}

static int test_kn08_validation(void) {
    static const double query[3] = {0.0, 0.0, 0.0};
    static const double reference[3] = {1.0, 0.0, 0.0};
    int32_t query_offsets[2] = {0, 1};
    static const int32_t reference_offsets[2] = {0, 1};
    double distance[2];
    int32_t index[2];
    uint8_t valid[2];

    /* K must be positive. */
    CHECK(run_knn(query, 1, reference, 1, query_offsets, reference_offsets, 1, 0,
                  GFFX_DTYPE_FLOAT64, distance, index, valid)
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_knn(query, 1, reference, 1, query_offsets, reference_offsets, 1, -1,
                  GFFX_DTYPE_FLOAT64, distance, index, valid)
          == GFFX_STATUS_INVALID_ARGUMENT);

    /* Offset rules. */
    query_offsets[0] = 1;
    CHECK(run_knn(query, 1, reference, 1, query_offsets, reference_offsets, 1, 1,
                  GFFX_DTYPE_FLOAT64, distance, index, valid)
          == GFFX_STATUS_INVALID_ARGUMENT);
    query_offsets[0] = 0; query_offsets[1] = 0;
    CHECK(run_knn(query, 1, reference, 1, query_offsets, reference_offsets, 1, 1,
                  GFFX_DTYPE_FLOAT64, distance, index, valid)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

/* ------------------------------------------------------- closest_point_on_mesh fixtures */

/* Unit right triangle in the z = 0 plane. */
static const double cp_vertices[9] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
};
static const int32_t cp_faces[3] = {0, 1, 2};
static const int32_t cp_point_offsets[2] = {0, 1};
static const int32_t cp_vertex_offsets[2] = {0, 3};
static const int32_t cp_face_offsets[2] = {0, 1};

static int test_cp01_cp03_regions(gffx_dtype dtype) {
    /* Interior: directly above the centroid-ish interior point (0.25, 0.25, 2). */
    static const double interior[3] = {0.25, 0.25, 2.0};
    /* Vertex region: beyond vertex 0. */
    static const double beyond_vertex[3] = {-1.0, -1.0, 0.0};
    /* Edge region: beyond the edge from v1 to v2, outside both vertex regions. */
    static const double beyond_edge[3] = {1.0, 1.0, 0.0};
    double dd[1]; float df[1];
    double bd[3]; float bf[3];
    double cd[3]; float cf[3];
    double vd[9]; float vf[9];
    double pd[3]; float pf[3];
    int32_t face_index[1];
    uint8_t valid[1];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *c = dtype == GFFX_DTYPE_FLOAT64 ? (void *)cd : (void *)cf;
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-14 : 1e-6;

    fill_components(v, dtype, cp_vertices, 9);

    /* CP-01 interior: closest is the in-plane projection, distance squared exactly 4. */
    fill_components(p, dtype, interior, 3);
    CHECK(run_closest(p, 1, v, 3, cp_faces, 1, cp_point_offsets, cp_vertex_offsets,
                      cp_face_offsets, 1, PX_EPS_DEFAULT, dtype, d, face_index, b, c, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u && face_index[0] == 0);
    CHECK(get_component(d, dtype, 0) == 4.0);
    CHECK(relative_close(get_component(c, dtype, 0), 0.25, tolerance));
    CHECK(relative_close(get_component(c, dtype, 1), 0.25, tolerance));
    CHECK(get_component(c, dtype, 2) == 0.0);
    CHECK(relative_close(get_component(b, dtype, 0) + get_component(b, dtype, 1) +
                         get_component(b, dtype, 2), 1.0, tolerance));

    /* CP-02 vertex region: closest is vertex 0 with barycentric (1,0,0). */
    fill_components(p, dtype, beyond_vertex, 3);
    CHECK(run_closest(p, 1, v, 3, cp_faces, 1, cp_point_offsets, cp_vertex_offsets,
                      cp_face_offsets, 1, PX_EPS_DEFAULT, dtype, d, face_index, b, c, valid)
          == GFFX_STATUS_OK);
    CHECK(get_component(c, dtype, 0) == 0.0 && get_component(c, dtype, 1) == 0.0);
    CHECK(get_component(b, dtype, 0) == 1.0);
    CHECK(get_component(b, dtype, 1) == 0.0 && get_component(b, dtype, 2) == 0.0);
    CHECK(get_component(d, dtype, 0) == 2.0);

    /* CP-03 edge region: closest lies on the v1-v2 edge, so the first barycentric is zero. */
    fill_components(p, dtype, beyond_edge, 3);
    CHECK(run_closest(p, 1, v, 3, cp_faces, 1, cp_point_offsets, cp_vertex_offsets,
                      cp_face_offsets, 1, PX_EPS_DEFAULT, dtype, d, face_index, b, c, valid)
          == GFFX_STATUS_OK);
    CHECK(get_component(b, dtype, 0) == 0.0);
    CHECK(relative_close(get_component(c, dtype, 0), 0.5, tolerance));
    CHECK(relative_close(get_component(c, dtype, 1), 0.5, tolerance));
    CHECK(relative_close(get_component(d, dtype, 0), 0.5, tolerance));
    return 0;
}

static int test_cp04_tie_breaking(gffx_dtype dtype) {
    /* Two coincident triangles: the lower face index must win. */
    static const int32_t faces[6] = {0, 1, 2, 0, 1, 2};
    static const int32_t face_offsets[2] = {0, 2};
    static const double above[3] = {0.25, 0.25, 1.0};
    double dd[1]; float df[1];
    double bd[3]; float bf[3];
    double cd[3]; float cf[3];
    double vd[9]; float vf[9];
    double pd[3]; float pf[3];
    int32_t face_index[1];
    uint8_t valid[1];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *c = dtype == GFFX_DTYPE_FLOAT64 ? (void *)cd : (void *)cf;

    fill_components(v, dtype, cp_vertices, 9);
    fill_components(p, dtype, above, 3);
    CHECK(run_closest(p, 1, v, 3, faces, 2, cp_point_offsets, cp_vertex_offsets, face_offsets,
                      1, PX_EPS_DEFAULT, dtype, d, face_index, b, c, valid) == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u);
    CHECK(face_index[0] == 0);
    return 0;
}

static int test_cp05_cp06_degenerate_and_empty(gffx_dtype dtype) {
    static const double collinear[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    static const int32_t empty_face_offsets[2] = {0, 0};
    static const double above[3] = {0.25, 0.25, 1.0};
    double dd[1]; float df[1];
    double bd[3]; float bf[3];
    double cd[3]; float cf[3];
    double vd[9]; float vf[9];
    double pd[3]; float pf[3];
    int32_t face_index[1];
    uint8_t valid[1];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *c = dtype == GFFX_DTYPE_FLOAT64 ? (void *)cd : (void *)cf;

    /* A wholly degenerate mesh returns the sentinels. */
    fill_components(v, dtype, collinear, 9);
    fill_components(p, dtype, above, 3);
    CHECK(run_closest(p, 1, v, 3, cp_faces, 1, cp_point_offsets, cp_vertex_offsets,
                      cp_face_offsets, 1, PX_EPS_DEFAULT, dtype, d, face_index, b, c, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 0u);
    CHECK(face_index[0] == -1);
    CHECK(isinf(get_component(d, dtype, 0)) && get_component(d, dtype, 0) > 0.0);
    CHECK(get_component(b, dtype, 0) == 0.0 && get_component(c, dtype, 0) == 0.0);

    /* An empty mesh does the same. */
    fill_components(v, dtype, cp_vertices, 9);
    CHECK(run_closest(p, 1, v, 3, NULL, 0, cp_point_offsets, cp_vertex_offsets,
                      empty_face_offsets, 1, PX_EPS_DEFAULT, dtype, d, face_index, b, c, valid)
          == GFFX_STATUS_OK);
    CHECK(valid[0] == 0u && face_index[0] == -1);

    /* Zero query points succeed with empty outputs. */
    {
        static const int32_t zero_point_offsets[2] = {0, 0};
        CHECK(run_closest(NULL, 0, v, 3, cp_faces, 1, zero_point_offsets, cp_vertex_offsets,
                          cp_face_offsets, 1, PX_EPS_DEFAULT, dtype, NULL, NULL, NULL, NULL,
                          NULL) == GFFX_STATUS_OK);
    }
    return 0;
}

static double closest_objective(
    const double *points, const double *vertices, const double *weights
) {
    double distance[2];
    int32_t face_index[2];
    double barycentric[6];
    double closest[6];
    uint8_t valid[2];
    static const int32_t point_offsets[2] = {0, 2};
    double total = 0.0;
    int64_t index;
    if (run_closest(points, 2, vertices, 3, cp_faces, 1, point_offsets, cp_vertex_offsets,
                    cp_face_offsets, 1, PX_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, distance,
                    face_index, barycentric, closest, valid) != GFFX_STATUS_OK) {
        return (double)NAN;
    }
    for (index = 0; index < 2; ++index) total += weights[index] * distance[index];
    return total;
}

static int test_cp07_gradients(void) {
    /* One interior-region point and one vertex-region point, both away from boundaries. */
    static const double points[6] = {0.3, 0.25, 1.5, -0.8, -0.7, 0.4};
    static const double weights[2] = {0.75, -0.5};
    static const int32_t point_offsets[2] = {0, 2};
    const double tolerance = 1e-6;
    double vertices[9];
    double distance[2];
    int32_t face_index[2];
    double barycentric[6];
    double closest[6];
    uint8_t valid[2];
    double grad_points[6];
    double grad_vertices[9];
    double perturbed_points[6];
    double perturbed_vertices[9];
    int64_t coordinate;

    memcpy(vertices, cp_vertices, sizeof(vertices));
    CHECK(run_closest(points, 2, vertices, 3, cp_faces, 1, point_offsets, cp_vertex_offsets,
                      cp_face_offsets, 1, PX_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, distance,
                      face_index, barycentric, closest, valid) == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u && valid[1] == 1u);
    CHECK(run_closest_backward(points, 2, vertices, 3, cp_faces, 1, face_index, barycentric,
                               closest, valid, weights, GFFX_DTYPE_FLOAT64, grad_points,
                               grad_vertices) == GFFX_STATUS_OK);

    for (coordinate = 0; coordinate < 6; ++coordinate) {
        double step = 1e-6;
        double forward_value;
        double backward_value;
        memcpy(perturbed_points, points, sizeof(perturbed_points));
        perturbed_points[coordinate] = points[coordinate] + step;
        forward_value = closest_objective(perturbed_points, vertices, weights);
        perturbed_points[coordinate] = points[coordinate] - step;
        backward_value = closest_objective(perturbed_points, vertices, weights);
        CHECK(relative_close(grad_points[coordinate],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }
    for (coordinate = 0; coordinate < 9; ++coordinate) {
        double step = 1e-6;
        double forward_value;
        double backward_value;
        memcpy(perturbed_vertices, vertices, sizeof(perturbed_vertices));
        perturbed_vertices[coordinate] = vertices[coordinate] + step;
        forward_value = closest_objective(points, perturbed_vertices, weights);
        perturbed_vertices[coordinate] = vertices[coordinate] - step;
        backward_value = closest_objective(points, perturbed_vertices, weights);
        CHECK(relative_close(grad_vertices[coordinate],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }
    return 0;
}

static int test_cp08_cp10_determinism_and_workspace(void) {
    static const double points[3] = {0.3, 0.25, 1.5};
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    double distance_a[1];
    double distance_b[1];
    double barycentric_a[3];
    double barycentric_b[3];
    double closest_a[3];
    double closest_b[3];
    int32_t face_a[1];
    int32_t face_b[1];
    uint8_t valid_a[1];
    uint8_t valid_b[1];

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    CHECK(run_closest(points, 1, cp_vertices, 3, cp_faces, 1, cp_point_offsets,
                      cp_vertex_offsets, cp_face_offsets, 1, PX_EPS_DEFAULT,
                      GFFX_DTYPE_FLOAT64, distance_a, face_a, barycentric_a, closest_a, valid_a)
          == GFFX_STATUS_OK);
    CHECK(run_closest(points, 1, cp_vertices, 3, cp_faces, 1, cp_point_offsets,
                      cp_vertex_offsets, cp_face_offsets, 1, PX_EPS_DEFAULT,
                      GFFX_DTYPE_FLOAT64, distance_b, face_b, barycentric_b, closest_b, valid_b)
          == GFFX_STATUS_OK);
    CHECK(memcmp(distance_a, distance_b, sizeof(distance_a)) == 0);
    CHECK(memcmp(barycentric_a, barycentric_b, sizeof(barycentric_a)) == 0);
    CHECK(memcmp(closest_a, closest_b, sizeof(closest_a)) == 0);
    CHECK(face_a[0] == face_b[0] && valid_a[0] == valid_b[0]);

    CHECK(gffx_points_closest_point_on_mesh_workspace(1, 3, 1, GFFX_DTYPE_FLOAT64, &context,
                                                      &required_bytes, &required_alignment,
                                                      &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    return 0;
}

static int test_cp09_validation(void) {
    static const double points[3] = {0.25, 0.25, 1.0};
    /* Two batch elements; a face in element 1 referencing element 0's vertices is invalid. */
    static const double vertices[18] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        5.0, 5.0, 0.0, 6.0, 5.0, 0.0, 5.0, 6.0, 0.0
    };
    static const int32_t cross_faces[6] = {0, 1, 2, 0, 1, 2};
    static const int32_t point_offsets[3] = {0, 1, 1};
    static const int32_t vertex_offsets[3] = {0, 3, 6};
    static const int32_t face_offsets[3] = {0, 1, 2};
    double distance[1];
    int32_t face_index[1];
    double barycentric[3];
    double closest[3];
    uint8_t valid[1];

    /* Face 1 belongs to batch element 1 but references vertices 0..2 of element 0. */
    CHECK(run_closest(points, 1, vertices, 6, cross_faces, 2, point_offsets, vertex_offsets,
                      face_offsets, 2, PX_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, distance, face_index,
                      barycentric, closest, valid) == GFFX_STATUS_INVALID_ARGUMENT);

    /* eps rules. */
    CHECK(run_closest(points, 1, cp_vertices, 3, cp_faces, 1, cp_point_offsets,
                      cp_vertex_offsets, cp_face_offsets, 1, -1.0, GFFX_DTYPE_FLOAT64, distance,
                      face_index, barycentric, closest, valid) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_closest(points, 1, cp_vertices, 3, cp_faces, 1, cp_point_offsets,
                      cp_vertex_offsets, cp_face_offsets, 1, (double)NAN, GFFX_DTYPE_FLOAT64,
                      distance, face_index, barycentric, closest, valid)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    size_t index;

    for (index = 0u; index < 2u; ++index) {
        gffx_dtype dtype = dtypes[index];
        result = test_kn01_kn02_basic_and_ties(dtype); if (result != 0) return result;
        result = test_kn03_kn05_padding(dtype); if (result != 0) return result;
        result = test_kn04_batching(dtype); if (result != 0) return result;
        result = test_kn07_kn09_determinism_and_workspace(dtype); if (result != 0) return result;
        result = test_cp01_cp03_regions(dtype); if (result != 0) return result;
        result = test_cp04_tie_breaking(dtype); if (result != 0) return result;
        result = test_cp05_cp06_degenerate_and_empty(dtype); if (result != 0) return result;
    }
    result = test_kn06_gradients(); if (result != 0) return result;
    result = test_kn08_validation(); if (result != 0) return result;
    result = test_cp07_gradients(); if (result != 0) return result;
    result = test_cp08_cp10_determinism_and_workspace(); if (result != 0) return result;
    result = test_cp09_validation(); if (result != 0) return result;
    return 0;
}
