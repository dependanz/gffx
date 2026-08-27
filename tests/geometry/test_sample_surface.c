/*
 * Phase 2 acceptance fixtures SS-01..SS-12 for mesh.sample_surface.
 *
 * Fixture numbers match the project acceptance record. Statistical fixtures use fixed keys and
 * counters, so their outcomes are deterministic; the bands keep them readable rather than
 * tolerating flakiness.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define SS_EPS_DEFAULT 9.5367431640625e-7 /* 2^-20 */
#define SS_MAX_SAMPLES 1024

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

static gffx_buffer make_workspace(void *data, uint64_t capacity) {
    gffx_buffer buffer = {0};
    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.abi_version = GFFX_ABI_VERSION;
    buffer.data = data;
    buffer.capacity_bytes = capacity;
    buffer.alignment = UINT64_C(8);
    buffer.device_type = GFFX_DEVICE_CPU;
    buffer.device_index = 0;
    return buffer;
}

static double get_component(const void *data, gffx_dtype dtype, int64_t index) {
    if (dtype == GFFX_DTYPE_FLOAT64) return ((const double *)data)[index];
    return (double)((const float *)data)[index];
}

static void fill_components(void *data, gffx_dtype dtype, const double *values, int64_t count) {
    int64_t index;
    for (index = 0; index < count; ++index) {
        if (dtype == GFFX_DTYPE_FLOAT64) ((double *)data)[index] = values[index];
        else ((float *)data)[index] = (float)values[index];
    }
}

static gffx_status run_sample(
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    const int32_t *vertex_offsets, const int32_t *face_offsets, int64_t batch_count,
    int64_t sample_count, const uint32_t *key, const uint32_t *counter, double eps,
    gffx_dtype dtype, void *points, int32_t *face_index, void *barycentric,
    uint32_t *next_counter, void *workspace_data, uint64_t workspace_capacity
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t offset_shape[1];
    int64_t rng_shape[1];
    int64_t point_shape[3];
    int64_t index_shape[2];
    int64_t point_strides[3];
    int64_t index_strides[2];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view vertex_offsets_view;
    gffx_tensor_view face_offsets_view;
    gffx_tensor_view key_view;
    gffx_tensor_view counter_view;
    gffx_tensor_view points_view;
    gffx_tensor_view face_index_view;
    gffx_tensor_view barycentric_view;
    gffx_tensor_view next_counter_view;
    gffx_buffer workspace;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    offset_shape[0] = batch_count + 1;
    rng_shape[0] = 2;
    point_shape[0] = batch_count; point_shape[1] = sample_count; point_shape[2] = 3;
    index_shape[0] = batch_count; index_shape[1] = sample_count;
    point_strides[0] = sample_count * 3; point_strides[1] = 3; point_strides[2] = 1;
    index_strides[0] = sample_count; index_strides[1] = 1;

    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    vertex_offsets_view = make_view((void *)vertex_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                    scalar_strides, GFFX_TENSOR_READ_ONLY);
    face_offsets_view = make_view((void *)face_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);
    key_view = make_view((void *)key, GFFX_DTYPE_UINT32, 1u, rng_shape, scalar_strides,
                         GFFX_TENSOR_READ_ONLY);
    counter_view = make_view((void *)counter, GFFX_DTYPE_UINT32, 1u, rng_shape, scalar_strides,
                             GFFX_TENSOR_READ_ONLY);
    points_view = make_view(points, dtype, 3u, point_shape, point_strides, GFFX_TENSOR_OUTPUT);
    face_index_view = make_view(face_index, GFFX_DTYPE_INT32, 2u, index_shape, index_strides,
                                GFFX_TENSOR_OUTPUT);
    barycentric_view = make_view(barycentric, dtype, 3u, point_shape, point_strides,
                                 GFFX_TENSOR_OUTPUT);
    next_counter_view = make_view(next_counter, GFFX_DTYPE_UINT32, 1u, rng_shape,
                                  scalar_strides, GFFX_TENSOR_OUTPUT);
    workspace = make_workspace(workspace_data, workspace_capacity);

    return gffx_mesh_sample_surface(&vertices_view, &faces_view, &vertex_offsets_view,
                                    &face_offsets_view, sample_count, &key_view, &counter_view,
                                    eps, &context, &points_view, &face_index_view,
                                    &barycentric_view, &next_counter_view,
                                    workspace_data != NULL ? &workspace : NULL, &diagnostic);
}

static gffx_status run_sample_backward(
    const int32_t *faces, int64_t face_count,
    const int32_t *face_index, const void *barycentric, const void *grad_points,
    int64_t batch_count, int64_t sample_count, int64_t vertex_count, gffx_dtype dtype,
    void *grad_vertices
) {
    int64_t face_shape[2];
    int64_t point_shape[3];
    int64_t index_shape[2];
    int64_t vertex_shape[2];
    int64_t point_strides[3];
    int64_t index_strides[2];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view faces_view;
    gffx_tensor_view face_index_view;
    gffx_tensor_view barycentric_view;
    gffx_tensor_view grad_points_view;
    gffx_tensor_view grad_vertices_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    face_shape[0] = face_count; face_shape[1] = 3;
    point_shape[0] = batch_count; point_shape[1] = sample_count; point_shape[2] = 3;
    index_shape[0] = batch_count; index_shape[1] = sample_count;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    point_strides[0] = sample_count * 3; point_strides[1] = 3; point_strides[2] = 1;
    index_strides[0] = sample_count; index_strides[1] = 1;

    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    face_index_view = make_view((void *)face_index, GFFX_DTYPE_INT32, 2u, index_shape,
                                index_strides, GFFX_TENSOR_READ_ONLY);
    barycentric_view = make_view((void *)barycentric, dtype, 3u, point_shape, point_strides,
                                 GFFX_TENSOR_READ_ONLY);
    grad_points_view = make_view((void *)grad_points, dtype, 3u, point_shape, point_strides,
                                 GFFX_TENSOR_READ_ONLY);
    grad_vertices_view = make_view(grad_vertices, dtype, 2u, vertex_shape, pair_strides,
                                   GFFX_TENSOR_OUTPUT);
    return gffx_mesh_sample_surface_backward(&faces_view, &face_index_view, &barycentric_view,
                                             &grad_points_view, &context, &grad_vertices_view,
                                             NULL, &diagnostic);
}

/* Unit right triangle, area 1/2. */
static const double one_triangle[9] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
};
static const int32_t one_face[3] = {0, 1, 2};
static const int32_t vertex_offsets_one[2] = {0, 3};
static const int32_t face_offsets_one[2] = {0, 1};
static const uint32_t key_default[2] = {0x13579BDFu, 0x2468ACE0u};
static const uint32_t counter_zero[2] = {0u, 0u};

static int test_ss01_single_triangle(gffx_dtype dtype) {
    enum { SAMPLES = 64 };
    double pd[SAMPLES * 3]; float pf[SAMPLES * 3];
    double bd[SAMPLES * 3]; float bf[SAMPLES * 3];
    double vd[9]; float vf[9];
    int32_t face_index[SAMPLES];
    uint32_t next_counter[2];
    double workspace[1];
    int64_t sample;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-12 : 1e-5;

    fill_components(v, dtype, one_triangle, 9);
    CHECK(run_sample(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, SAMPLES,
                     key_default, counter_zero, SS_EPS_DEFAULT, dtype, p, face_index, b,
                     next_counter, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    for (sample = 0; sample < SAMPLES; ++sample) {
        double b0 = get_component(b, dtype, sample * 3 + 0);
        double b1 = get_component(b, dtype, sample * 3 + 1);
        double b2 = get_component(b, dtype, sample * 3 + 2);
        double x = get_component(p, dtype, sample * 3 + 0);
        double y = get_component(p, dtype, sample * 3 + 1);
        double z = get_component(p, dtype, sample * 3 + 2);
        CHECK(face_index[sample] == 0);
        CHECK(b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0);
        CHECK(fabs(b0 + b1 + b2 - 1.0) <= tolerance);
        /* points reconstruct from the reported weights: v0 is the origin, v1 = +x, v2 = +y. */
        CHECK(fabs(x - b1) <= tolerance);
        CHECK(fabs(y - b2) <= tolerance);
        CHECK(fabs(z) <= tolerance);
    }
    return 0;
}

static int test_ss02_ss04_reproducibility_and_counter(gffx_dtype dtype) {
    enum { SAMPLES = 16 };
    static const uint32_t counter_one[2] = {1u, 0u};
    static const uint32_t counter_wrap[2] = {0xFFFFFFFFu, 5u};
    double p1_d[SAMPLES * 3]; float p1_f[SAMPLES * 3];
    double p2_d[SAMPLES * 3]; float p2_f[SAMPLES * 3];
    double b1_d[SAMPLES * 3]; float b1_f[SAMPLES * 3];
    double b2_d[SAMPLES * 3]; float b2_f[SAMPLES * 3];
    double vd[9]; float vf[9];
    int32_t index1[SAMPLES];
    int32_t index2[SAMPLES];
    uint32_t next1[2];
    uint32_t next2[2];
    double workspace[1];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    int differs = 0;
    int64_t entry;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *p1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)p1_d : (void *)p1_f;
    void *p2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)p2_d : (void *)p2_f;
    void *b1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)b1_d : (void *)b1_f;
    void *b2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)b2_d : (void *)b2_f;

    fill_components(v, dtype, one_triangle, 9);

    /* SS-02: identical key and counter reproduce bitwise. */
    CHECK(run_sample(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, SAMPLES,
                     key_default, counter_zero, SS_EPS_DEFAULT, dtype, p1, index1, b1, next1,
                     workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(run_sample(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, SAMPLES,
                     key_default, counter_zero, SS_EPS_DEFAULT, dtype, p2, index2, b2, next2,
                     workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(memcmp(p1, p2, SAMPLES * 3u * element) == 0);
    CHECK(memcmp(b1, b2, SAMPLES * 3u * element) == 0);
    CHECK(memcmp(index1, index2, sizeof(index1)) == 0);
    CHECK(next1[0] == next2[0] && next1[1] == next2[1]);

    /* SS-04: the counter advances as a 64-bit little-endian increment. */
    CHECK(next1[0] == 1u && next1[1] == 0u);

    /* SS-03: a different counter changes at least one sample. */
    CHECK(run_sample(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, SAMPLES,
                     key_default, counter_one, SS_EPS_DEFAULT, dtype, p2, index2, b2, next2,
                     workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    for (entry = 0; entry < SAMPLES * 3; ++entry) {
        if (get_component(p1, dtype, entry) != get_component(p2, dtype, entry)) differs = 1;
    }
    CHECK(differs == 1);
    CHECK(next2[0] == 2u && next2[1] == 0u);

    /* SS-04 wrap: the low word carries into the high word. */
    CHECK(run_sample(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, SAMPLES,
                     key_default, counter_wrap, SS_EPS_DEFAULT, dtype, p2, index2, b2, next2,
                     workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(next2[0] == 0u && next2[1] == 6u);
    return 0;
}

static int test_ss05_ss06_area_weighting(void) {
    enum { SAMPLES = 1000 };
    /* Face 0 has area 3/2, face 1 has area 1/2, and face 2 is degenerate. */
    static const double vertices[18] = {
        0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 10.0, 1.0, 0.0
    };
    static const int32_t faces[9] = {0, 1, 2, 3, 4, 5, 0, 1, 1};
    static const int32_t vertex_offsets[2] = {0, 6};
    static const int32_t face_offsets[2] = {0, 3};
    double points[SAMPLES * 3];
    double barycentric[SAMPLES * 3];
    int32_t face_index[SAMPLES];
    uint32_t next_counter[2];
    double workspace[3];
    int64_t counts[3] = {0, 0, 0};
    int64_t sample;

    CHECK(run_sample(vertices, 6, faces, 3, vertex_offsets, face_offsets, 1, SAMPLES,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, sizeof(workspace))
          == GFFX_STATUS_OK);
    for (sample = 0; sample < SAMPLES; ++sample) {
        CHECK(face_index[sample] >= 0 && face_index[sample] < 3);
        counts[face_index[sample]] += 1;
    }
    /* SS-06: the degenerate face is never selected. */
    CHECK(counts[2] == 0);
    /* SS-05: selection follows the 3:1 area ratio inside a wide deterministic band. A uniform
     * over eligible faces would give about 500 each and fail this. */
    CHECK(counts[0] > 650 && counts[0] < 850);
    CHECK(counts[1] > 150 && counts[1] < 350);
    CHECK(counts[0] + counts[1] == SAMPLES);
    return 0;
}

static int test_ss07_no_eligible_face(void) {
    static const double collinear[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    double points[3];
    double barycentric[3];
    int32_t face_index[1];
    uint32_t next_counter[2];
    double workspace[1];

    CHECK(run_sample(collinear, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, 1,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Zero samples request nothing, so a fully degenerate mesh is acceptable. */
    CHECK(run_sample(collinear, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, 0,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, NULL,
                     NULL, NULL, next_counter, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    return 0;
}

static int test_ss08_batching(void) {
    enum { SAMPLES = 32 };
    static const double vertices[18] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 10.0, 1.0, 0.0
    };
    static const int32_t faces[6] = {0, 1, 2, 3, 4, 5};
    static const int32_t vertex_offsets[3] = {0, 3, 6};
    static const int32_t face_offsets[3] = {0, 1, 2};
    double points[2 * SAMPLES * 3];
    double barycentric[2 * SAMPLES * 3];
    int32_t face_index[2 * SAMPLES];
    uint32_t next_counter[2];
    double workspace[2];
    int64_t sample;
    int differs = 0;

    CHECK(run_sample(vertices, 6, faces, 2, vertex_offsets, face_offsets, 2, SAMPLES,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, sizeof(workspace))
          == GFFX_STATUS_OK);
    for (sample = 0; sample < SAMPLES; ++sample) {
        /* Element 0 selects face 0; element 1 selects the global index 1. */
        CHECK(face_index[sample] == 0);
        CHECK(face_index[SAMPLES + sample] == 1);
        /* Element 1's triangle sits near x = 10. */
        CHECK(points[(SAMPLES + sample) * 3 + 0] >= 9.0);
        if (barycentric[sample * 3 + 0] !=
            barycentric[(SAMPLES + sample) * 3 + 0]) {
            differs = 1;
        }
    }
    /* Embedding the batch index in the counter gives each element its own stream. */
    CHECK(differs == 1);
    return 0;
}

static int test_ss09_gradients(gffx_dtype dtype) {
    enum { SAMPLES = 8 };
    double pd[SAMPLES * 3]; float pf[SAMPLES * 3];
    double bd[SAMPLES * 3]; float bf[SAMPLES * 3];
    double gd[SAMPLES * 3]; float gf[SAMPLES * 3];
    double vd[9]; float vf[9];
    double grad_d[9]; float grad_f[9];
    double expected[9] = {0};
    double cotangent[SAMPLES * 3];
    int32_t face_index[SAMPLES];
    uint32_t next_counter[2];
    double workspace[1];
    int64_t sample;
    int64_t axis;
    int64_t index;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *p = dtype == GFFX_DTYPE_FLOAT64 ? (void *)pd : (void *)pf;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *g = dtype == GFFX_DTYPE_FLOAT64 ? (void *)gd : (void *)gf;
    void *grad = dtype == GFFX_DTYPE_FLOAT64 ? (void *)grad_d : (void *)grad_f;
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-12 : 1e-5;

    fill_components(v, dtype, one_triangle, 9);
    CHECK(run_sample(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, SAMPLES,
                     key_default, counter_zero, SS_EPS_DEFAULT, dtype, p, face_index, b,
                     next_counter, workspace, sizeof(workspace)) == GFFX_STATUS_OK);

    for (index = 0; index < SAMPLES * 3; ++index) cotangent[index] = (double)(index + 1) * 0.125;
    fill_components(g, dtype, cotangent, SAMPLES * 3);

    /* The map is linear, so the expected scatter is computable exactly. */
    for (sample = 0; sample < SAMPLES; ++sample) {
        int corner;
        for (corner = 0; corner < 3; ++corner) {
            double weight = get_component(b, dtype, sample * 3 + corner);
            for (axis = 0; axis < 3; ++axis) {
                expected[one_face[corner] * 3 + axis] +=
                    weight * cotangent[sample * 3 + axis];
            }
        }
    }
    CHECK(run_sample_backward(one_face, 1, face_index, b, g, 1, SAMPLES, 3, dtype, grad)
          == GFFX_STATUS_OK);
    for (index = 0; index < 9; ++index) {
        CHECK(fabs(get_component(grad, dtype, index) - expected[index]) <= tolerance);
    }
    return 0;
}

static int test_ss10_validation(void) {
    double points[3];
    double barycentric[3];
    int32_t face_index[1];
    uint32_t next_counter[2];
    double workspace[1];
    int32_t bad_offsets[2] = {0, 0};

    /* Negative sample counts. */
    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, -1,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* eps rules. */
    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, 1,
                     key_default, counter_zero, -1.0, GFFX_DTYPE_FLOAT64, points, face_index,
                     barycentric, next_counter, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, 1,
                     key_default, counter_zero, (double)NAN, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Offset rules: the final face offset must equal F. */
    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, bad_offsets, 1, 1,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static int test_ss11_workspace(void) {
    uint64_t required_bytes = 0;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    double points[3];
    double barycentric[3];
    int32_t face_index[1];
    uint32_t next_counter[2];
    double workspace[1];

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    CHECK(gffx_mesh_sample_surface_workspace(3, 1, 4, GFFX_DTYPE_FLOAT64, &context,
                                             &required_bytes, &required_alignment, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(required_bytes == UINT64_C(8));
    CHECK(required_alignment == UINT64_C(8));
    /* The requirement is dtype-independent because the table is always double. */
    CHECK(gffx_mesh_sample_surface_workspace(3, 5, 4, GFFX_DTYPE_FLOAT32, &context,
                                             &required_bytes, &required_alignment, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(required_bytes == UINT64_C(40));

    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, 1,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, NULL, 0)
          == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, 1,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, UINT64_C(7))
          == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, 1,
                     key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, points,
                     face_index, barycentric, next_counter, workspace, UINT64_C(8))
          == GFFX_STATUS_OK);
    return 0;
}

static int test_ss12_uniformity(void) {
    enum { SAMPLES = 1000 };
    double points[SAMPLES * 3];
    double barycentric[SAMPLES * 3];
    int32_t face_index[SAMPLES];
    uint32_t next_counter[2];
    double workspace[1];
    double mean[3] = {0.0, 0.0, 0.0};
    int64_t sample;
    int axis;

    CHECK(run_sample(one_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                     SAMPLES, key_default, counter_zero, SS_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                     points, face_index, barycentric, next_counter, workspace,
                     sizeof(workspace)) == GFFX_STATUS_OK);
    for (sample = 0; sample < SAMPLES; ++sample) {
        for (axis = 0; axis < 3; ++axis) mean[axis] += barycentric[sample * 3 + axis];
    }
    for (axis = 0; axis < 3; ++axis) {
        mean[axis] /= (double)SAMPLES;
        /* A map collapsing toward a vertex or edge would leave this band. */
        CHECK(mean[axis] > 0.28 && mean[axis] < 0.39);
    }
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    size_t index;

    for (index = 0u; index < 2u; ++index) {
        gffx_dtype dtype = dtypes[index];
        result = test_ss01_single_triangle(dtype); if (result != 0) return result;
        result = test_ss02_ss04_reproducibility_and_counter(dtype); if (result != 0) return result;
        result = test_ss09_gradients(dtype); if (result != 0) return result;
    }
    result = test_ss05_ss06_area_weighting(); if (result != 0) return result;
    result = test_ss07_no_eligible_face(); if (result != 0) return result;
    result = test_ss08_batching(); if (result != 0) return result;
    result = test_ss10_validation(); if (result != 0) return result;
    result = test_ss11_workspace(); if (result != 0) return result;
    result = test_ss12_uniformity(); if (result != 0) return result;
    return 0;
}
