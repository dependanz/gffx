/*
 * Phase 2 acceptance fixtures GF-01..GF-09 for mesh.gather_faces.
 *
 * Fixture numbers match the project acceptance record. Failures return the source line. The
 * operation only copies and adds, so every comparison here is exact in both dtypes.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

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

static const int64_t pair_strides[2] = {3, 1};
static const int64_t triple_strides[3] = {9, 3, 1};

static gffx_status run_forward(
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    gffx_dtype dtype, void *face_vertices
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t out_shape[3];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view out_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    out_shape[0] = face_count; out_shape[1] = 3; out_shape[2] = 3;

    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    out_view = make_view(face_vertices, dtype, 3u, out_shape, triple_strides,
                         GFFX_TENSOR_OUTPUT);
    return gffx_mesh_gather_faces(&vertices_view, &faces_view, &context, &out_view, NULL,
                                  &diagnostic);
}

static gffx_status run_backward(
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    gffx_dtype dtype, const void *grad_face_vertices, void *grad_vertices
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t out_shape[3];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view cotangent_view;
    gffx_tensor_view gradient_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    out_shape[0] = face_count; out_shape[1] = 3; out_shape[2] = 3;

    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    cotangent_view = make_view((void *)grad_face_vertices, dtype, 3u, out_shape, triple_strides,
                               GFFX_TENSOR_READ_ONLY);
    gradient_view = make_view(grad_vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    return gffx_mesh_gather_faces_backward(&vertices_view, &faces_view, &cotangent_view,
                                           &context, &gradient_view, NULL, &diagnostic);
}

static const double unit_triangle[9] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
};
static const int32_t one_face[3] = {0, 1, 2};

/* Verifies face k corner j equals vertex `vertex_index` of the source array, exactly. */
static int corner_matches(
    const void *out, gffx_dtype dtype, int64_t face, int64_t corner,
    const double *source, int64_t vertex_index
) {
    int64_t axis;
    for (axis = 0; axis < 3; ++axis) {
        double actual = get_component(out, dtype, face * 9 + corner * 3 + axis);
        double expected = source[vertex_index * 3 + axis];
        if (actual != expected) return 0;
    }
    return 1;
}

static int test_gf01_gf02_order(gffx_dtype dtype) {
    static const int32_t permuted[3] = {2, 0, 1};
    double od[9]; float of[9];
    double vd[9]; float vf[9];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *o = dtype == GFFX_DTYPE_FLOAT64 ? (void *)od : (void *)of;

    fill_components(v, dtype, unit_triangle, 9);
    CHECK(run_forward(v, 3, one_face, 1, dtype, o) == GFFX_STATUS_OK);
    CHECK(corner_matches(o, dtype, 0, 0, unit_triangle, 0));
    CHECK(corner_matches(o, dtype, 0, 1, unit_triangle, 1));
    CHECK(corner_matches(o, dtype, 0, 2, unit_triangle, 2));

    CHECK(run_forward(v, 3, permuted, 1, dtype, o) == GFFX_STATUS_OK);
    CHECK(corner_matches(o, dtype, 0, 0, unit_triangle, 2));
    CHECK(corner_matches(o, dtype, 0, 1, unit_triangle, 0));
    CHECK(corner_matches(o, dtype, 0, 2, unit_triangle, 1));
    return 0;
}

static int test_gf03_gf04_sharing_and_repeats(gffx_dtype dtype) {
    static const double quad[12] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0
    };
    static const int32_t shared[6] = {0, 1, 2, 1, 3, 2};
    static const int32_t repeated[3] = {0, 0, 1};
    double od[18]; float of[18];
    double vd[12]; float vf[12];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *o = dtype == GFFX_DTYPE_FLOAT64 ? (void *)od : (void *)of;

    fill_components(v, dtype, quad, 12);
    CHECK(run_forward(v, 4, shared, 2, dtype, o) == GFFX_STATUS_OK);
    CHECK(corner_matches(o, dtype, 0, 1, quad, 1));
    CHECK(corner_matches(o, dtype, 0, 2, quad, 2));
    CHECK(corner_matches(o, dtype, 1, 0, quad, 1));
    CHECK(corner_matches(o, dtype, 1, 2, quad, 2));

    CHECK(run_forward(v, 4, repeated, 1, dtype, o) == GFFX_STATUS_OK);
    CHECK(corner_matches(o, dtype, 0, 0, quad, 0));
    CHECK(corner_matches(o, dtype, 0, 1, quad, 0));
    CHECK(corner_matches(o, dtype, 0, 2, quad, 1));
    return 0;
}

static int test_gf05_empty(gffx_dtype dtype) {
    double od[9]; float of[9];
    double vd[9]; float vf[9];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *o = dtype == GFFX_DTYPE_FLOAT64 ? (void *)od : (void *)of;

    fill_components(v, dtype, unit_triangle, 9);
    CHECK(run_forward(v, 3, NULL, 0, dtype, o) == GFFX_STATUS_OK);
    CHECK(run_forward(NULL, 0, NULL, 0, dtype, o) == GFFX_STATUS_OK);
    return 0;
}

static int test_gf06_nonfinite_bitwise(void) {
    double vertices[9];
    double out[9];
    int64_t axis;
    fill_components(vertices, GFFX_DTYPE_FLOAT64, unit_triangle, 9);
    vertices[0] = (double)NAN;
    vertices[4] = (double)INFINITY;
    vertices[8] = -(double)INFINITY;
    CHECK(run_forward(vertices, 3, one_face, 1, GFFX_DTYPE_FLOAT64, out) == GFFX_STATUS_OK);
    CHECK(isnan(out[0]));
    CHECK(isinf(out[4]) && out[4] > 0.0);
    CHECK(isinf(out[8]) && out[8] < 0.0);
    for (axis = 1; axis < 3; ++axis) CHECK(out[axis] == vertices[axis]);
    return 0;
}

static int test_gf07_validation(void) {
    double vertices[9];
    double out[9];
    double gradient[9];
    double cotangent[9] = {0};
    int32_t faces[3] = {0, 1, 2};
    int64_t vertex_shape[2] = {3, 3};
    int64_t face_shape[2] = {1, 3};
    int64_t out_shape[3] = {1, 3, 3};
    static const int64_t wrong_out_shape[3] = {1, 3, 2};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view out_view;
    gffx_tensor_view broken;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    fill_components(vertices, GFFX_DTYPE_FLOAT64, unit_triangle, 9);
    vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view(faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    out_view = make_view(out, GFFX_DTYPE_FLOAT64, 3u, out_shape, triple_strides,
                         GFFX_TENSOR_OUTPUT);

    /* Index range. */
    faces[2] = 3;
    CHECK(run_forward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, out)
          == GFFX_STATUS_INVALID_ARGUMENT);
    faces[2] = -1;
    CHECK(run_forward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, out)
          == GFFX_STATUS_INVALID_ARGUMENT);
    faces[2] = 2;

    /* Output shape and dtype. */
    broken = out_view;
    broken.shape = wrong_out_shape;
    CHECK(gffx_mesh_gather_faces(&vertices_view, &faces_view, &context, &broken, NULL,
                                 &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    broken = out_view;
    broken.dtype = GFFX_DTYPE_FLOAT32;
    CHECK(gffx_mesh_gather_faces(&vertices_view, &faces_view, &context, &broken, NULL,
                                 &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    /* Output aliasing an input. */
    broken = out_view;
    broken.data = vertices;
    CHECK(gffx_mesh_gather_faces(&vertices_view, &faces_view, &context, &broken, NULL,
                                 &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    /* Null required arguments. */
    CHECK(gffx_mesh_gather_faces(NULL, &faces_view, &context, &out_view, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_mesh_gather_faces(&vertices_view, &faces_view, &context, NULL, NULL, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);

    /* Integer vertices are well-formed but unsupported. */
    broken = vertices_view;
    broken.dtype = GFFX_DTYPE_INT32;
    CHECK(gffx_mesh_gather_faces(&broken, &faces_view, &context, &out_view, NULL, &diagnostic)
          == GFFX_STATUS_UNSUPPORTED);

    /* Backward rejects a null cotangent. */
    CHECK(run_backward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, NULL, gradient)
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_backward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, cotangent, NULL)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static int test_gf08_gradients(gffx_dtype dtype) {
    /* Two faces sharing vertices 1 and 2, plus a face repeating vertex 0, so the scatter-add
     * accumulates across faces and within a face. */
    static const double quad[12] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0
    };
    static const int32_t faces[9] = {0, 1, 2, 1, 3, 2, 0, 0, 3};
    double expected[12] = {0};
    double cotangent[27];
    double gd[12]; float gf[12];
    double cd[27]; float cf[27];
    double vd[12]; float vf[12];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *c = dtype == GFFX_DTYPE_FLOAT64 ? (void *)cd : (void *)cf;
    void *g = dtype == GFFX_DTYPE_FLOAT64 ? (void *)gd : (void *)gf;
    int64_t index;
    int64_t face;
    int64_t corner;
    int64_t axis;

    /* Distinct, exactly representable cotangent per corner component. */
    for (index = 0; index < 27; ++index) cotangent[index] = (double)(index + 1) * 0.25;
    for (face = 0; face < 3; ++face) {
        for (corner = 0; corner < 3; ++corner) {
            int64_t vertex = (int64_t)faces[face * 3 + corner];
            for (axis = 0; axis < 3; ++axis) {
                expected[vertex * 3 + axis] += cotangent[face * 9 + corner * 3 + axis];
            }
        }
    }

    fill_components(v, dtype, quad, 12);
    fill_components(c, dtype, cotangent, 27);
    CHECK(run_backward(v, 4, faces, 3, dtype, c, g) == GFFX_STATUS_OK);
    for (index = 0; index < 12; ++index) {
        CHECK(get_component(g, dtype, index) == expected[index]);
    }

    /* A vertex referenced by no face receives exactly zero. */
    {
        static const int32_t only_first[3] = {0, 1, 2};
        CHECK(run_backward(v, 4, only_first, 1, dtype, c, g) == GFFX_STATUS_OK);
        for (axis = 0; axis < 3; ++axis) CHECK(get_component(g, dtype, 9 + axis) == 0.0);
    }
    return 0;
}

static int test_gf09_determinism_and_packing(gffx_dtype dtype) {
    static const int32_t second_face[3] = {0, 1, 2};
    double packed_vertices[18];
    int32_t packed_faces[6];
    double o1_d[9]; float o1_f[9];
    double o2_d[9]; float o2_f[9];
    double op_d[18]; float op_f[18];
    double vp_d[18]; float vp_f[18];
    double v1_d[9]; float v1_f[9];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    int64_t index;
    void *vp = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vp_d : (void *)vp_f;
    void *v1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)v1_d : (void *)v1_f;
    void *o1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)o1_d : (void *)o1_f;
    void *o2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)o2_d : (void *)o2_f;
    void *op = dtype == GFFX_DTYPE_FLOAT64 ? (void *)op_d : (void *)op_f;

    fill_components(v1, dtype, unit_triangle, 9);
    CHECK(run_forward(v1, 3, one_face, 1, dtype, o1) == GFFX_STATUS_OK);
    CHECK(run_forward(v1, 3, one_face, 1, dtype, o2) == GFFX_STATUS_OK);
    CHECK(memcmp(o1, o2, 9u * element) == 0);

    for (index = 0; index < 9; ++index) {
        packed_vertices[index] = unit_triangle[index];
        packed_vertices[9 + index] = unit_triangle[index] + 4.0;
    }
    for (index = 0; index < 3; ++index) {
        packed_faces[index] = one_face[index];
        packed_faces[3 + index] = second_face[index] + 3;
    }
    fill_components(vp, dtype, packed_vertices, 18);
    CHECK(run_forward(vp, 6, packed_faces, 2, dtype, op) == GFFX_STATUS_OK);
    CHECK(memcmp(op, o1, 9u * element) == 0);
    return 0;
}

static int test_workspace_query(void) {
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;

    CHECK(gffx_mesh_gather_faces_workspace(3, 1, GFFX_DTYPE_FLOAT64, &context, &required_bytes,
                                           &required_alignment, &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    CHECK(gffx_mesh_gather_faces_workspace(3, 1, GFFX_DTYPE_INT32, &context, &required_bytes,
                                           &required_alignment, &diagnostic)
          == GFFX_STATUS_UNSUPPORTED);
    CHECK(gffx_mesh_gather_faces_workspace(-1, 1, GFFX_DTYPE_FLOAT32, &context, &required_bytes,
                                           &required_alignment, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    size_t index;

    for (index = 0u; index < 2u; ++index) {
        gffx_dtype dtype = dtypes[index];
        result = test_gf01_gf02_order(dtype); if (result != 0) return result;
        result = test_gf03_gf04_sharing_and_repeats(dtype); if (result != 0) return result;
        result = test_gf05_empty(dtype); if (result != 0) return result;
        result = test_gf08_gradients(dtype); if (result != 0) return result;
        result = test_gf09_determinism_and_packing(dtype); if (result != 0) return result;
    }
    result = test_gf06_nonfinite_bitwise(); if (result != 0) return result;
    result = test_gf07_validation(); if (result != 0) return result;
    result = test_workspace_query(); if (result != 0) return result;
    return 0;
}
