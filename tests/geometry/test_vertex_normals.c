/*
 * Phase 2 acceptance fixtures VN-01..VN-14 for mesh.vertex_normals.
 *
 * Fixture numbers match the project acceptance record. Failures return the source line. Exact
 * rows compare bit-exactly in both dtypes and both weightings; VN-05 and VN-08 use the
 * documented tolerances; VN-10 checks reverse-mode results against central finite differences
 * in FLOAT64.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define VN_EPS_DEFAULT 9.5367431640625e-7 /* 2^-20 */
#define VN_MAX_V 16
#define VN_MAX_F 16

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

static gffx_buffer make_workspace(void *data, uint64_t capacity, uint64_t alignment) {
    gffx_buffer buffer = {0};
    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.abi_version = GFFX_ABI_VERSION;
    buffer.data = data;
    buffer.capacity_bytes = capacity;
    buffer.alignment = alignment;
    buffer.device_type = GFFX_DEVICE_CPU;
    buffer.device_index = 0;
    return buffer;
}

static void set_component(void *data, gffx_dtype dtype, int64_t index, double value) {
    if (dtype == GFFX_DTYPE_FLOAT64) {
        ((double *)data)[index] = value;
    } else {
        ((float *)data)[index] = (float)value;
    }
}

static double get_component(const void *data, gffx_dtype dtype, int64_t index) {
    if (dtype == GFFX_DTYPE_FLOAT64) {
        return ((const double *)data)[index];
    }
    return (double)((const float *)data)[index];
}

static void fill_components(void *data, gffx_dtype dtype, const double *values, int64_t count) {
    int64_t index;
    for (index = 0; index < count; ++index) {
        set_component(data, dtype, index, values[index]);
    }
}

static gffx_status run_forward(
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    gffx_dtype dtype, double eps, uint32_t weighting,
    void *unit_normals
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    static const int64_t pair_strides[2] = {3, 1};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view normals_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;

    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    normals_view = make_view(unit_normals, dtype, 2u, vertex_shape, pair_strides,
                             GFFX_TENSOR_OUTPUT);
    return gffx_mesh_vertex_normals(&vertices_view, &faces_view, eps, weighting, &context,
                                    &normals_view, NULL, &diagnostic);
}

static gffx_status run_backward(
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    gffx_dtype dtype, double eps, uint32_t weighting,
    const void *grad_unit_normals, void *grad_vertices,
    void *workspace_data, uint64_t workspace_capacity
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    static const int64_t pair_strides[2] = {3, 1};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view cotangent_view;
    gffx_tensor_view gradient_view;
    gffx_buffer workspace;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;

    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    cotangent_view = make_view((void *)grad_unit_normals, dtype, 2u, vertex_shape, pair_strides,
                               GFFX_TENSOR_READ_ONLY);
    gradient_view = make_view(grad_vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_OUTPUT);
    workspace = make_workspace(workspace_data, workspace_capacity,
                               dtype == GFFX_DTYPE_FLOAT64 ? UINT64_C(8) : UINT64_C(4));
    return gffx_mesh_vertex_normals_backward(
        &vertices_view, &faces_view, eps, weighting, &cotangent_view, &context,
        &gradient_view, workspace_data != NULL ? &workspace : NULL, &diagnostic);
}

static int vertex_is_exact(const void *normals, gffx_dtype dtype, int64_t vertex,
                           double x, double y, double z) {
    if (get_component(normals, dtype, vertex * 3 + 0) != x) return 0;
    if (get_component(normals, dtype, vertex * 3 + 1) != y) return 0;
    if (get_component(normals, dtype, vertex * 3 + 2) != z) return 0;
    return 1;
}

static int relative_close(double actual, double expected, double tolerance) {
    double magnitude = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
    return fabs(actual - expected) <= tolerance * magnitude;
}

static const double unit_triangle[9] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
};
static const int32_t one_face[3] = {0, 1, 2};

static const double roof_vertices[12] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0
};
static const int32_t roof_faces[6] = {0, 1, 2, 0, 2, 3};

static const double octahedron_vertices[18] = {
    1.0, 0.0, 0.0,   -1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,    0.0, -1.0, 0.0,
    0.0, 0.0, 1.0,    0.0, 0.0, -1.0
};
static const int32_t octahedron_faces[24] = {
    0, 2, 4,   2, 1, 4,   1, 3, 4,   3, 0, 4,
    2, 0, 5,   1, 2, 5,   3, 1, 5,   0, 3, 5
};

static int test_vn01_02_unit_triangle(gffx_dtype dtype, uint32_t weighting) {
    static const int32_t flipped[3] = {0, 2, 1};
    double nd[9]; float nf[9];
    double vd[9]; float vf[9];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    int64_t vertex;

    fill_components(v, dtype, unit_triangle, 9);
    CHECK(run_forward(v, 3, one_face, 1, dtype, VN_EPS_DEFAULT, weighting, n)
          == GFFX_STATUS_OK);
    for (vertex = 0; vertex < 3; ++vertex) {
        CHECK(vertex_is_exact(n, dtype, vertex, 0.0, 0.0, 1.0));
    }
    CHECK(run_forward(v, 3, flipped, 1, dtype, VN_EPS_DEFAULT, weighting, n)
          == GFFX_STATUS_OK);
    for (vertex = 0; vertex < 3; ++vertex) {
        CHECK(vertex_is_exact(n, dtype, vertex, 0.0, 0.0, -1.0));
    }
    return 0;
}

static int test_vn03_isolated_vertex(gffx_dtype dtype, uint32_t weighting) {
    static const double with_isolated[12] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0
    };
    double nd[12]; float nf[12];
    double vd[12]; float vf[12];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;

    fill_components(v, dtype, with_isolated, 12);
    CHECK(run_forward(v, 4, one_face, 1, dtype, VN_EPS_DEFAULT, weighting, n)
          == GFFX_STATUS_OK);
    CHECK(vertex_is_exact(n, dtype, 0, 0.0, 0.0, 1.0));
    CHECK(vertex_is_exact(n, dtype, 3, 0.0, 0.0, 0.0));
    return 0;
}

static int test_vn04_coplanar_square(gffx_dtype dtype, uint32_t weighting) {
    static const double square[12] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0
    };
    static const int32_t square_faces[6] = {0, 1, 2, 0, 2, 3};
    double nd[12]; float nf[12];
    double vd[12]; float vf[12];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    int64_t vertex;

    fill_components(v, dtype, square, 12);
    CHECK(run_forward(v, 4, square_faces, 2, dtype, VN_EPS_DEFAULT, weighting, n)
          == GFFX_STATUS_OK);
    for (vertex = 0; vertex < 4; ++vertex) {
        CHECK(vertex_is_exact(n, dtype, vertex, 0.0, 0.0, 1.0));
    }
    return 0;
}

static int test_vn05_weighting_discrimination(gffx_dtype dtype) {
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-14 : 1e-6;
    const double inv_sqrt2 = 1.0 / sqrt(2.0);
    const double area_x = 1.0 / sqrt(1.25);
    const double area_z = 0.5 / sqrt(1.25);
    double n_uniform[12]; float n_uniform_f[12];
    double n_area[12]; float n_area_f[12];
    double vd[12]; float vf[12];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *nu = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n_uniform : (void *)n_uniform_f;
    void *na = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n_area : (void *)n_area_f;
    int64_t shared;

    fill_components(v, dtype, roof_vertices, 12);
    CHECK(run_forward(v, 4, roof_faces, 2, dtype, VN_EPS_DEFAULT,
                      GFFX_MESH_WEIGHTING_UNIFORM, nu) == GFFX_STATUS_OK);
    CHECK(run_forward(v, 4, roof_faces, 2, dtype, VN_EPS_DEFAULT,
                      GFFX_MESH_WEIGHTING_AREA, na) == GFFX_STATUS_OK);

    /* Single-face vertices are exact in both modes. */
    CHECK(vertex_is_exact(nu, dtype, 1, 0.0, 0.0, 1.0));
    CHECK(vertex_is_exact(na, dtype, 1, 0.0, 0.0, 1.0));
    CHECK(vertex_is_exact(nu, dtype, 3, 1.0, 0.0, 0.0));
    CHECK(vertex_is_exact(na, dtype, 3, 1.0, 0.0, 0.0));

    /* Shared-edge vertices differ by weighting. */
    for (shared = 0; shared < 3; shared += 2) {
        CHECK(relative_close(get_component(nu, dtype, shared * 3 + 0), inv_sqrt2, tolerance));
        CHECK(get_component(nu, dtype, shared * 3 + 1) == 0.0);
        CHECK(relative_close(get_component(nu, dtype, shared * 3 + 2), inv_sqrt2, tolerance));
        CHECK(relative_close(get_component(na, dtype, shared * 3 + 0), area_x, tolerance));
        CHECK(get_component(na, dtype, shared * 3 + 1) == 0.0);
        CHECK(relative_close(get_component(na, dtype, shared * 3 + 2), area_z, tolerance));
    }
    CHECK(get_component(nu, dtype, 0) != get_component(na, dtype, 0));
    return 0;
}

static int test_vn06_degenerate_inert(gffx_dtype dtype, uint32_t weighting) {
    static const int32_t with_repeated[6] = {0, 1, 2, 0, 0, 1};
    static const double collinear[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    double base_d[9]; float base_f[9];
    double mixed_d[9]; float mixed_f[9];
    double vd[9]; float vf[9];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *base = dtype == GFFX_DTYPE_FLOAT64 ? (void *)base_d : (void *)base_f;
    void *mixed = dtype == GFFX_DTYPE_FLOAT64 ? (void *)mixed_d : (void *)mixed_f;
    int64_t vertex;

    fill_components(v, dtype, unit_triangle, 9);
    CHECK(run_forward(v, 3, one_face, 1, dtype, VN_EPS_DEFAULT, weighting, base)
          == GFFX_STATUS_OK);
    CHECK(run_forward(v, 3, with_repeated, 2, dtype, VN_EPS_DEFAULT, weighting, mixed)
          == GFFX_STATUS_OK);
    CHECK(memcmp(base, mixed, 9u * element) == 0);

    fill_components(v, dtype, collinear, 9);
    CHECK(run_forward(v, 3, one_face, 1, dtype, VN_EPS_DEFAULT, weighting, mixed)
          == GFFX_STATUS_OK);
    for (vertex = 0; vertex < 3; ++vertex) {
        CHECK(vertex_is_exact(mixed, dtype, vertex, 0.0, 0.0, 0.0));
    }
    return 0;
}

static int test_vn07_exact_cancellation(gffx_dtype dtype, uint32_t weighting) {
    static const int32_t opposed[6] = {0, 1, 2, 0, 2, 1};
    double nd[9]; float nf[9];
    double vd[9]; float vf[9];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    int64_t vertex;

    fill_components(v, dtype, unit_triangle, 9);
    CHECK(run_forward(v, 3, opposed, 2, dtype, VN_EPS_DEFAULT, weighting, n)
          == GFFX_STATUS_OK);
    for (vertex = 0; vertex < 3; ++vertex) {
        CHECK(vertex_is_exact(n, dtype, vertex, 0.0, 0.0, 0.0));
    }
    return 0;
}

static int test_vn08_octahedron(gffx_dtype dtype, uint32_t weighting) {
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-14 : 1e-6;
    static const double axis_signs[18] = {
        1.0, 0.0, 0.0,  -1.0, 0.0, 0.0,  0.0, 1.0, 0.0,
        0.0, -1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, -1.0
    };
    double nd[18]; float nf[18];
    double vd[18]; float vf[18];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    int64_t index;

    fill_components(v, dtype, octahedron_vertices, 18);
    CHECK(run_forward(v, 6, octahedron_faces, 8, dtype, VN_EPS_DEFAULT, weighting, n)
          == GFFX_STATUS_OK);
    for (index = 0; index < 18; ++index) {
        if (axis_signs[index] == 0.0) {
            CHECK(get_component(n, dtype, index) == 0.0);
        } else {
            CHECK(relative_close(get_component(n, dtype, index), axis_signs[index], tolerance));
        }
    }
    return 0;
}

static int test_vn09_validation(void) {
    double vertices[9];
    double normals[9];
    double gradient[9];
    double workspace[9];
    double cotangent[9] = {0};

    fill_components(vertices, GFFX_DTYPE_FLOAT64, unit_triangle, 9);

    /* Unknown weighting values are invalid arguments. */
    CHECK(run_forward(vertices, 3, one_face, 1, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                      UINT32_C(0), normals) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_forward(vertices, 3, one_face, 1, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                      UINT32_C(3), normals) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_backward(vertices, 3, one_face, 1, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                       UINT32_C(0), cotangent, gradient, workspace,
                       sizeof(workspace)) == GFFX_STATUS_INVALID_ARGUMENT);

    /* eps rules and index-range rules follow face_geometry. */
    CHECK(run_forward(vertices, 3, one_face, 1, GFFX_DTYPE_FLOAT64, -1.0,
                      GFFX_MESH_WEIGHTING_AREA, normals) == GFFX_STATUS_INVALID_ARGUMENT);
    {
        static const int32_t out_of_range[3] = {0, 1, 3};
        CHECK(run_forward(vertices, 3, out_of_range, 1, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                          GFFX_MESH_WEIGHTING_AREA, normals) == GFFX_STATUS_INVALID_ARGUMENT);
    }

    /* Output aliasing an input is rejected. */
    {
        int64_t vertex_shape[2] = {3, 3};
        int64_t face_shape[2] = {1, 3};
        static const int64_t pair_strides[2] = {3, 1};
        gffx_execution_context context = cpu_context();
        gffx_diagnostic_buffer diagnostic = {0};
        gffx_tensor_view vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u,
                                                   vertex_shape, pair_strides,
                                                   GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view faces_view = make_view((void *)one_face, GFFX_DTYPE_INT32, 2u,
                                                face_shape, pair_strides,
                                                GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view aliased = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                             pair_strides, GFFX_TENSOR_OUTPUT);
        diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
        diagnostic.abi_version = GFFX_ABI_VERSION;
        CHECK(gffx_mesh_vertex_normals(&vertices_view, &faces_view, VN_EPS_DEFAULT,
                                       GFFX_MESH_WEIGHTING_AREA, &context, &aliased, NULL,
                                       &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    }
    return 0;
}

static double objective_weighted_vertex_normals(
    const double *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    uint32_t weighting, const double *weights
) {
    double normals[VN_MAX_V * 3];
    double total = 0.0;
    int64_t index;
    if (run_forward(vertices, vertex_count, faces, face_count, GFFX_DTYPE_FLOAT64,
                    VN_EPS_DEFAULT, weighting, normals) != GFFX_STATUS_OK) {
        return (double)NAN;
    }
    for (index = 0; index < vertex_count * 3; ++index) total += weights[index] * normals[index];
    return total;
}

static int test_vn10_gradients(void) {
    static const double weights[12] = {
        0.3, -0.7, 0.2, 0.5, 0.1, -0.4, -0.6, 0.8, 0.3, 0.2, -0.5, 0.9
    };
    const double tolerance = 1e-6;
    uint32_t weightings[2] = {GFFX_MESH_WEIGHTING_AREA, GFFX_MESH_WEIGHTING_UNIFORM};
    double gradient[12];
    double workspace[12];
    double perturbed[12];
    int64_t coordinate;
    size_t mode;

    for (mode = 0u; mode < 2u; ++mode) {
        CHECK(run_backward(roof_vertices, 4, roof_faces, 2, GFFX_DTYPE_FLOAT64,
                           VN_EPS_DEFAULT, weightings[mode], weights, gradient,
                           workspace, sizeof(workspace)) == GFFX_STATUS_OK);
        for (coordinate = 0; coordinate < 12; ++coordinate) {
            double step = 1e-6 * (fabs(roof_vertices[coordinate]) > 1.0
                                  ? fabs(roof_vertices[coordinate]) : 1.0);
            double forward_value;
            double backward_value;
            memcpy(perturbed, roof_vertices, sizeof(perturbed));
            perturbed[coordinate] = roof_vertices[coordinate] + step;
            forward_value = objective_weighted_vertex_normals(perturbed, 4, roof_faces, 2,
                                                              weightings[mode], weights);
            perturbed[coordinate] = roof_vertices[coordinate] - step;
            backward_value = objective_weighted_vertex_normals(perturbed, 4, roof_faces, 2,
                                                               weightings[mode], weights);
            CHECK(relative_close(gradient[coordinate],
                                 (forward_value - backward_value) / (2.0 * step), tolerance));
        }
    }

    /* A fully degenerate mesh produces exactly zero gradients. */
    {
        static const double collinear[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
        double small_gradient[9];
        double small_workspace[9];
        double ones[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        CHECK(run_backward(collinear, 3, one_face, 1, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                           GFFX_MESH_WEIGHTING_AREA, ones, small_gradient,
                           small_workspace, sizeof(small_workspace)) == GFFX_STATUS_OK);
        for (coordinate = 0; coordinate < 9; ++coordinate) {
            CHECK(small_gradient[coordinate] == 0.0);
        }
    }
    return 0;
}

static int test_vn11_determinism(gffx_dtype dtype, uint32_t weighting) {
    double n1d[18]; float n1f[18];
    double n2d[18]; float n2f[18];
    double vd[18]; float vf[18];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n1d : (void *)n1f;
    void *n2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n2d : (void *)n2f;

    fill_components(v, dtype, octahedron_vertices, 18);
    CHECK(run_forward(v, 6, octahedron_faces, 8, dtype, VN_EPS_DEFAULT, weighting, n1)
          == GFFX_STATUS_OK);
    CHECK(run_forward(v, 6, octahedron_faces, 8, dtype, VN_EPS_DEFAULT, weighting, n2)
          == GFFX_STATUS_OK);
    CHECK(memcmp(n1, n2, 18u * element) == 0);

    if (dtype == GFFX_DTYPE_FLOAT64) {
        double g1[18];
        double g2[18];
        double workspace[18];
        double ones[18];
        int64_t index;
        for (index = 0; index < 18; ++index) ones[index] = 1.0;
        CHECK(run_backward(vd, 6, octahedron_faces, 8, dtype, VN_EPS_DEFAULT, weighting,
                           ones, g1, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
        CHECK(run_backward(vd, 6, octahedron_faces, 8, dtype, VN_EPS_DEFAULT, weighting,
                           ones, g2, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
        CHECK(memcmp(g1, g2, sizeof(g1)) == 0);
    }
    return 0;
}

static int test_vn12_workspace(void) {
    uint64_t required_bytes = 0;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    double gradient[12];
    double workspace[12];
    double cotangent[12] = {0};

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;

    CHECK(gffx_mesh_vertex_normals_workspace(4, 2, GFFX_DTYPE_FLOAT64, &context,
                                             &required_bytes, &required_alignment,
                                             &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == UINT64_C(96));
    CHECK(required_alignment == UINT64_C(8));
    CHECK(gffx_mesh_vertex_normals_workspace(4, 2, GFFX_DTYPE_FLOAT32, &context,
                                             &required_bytes, &required_alignment,
                                             &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == UINT64_C(48));
    CHECK(required_alignment == UINT64_C(4));

    /* Backward requires the reported capacity; the forward pass accepts a null workspace. */
    CHECK(run_backward(roof_vertices, 4, roof_faces, 2, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                       GFFX_MESH_WEIGHTING_AREA, cotangent, gradient, NULL, UINT64_C(0))
          == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(run_backward(roof_vertices, 4, roof_faces, 2, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                       GFFX_MESH_WEIGHTING_AREA, cotangent, gradient, workspace, UINT64_C(95))
          == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(run_backward(roof_vertices, 4, roof_faces, 2, GFFX_DTYPE_FLOAT64, VN_EPS_DEFAULT,
                       GFFX_MESH_WEIGHTING_AREA, cotangent, gradient, workspace, UINT64_C(96))
          == GFFX_STATUS_OK);
    return 0;
}

static int test_vn13_packed_concatenation(gffx_dtype dtype, uint32_t weighting) {
    double packed_vertices[30];
    int32_t packed_faces[30];
    double np_d[30]; float np_f[30];
    double n1_d[12]; float n1_f[12];
    double n2_d[18]; float n2_f[18];
    double vp_d[30]; float vp_f[30];
    double v1_d[12]; float v1_f[12];
    double v2_d[18]; float v2_f[18];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    int64_t index;
    void *vp = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vp_d : (void *)vp_f;
    void *v1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)v1_d : (void *)v1_f;
    void *v2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)v2_d : (void *)v2_f;
    void *np = dtype == GFFX_DTYPE_FLOAT64 ? (void *)np_d : (void *)np_f;
    void *n1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n1_d : (void *)n1_f;
    void *n2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n2_d : (void *)n2_f;

    for (index = 0; index < 12; ++index) packed_vertices[index] = roof_vertices[index];
    for (index = 0; index < 18; ++index) packed_vertices[12 + index] = octahedron_vertices[index];
    for (index = 0; index < 6; ++index) packed_faces[index] = roof_faces[index];
    for (index = 0; index < 24; ++index) packed_faces[6 + index] = octahedron_faces[index] + 4;

    fill_components(vp, dtype, packed_vertices, 30);
    fill_components(v1, dtype, roof_vertices, 12);
    fill_components(v2, dtype, octahedron_vertices, 18);

    CHECK(run_forward(vp, 10, packed_faces, 10, dtype, VN_EPS_DEFAULT, weighting, np)
          == GFFX_STATUS_OK);
    CHECK(run_forward(v1, 4, roof_faces, 2, dtype, VN_EPS_DEFAULT, weighting, n1)
          == GFFX_STATUS_OK);
    CHECK(run_forward(v2, 6, octahedron_faces, 8, dtype, VN_EPS_DEFAULT, weighting, n2)
          == GFFX_STATUS_OK);
    CHECK(memcmp(np, n1, 12u * element) == 0);
    CHECK(memcmp((const char *)np + 12u * element, n2, 18u * element) == 0);
    return 0;
}

static int test_vn14_empty(gffx_dtype dtype, uint32_t weighting) {
    double nd[9]; float nf[9];
    double vd[9]; float vf[9];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    int64_t vertex;

    CHECK(run_forward(NULL, 0, NULL, 0, dtype, VN_EPS_DEFAULT, weighting, NULL)
          == GFFX_STATUS_OK);
    fill_components(v, dtype, unit_triangle, 9);
    CHECK(run_forward(v, 3, NULL, 0, dtype, VN_EPS_DEFAULT, weighting, n)
          == GFFX_STATUS_OK);
    for (vertex = 0; vertex < 3; ++vertex) {
        CHECK(vertex_is_exact(n, dtype, vertex, 0.0, 0.0, 0.0));
    }
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    uint32_t weightings[2] = {GFFX_MESH_WEIGHTING_AREA, GFFX_MESH_WEIGHTING_UNIFORM};
    size_t dtype_index;
    size_t mode;

    for (dtype_index = 0u; dtype_index < 2u; ++dtype_index) {
        gffx_dtype dtype = dtypes[dtype_index];
        for (mode = 0u; mode < 2u; ++mode) {
            uint32_t weighting = weightings[mode];
            result = test_vn01_02_unit_triangle(dtype, weighting); if (result != 0) return result;
            result = test_vn03_isolated_vertex(dtype, weighting); if (result != 0) return result;
            result = test_vn04_coplanar_square(dtype, weighting); if (result != 0) return result;
            result = test_vn06_degenerate_inert(dtype, weighting); if (result != 0) return result;
            result = test_vn07_exact_cancellation(dtype, weighting); if (result != 0) return result;
            result = test_vn08_octahedron(dtype, weighting); if (result != 0) return result;
            result = test_vn11_determinism(dtype, weighting); if (result != 0) return result;
            result = test_vn13_packed_concatenation(dtype, weighting); if (result != 0) return result;
            result = test_vn14_empty(dtype, weighting); if (result != 0) return result;
        }
        result = test_vn05_weighting_discrimination(dtype); if (result != 0) return result;
    }
    result = test_vn09_validation(); if (result != 0) return result;
    result = test_vn10_gradients(); if (result != 0) return result;
    result = test_vn12_workspace(); if (result != 0) return result;
    return 0;
}
