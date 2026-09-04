/*
 * Phase 2 acceptance fixtures FG-01..FG-15 for mesh.face_geometry.
 *
 * Each fixture number below matches the project acceptance record. Failures return the source
 * line, matching the existing ABI test convention. Exact rows compare bit-exactly in both
 * FLOAT32 and FLOAT64; FG-10 uses the documented tolerances; FG-12 checks reverse-mode results
 * against central finite differences in FLOAT64.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define FG_EPS_DEFAULT 9.5367431640625e-7 /* 2^-20, exact in both dtypes */
#define FG_MAX_V 16
#define FG_MAX_F 16

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

static void fill_vertices(void *data, gffx_dtype dtype, const double *xyz, int64_t vertex_count) {
    int64_t index;
    for (index = 0; index < vertex_count * 3; ++index) {
        set_component(data, dtype, index, xyz[index]);
    }
}

/* Runs the forward operation over dense CPU views; returns the operation status. */
static gffx_status run_forward(
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    gffx_dtype dtype, double eps,
    void *unit_normals, void *areas, uint8_t *valid
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t normal_shape[2];
    int64_t scalar_shape[1];
    static const int64_t pair_strides[2] = {3, 1};
    static const int64_t scalar_strides[1] = {1};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view normals_view;
    gffx_tensor_view areas_view;
    gffx_tensor_view valid_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;

    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    normal_shape[0] = face_count; normal_shape[1] = 3;
    scalar_shape[0] = face_count;

    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    normals_view = make_view(unit_normals, dtype, 2u, normal_shape, pair_strides,
                             GFFX_TENSOR_OUTPUT);
    areas_view = make_view(areas, dtype, 1u, scalar_shape, scalar_strides, GFFX_TENSOR_OUTPUT);
    valid_view = make_view(valid, GFFX_DTYPE_BOOL, 1u, scalar_shape, scalar_strides,
                           GFFX_TENSOR_OUTPUT);

    return gffx_mesh_face_geometry(&vertices_view, &faces_view, eps, &context,
                                   &normals_view, &areas_view, &valid_view, NULL, &diagnostic);
}

/* Runs the backward operation; NULL cotangent pointers are forwarded as NULL views. */
static gffx_status run_backward(
    const void *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    gffx_dtype dtype, double eps,
    const void *grad_unit_normals, const void *grad_areas,
    void *grad_vertices
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t normal_shape[2];
    int64_t scalar_shape[1];
    static const int64_t pair_strides[2] = {3, 1};
    static const int64_t scalar_strides[1] = {1};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view grad_normals_view;
    gffx_tensor_view grad_areas_view;
    gffx_tensor_view grad_vertices_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;

    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    normal_shape[0] = face_count; normal_shape[1] = 3;
    scalar_shape[0] = face_count;

    vertices_view = make_view((void *)vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    grad_normals_view = make_view((void *)grad_unit_normals, dtype, 2u, normal_shape,
                                  pair_strides, GFFX_TENSOR_READ_ONLY);
    grad_areas_view = make_view((void *)grad_areas, dtype, 1u, scalar_shape, scalar_strides,
                                GFFX_TENSOR_READ_ONLY);
    grad_vertices_view = make_view(grad_vertices, dtype, 2u, vertex_shape, pair_strides,
                                   GFFX_TENSOR_OUTPUT);

    return gffx_mesh_face_geometry_backward(
        &vertices_view, &faces_view, eps,
        grad_unit_normals != NULL ? &grad_normals_view : NULL,
        grad_areas != NULL ? &grad_areas_view : NULL,
        &context, &grad_vertices_view, NULL, &diagnostic);
}

static int check_face_exact(
    const void *unit_normals, const void *areas, const uint8_t *valid,
    gffx_dtype dtype, int64_t face,
    double nx, double ny, double nz, double area, int expect_valid
) {
    if (valid[face] != (expect_valid ? 1u : 0u)) return 0;
    if (get_component(unit_normals, dtype, face * 3 + 0) != nx) return 0;
    if (get_component(unit_normals, dtype, face * 3 + 1) != ny) return 0;
    if (get_component(unit_normals, dtype, face * 3 + 2) != nz) return 0;
    if (get_component(areas, dtype, face) != area) return 0;
    return 1;
}

static int relative_close(double actual, double expected, double relative_tolerance) {
    double magnitude = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
    return fabs(actual - expected) <= relative_tolerance * magnitude;
}

static const double unit_triangle[9] = {
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0
};
static const int32_t one_face[3] = {0, 1, 2};

/* Octahedron with exact +-1 axis coordinates and eight outward-wound faces. */
static const double octahedron_vertices[18] = {
    1.0, 0.0, 0.0,   -1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,    0.0, -1.0, 0.0,
    0.0, 0.0, 1.0,    0.0, 0.0, -1.0
};
static const int32_t octahedron_faces[24] = {
    0, 2, 4,   2, 1, 4,   1, 3, 4,   3, 0, 4,
    2, 0, 5,   1, 2, 5,   3, 1, 5,   0, 3, 5
};
static const double octahedron_normal_signs[24] = {
    1.0, 1.0, 1.0,   -1.0, 1.0, 1.0,   -1.0, -1.0, 1.0,   1.0, -1.0, 1.0,
    1.0, 1.0, -1.0,  -1.0, 1.0, -1.0,  -1.0, -1.0, -1.0,  1.0, -1.0, -1.0
};

static int test_fg01_unit_triangle(gffx_dtype dtype) {
    double storage_n[3]; float storage_nf[3];
    double storage_a[1]; float storage_af[1];
    double vertex_data[9]; float vertex_dataf[9];
    uint8_t valid[1] = {9};
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vertex_data : (void *)vertex_dataf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)storage_n : (void *)storage_nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)storage_a : (void *)storage_af;

    fill_vertices(v, dtype, unit_triangle, 3);
    CHECK(run_forward(v, 3, one_face, 1, dtype, FG_EPS_DEFAULT, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 1.0, 0.5, 1));
    return 0;
}

static int test_fg02_winding(gffx_dtype dtype) {
    static const int32_t flipped[3] = {0, 2, 1};
    double nd[3]; float nf[3]; double ad[1]; float af[1];
    double vd[9]; float vf[9];
    uint8_t valid[1] = {9};
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;

    fill_vertices(v, dtype, unit_triangle, 3);
    CHECK(run_forward(v, 3, flipped, 1, dtype, FG_EPS_DEFAULT, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, -1.0, 0.5, 1));
    return 0;
}

static int test_fg03_translation(gffx_dtype dtype) {
    double translated[9];
    double nd[3]; float nf[3]; double ad[1]; float af[1];
    double vd[9]; float vf[9];
    uint8_t valid[1] = {9};
    int64_t index;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;

    for (index = 0; index < 3; ++index) {
        translated[index * 3 + 0] = unit_triangle[index * 3 + 0] + 8.0;
        translated[index * 3 + 1] = unit_triangle[index * 3 + 1] - 4.0;
        translated[index * 3 + 2] = unit_triangle[index * 3 + 2] + 2.5;
    }
    fill_vertices(v, dtype, translated, 3);
    CHECK(run_forward(v, 3, one_face, 1, dtype, FG_EPS_DEFAULT, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 1.0, 0.5, 1));
    return 0;
}

static int test_fg04_rotation(gffx_dtype dtype) {
    /* Exact 90-degree rotation about z: (x, y, z) -> (-y, x, z). */
    double rotated[9];
    double nd[3]; float nf[3]; double ad[1]; float af[1];
    double vd[9]; float vf[9];
    uint8_t valid[1] = {9};
    int64_t index;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;

    for (index = 0; index < 3; ++index) {
        rotated[index * 3 + 0] = -unit_triangle[index * 3 + 1];
        rotated[index * 3 + 1] = unit_triangle[index * 3 + 0];
        rotated[index * 3 + 2] = unit_triangle[index * 3 + 2];
    }
    fill_vertices(v, dtype, rotated, 3);
    CHECK(run_forward(v, 3, one_face, 1, dtype, FG_EPS_DEFAULT, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 1.0, 0.5, 1));
    return 0;
}

static int test_fg05_scaling(gffx_dtype dtype) {
    double scaled[9];
    double nd[3]; float nf[3]; double ad[1]; float af[1];
    double vd[9]; float vf[9];
    uint8_t valid[1] = {9};
    int64_t index;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;

    for (index = 0; index < 9; ++index) {
        scaled[index] = unit_triangle[index] * 2.0;
    }
    fill_vertices(v, dtype, scaled, 3);
    CHECK(run_forward(v, 3, one_face, 1, dtype, FG_EPS_DEFAULT, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 1.0, 2.0, 1));
    return 0;
}

static int test_fg06_degenerate(gffx_dtype dtype) {
    static const int32_t repeated[3] = {0, 0, 1};
    static const double collinear[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    double nd[3]; float nf[3]; double ad[1]; float af[1];
    double vd[9]; float vf[9];
    uint8_t valid[1] = {9};
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;

    fill_vertices(v, dtype, unit_triangle, 3);
    CHECK(run_forward(v, 3, repeated, 1, dtype, FG_EPS_DEFAULT, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 0.0, 0.0, 0));

    fill_vertices(v, dtype, collinear, 3);
    CHECK(run_forward(v, 3, one_face, 1, dtype, FG_EPS_DEFAULT, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 0.0, 0.0, 0));
    return 0;
}

static int test_fg07_eps_boundary(gffx_dtype dtype) {
    /* Legs 1 and 2 give doubled area d = 2 exactly; validity is the strict d > eps. */
    static const double legs[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0};
    double nd[3]; float nf[3]; double ad[1]; float af[1];
    double vd[9]; float vf[9];
    uint8_t valid[1] = {9};
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;

    fill_vertices(v, dtype, legs, 3);
    CHECK(run_forward(v, 3, one_face, 1, dtype, 2.0, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 0.0, 0.0, 0));

    CHECK(run_forward(v, 3, one_face, 1, dtype, 1.75, n, a, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(n, a, valid, dtype, 0, 0.0, 0.0, 1.0, 1.0, 1));
    return 0;
}

static int test_fg08_empty(gffx_dtype dtype) {
    double vd[9]; float vf[9];
    double nd[1]; double ad[1];
    uint8_t valid[1];
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;

    fill_vertices(v, dtype, unit_triangle, 3);
    CHECK(run_forward(v, 3, NULL, 0, dtype, FG_EPS_DEFAULT, nd, ad, valid) == GFFX_STATUS_OK);
    CHECK(run_forward(NULL, 0, NULL, 0, dtype, FG_EPS_DEFAULT, nd, ad, valid) == GFFX_STATUS_OK);
    return 0;
}

static int test_fg09_validation(void) {
    double vertices[9];
    double normals[3];
    double areas[1];
    uint8_t valid[1];
    int32_t faces[3];

    fill_vertices(vertices, GFFX_DTYPE_FLOAT64, unit_triangle, 3);

    /* Out-of-range and negative indices are rejected before any dereference. */
    faces[0] = 0; faces[1] = 1; faces[2] = 3;
    CHECK(run_forward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, FG_EPS_DEFAULT,
                      normals, areas, valid) == GFFX_STATUS_INVALID_ARGUMENT);
    faces[2] = -1;
    CHECK(run_forward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, FG_EPS_DEFAULT,
                      normals, areas, valid) == GFFX_STATUS_INVALID_ARGUMENT);

    /* eps must be finite and nonnegative. */
    faces[2] = 2;
    CHECK(run_forward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, -1.0,
                      normals, areas, valid) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_forward(vertices, 3, faces, 1, GFFX_DTYPE_FLOAT64, (double)NAN,
                      normals, areas, valid) == GFFX_STATUS_INVALID_ARGUMENT);

    /* Wrong shapes, ranks, and dtypes. */
    {
        static const int64_t bad_vertex_shape[2] = {3, 2};
        static const int64_t bad_face_shape[2] = {1, 4};
        static const int64_t pair_strides[2] = {3, 1};
        static const int64_t normal_shape[2] = {1, 3};
        static const int64_t scalar_shape[1] = {1};
        static const int64_t scalar_strides[1] = {1};
        int64_t vertex_shape[2] = {3, 3};
        int64_t face_shape[2] = {1, 3};
        gffx_execution_context context = cpu_context();
        gffx_diagnostic_buffer diagnostic = {0};
        gffx_tensor_view vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u,
                                                   vertex_shape, pair_strides,
                                                   GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view faces_view = make_view(faces, GFFX_DTYPE_INT32, 2u, face_shape,
                                                pair_strides, GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view normals_view = make_view(normals, GFFX_DTYPE_FLOAT64, 2u, normal_shape,
                                                  pair_strides, GFFX_TENSOR_OUTPUT);
        gffx_tensor_view areas_view = make_view(areas, GFFX_DTYPE_FLOAT64, 1u, scalar_shape,
                                                scalar_strides, GFFX_TENSOR_OUTPUT);
        gffx_tensor_view valid_view = make_view(valid, GFFX_DTYPE_BOOL, 1u, scalar_shape,
                                                scalar_strides, GFFX_TENSOR_OUTPUT);
        gffx_tensor_view broken;
        diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
        diagnostic.abi_version = GFFX_ABI_VERSION;

        /* Vertices [V,2]. */
        broken = vertices_view;
        broken.shape = bad_vertex_shape;
        CHECK(gffx_mesh_face_geometry(&broken, &faces_view, FG_EPS_DEFAULT, &context,
                                      &normals_view, &areas_view, &valid_view, NULL,
                                      &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

        /* Faces [F,4]. */
        broken = faces_view;
        broken.shape = bad_face_shape;
        CHECK(gffx_mesh_face_geometry(&vertices_view, &broken, FG_EPS_DEFAULT, &context,
                                      &normals_view, &areas_view, &valid_view, NULL,
                                      &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

        /* Faces carrying a floating dtype: well-formed view, unsupported here. */
        broken = faces_view;
        broken.dtype = GFFX_DTYPE_FLOAT32;
        CHECK(gffx_mesh_face_geometry(&vertices_view, &broken, FG_EPS_DEFAULT, &context,
                                      &normals_view, &areas_view, &valid_view, NULL,
                                      &diagnostic) == GFFX_STATUS_UNSUPPORTED);

        /* Vertices carrying an integer dtype: unsupported here. */
        broken = vertices_view;
        broken.dtype = GFFX_DTYPE_INT32;
        CHECK(gffx_mesh_face_geometry(&broken, &faces_view, FG_EPS_DEFAULT, &context,
                                      &normals_view, &areas_view, &valid_view, NULL,
                                      &diagnostic) == GFFX_STATUS_UNSUPPORTED);

        /* Null required arguments. */
        CHECK(gffx_mesh_face_geometry(NULL, &faces_view, FG_EPS_DEFAULT, &context,
                                      &normals_view, &areas_view, &valid_view, NULL,
                                      &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
        CHECK(gffx_mesh_face_geometry(&vertices_view, &faces_view, FG_EPS_DEFAULT, &context,
                                      NULL, &areas_view, &valid_view, NULL,
                                      &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    }
    return 0;
}

static int test_fg10_octahedron(gffx_dtype dtype) {
    const double inv_sqrt3 = 1.0 / sqrt(3.0);
    const double face_area = sqrt(3.0) / 2.0;
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-14 : 1e-6;
    double nd[24]; float nf[24]; double ad[8]; float af[8];
    double vd[18]; float vf[18];
    uint8_t valid[8];
    int64_t face;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n = dtype == GFFX_DTYPE_FLOAT64 ? (void *)nd : (void *)nf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;

    fill_vertices(v, dtype, octahedron_vertices, 6);
    CHECK(run_forward(v, 6, octahedron_faces, 8, dtype, FG_EPS_DEFAULT, n, a, valid)
          == GFFX_STATUS_OK);
    for (face = 0; face < 8; ++face) {
        int64_t axis;
        CHECK(valid[face] == 1u);
        CHECK(relative_close(get_component(a, dtype, face), face_area, tolerance));
        for (axis = 0; axis < 3; ++axis) {
            double expected = octahedron_normal_signs[face * 3 + axis] * inv_sqrt3;
            CHECK(relative_close(get_component(n, dtype, face * 3 + axis), expected, tolerance));
        }
    }
    return 0;
}

static int test_fg11_nonfinite(void) {
    double nd[3]; double ad[1];
    float nf[3]; float af[1];
    double vertices_nan[9];
    float vertices_inf[9];
    uint8_t valid[1];
    int64_t index;

    /* NaN input: the d > eps comparison is false, so the invalid branch produces exact zeros. */
    fill_vertices(vertices_nan, GFFX_DTYPE_FLOAT64, unit_triangle, 3);
    vertices_nan[0] = (double)NAN;
    CHECK(run_forward(vertices_nan, 3, one_face, 1, GFFX_DTYPE_FLOAT64, FG_EPS_DEFAULT,
                      nd, ad, valid) == GFFX_STATUS_OK);
    CHECK(check_face_exact(nd, ad, valid, GFFX_DTYPE_FLOAT64, 0, 0.0, 0.0, 0.0, 0.0, 0));

    /* Float32 overflow: d is +inf, the face is valid, area is +inf, no crash or sanitization. */
    for (index = 0; index < 9; ++index) {
        vertices_inf[index] = (float)(unit_triangle[index] * 1e30);
    }
    CHECK(run_forward(vertices_inf, 3, one_face, 1, GFFX_DTYPE_FLOAT32, FG_EPS_DEFAULT,
                      nf, af, valid) == GFFX_STATUS_OK);
    CHECK(valid[0] == 1u);
    CHECK(isinf((double)af[0]) && af[0] > 0.0f);
    return 0;
}

/* FG-12 helpers: scalar objectives for finite differences in FLOAT64. */
static double objective_area_sum(const double *vertices, int64_t vertex_count,
                                 const int32_t *faces, int64_t face_count) {
    double normals[FG_MAX_F * 3];
    double areas[FG_MAX_F];
    uint8_t valid[FG_MAX_F];
    double total = 0.0;
    int64_t face;
    if (run_forward(vertices, vertex_count, faces, face_count, GFFX_DTYPE_FLOAT64,
                    FG_EPS_DEFAULT, normals, areas, valid) != GFFX_STATUS_OK) {
        return (double)NAN;
    }
    for (face = 0; face < face_count; ++face) total += areas[face];
    return total;
}

static double objective_weighted_normals(const double *vertices, int64_t vertex_count,
                                         const int32_t *faces, int64_t face_count,
                                         const double *weights) {
    double normals[FG_MAX_F * 3];
    double areas[FG_MAX_F];
    uint8_t valid[FG_MAX_F];
    double total = 0.0;
    int64_t index;
    if (run_forward(vertices, vertex_count, faces, face_count, GFFX_DTYPE_FLOAT64,
                    FG_EPS_DEFAULT, normals, areas, valid) != GFFX_STATUS_OK) {
        return (double)NAN;
    }
    for (index = 0; index < face_count * 3; ++index) total += weights[index] * normals[index];
    return total;
}

static int test_fg12_gradients(void) {
    /* Irregular single triangle plus a two-face mesh sharing an edge (scatter-add coverage). */
    static const double single[9] = {0.1, 0.2, 0.3, 1.1, -0.4, 0.25, -0.3, 0.9, 1.7};
    static const double shared[12] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5
    };
    static const int32_t shared_faces[6] = {0, 1, 2, 1, 3, 2};
    static const double weights[6] = {0.7, -0.3, 0.5, -0.2, 0.4, 0.9};
    const double tolerance = 1e-6;
    double gradient[12];
    double perturbed[12];
    int64_t coordinate;
    int mesh;

    for (mesh = 0; mesh < 2; ++mesh) {
        const double *vertices = mesh == 0 ? single : shared;
        const int32_t *faces = mesh == 0 ? one_face : shared_faces;
        int64_t vertex_count = mesh == 0 ? 3 : 4;
        int64_t face_count = mesh == 0 ? 1 : 2;
        double ones[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        /* Analytic gradient of sum(areas) from the backward entry point. */
        CHECK(run_backward(vertices, vertex_count, faces, face_count, GFFX_DTYPE_FLOAT64,
                           FG_EPS_DEFAULT, NULL, ones, gradient) == GFFX_STATUS_OK);
        for (coordinate = 0; coordinate < vertex_count * 3; ++coordinate) {
            double step = 1e-6 * (fabs(vertices[coordinate]) > 1.0
                                  ? fabs(vertices[coordinate]) : 1.0);
            double forward_value;
            double backward_value;
            memcpy(perturbed, vertices, (size_t)(vertex_count * 3) * sizeof(double));
            perturbed[coordinate] = vertices[coordinate] + step;
            forward_value = objective_area_sum(perturbed, vertex_count, faces, face_count);
            perturbed[coordinate] = vertices[coordinate] - step;
            backward_value = objective_area_sum(perturbed, vertex_count, faces, face_count);
            CHECK(relative_close(gradient[coordinate],
                                 (forward_value - backward_value) / (2.0 * step), tolerance));
        }
    }

    /* Gradient of a fixed linear functional of the unit normals, single triangle. */
    CHECK(run_backward(single, 3, one_face, 1, GFFX_DTYPE_FLOAT64, FG_EPS_DEFAULT,
                       weights, NULL, gradient) == GFFX_STATUS_OK);
    for (coordinate = 0; coordinate < 9; ++coordinate) {
        double step = 1e-6 * (fabs(single[coordinate]) > 1.0 ? fabs(single[coordinate]) : 1.0);
        double forward_value;
        double backward_value;
        memcpy(perturbed, single, 9 * sizeof(double));
        perturbed[coordinate] = single[coordinate] + step;
        forward_value = objective_weighted_normals(perturbed, 3, one_face, 1, weights);
        perturbed[coordinate] = single[coordinate] - step;
        backward_value = objective_weighted_normals(perturbed, 3, one_face, 1, weights);
        CHECK(relative_close(gradient[coordinate],
                             (forward_value - backward_value) / (2.0 * step), tolerance));
    }

    /* Degenerate faces contribute exactly zero gradient. */
    {
        static const int32_t repeated[3] = {0, 0, 1};
        double vertices[9];
        double ones_n[3] = {1.0, 1.0, 1.0};
        double ones_a[1] = {1.0};
        fill_vertices(vertices, GFFX_DTYPE_FLOAT64, unit_triangle, 3);
        CHECK(run_backward(vertices, 3, repeated, 1, GFFX_DTYPE_FLOAT64, FG_EPS_DEFAULT,
                           ones_n, ones_a, gradient) == GFFX_STATUS_OK);
        for (coordinate = 0; coordinate < 9; ++coordinate) {
            CHECK(gradient[coordinate] == 0.0);
        }
    }
    return 0;
}

static int test_fg13_determinism(gffx_dtype dtype) {
    double n1d[24]; float n1f[24]; double a1d[8]; float a1f[8];
    double n2d[24]; float n2f[24]; double a2d[8]; float a2f[8];
    double vd[18]; float vf[18];
    uint8_t valid1[8];
    uint8_t valid2[8];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *n1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n1d : (void *)n1f;
    void *a1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)a1d : (void *)a1f;
    void *n2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n2d : (void *)n2f;
    void *a2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)a2d : (void *)a2f;

    fill_vertices(v, dtype, octahedron_vertices, 6);
    CHECK(run_forward(v, 6, octahedron_faces, 8, dtype, FG_EPS_DEFAULT, n1, a1, valid1)
          == GFFX_STATUS_OK);
    CHECK(run_forward(v, 6, octahedron_faces, 8, dtype, FG_EPS_DEFAULT, n2, a2, valid2)
          == GFFX_STATUS_OK);
    CHECK(memcmp(n1, n2, 24u * element) == 0);
    CHECK(memcmp(a1, a2, 8u * element) == 0);
    CHECK(memcmp(valid1, valid2, sizeof(valid1)) == 0);

    if (dtype == GFFX_DTYPE_FLOAT64) {
        double g1[18];
        double g2[18];
        double ones[24];
        int64_t index;
        for (index = 0; index < 24; ++index) ones[index] = 1.0;
        CHECK(run_backward(vd, 6, octahedron_faces, 8, dtype, FG_EPS_DEFAULT, ones, ones, g1)
              == GFFX_STATUS_OK);
        CHECK(run_backward(vd, 6, octahedron_faces, 8, dtype, FG_EPS_DEFAULT, ones, ones, g2)
              == GFFX_STATUS_OK);
        CHECK(memcmp(g1, g2, sizeof(g1)) == 0);
    }
    return 0;
}

static int test_fg14_alias_and_shape_rejection(void) {
    double vertices[9];
    double normals[3];
    double areas[1];
    uint8_t valid[1];
    int64_t vertex_shape[2] = {3, 3};
    int64_t face_shape[2] = {1, 3};
    static const int64_t pair_strides[2] = {3, 1};
    static const int64_t normal_shape[2] = {1, 3};
    static const int64_t scalar_shape[1] = {1};
    static const int64_t scalar_strides[1] = {1};
    static const int64_t wrong_scalar_shape[1] = {2};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view normals_view;
    gffx_tensor_view areas_view;
    gffx_tensor_view valid_view;
    gffx_tensor_view broken;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    fill_vertices(vertices, GFFX_DTYPE_FLOAT64, unit_triangle, 3);
    vertices_view = make_view(vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)one_face, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    normals_view = make_view(normals, GFFX_DTYPE_FLOAT64, 2u, normal_shape, pair_strides,
                             GFFX_TENSOR_OUTPUT);
    areas_view = make_view(areas, GFFX_DTYPE_FLOAT64, 1u, scalar_shape, scalar_strides,
                           GFFX_TENSOR_OUTPUT);
    valid_view = make_view(valid, GFFX_DTYPE_BOOL, 1u, scalar_shape, scalar_strides,
                           GFFX_TENSOR_OUTPUT);

    /* Output aliasing an input is rejected. */
    broken = normals_view;
    broken.data = vertices;
    CHECK(gffx_mesh_face_geometry(&vertices_view, &faces_view, FG_EPS_DEFAULT, &context,
                                  &broken, &areas_view, &valid_view, NULL,
                                  &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    /* Outputs aliasing each other are rejected. */
    broken = areas_view;
    broken.data = normals;
    CHECK(gffx_mesh_face_geometry(&vertices_view, &faces_view, FG_EPS_DEFAULT, &context,
                                  &normals_view, &broken, &valid_view, NULL,
                                  &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    /* An output whose extent disagrees with F is rejected. */
    broken = areas_view;
    broken.shape = wrong_scalar_shape;
    CHECK(gffx_mesh_face_geometry(&vertices_view, &faces_view, FG_EPS_DEFAULT, &context,
                                  &normals_view, &broken, &valid_view, NULL,
                                  &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);

    /* An output whose dtype disagrees with the computation dtype is rejected. */
    broken = areas_view;
    broken.dtype = GFFX_DTYPE_FLOAT32;
    CHECK(gffx_mesh_face_geometry(&vertices_view, &faces_view, FG_EPS_DEFAULT, &context,
                                  &normals_view, &broken, &valid_view, NULL,
                                  &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static int test_fg15_packed_concatenation(gffx_dtype dtype) {
    /* Octahedron plus the unit triangle packed with global indices; per-face outputs must be
     * bitwise identical to the two separate executions. */
    double packed_vertices[27];
    int32_t packed_faces[27];
    double np_d[27]; float np_f[27]; double ap_d[9]; float ap_f[9];
    double n1_d[24]; float n1_f[24]; double a1_d[8]; float a1_f[8];
    double n2_d[3]; float n2_f[3]; double a2_d[1]; float a2_f[1];
    double vp_d[27]; float vp_f[27];
    double v1_d[18]; float v1_f[18];
    double v2_d[9]; float v2_f[9];
    uint8_t valid_packed[9];
    uint8_t valid_first[8];
    uint8_t valid_second[1];
    size_t element = dtype == GFFX_DTYPE_FLOAT64 ? sizeof(double) : sizeof(float);
    int64_t index;
    void *vp = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vp_d : (void *)vp_f;
    void *v1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)v1_d : (void *)v1_f;
    void *v2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)v2_d : (void *)v2_f;
    void *np = dtype == GFFX_DTYPE_FLOAT64 ? (void *)np_d : (void *)np_f;
    void *ap = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ap_d : (void *)ap_f;
    void *n1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n1_d : (void *)n1_f;
    void *a1 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)a1_d : (void *)a1_f;
    void *n2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)n2_d : (void *)n2_f;
    void *a2 = dtype == GFFX_DTYPE_FLOAT64 ? (void *)a2_d : (void *)a2_f;

    for (index = 0; index < 18; ++index) packed_vertices[index] = octahedron_vertices[index];
    for (index = 0; index < 9; ++index) packed_vertices[18 + index] = unit_triangle[index];
    for (index = 0; index < 24; ++index) packed_faces[index] = octahedron_faces[index];
    for (index = 0; index < 3; ++index) packed_faces[24 + index] = one_face[index] + 6;

    fill_vertices(vp, dtype, packed_vertices, 9);
    fill_vertices(v1, dtype, octahedron_vertices, 6);
    fill_vertices(v2, dtype, unit_triangle, 3);

    CHECK(run_forward(vp, 9, packed_faces, 9, dtype, FG_EPS_DEFAULT, np, ap, valid_packed)
          == GFFX_STATUS_OK);
    CHECK(run_forward(v1, 6, octahedron_faces, 8, dtype, FG_EPS_DEFAULT, n1, a1, valid_first)
          == GFFX_STATUS_OK);
    CHECK(run_forward(v2, 3, one_face, 1, dtype, FG_EPS_DEFAULT, n2, a2, valid_second)
          == GFFX_STATUS_OK);

    CHECK(memcmp(np, n1, 24u * element) == 0);
    CHECK(memcmp((const char *)np + 24u * element, n2, 3u * element) == 0);
    CHECK(memcmp(ap, a1, 8u * element) == 0);
    CHECK(memcmp((const char *)ap + 8u * element, a2, 1u * element) == 0);
    CHECK(memcmp(valid_packed, valid_first, 8u) == 0);
    CHECK(valid_packed[8] == valid_second[0]);
    return 0;
}

static int test_workspace_query(void) {
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;

    CHECK(gffx_mesh_face_geometry_workspace(6, 8, GFFX_DTYPE_FLOAT64, &context,
                                            &required_bytes, &required_alignment,
                                            &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    CHECK(gffx_mesh_face_geometry_workspace(6, 8, GFFX_DTYPE_INT32, &context,
                                            &required_bytes, &required_alignment,
                                            &diagnostic) == GFFX_STATUS_UNSUPPORTED);
    CHECK(gffx_mesh_face_geometry_workspace(-1, 8, GFFX_DTYPE_FLOAT32, &context,
                                            &required_bytes, &required_alignment,
                                            &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(gffx_mesh_face_geometry_workspace(6, 8, GFFX_DTYPE_FLOAT32, &context,
                                            NULL, &required_alignment,
                                            &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    size_t dtype_index;

    for (dtype_index = 0u; dtype_index < 2u; ++dtype_index) {
        gffx_dtype dtype = dtypes[dtype_index];
        result = test_fg01_unit_triangle(dtype); if (result != 0) return result;
        result = test_fg02_winding(dtype); if (result != 0) return result;
        result = test_fg03_translation(dtype); if (result != 0) return result;
        result = test_fg04_rotation(dtype); if (result != 0) return result;
        result = test_fg05_scaling(dtype); if (result != 0) return result;
        result = test_fg06_degenerate(dtype); if (result != 0) return result;
        result = test_fg07_eps_boundary(dtype); if (result != 0) return result;
        result = test_fg08_empty(dtype); if (result != 0) return result;
        result = test_fg10_octahedron(dtype); if (result != 0) return result;
        result = test_fg13_determinism(dtype); if (result != 0) return result;
        result = test_fg15_packed_concatenation(dtype); if (result != 0) return result;
    }
    result = test_fg09_validation(); if (result != 0) return result;
    result = test_fg11_nonfinite(); if (result != 0) return result;
    result = test_fg12_gradients(); if (result != 0) return result;
    result = test_fg14_alias_and_shape_rejection(); if (result != 0) return result;
    result = test_workspace_query(); if (result != 0) return result;
    return 0;
}
