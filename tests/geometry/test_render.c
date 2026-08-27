/*
 * Phase 2 acceptance fixtures RA-01..RA-14 and IN-01..IN-05 for render.rasterize and
 * render.interpolate. Fixture numbers match the project acceptance record.
 */

#include <gffx/execution.h>
#include <gffx/render.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define RA_EPS_DEFAULT 9.5367431640625e-7 /* 2^-20 */

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

static void fill_double(void *data, gffx_dtype dtype, const double *values, int64_t count) {
    int64_t index;
    for (index = 0; index < count; ++index) {
        if (dtype == GFFX_DTYPE_FLOAT64) ((double *)data)[index] = values[index];
        else ((float *)data)[index] = (float)values[index];
    }
}

static double get_double(const void *data, gffx_dtype dtype, int64_t index) {
    if (dtype == GFFX_DTYPE_FLOAT64) return ((const double *)data)[index];
    return (double)((const float *)data)[index];
}

static int relative_close(double actual, double expected, double tolerance) {
    double magnitude = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
    return fabs(actual - expected) <= tolerance * magnitude;
}

/* ------------------------------------------------------------------ rasterize helper */

static gffx_status run_rasterize(
    const void *ndc_vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    const int32_t *vertex_offsets, const int32_t *face_offsets, int64_t batch_count,
    int64_t height, int64_t width, int64_t neighbors,
    double blur_radius, uint32_t cull_mode, double eps, gffx_dtype dtype,
    int32_t *face_index, void *barycentric, void *depth, void *signed_distance
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t offset_shape[1];
    int64_t fragment_shape[4];
    int64_t bary_shape[5];
    int64_t fragment_strides[4];
    int64_t bary_strides[5];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view vertex_offsets_view;
    gffx_tensor_view face_offsets_view;
    gffx_tensor_view face_index_view;
    gffx_tensor_view bary_view;
    gffx_tensor_view depth_view;
    gffx_tensor_view distance_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    offset_shape[0] = batch_count + 1;
    fragment_shape[0] = batch_count; fragment_shape[1] = height;
    fragment_shape[2] = width; fragment_shape[3] = neighbors;
    bary_shape[0] = batch_count; bary_shape[1] = height; bary_shape[2] = width;
    bary_shape[3] = neighbors; bary_shape[4] = 3;
    fragment_strides[3] = 1;
    fragment_strides[2] = neighbors;
    fragment_strides[1] = width * neighbors;
    fragment_strides[0] = height * width * neighbors;
    bary_strides[4] = 1;
    bary_strides[3] = 3;
    bary_strides[2] = neighbors * 3;
    bary_strides[1] = width * neighbors * 3;
    bary_strides[0] = height * width * neighbors * 3;

    vertices_view = make_view((void *)ndc_vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    vertex_offsets_view = make_view((void *)vertex_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                    scalar_strides, GFFX_TENSOR_READ_ONLY);
    face_offsets_view = make_view((void *)face_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);
    face_index_view = make_view(face_index, GFFX_DTYPE_INT32, 4u, fragment_shape,
                                fragment_strides, GFFX_TENSOR_OUTPUT);
    bary_view = make_view(barycentric, dtype, 5u, bary_shape, bary_strides, GFFX_TENSOR_OUTPUT);
    depth_view = make_view(depth, dtype, 4u, fragment_shape, fragment_strides,
                           GFFX_TENSOR_OUTPUT);
    distance_view = make_view(signed_distance, dtype, 4u, fragment_shape, fragment_strides,
                              GFFX_TENSOR_OUTPUT);
    return gffx_render_rasterize(&vertices_view, &faces_view, &vertex_offsets_view,
                                 &face_offsets_view, height, width, neighbors, blur_radius,
                                 cull_mode, eps, &context, &face_index_view, &bary_view,
                                 &depth_view, &distance_view, NULL, &diagnostic);
}

static gffx_status run_rasterize_backward(
    const void *ndc_vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    int64_t batch_count, int64_t height, int64_t width, int64_t neighbors,
    const int32_t *face_index, const void *grad_barycentric, const void *grad_depth,
    const void *grad_signed_distance, gffx_dtype dtype, void *grad_ndc_vertices
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t fragment_shape[4];
    int64_t bary_shape[5];
    int64_t fragment_strides[4];
    int64_t bary_strides[5];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view face_index_view;
    gffx_tensor_view grad_bary_view;
    gffx_tensor_view grad_depth_view;
    gffx_tensor_view grad_distance_view;
    gffx_tensor_view grad_vertices_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    fragment_shape[0] = batch_count; fragment_shape[1] = height;
    fragment_shape[2] = width; fragment_shape[3] = neighbors;
    bary_shape[0] = batch_count; bary_shape[1] = height; bary_shape[2] = width;
    bary_shape[3] = neighbors; bary_shape[4] = 3;
    fragment_strides[3] = 1;
    fragment_strides[2] = neighbors;
    fragment_strides[1] = width * neighbors;
    fragment_strides[0] = height * width * neighbors;
    bary_strides[4] = 1;
    bary_strides[3] = 3;
    bary_strides[2] = neighbors * 3;
    bary_strides[1] = width * neighbors * 3;
    bary_strides[0] = height * width * neighbors * 3;

    vertices_view = make_view((void *)ndc_vertices, dtype, 2u, vertex_shape, pair_strides,
                              GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    face_index_view = make_view((void *)face_index, GFFX_DTYPE_INT32, 4u, fragment_shape,
                                fragment_strides, GFFX_TENSOR_READ_ONLY);
    grad_bary_view = make_view((void *)grad_barycentric, dtype, 5u, bary_shape, bary_strides,
                               GFFX_TENSOR_READ_ONLY);
    grad_depth_view = make_view((void *)grad_depth, dtype, 4u, fragment_shape, fragment_strides,
                                GFFX_TENSOR_READ_ONLY);
    grad_distance_view = make_view((void *)grad_signed_distance, dtype, 4u, fragment_shape,
                                   fragment_strides, GFFX_TENSOR_READ_ONLY);
    grad_vertices_view = make_view(grad_ndc_vertices, dtype, 2u, vertex_shape, pair_strides,
                                   GFFX_TENSOR_OUTPUT);
    return gffx_render_rasterize_backward(
        &vertices_view, &faces_view, height, width, &face_index_view,
        grad_barycentric != NULL ? &grad_bary_view : NULL,
        grad_depth != NULL ? &grad_depth_view : NULL,
        grad_signed_distance != NULL ? &grad_distance_view : NULL,
        &context, &grad_vertices_view, NULL, &diagnostic);
}

/* A triangle covering the whole NDC square, wound counter-clockwise with y up. */
static const double full_triangle[9] = {
    -3.0, -1.0, 0.5,    3.0, -1.0, 0.5,    0.0, 3.0, 0.5
};
static const int32_t one_face[3] = {0, 1, 2};
static const int32_t vertex_offsets_one[2] = {0, 3};
static const int32_t face_offsets_one[2] = {0, 1};

static int test_ra01_full_coverage(gffx_dtype dtype) {
    enum { H = 4, W = 4, K = 1 };
    double vd[9]; float vf[9];
    double bd[H * W * K * 3]; float bf[H * W * K * 3];
    double dd[H * W * K]; float df[H * W * K];
    double sd[H * W * K]; float sf[H * W * K];
    int32_t face_index[H * W * K];
    int64_t pixel;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;
    void *s = dtype == GFFX_DTYPE_FLOAT64 ? (void *)sd : (void *)sf;
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-12 : 1e-5;

    fill_double(v, dtype, full_triangle, 9);
    CHECK(run_rasterize(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, H, W, K,
                        0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, dtype, face_index, b, d, s)
          == GFFX_STATUS_OK);
    for (pixel = 0; pixel < H * W; ++pixel) {
        double b0 = get_double(b, dtype, pixel * 3 + 0);
        double b1 = get_double(b, dtype, pixel * 3 + 1);
        double b2 = get_double(b, dtype, pixel * 3 + 2);
        CHECK(face_index[pixel] == 0);
        CHECK(b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0);
        CHECK(relative_close(b0 + b1 + b2, 1.0, tolerance));
        /* All three vertices share z = 0.5, so the interpolated depth is 0.5 everywhere. */
        CHECK(relative_close(get_double(d, dtype, pixel), 0.5, tolerance));
        /* Inside pixels carry a negative signed distance. */
        CHECK(get_double(s, dtype, pixel) < 0.0);
    }
    return 0;
}

static int test_ra02_background(gffx_dtype dtype) {
    enum { H = 2, W = 2, K = 2 };
    /* A tiny triangle far outside the NDC square. */
    static const double offscreen[9] = {
        5.0, 5.0, 0.5,   5.1, 5.0, 0.5,   5.0, 5.1, 0.5
    };
    double vd[9]; float vf[9];
    double bd[H * W * K * 3]; float bf[H * W * K * 3];
    double dd[H * W * K]; float df[H * W * K];
    double sd[H * W * K]; float sf[H * W * K];
    int32_t face_index[H * W * K];
    int64_t slot;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;
    void *s = dtype == GFFX_DTYPE_FLOAT64 ? (void *)sd : (void *)sf;

    fill_double(v, dtype, offscreen, 9);
    CHECK(run_rasterize(v, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, H, W, K,
                        0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, dtype, face_index, b, d, s)
          == GFFX_STATUS_OK);
    for (slot = 0; slot < H * W * K; ++slot) {
        CHECK(face_index[slot] == -1);
        CHECK(get_double(b, dtype, slot * 3 + 0) == 0.0);
        CHECK(get_double(b, dtype, slot * 3 + 1) == 0.0);
        CHECK(get_double(b, dtype, slot * 3 + 2) == 0.0);
        CHECK(isinf(get_double(d, dtype, slot)) && get_double(d, dtype, slot) > 0.0);
        CHECK(isinf(get_double(s, dtype, slot)) && get_double(s, dtype, slot) > 0.0);
    }
    return 0;
}

static int test_ra03_ra04_depth_ordering(gffx_dtype dtype) {
    enum { H = 2, W = 2, K = 2 };
    /* Two full-coverage triangles, the second nearer than the first. */
    static const double layered[18] = {
        -3.0, -1.0, 0.8,   3.0, -1.0, 0.8,   0.0, 3.0, 0.8,
        -3.0, -1.0, 0.2,   3.0, -1.0, 0.2,   0.0, 3.0, 0.2
    };
    static const double coincident[18] = {
        -3.0, -1.0, 0.5,   3.0, -1.0, 0.5,   0.0, 3.0, 0.5,
        -3.0, -1.0, 0.5,   3.0, -1.0, 0.5,   0.0, 3.0, 0.5
    };
    static const int32_t two_faces[6] = {0, 1, 2, 3, 4, 5};
    static const int32_t vertex_offsets[2] = {0, 6};
    static const int32_t face_offsets[2] = {0, 2};
    double vd[18]; float vf[18];
    double bd[H * W * K * 3]; float bf[H * W * K * 3];
    double dd[H * W * K]; float df[H * W * K];
    double sd[H * W * K]; float sf[H * W * K];
    int32_t face_index[H * W * K];
    int64_t pixel;
    void *v = dtype == GFFX_DTYPE_FLOAT64 ? (void *)vd : (void *)vf;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *d = dtype == GFFX_DTYPE_FLOAT64 ? (void *)dd : (void *)df;
    void *s = dtype == GFFX_DTYPE_FLOAT64 ? (void *)sd : (void *)sf;
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-12 : 1e-5;

    /* RA-03: face 1 is nearer and must occupy slot 0 even though it is scanned second. */
    fill_double(v, dtype, layered, 18);
    CHECK(run_rasterize(v, 6, two_faces, 2, vertex_offsets, face_offsets, 1, H, W, K,
                        0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, dtype, face_index, b, d, s)
          == GFFX_STATUS_OK);
    for (pixel = 0; pixel < H * W; ++pixel) {
        CHECK(face_index[pixel * K + 0] == 1);
        CHECK(face_index[pixel * K + 1] == 0);
        CHECK(relative_close(get_double(d, dtype, pixel * K + 0), 0.2, tolerance));
        CHECK(relative_close(get_double(d, dtype, pixel * K + 1), 0.8, tolerance));
    }

    /* RA-04: identical depth resolves to the lower global face index. */
    fill_double(v, dtype, coincident, 18);
    CHECK(run_rasterize(v, 6, two_faces, 2, vertex_offsets, face_offsets, 1, H, W, K,
                        0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, dtype, face_index, b, d, s)
          == GFFX_STATUS_OK);
    for (pixel = 0; pixel < H * W; ++pixel) {
        CHECK(face_index[pixel * K + 0] == 0);
        CHECK(face_index[pixel * K + 1] == 1);
    }
    return 0;
}

static int test_ra05_ra06_culling_and_degenerate(void) {
    enum { H = 2, W = 2, K = 1 };
    /* Face 0 is counter-clockwise in NDC (front); face 1 reverses two corners (back). */
    static const int32_t ccw_face[3] = {0, 1, 2};
    static const int32_t cw_face[3] = {0, 2, 1};
    static const int32_t degenerate_face[3] = {0, 1, 1};
    double barycentric[H * W * K * 3];
    double depth[H * W * K];
    double distance[H * W * K];
    int32_t face_index[H * W * K];

    /* Front-facing triangle. */
    CHECK(run_rasterize(full_triangle, 3, ccw_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    CHECK(face_index[0] == 0);
    CHECK(run_rasterize(full_triangle, 3, ccw_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_BACK, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    CHECK(face_index[0] == 0);
    CHECK(run_rasterize(full_triangle, 3, ccw_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_FRONT, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    CHECK(face_index[0] == -1);

    /* Back-facing triangle: the cull modes swap roles. */
    CHECK(run_rasterize(full_triangle, 3, cw_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    CHECK(face_index[0] == 0);
    CHECK(run_rasterize(full_triangle, 3, cw_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_BACK, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    CHECK(face_index[0] == -1);
    CHECK(run_rasterize(full_triangle, 3, cw_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_FRONT, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    CHECK(face_index[0] == 0);

    /* RA-06: a zero-area face is never rasterized under any cull mode. */
    CHECK(run_rasterize(full_triangle, 3, degenerate_face, 1, vertex_offsets_one,
                        face_offsets_one, 1, H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT,
                        GFFX_DTYPE_FLOAT64, face_index, barycentric, depth, distance)
          == GFFX_STATUS_OK);
    CHECK(face_index[0] == -1);
    return 0;
}

static int test_ra07_signed_distance_and_blur(void) {
    enum { H = 1, W = 8, K = 1 };
    /* A triangle covering the left half of a 1x8 image. Pixel centres sit at x = 0.5, 1.5,
     * ... 7.5. Two vertices share pixel x = 4 so the right edge is exactly vertical there and
     * the covered span does not taper across the sampled row; the third sits far to the left
     * on the row itself, so pixels 0..3 are inside and 4..7 are outside. */
    static const double left_half[9] = {
        0.0, 21.0, 0.5,   0.0, -19.0, 0.5,   -6.0, 0.0, 0.5
    };
    double barycentric[H * W * K * 3];
    double depth[H * W * K];
    double distance[H * W * K];
    int32_t face_index[H * W * K];
    int64_t pixel;

    /* With no blur only inside pixels are candidates. */
    CHECK(run_rasterize(left_half, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    for (pixel = 0; pixel < 4; ++pixel) {
        CHECK(face_index[pixel] == 0);
        CHECK(distance[pixel] < 0.0);
    }
    for (pixel = 4; pixel < 8; ++pixel) {
        CHECK(face_index[pixel] == -1);
    }

    /* A blur radius of 1 pixel admits exactly the first outside pixel, whose centre lies 0.5
     * pixels beyond the edge; the next centre is 1.5 pixels out and stays excluded. */
    CHECK(run_rasterize(left_half, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 1.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    CHECK(face_index[4] == 0);
    CHECK(distance[4] > 0.0);
    CHECK(relative_close(distance[4], 0.25, 1e-9));
    CHECK(face_index[5] == -1);
    return 0;
}

static int test_ra08_ra09_orientation_and_centres(void) {
    enum { H = 4, W = 4, K = 1 };
    /* A triangle occupying the upper-left NDC quadrant: x in [-1,0], y in [0,1]. With row 0 at
     * the top, that is rows 0..1 and columns 0..1. */
    static const double upper_left[9] = {
        -1.0, 0.0, 0.5,   0.0, 0.0, 0.5,   -1.0, 1.0, 0.5
    };
    double barycentric[H * W * K * 3];
    double depth[H * W * K];
    double distance[H * W * K];
    int32_t face_index[H * W * K];
    int64_t row;
    int64_t column;

    CHECK(run_rasterize(upper_left, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    for (row = 0; row < H; ++row) {
        for (column = 0; column < W; ++column) {
            int32_t hit = face_index[row * W + column];
            if (row >= 2 || column >= 2) {
                /* Nothing in the lower or right half is covered. */
                CHECK(hit == -1);
            }
        }
    }
    /* The top-left pixel centre lies inside the quadrant triangle. */
    CHECK(face_index[0] == 0);
    return 0;
}

static int test_ra10_gradients(void) {
    enum { H = 2, W = 2, K = 1 };
    /* An irregular triangle covering the image, chosen away from edges and coverage flips. */
    static const double vertices[9] = {
        -2.3, -1.7, 0.25,   2.1, -1.4, 0.6,   0.15, 2.6, 0.45
    };
    static const double bary_weights[H * W * K * 3] = {
        0.7, -0.3, 0.5,  -0.2, 0.9, 0.1,  0.4, 0.25, -0.6,  0.15, -0.5, 0.8
    };
    static const double depth_weights[H * W * K] = {0.6, -0.4, 0.9, 0.2};
    static const double distance_weights[H * W * K] = {0.3, 0.7, -0.5, 0.45};
    const double tolerance = 1e-6;
    double barycentric[H * W * K * 3];
    double depth[H * W * K];
    double distance[H * W * K];
    int32_t face_index[H * W * K];
    double gradient[9];
    double perturbed[9];
    int64_t coordinate;

    CHECK(run_rasterize(vertices, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance) == GFFX_STATUS_OK);
    for (coordinate = 0; coordinate < H * W * K; ++coordinate) CHECK(face_index[coordinate] == 0);

    CHECK(run_rasterize_backward(vertices, 3, one_face, 1, 1, H, W, K, face_index, bary_weights,
                                 depth_weights, distance_weights, GFFX_DTYPE_FLOAT64, gradient)
          == GFFX_STATUS_OK);

    for (coordinate = 0; coordinate < 9; ++coordinate) {
        double step = 1e-6;
        double plus_total = 0.0;
        double minus_total = 0.0;
        double local_bary[H * W * K * 3];
        double local_depth[H * W * K];
        double local_distance[H * W * K];
        int32_t local_index[H * W * K];
        int64_t entry;
        memcpy(perturbed, vertices, sizeof(perturbed));

        perturbed[coordinate] = vertices[coordinate] + step;
        CHECK(run_rasterize(perturbed, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                            H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                            local_index, local_bary, local_depth, local_distance)
              == GFFX_STATUS_OK);
        for (entry = 0; entry < H * W * K; ++entry) {
            plus_total += depth_weights[entry] * local_depth[entry];
            plus_total += distance_weights[entry] * local_distance[entry];
            plus_total += bary_weights[entry * 3 + 0] * local_bary[entry * 3 + 0];
            plus_total += bary_weights[entry * 3 + 1] * local_bary[entry * 3 + 1];
            plus_total += bary_weights[entry * 3 + 2] * local_bary[entry * 3 + 2];
        }

        perturbed[coordinate] = vertices[coordinate] - step;
        CHECK(run_rasterize(perturbed, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                            H, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                            local_index, local_bary, local_depth, local_distance)
              == GFFX_STATUS_OK);
        for (entry = 0; entry < H * W * K; ++entry) {
            minus_total += depth_weights[entry] * local_depth[entry];
            minus_total += distance_weights[entry] * local_distance[entry];
            minus_total += bary_weights[entry * 3 + 0] * local_bary[entry * 3 + 0];
            minus_total += bary_weights[entry * 3 + 1] * local_bary[entry * 3 + 1];
            minus_total += bary_weights[entry * 3 + 2] * local_bary[entry * 3 + 2];
        }
        CHECK(relative_close(gradient[coordinate],
                             (plus_total - minus_total) / (2.0 * step), tolerance));
    }
    return 0;
}

static int test_ra11_ra12_determinism_and_batching(void) {
    enum { H = 2, W = 2, K = 1 };
    static const double two_meshes[18] = {
        -3.0, -1.0, 0.5,   3.0, -1.0, 0.5,   0.0, 3.0, 0.5,
        -3.0, -1.0, 0.9,   3.0, -1.0, 0.9,   0.0, 3.0, 0.9
    };
    static const int32_t faces[6] = {0, 1, 2, 3, 4, 5};
    static const int32_t vertex_offsets[3] = {0, 3, 6};
    static const int32_t face_offsets[3] = {0, 1, 2};
    double bary_a[2 * H * W * K * 3];
    double bary_b[2 * H * W * K * 3];
    double depth_a[2 * H * W * K];
    double depth_b[2 * H * W * K];
    double distance_a[2 * H * W * K];
    double distance_b[2 * H * W * K];
    int32_t index_a[2 * H * W * K];
    int32_t index_b[2 * H * W * K];
    int64_t pixel;

    CHECK(run_rasterize(two_meshes, 6, faces, 2, vertex_offsets, face_offsets, 2, H, W, K,
                        0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, index_a,
                        bary_a, depth_a, distance_a) == GFFX_STATUS_OK);
    CHECK(run_rasterize(two_meshes, 6, faces, 2, vertex_offsets, face_offsets, 2, H, W, K,
                        0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64, index_b,
                        bary_b, depth_b, distance_b) == GFFX_STATUS_OK);
    CHECK(memcmp(bary_a, bary_b, sizeof(bary_a)) == 0);
    CHECK(memcmp(depth_a, depth_b, sizeof(depth_a)) == 0);
    CHECK(memcmp(distance_a, distance_b, sizeof(distance_a)) == 0);
    CHECK(memcmp(index_a, index_b, sizeof(index_a)) == 0);

    /* Element 0 sees only face 0; element 1 sees only the global face 1. */
    for (pixel = 0; pixel < H * W; ++pixel) {
        CHECK(index_a[pixel] == 0);
        CHECK(index_a[H * W + pixel] == 1);
    }
    return 0;
}

static int test_ra13_ra14_validation_and_workspace(void) {
    enum { H = 2, W = 2, K = 1 };
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    double barycentric[H * W * K * 3];
    double depth[H * W * K];
    double distance[H * W * K];
    int32_t face_index[H * W * K];

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;

    /* K must be positive. */
    CHECK(run_rasterize(full_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, 0, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance)
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Image dimensions must be positive. */
    CHECK(run_rasterize(full_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        0, W, K, 0.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance)
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* The blur radius must be finite and nonnegative. */
    CHECK(run_rasterize(full_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, -1.0, GFFX_CULL_NONE, RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance)
          == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(run_rasterize(full_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, (double)NAN, GFFX_CULL_NONE, RA_EPS_DEFAULT,
                        GFFX_DTYPE_FLOAT64, face_index, barycentric, depth, distance)
          == GFFX_STATUS_INVALID_ARGUMENT);
    /* Unknown cull modes are rejected. */
    CHECK(run_rasterize(full_triangle, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                        H, W, K, 0.0, UINT32_C(99), RA_EPS_DEFAULT, GFFX_DTYPE_FLOAT64,
                        face_index, barycentric, depth, distance)
          == GFFX_STATUS_INVALID_ARGUMENT);

    CHECK(gffx_render_rasterize_workspace(3, 1, H, W, K, GFFX_DTYPE_FLOAT64, &context,
                                          &required_bytes, &required_alignment, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    return 0;
}

/* ---------------------------------------------------------------- interpolate helpers */

static gffx_status run_interpolate(
    const int32_t *face_index, const void *barycentric, const void *face_attributes,
    int64_t batch_count, int64_t height, int64_t width, int64_t neighbors,
    int64_t face_count, int64_t channels, gffx_dtype dtype, void *attributes
) {
    int64_t fragment_shape[4];
    int64_t bary_shape[5];
    int64_t attribute_shape[3];
    int64_t output_shape[5];
    int64_t fragment_strides[4];
    int64_t bary_strides[5];
    int64_t attribute_strides[3];
    int64_t output_strides[5];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view face_index_view;
    gffx_tensor_view bary_view;
    gffx_tensor_view attribute_view;
    gffx_tensor_view output_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    fragment_shape[0] = batch_count; fragment_shape[1] = height;
    fragment_shape[2] = width; fragment_shape[3] = neighbors;
    bary_shape[0] = batch_count; bary_shape[1] = height; bary_shape[2] = width;
    bary_shape[3] = neighbors; bary_shape[4] = 3;
    attribute_shape[0] = face_count; attribute_shape[1] = 3; attribute_shape[2] = channels;
    output_shape[0] = batch_count; output_shape[1] = height; output_shape[2] = width;
    output_shape[3] = neighbors; output_shape[4] = channels;
    fragment_strides[3] = 1;
    fragment_strides[2] = neighbors;
    fragment_strides[1] = width * neighbors;
    fragment_strides[0] = height * width * neighbors;
    bary_strides[4] = 1; bary_strides[3] = 3; bary_strides[2] = neighbors * 3;
    bary_strides[1] = width * neighbors * 3;
    bary_strides[0] = height * width * neighbors * 3;
    attribute_strides[2] = 1; attribute_strides[1] = channels;
    attribute_strides[0] = 3 * channels;
    output_strides[4] = 1; output_strides[3] = channels;
    output_strides[2] = neighbors * channels;
    output_strides[1] = width * neighbors * channels;
    output_strides[0] = height * width * neighbors * channels;

    face_index_view = make_view((void *)face_index, GFFX_DTYPE_INT32, 4u, fragment_shape,
                                fragment_strides, GFFX_TENSOR_READ_ONLY);
    bary_view = make_view((void *)barycentric, dtype, 5u, bary_shape, bary_strides,
                          GFFX_TENSOR_READ_ONLY);
    attribute_view = make_view((void *)face_attributes, dtype, 3u, attribute_shape,
                               attribute_strides, GFFX_TENSOR_READ_ONLY);
    output_view = make_view(attributes, dtype, 5u, output_shape, output_strides,
                            GFFX_TENSOR_OUTPUT);
    return gffx_render_interpolate(&face_index_view, &bary_view, &attribute_view, &context,
                                   &output_view, NULL, &diagnostic);
}

static gffx_status run_interpolate_backward(
    const int32_t *face_index, const void *barycentric, const void *face_attributes,
    const void *grad_attributes, int64_t batch_count, int64_t height, int64_t width,
    int64_t neighbors, int64_t face_count, int64_t channels, gffx_dtype dtype,
    void *grad_barycentric, void *grad_face_attributes
) {
    int64_t fragment_shape[4];
    int64_t bary_shape[5];
    int64_t attribute_shape[3];
    int64_t output_shape[5];
    int64_t fragment_strides[4];
    int64_t bary_strides[5];
    int64_t attribute_strides[3];
    int64_t output_strides[5];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view face_index_view;
    gffx_tensor_view bary_view;
    gffx_tensor_view attribute_view;
    gffx_tensor_view cotangent_view;
    gffx_tensor_view grad_bary_view;
    gffx_tensor_view grad_attribute_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    fragment_shape[0] = batch_count; fragment_shape[1] = height;
    fragment_shape[2] = width; fragment_shape[3] = neighbors;
    bary_shape[0] = batch_count; bary_shape[1] = height; bary_shape[2] = width;
    bary_shape[3] = neighbors; bary_shape[4] = 3;
    attribute_shape[0] = face_count; attribute_shape[1] = 3; attribute_shape[2] = channels;
    output_shape[0] = batch_count; output_shape[1] = height; output_shape[2] = width;
    output_shape[3] = neighbors; output_shape[4] = channels;
    fragment_strides[3] = 1; fragment_strides[2] = neighbors;
    fragment_strides[1] = width * neighbors;
    fragment_strides[0] = height * width * neighbors;
    bary_strides[4] = 1; bary_strides[3] = 3; bary_strides[2] = neighbors * 3;
    bary_strides[1] = width * neighbors * 3;
    bary_strides[0] = height * width * neighbors * 3;
    attribute_strides[2] = 1; attribute_strides[1] = channels;
    attribute_strides[0] = 3 * channels;
    output_strides[4] = 1; output_strides[3] = channels;
    output_strides[2] = neighbors * channels;
    output_strides[1] = width * neighbors * channels;
    output_strides[0] = height * width * neighbors * channels;

    face_index_view = make_view((void *)face_index, GFFX_DTYPE_INT32, 4u, fragment_shape,
                                fragment_strides, GFFX_TENSOR_READ_ONLY);
    bary_view = make_view((void *)barycentric, dtype, 5u, bary_shape, bary_strides,
                          GFFX_TENSOR_READ_ONLY);
    attribute_view = make_view((void *)face_attributes, dtype, 3u, attribute_shape,
                               attribute_strides, GFFX_TENSOR_READ_ONLY);
    cotangent_view = make_view((void *)grad_attributes, dtype, 5u, output_shape, output_strides,
                               GFFX_TENSOR_READ_ONLY);
    grad_bary_view = make_view(grad_barycentric, dtype, 5u, bary_shape, bary_strides,
                               GFFX_TENSOR_OUTPUT);
    grad_attribute_view = make_view(grad_face_attributes, dtype, 3u, attribute_shape,
                                    attribute_strides, GFFX_TENSOR_OUTPUT);
    return gffx_render_interpolate_backward(
        &face_index_view, &bary_view, &attribute_view, &cotangent_view, &context,
        grad_barycentric != NULL ? &grad_bary_view : NULL,
        grad_face_attributes != NULL ? &grad_attribute_view : NULL, NULL, &diagnostic);
}

static int test_in01_in03_interpolation(gffx_dtype dtype) {
    enum { CHANNELS = 2 };
    /* Two fragments: one on face 0, one background. */
    static const int32_t face_index[2] = {0, -1};
    static const double barycentric[6] = {0.25, 0.5, 0.25,   0.0, 0.0, 0.0};
    static const double attributes[6] = {
        1.0, 10.0,   2.0, 20.0,   4.0, 40.0
    };
    double bd[6]; float bf[6];
    double ad[6]; float af[6];
    double od[4]; float of[4];
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;
    void *o = dtype == GFFX_DTYPE_FLOAT64 ? (void *)od : (void *)of;

    fill_double(b, dtype, barycentric, 6);
    fill_double(a, dtype, attributes, 6);
    CHECK(run_interpolate(face_index, b, a, 1, 1, 2, 1, 1, CHANNELS, dtype, o)
          == GFFX_STATUS_OK);
    /* 0.25*1 + 0.5*2 + 0.25*4 = 2.25; the second channel is ten times the first. */
    CHECK(get_double(o, dtype, 0) == 2.25);
    CHECK(get_double(o, dtype, 1) == 22.5);
    /* IN-02: the background fragment is exactly zero in every channel. */
    CHECK(get_double(o, dtype, 2) == 0.0);
    CHECK(get_double(o, dtype, 3) == 0.0);
    return 0;
}

static int test_in04_gradients(gffx_dtype dtype) {
    enum { CHANNELS = 2 };
    /* Two fragments selecting the same face, so attribute gradients accumulate. */
    static const int32_t face_index[3] = {0, 0, -1};
    static const double barycentric[9] = {
        0.25, 0.5, 0.25,   0.5, 0.25, 0.25,   0.0, 0.0, 0.0
    };
    static const double attributes[6] = {1.0, 10.0, 2.0, 20.0, 4.0, 40.0};
    static const double cotangent[6] = {1.0, 0.5,   2.0, 0.25,   8.0, 8.0};
    double expected_bary[9] = {0};
    double expected_attributes[6] = {0};
    double bd[9]; float bf[9];
    double ad[6]; float af[6];
    double cd[6]; float cf[6];
    double gbd[9]; float gbf[9];
    double gad[6]; float gaf[6];
    int64_t fragment;
    int64_t corner;
    int64_t channel;
    void *b = dtype == GFFX_DTYPE_FLOAT64 ? (void *)bd : (void *)bf;
    void *a = dtype == GFFX_DTYPE_FLOAT64 ? (void *)ad : (void *)af;
    void *c = dtype == GFFX_DTYPE_FLOAT64 ? (void *)cd : (void *)cf;
    void *gb = dtype == GFFX_DTYPE_FLOAT64 ? (void *)gbd : (void *)gbf;
    void *ga = dtype == GFFX_DTYPE_FLOAT64 ? (void *)gad : (void *)gaf;
    const double tolerance = dtype == GFFX_DTYPE_FLOAT64 ? 1e-12 : 1e-5;

    fill_double(b, dtype, barycentric, 9);
    fill_double(a, dtype, attributes, 6);
    fill_double(c, dtype, cotangent, 6);

    /* The map is bilinear, so both gradients are computable in closed form. */
    for (fragment = 0; fragment < 2; ++fragment) {
        for (corner = 0; corner < 3; ++corner) {
            for (channel = 0; channel < CHANNELS; ++channel) {
                double weight = barycentric[fragment * 3 + corner];
                double g = cotangent[fragment * CHANNELS + channel];
                expected_bary[fragment * 3 + corner] +=
                    attributes[corner * CHANNELS + channel] * g;
                expected_attributes[corner * CHANNELS + channel] += weight * g;
            }
        }
    }

    CHECK(run_interpolate_backward(face_index, b, a, c, 1, 1, 3, 1, 1, CHANNELS, dtype, gb, ga)
          == GFFX_STATUS_OK);
    for (fragment = 0; fragment < 9; ++fragment) {
        CHECK(fabs(get_double(gb, dtype, fragment) - expected_bary[fragment]) <= tolerance);
    }
    for (fragment = 0; fragment < 6; ++fragment) {
        CHECK(fabs(get_double(ga, dtype, fragment) - expected_attributes[fragment]) <= tolerance);
    }
    return 0;
}

static int test_in05_validation(void) {
    static const int32_t face_index[1] = {0};
    static const double barycentric[3] = {0.25, 0.5, 0.25};
    static const double attributes[3] = {1.0, 2.0, 4.0};
    static const int32_t out_of_range[1] = {5};
    double output[1];
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    /* A fragment naming a face outside the attribute range is rejected. */
    CHECK(run_interpolate(out_of_range, barycentric, attributes, 1, 1, 1, 1, 1, 1,
                          GFFX_DTYPE_FLOAT64, output) == GFFX_STATUS_INVALID_ARGUMENT);
    /* The valid case still succeeds. */
    CHECK(run_interpolate(face_index, barycentric, attributes, 1, 1, 1, 1, 1, 1,
                          GFFX_DTYPE_FLOAT64, output) == GFFX_STATUS_OK);
    CHECK(output[0] == 2.25);

    CHECK(gffx_render_interpolate_workspace(4, 3, GFFX_DTYPE_FLOAT64, &context, &required_bytes,
                                            &required_alignment, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);
    return 0;
}

int main(void) {
    int result;
    gffx_dtype dtypes[2] = {GFFX_DTYPE_FLOAT32, GFFX_DTYPE_FLOAT64};
    size_t index;

    for (index = 0u; index < 2u; ++index) {
        gffx_dtype dtype = dtypes[index];
        result = test_ra01_full_coverage(dtype); if (result != 0) return result;
        result = test_ra02_background(dtype); if (result != 0) return result;
        result = test_ra03_ra04_depth_ordering(dtype); if (result != 0) return result;
        result = test_in01_in03_interpolation(dtype); if (result != 0) return result;
        result = test_in04_gradients(dtype); if (result != 0) return result;
    }
    result = test_ra05_ra06_culling_and_degenerate(); if (result != 0) return result;
    result = test_ra07_signed_distance_and_blur(); if (result != 0) return result;
    result = test_ra08_ra09_orientation_and_centres(); if (result != 0) return result;
    result = test_ra10_gradients(); if (result != 0) return result;
    result = test_ra11_ra12_determinism_and_batching(); if (result != 0) return result;
    result = test_ra13_ra14_validation_and_workspace(); if (result != 0) return result;
    result = test_in05_validation(); if (result != 0) return result;
    return 0;
}
