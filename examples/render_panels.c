/*
 * A six-panel visual survey of the Phase 2 CPU kernels.
 *
 * Developer demo only: not part of the library surface, installed by no packaging rule,
 * registered as no test, and advertising no operation. It exists so each kernel family can be
 * seen doing something distinct rather than only asserted in a test.
 *
 *   panel 1  smooth shading      mesh.vertex_normals area weighted
 *   panel 2  flat shading        mesh.face_geometry normals, same geometry, visibly faceted
 *   panel 3  soft silhouette     render.rasterize blur radius and signed distance
 *   panel 4  depth ordering      two interpenetrating spheres coloured by winning face
 *   panel 5  surface sampling    mesh.sample_surface points splatted through the camera
 *   panel 6  distance field      points.closest_point_on_mesh over a slice plane
 */

#include <gffx/mesh.h>
#include <gffx/points.h>
#include <gffx/render.h>
#include <gffx/transforms.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PANEL 200
#define COLUMNS 3
#define ROWS 2
#define IMAGE_W (PANEL * COLUMNS)
#define IMAGE_H (PANEL * ROWS)

static const int64_t pair_strides[2] = {3, 1};
static const int64_t quad_strides[2] = {4, 1};
static const int64_t scalar_strides[1] = {1};
static const int64_t matrix_strides[3] = {16, 4, 1};
static const int64_t triple_strides[3] = {9, 3, 1};

static gffx_execution_context cpu_context(void) {
    gffx_execution_context context = {0};
    context.struct_size = (uint32_t)sizeof(context);
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    return context;
}

static gffx_tensor_view view_of(
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
    view.flags = flags;
    return view;
}

/* ------------------------------------------------------------------ icosphere builder */

typedef struct sphere {
    double *positions;
    int32_t *faces;
    int64_t vertex_count;
    int64_t face_count;
} sphere;

typedef struct midpoint_cache {
    int32_t *keys_a;
    int32_t *keys_b;
    int32_t *values;
    int64_t count;
} midpoint_cache;

static int32_t midpoint_of(
    sphere *s, midpoint_cache *cache, int32_t a, int32_t b
) {
    int32_t low = a < b ? a : b;
    int32_t high = a < b ? b : a;
    int64_t index;
    double x, y, z, length;
    for (index = 0; index < cache->count; ++index) {
        if (cache->keys_a[index] == low && cache->keys_b[index] == high) {
            return cache->values[index];
        }
    }
    x = (s->positions[low * 3 + 0] + s->positions[high * 3 + 0]) * 0.5;
    y = (s->positions[low * 3 + 1] + s->positions[high * 3 + 1]) * 0.5;
    z = (s->positions[low * 3 + 2] + s->positions[high * 3 + 2]) * 0.5;
    length = sqrt(x * x + y * y + z * z);
    s->positions[s->vertex_count * 3 + 0] = x / length;
    s->positions[s->vertex_count * 3 + 1] = y / length;
    s->positions[s->vertex_count * 3 + 2] = z / length;
    cache->keys_a[cache->count] = low;
    cache->keys_b[cache->count] = high;
    cache->values[cache->count] = (int32_t)s->vertex_count;
    cache->count += 1;
    s->vertex_count += 1;
    return cache->values[cache->count - 1];
}

/* Icosahedron subdivided twice and projected to the unit sphere: 162 vertices, 320 faces. */
static int build_icosphere(sphere *out) {
    /* Golden ratio, written as a literal so the table is a constant initializer. */
    static const double base[36] = {
        -1, 1.6180339887498949, 0,    1, 1.6180339887498949, 0,
        -1, -1.6180339887498949, 0,   1, -1.6180339887498949, 0,
        0, -1, 1.6180339887498949,    0, 1, 1.6180339887498949,
        0, -1, -1.6180339887498949,   0, 1, -1.6180339887498949,
        1.6180339887498949, 0, -1,    1.6180339887498949, 0, 1,
        -1.6180339887498949, 0, -1,   -1.6180339887498949, 0, 1
    };
    static const int32_t base_faces[60] = {
        0, 11, 5,  0, 5, 1,   0, 1, 7,   0, 7, 10,  0, 10, 11,
        1, 5, 9,   5, 11, 4,  11, 10, 2, 10, 7, 6,  7, 1, 8,
        3, 9, 4,   3, 4, 2,   3, 2, 6,   3, 6, 8,   3, 8, 9,
        4, 9, 5,   2, 4, 11,  6, 2, 10,  8, 6, 7,   9, 8, 1
    };
    midpoint_cache cache;
    int32_t *next_faces;
    int level;
    int64_t index;

    out->positions = (double *)malloc(200 * 3 * sizeof(double));
    out->faces = (int32_t *)malloc(400 * 3 * sizeof(int32_t));
    next_faces = (int32_t *)malloc(400 * 3 * sizeof(int32_t));
    cache.keys_a = (int32_t *)malloc(1200 * sizeof(int32_t));
    cache.keys_b = (int32_t *)malloc(1200 * sizeof(int32_t));
    cache.values = (int32_t *)malloc(1200 * sizeof(int32_t));
    if (out->positions == NULL || out->faces == NULL || next_faces == NULL ||
        cache.keys_a == NULL || cache.keys_b == NULL || cache.values == NULL) {
        return 1;
    }
    for (index = 0; index < 12; ++index) {
        double x = base[index * 3 + 0], y = base[index * 3 + 1], z = base[index * 3 + 2];
        double length = sqrt(x * x + y * y + z * z);
        out->positions[index * 3 + 0] = x / length;
        out->positions[index * 3 + 1] = y / length;
        out->positions[index * 3 + 2] = z / length;
    }
    memcpy(out->faces, base_faces, sizeof(base_faces));
    out->vertex_count = 12;
    out->face_count = 20;

    for (level = 0; level < 2; ++level) {
        int64_t face;
        int64_t produced = 0;
        cache.count = 0;
        for (face = 0; face < out->face_count; ++face) {
            int32_t a = out->faces[face * 3 + 0];
            int32_t b = out->faces[face * 3 + 1];
            int32_t c = out->faces[face * 3 + 2];
            int32_t ab = midpoint_of(out, &cache, a, b);
            int32_t bc = midpoint_of(out, &cache, b, c);
            int32_t ca = midpoint_of(out, &cache, c, a);
            int32_t quad[12] = {a, ab, ca,  b, bc, ab,  c, ca, bc,  ab, bc, ca};
            memcpy(next_faces + produced * 3, quad, sizeof(quad));
            produced += 4;
        }
        memcpy(out->faces, next_faces, (size_t)produced * 3 * sizeof(int32_t));
        out->face_count = produced;
    }
    free(next_faces);
    free(cache.keys_a);
    free(cache.keys_b);
    free(cache.values);
    return 0;
}

static void rotate_translate(
    const double *source, int64_t count, double ax, double ay, double dx, double dz, double *out
) {
    double cx = cos(ax), sx = sin(ax), cy = cos(ay), sy = sin(ay);
    int64_t index;
    for (index = 0; index < count; ++index) {
        double x = source[index * 3 + 0], y = source[index * 3 + 1], z = source[index * 3 + 2];
        double y1 = y * cx - z * sx;
        double z1 = y * sx + z * cx;
        double x2 = x * cy + z1 * sy;
        double z2 = -x * sy + z1 * cy;
        out[index * 3 + 0] = x2 + dx;
        out[index * 3 + 1] = y1;
        out[index * 3 + 2] = z2 + dz;
    }
}

static void projection_matrix(double focal, double size, double *m) {
    int index;
    for (index = 0; index < 16; ++index) m[index] = 0.0;
    m[0] = 2.0 * focal / size;
    m[5] = 2.0 * focal / size;
    m[10] = (12.0 + 0.1) / (0.1 - 12.0);
    m[11] = (2.0 * 12.0 * 0.1) / (0.1 - 12.0);
    m[14] = -1.0;
}

/* --------------------------------------------------------------------- panel plotting */

static unsigned char *canvas;

static void put_pixel(int panel_index, int64_t row, int64_t column, double r, double g, double b) {
    int64_t px = (panel_index % COLUMNS) * PANEL + column;
    int64_t py = (panel_index / COLUMNS) * PANEL + row;
    int64_t offset;
    if (row < 0 || row >= PANEL || column < 0 || column >= PANEL) return;
    offset = (py * IMAGE_W + px) * 3;
    canvas[offset + 0] = (unsigned char)(r < 0 ? 0 : (r > 1 ? 255 : r * 255.0));
    canvas[offset + 1] = (unsigned char)(g < 0 ? 0 : (g > 1 ? 255 : g * 255.0));
    canvas[offset + 2] = (unsigned char)(b < 0 ? 0 : (b > 1 ? 255 : b * 255.0));
}

/* Projects world vertices to NDC through the shared camera. */
static gffx_status project(
    const double *world, int64_t vertex_count, double *ndc, double focal
) {
    static double matrix[16];
    double *homogeneous = (double *)malloc((size_t)vertex_count * 4 * sizeof(double));
    uint8_t *valid = (uint8_t *)malloc((size_t)vertex_count);
    int32_t offsets[2];
    int64_t vertex_shape[2];
    int64_t homogeneous_shape[2];
    int64_t valid_shape[1];
    int64_t offset_shape[1] = {2};
    int64_t matrix_shape[3] = {1, 4, 4};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_status status;
    gffx_tensor_view world_view, matrix_view, offsets_view, homogeneous_view, ndc_view,
        valid_view, homogeneous_input;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    offsets[0] = 0; offsets[1] = (int32_t)vertex_count;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    homogeneous_shape[0] = vertex_count; homogeneous_shape[1] = 4;
    valid_shape[0] = vertex_count;
    projection_matrix(focal, (double)PANEL, matrix);

    world_view = view_of((void *)world, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                         GFFX_TENSOR_READ_ONLY);
    matrix_view = view_of(matrix, GFFX_DTYPE_FLOAT64, 3u, matrix_shape, matrix_strides,
                          GFFX_TENSOR_READ_ONLY);
    offsets_view = view_of(offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                           GFFX_TENSOR_READ_ONLY);
    homogeneous_view = view_of(homogeneous, GFFX_DTYPE_FLOAT64, 2u, homogeneous_shape,
                               quad_strides, GFFX_TENSOR_OUTPUT);
    status = gffx_transforms_transform_points(&world_view, &matrix_view, &offsets_view,
                                              &context, &homogeneous_view, NULL, &diagnostic);
    if (status == GFFX_STATUS_OK) {
        homogeneous_input = view_of(homogeneous, GFFX_DTYPE_FLOAT64, 2u, homogeneous_shape,
                                    quad_strides, GFFX_TENSOR_READ_ONLY);
        ndc_view = view_of(ndc, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
        valid_view = view_of(valid, GFFX_DTYPE_BOOL, 1u, valid_shape, scalar_strides,
                             GFFX_TENSOR_OUTPUT);
        status = gffx_transforms_perspective_divide(&homogeneous_input, 1e-9, &context,
                                                    &ndc_view, &valid_view, NULL, &diagnostic);
    }
    free(homogeneous);
    free(valid);
    return status;
}

/* Rasterizes and returns the fragment buffers; caller frees. */
static gffx_status rasterize_panel(
    const double *ndc, int64_t vertex_count, const int32_t *faces, int64_t face_count,
    double blur, uint32_t cull, int32_t **out_index, double **out_bary, double **out_distance
) {
    int32_t vertex_offsets[2];
    int32_t face_offsets[2];
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t offset_shape[1] = {2};
    int64_t fragment_shape[4] = {1, PANEL, PANEL, 1};
    int64_t bary_shape[5] = {1, PANEL, PANEL, 1, 3};
    int64_t fragment_strides[4] = {PANEL * PANEL, PANEL, 1, 1};
    int64_t bary_strides[5] = {PANEL * PANEL * 3, PANEL * 3, 3, 3, 1};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    double *depth = (double *)malloc(PANEL * PANEL * sizeof(double));
    gffx_tensor_view ndc_view, faces_view, vertex_offsets_view, face_offsets_view,
        index_view, bary_view, depth_view, distance_view;
    gffx_status status;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    *out_index = (int32_t *)malloc(PANEL * PANEL * sizeof(int32_t));
    *out_bary = (double *)malloc(PANEL * PANEL * 3 * sizeof(double));
    *out_distance = (double *)malloc(PANEL * PANEL * sizeof(double));
    vertex_offsets[0] = 0; vertex_offsets[1] = (int32_t)vertex_count;
    face_offsets[0] = 0; face_offsets[1] = (int32_t)face_count;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;

    ndc_view = view_of((void *)ndc, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                       GFFX_TENSOR_READ_ONLY);
    faces_view = view_of((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                         GFFX_TENSOR_READ_ONLY);
    vertex_offsets_view = view_of(vertex_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);
    face_offsets_view = view_of(face_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                scalar_strides, GFFX_TENSOR_READ_ONLY);
    index_view = view_of(*out_index, GFFX_DTYPE_INT32, 4u, fragment_shape, fragment_strides,
                         GFFX_TENSOR_OUTPUT);
    bary_view = view_of(*out_bary, GFFX_DTYPE_FLOAT64, 5u, bary_shape, bary_strides,
                        GFFX_TENSOR_OUTPUT);
    depth_view = view_of(depth, GFFX_DTYPE_FLOAT64, 4u, fragment_shape, fragment_strides,
                         GFFX_TENSOR_OUTPUT);
    distance_view = view_of(*out_distance, GFFX_DTYPE_FLOAT64, 4u, fragment_shape,
                            fragment_strides, GFFX_TENSOR_OUTPUT);
    status = gffx_render_rasterize(&ndc_view, &faces_view, &vertex_offsets_view,
                                   &face_offsets_view, PANEL, PANEL, 1, blur, cull, 1e-12,
                                   &context, &index_view, &bary_view, &depth_view,
                                   &distance_view, NULL, &diagnostic);
    free(depth);
    return status;
}

/* Interpolates [F,3,3] corner attributes across the fragments. */
static gffx_status interpolate_corners(
    const int32_t *face_index, const double *bary, const double *corners, int64_t face_count,
    double *out
) {
    int64_t fragment_shape[4] = {1, PANEL, PANEL, 1};
    int64_t bary_shape[5] = {1, PANEL, PANEL, 1, 3};
    int64_t out_shape[5] = {1, PANEL, PANEL, 1, 3};
    int64_t fragment_strides[4] = {PANEL * PANEL, PANEL, 1, 1};
    int64_t bary_strides[5] = {PANEL * PANEL * 3, PANEL * 3, 3, 3, 1};
    int64_t corner_shape[3];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view index_view, bary_view, corner_view, out_view;
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    corner_shape[0] = face_count; corner_shape[1] = 3; corner_shape[2] = 3;
    index_view = view_of((void *)face_index, GFFX_DTYPE_INT32, 4u, fragment_shape,
                         fragment_strides, GFFX_TENSOR_READ_ONLY);
    bary_view = view_of((void *)bary, GFFX_DTYPE_FLOAT64, 5u, bary_shape, bary_strides,
                        GFFX_TENSOR_READ_ONLY);
    corner_view = view_of((void *)corners, GFFX_DTYPE_FLOAT64, 3u, corner_shape,
                          triple_strides, GFFX_TENSOR_READ_ONLY);
    out_view = view_of(out, GFFX_DTYPE_FLOAT64, 5u, out_shape, bary_strides,
                       GFFX_TENSOR_OUTPUT);
    return gffx_render_interpolate(&index_view, &bary_view, &corner_view, &context, &out_view,
                                   NULL, &diagnostic);
}

static void shade_normals(int panel_index, const int32_t *face_index, const double *normals,
                          double tint_r, double tint_g, double tint_b) {
    static const double light[3] = {0.4082482904638631, 0.4082482904638631, 0.8164965809277261};
    int64_t row, column;
    for (row = 0; row < PANEL; ++row) {
        for (column = 0; column < PANEL; ++column) {
            int64_t entry = row * PANEL + column;
            double nx, ny, nz, length, lambert, value;
            if (face_index[entry] < 0) {
                put_pixel(panel_index, row, column, 0.09, 0.10, 0.13);
                continue;
            }
            nx = normals[entry * 3 + 0];
            ny = normals[entry * 3 + 1];
            nz = normals[entry * 3 + 2];
            length = sqrt(nx * nx + ny * ny + nz * nz);
            if (length <= 0.0) { put_pixel(panel_index, row, column, 0.09, 0.10, 0.13); continue; }
            lambert = (nx * light[0] + ny * light[1] + nz * light[2]) / length;
            if (lambert < 0.0) lambert = 0.0;
            value = 0.12 + 0.88 * lambert;
            put_pixel(panel_index, row, column, value * tint_r, value * tint_g, value * tint_b);
        }
    }
}

int main(void) {
    sphere mesh;
    double *world = NULL;
    double *ndc = NULL;
    double *vertex_normals = NULL;
    double *face_normals = NULL;
    double *corner_smooth = NULL;
    double *corner_flat = NULL;
    double *pixel_normals = NULL;
    int32_t *face_index = NULL;
    double *bary = NULL;
    double *distance = NULL;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_status status;
    int64_t index, row, column;
    FILE *file;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    if (build_icosphere(&mesh) != 0) return 1;
    printf("  icosphere: %lld vertices, %lld faces\n",
           (long long)mesh.vertex_count, (long long)mesh.face_count);

    canvas = (unsigned char *)malloc((size_t)IMAGE_W * IMAGE_H * 3);
    world = (double *)malloc((size_t)mesh.vertex_count * 3 * sizeof(double) * 2);
    ndc = (double *)malloc((size_t)mesh.vertex_count * 3 * sizeof(double) * 2);
    vertex_normals = (double *)malloc((size_t)mesh.vertex_count * 3 * sizeof(double));
    face_normals = (double *)malloc((size_t)mesh.face_count * 3 * sizeof(double));
    corner_smooth = (double *)malloc((size_t)mesh.face_count * 9 * sizeof(double));
    corner_flat = (double *)malloc((size_t)mesh.face_count * 9 * sizeof(double));
    pixel_normals = (double *)malloc(PANEL * PANEL * 3 * sizeof(double));
    if (canvas == NULL || world == NULL || ndc == NULL || vertex_normals == NULL ||
        face_normals == NULL || corner_smooth == NULL || corner_flat == NULL ||
        pixel_normals == NULL) {
        return 1;
    }
    memset(canvas, 20, (size_t)IMAGE_W * IMAGE_H * 3);
    rotate_translate(mesh.positions, mesh.vertex_count, 0.5, 0.8, 0.0, -3.2, world);

    {
        int64_t vertex_shape[2] = {mesh.vertex_count, 3};
        int64_t face_shape[2] = {mesh.face_count, 3};
        int64_t corner_shape[3] = {mesh.face_count, 3, 3};
        int64_t area_shape[1] = {mesh.face_count};
        gffx_tensor_view world_view = view_of(world, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                              pair_strides, GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view faces_view = view_of(mesh.faces, GFFX_DTYPE_INT32, 2u, face_shape,
                                              pair_strides, GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view normals_view = view_of(vertex_normals, GFFX_DTYPE_FLOAT64, 2u,
                                                vertex_shape, pair_strides, GFFX_TENSOR_OUTPUT);
        double *areas = (double *)malloc((size_t)mesh.face_count * sizeof(double));
        uint8_t *valid = (uint8_t *)malloc((size_t)mesh.face_count);
        gffx_tensor_view face_normal_view = view_of(face_normals, GFFX_DTYPE_FLOAT64, 2u,
                                                    face_shape, pair_strides,
                                                    GFFX_TENSOR_OUTPUT);
        gffx_tensor_view area_view = view_of(areas, GFFX_DTYPE_FLOAT64, 1u, area_shape,
                                             scalar_strides, GFFX_TENSOR_OUTPUT);
        gffx_tensor_view valid_view = view_of(valid, GFFX_DTYPE_BOOL, 1u, area_shape,
                                              scalar_strides, GFFX_TENSOR_OUTPUT);
        gffx_tensor_view corner_view = view_of(corner_smooth, GFFX_DTYPE_FLOAT64, 3u,
                                               corner_shape, triple_strides,
                                               GFFX_TENSOR_OUTPUT);
        gffx_tensor_view normals_input;

        status = gffx_mesh_vertex_normals(&world_view, &faces_view, 1e-12,
                                          GFFX_MESH_WEIGHTING_AREA, &context, &normals_view,
                                          NULL, &diagnostic);
        if (status != GFFX_STATUS_OK) { printf("vertex_normals %u\n", status); return 1; }
        status = gffx_mesh_face_geometry(&world_view, &faces_view, 1e-12, &context,
                                         &face_normal_view, &area_view, &valid_view, NULL,
                                         &diagnostic);
        if (status != GFFX_STATUS_OK) { printf("face_geometry %u\n", status); return 1; }
        normals_input = view_of(vertex_normals, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                pair_strides, GFFX_TENSOR_READ_ONLY);
        status = gffx_mesh_gather_faces(&normals_input, &faces_view, &context, &corner_view,
                                        NULL, &diagnostic);
        if (status != GFFX_STATUS_OK) { printf("gather_faces %u\n", status); return 1; }
        /* Flat shading repeats each face normal at all three corners. */
        for (index = 0; index < mesh.face_count; ++index) {
            int corner;
            for (corner = 0; corner < 3; ++corner) {
                corner_flat[index * 9 + corner * 3 + 0] = face_normals[index * 3 + 0];
                corner_flat[index * 9 + corner * 3 + 1] = face_normals[index * 3 + 1];
                corner_flat[index * 9 + corner * 3 + 2] = face_normals[index * 3 + 2];
            }
        }
        free(areas);
        free(valid);
    }

    if (project(world, mesh.vertex_count, ndc, PANEL * 0.85) != GFFX_STATUS_OK) return 1;

    /* Panel 1: smooth. Panel 2: flat. */
    if (rasterize_panel(ndc, mesh.vertex_count, mesh.faces, mesh.face_count, 0.0,
                        GFFX_CULL_BACK, &face_index, &bary, &distance) != GFFX_STATUS_OK) {
        return 1;
    }
    if (interpolate_corners(face_index, bary, corner_smooth, mesh.face_count,
                            pixel_normals) != GFFX_STATUS_OK) return 1;
    shade_normals(0, face_index, pixel_normals, 1.0, 0.93, 0.80);
    if (interpolate_corners(face_index, bary, corner_flat, mesh.face_count,
                            pixel_normals) != GFFX_STATUS_OK) return 1;
    shade_normals(1, face_index, pixel_normals, 0.80, 0.90, 1.0);

    /* Panel 3: soft silhouette from the blur radius and signed distance. */
    {
        int32_t *soft_index;
        double *soft_bary;
        double *soft_distance;
        if (rasterize_panel(ndc, mesh.vertex_count, mesh.faces, mesh.face_count, 14.0,
                            GFFX_CULL_BACK, &soft_index, &soft_bary,
                            &soft_distance) != GFFX_STATUS_OK) return 1;
        for (row = 0; row < PANEL; ++row) {
            for (column = 0; column < PANEL; ++column) {
                int64_t entry = row * PANEL + column;
                double sd, alpha;
                if (soft_index[entry] < 0) {
                    put_pixel(2, row, column, 0.09, 0.10, 0.13);
                    continue;
                }
                sd = soft_distance[entry];
                /* Negative inside, positive outside; map to an alpha ramp over the blur band. */
                alpha = sd <= 0.0 ? 1.0 : 1.0 - sqrt(sd) / 14.0;
                if (alpha < 0.0) alpha = 0.0;
                put_pixel(2, row, column,
                          0.09 + alpha * 0.85, 0.10 + alpha * 0.45, 0.13 + alpha * 0.75);
            }
        }
        free(soft_index); free(soft_bary); free(soft_distance);
    }

    /* Panel 4: two interpenetrating spheres in one element, coloured by winning face. */
    {
        double *pair_world = (double *)malloc((size_t)mesh.vertex_count * 6 * sizeof(double));
        double *pair_ndc = (double *)malloc((size_t)mesh.vertex_count * 6 * sizeof(double));
        int32_t *pair_faces = (int32_t *)malloc((size_t)mesh.face_count * 6 * sizeof(int32_t));
        int32_t *pair_index; double *pair_bary; double *pair_distance;
        rotate_translate(mesh.positions, mesh.vertex_count, 0.5, 0.8, -0.45, -3.2, pair_world);
        rotate_translate(mesh.positions, mesh.vertex_count, 1.1, 2.0, 0.45, -3.0,
                         pair_world + mesh.vertex_count * 3);
        memcpy(pair_faces, mesh.faces, (size_t)mesh.face_count * 3 * sizeof(int32_t));
        for (index = 0; index < mesh.face_count * 3; ++index) {
            pair_faces[mesh.face_count * 3 + index] =
                mesh.faces[index] + (int32_t)mesh.vertex_count;
        }
        if (project(pair_world, mesh.vertex_count * 2, pair_ndc, PANEL * 0.85) !=
            GFFX_STATUS_OK) return 1;
        if (rasterize_panel(pair_ndc, mesh.vertex_count * 2, pair_faces, mesh.face_count * 2,
                            0.0, GFFX_CULL_BACK, &pair_index, &pair_bary,
                            &pair_distance) != GFFX_STATUS_OK) return 1;
        for (row = 0; row < PANEL; ++row) {
            for (column = 0; column < PANEL; ++column) {
                int64_t entry = row * PANEL + column;
                int32_t face = pair_index[entry];
                double shade;
                if (face < 0) { put_pixel(3, row, column, 0.09, 0.10, 0.13); continue; }
                /* Which sphere won the depth test is directly visible in the face index. */
                shade = 0.35 + 0.65 * ((double)((face * 37) % 64) / 64.0);
                if (face < (int32_t)mesh.face_count) {
                    put_pixel(3, row, column, shade * 1.0, shade * 0.55, shade * 0.35);
                } else {
                    put_pixel(3, row, column, shade * 0.35, shade * 0.75, shade * 1.0);
                }
            }
        }
        free(pair_world); free(pair_ndc); free(pair_faces);
        free(pair_index); free(pair_bary); free(pair_distance);
    }

    /* Panel 5: area-weighted surface sampling, projected and splatted. */
    {
        enum { SAMPLES = 6000 };
        double *points = (double *)malloc(SAMPLES * 3 * sizeof(double));
        double *sample_bary = (double *)malloc(SAMPLES * 3 * sizeof(double));
        double *sample_ndc = (double *)malloc(SAMPLES * 3 * sizeof(double));
        int32_t *sample_face = (int32_t *)malloc(SAMPLES * sizeof(int32_t));
        double *workspace = (double *)malloc((size_t)mesh.face_count * sizeof(double));
        uint32_t key[2] = {0x9E3779B9u, 0x243F6A88u};
        uint32_t counter[2] = {0u, 0u};
        uint32_t next_counter[2];
        int64_t vertex_shape[2] = {mesh.vertex_count, 3};
        int64_t face_shape[2] = {mesh.face_count, 3};
        int64_t offset_shape[1] = {2};
        int64_t rng_shape[1] = {2};
        int64_t point_shape[3] = {1, SAMPLES, 3};
        int64_t index_shape[2] = {1, SAMPLES};
        int64_t point_strides[3] = {SAMPLES * 3, 3, 1};
        int64_t index_strides[2] = {SAMPLES, 1};
        int32_t vertex_offsets[2] = {0, 0};
        int32_t face_offsets[2] = {0, 0};
        gffx_buffer workspace_buffer = {0};
        gffx_tensor_view world_view, faces_view, vo_view, fo_view, key_view, counter_view,
            points_view, index_view, bary_view, next_view;
        vertex_offsets[1] = (int32_t)mesh.vertex_count;
        face_offsets[1] = (int32_t)mesh.face_count;
        workspace_buffer.struct_size = (uint32_t)sizeof(workspace_buffer);
        workspace_buffer.abi_version = GFFX_ABI_VERSION;
        workspace_buffer.data = workspace;
        workspace_buffer.capacity_bytes = (uint64_t)mesh.face_count * sizeof(double);
        workspace_buffer.alignment = 8u;
        workspace_buffer.device_type = GFFX_DEVICE_CPU;

        world_view = view_of(world, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                             GFFX_TENSOR_READ_ONLY);
        faces_view = view_of(mesh.faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                             GFFX_TENSOR_READ_ONLY);
        vo_view = view_of(vertex_offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                          GFFX_TENSOR_READ_ONLY);
        fo_view = view_of(face_offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                          GFFX_TENSOR_READ_ONLY);
        key_view = view_of(key, GFFX_DTYPE_UINT32, 1u, rng_shape, scalar_strides,
                           GFFX_TENSOR_READ_ONLY);
        counter_view = view_of(counter, GFFX_DTYPE_UINT32, 1u, rng_shape, scalar_strides,
                               GFFX_TENSOR_READ_ONLY);
        points_view = view_of(points, GFFX_DTYPE_FLOAT64, 3u, point_shape, point_strides,
                              GFFX_TENSOR_OUTPUT);
        index_view = view_of(sample_face, GFFX_DTYPE_INT32, 2u, index_shape, index_strides,
                             GFFX_TENSOR_OUTPUT);
        bary_view = view_of(sample_bary, GFFX_DTYPE_FLOAT64, 3u, point_shape, point_strides,
                            GFFX_TENSOR_OUTPUT);
        next_view = view_of(next_counter, GFFX_DTYPE_UINT32, 1u, rng_shape, scalar_strides,
                            GFFX_TENSOR_OUTPUT);
        status = gffx_mesh_sample_surface(&world_view, &faces_view, &vo_view, &fo_view, SAMPLES,
                                          &key_view, &counter_view, 1e-12, &context,
                                          &points_view, &index_view, &bary_view, &next_view,
                                          &workspace_buffer, &diagnostic);
        if (status != GFFX_STATUS_OK) { printf("sample_surface %u\n", status); return 1; }
        if (project(points, SAMPLES, sample_ndc, PANEL * 0.85) != GFFX_STATUS_OK) return 1;
        for (row = 0; row < PANEL; ++row) {
            for (column = 0; column < PANEL; ++column) put_pixel(4, row, column, 0.09, 0.10, 0.13);
        }
        for (index = 0; index < SAMPLES; ++index) {
            double sx = (sample_ndc[index * 3 + 0] + 1.0) * PANEL * 0.5;
            double sy = (1.0 - sample_ndc[index * 3 + 1]) * PANEL * 0.5;
            /* Depth-tint the dots so the near hemisphere reads brighter. */
            double near = 1.0 - (sample_ndc[index * 3 + 2] + 1.0) * 0.5;
            double value = 0.35 + 0.65 * near;
            put_pixel(4, (int64_t)sy, (int64_t)sx, value * 0.65, value * 1.0, value * 0.72);
        }
        free(points); free(sample_bary); free(sample_ndc); free(sample_face); free(workspace);
    }

    /* Panel 6: distance field over a slice plane through the sphere. */
    {
        enum { GRID = 100 };
        int64_t query_count = GRID * GRID;
        double *queries = (double *)malloc((size_t)query_count * 3 * sizeof(double));
        double *dist2 = (double *)malloc((size_t)query_count * sizeof(double));
        double *closest = (double *)malloc((size_t)query_count * 3 * sizeof(double));
        double *cp_bary = (double *)malloc((size_t)query_count * 3 * sizeof(double));
        int32_t *cp_face = (int32_t *)malloc((size_t)query_count * sizeof(int32_t));
        uint8_t *cp_valid = (uint8_t *)malloc((size_t)query_count);
        int32_t point_offsets[2]; int32_t vertex_offsets[2]; int32_t face_offsets[2];
        int64_t query_shape[2]; int64_t vertex_shape[2]; int64_t face_shape[2];
        int64_t offset_shape[1] = {2}; int64_t scalar_shape[1];
        gffx_tensor_view q_view, v_view, f_view, po_view, vo_view, fo_view, d_view, fi_view,
            b_view, c_view, valid_view;
        int64_t gx, gy;
        for (gy = 0; gy < GRID; ++gy) {
            for (gx = 0; gx < GRID; ++gx) {
                int64_t entry = gy * GRID + gx;
                queries[entry * 3 + 0] = -2.0 + 4.0 * ((double)gx + 0.5) / (double)GRID;
                queries[entry * 3 + 1] = -2.0 + 4.0 * ((double)gy + 0.5) / (double)GRID;
                queries[entry * 3 + 2] = -3.2;   /* slice through the sphere centre */
            }
        }
        point_offsets[0] = 0; point_offsets[1] = (int32_t)query_count;
        vertex_offsets[0] = 0; vertex_offsets[1] = (int32_t)mesh.vertex_count;
        face_offsets[0] = 0; face_offsets[1] = (int32_t)mesh.face_count;
        query_shape[0] = query_count; query_shape[1] = 3;
        vertex_shape[0] = mesh.vertex_count; vertex_shape[1] = 3;
        face_shape[0] = mesh.face_count; face_shape[1] = 3;
        scalar_shape[0] = query_count;
        q_view = view_of(queries, GFFX_DTYPE_FLOAT64, 2u, query_shape, pair_strides,
                         GFFX_TENSOR_READ_ONLY);
        v_view = view_of(world, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                         GFFX_TENSOR_READ_ONLY);
        f_view = view_of(mesh.faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                         GFFX_TENSOR_READ_ONLY);
        po_view = view_of(point_offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                          GFFX_TENSOR_READ_ONLY);
        vo_view = view_of(vertex_offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                          GFFX_TENSOR_READ_ONLY);
        fo_view = view_of(face_offsets, GFFX_DTYPE_INT32, 1u, offset_shape, scalar_strides,
                          GFFX_TENSOR_READ_ONLY);
        d_view = view_of(dist2, GFFX_DTYPE_FLOAT64, 1u, scalar_shape, scalar_strides,
                         GFFX_TENSOR_OUTPUT);
        fi_view = view_of(cp_face, GFFX_DTYPE_INT32, 1u, scalar_shape, scalar_strides,
                          GFFX_TENSOR_OUTPUT);
        b_view = view_of(cp_bary, GFFX_DTYPE_FLOAT64, 2u, query_shape, pair_strides,
                         GFFX_TENSOR_OUTPUT);
        c_view = view_of(closest, GFFX_DTYPE_FLOAT64, 2u, query_shape, pair_strides,
                         GFFX_TENSOR_OUTPUT);
        valid_view = view_of(cp_valid, GFFX_DTYPE_BOOL, 1u, scalar_shape, scalar_strides,
                             GFFX_TENSOR_OUTPUT);
        status = gffx_points_closest_point_on_mesh(&q_view, &v_view, &f_view, &po_view,
                                                   &vo_view, &fo_view, 1e-12, &context, &d_view,
                                                   &fi_view, &b_view, &c_view, &valid_view,
                                                   NULL, &diagnostic);
        if (status != GFFX_STATUS_OK) { printf("closest_point %u\n", status); return 1; }
        for (row = 0; row < PANEL; ++row) {
            for (column = 0; column < PANEL; ++column) {
                int64_t sx = column * GRID / PANEL;
                int64_t sy = row * GRID / PANEL;
                double d = sqrt(dist2[sy * GRID + sx]);
                /* Banded colour ramp so the iso-contours of the distance field are visible. */
                double band = fmod(d * 9.0, 1.0);
                double base = 1.0 / (1.0 + d * 1.6);
                put_pixel(5, row, column,
                          0.10 + base * 0.9, 0.12 + base * 0.35 + band * 0.18,
                          0.16 + band * 0.55);
            }
        }
        free(queries); free(dist2); free(closest); free(cp_bary); free(cp_face); free(cp_valid);
    }

    file = fopen("gffx_panels.ppm", "wb");
    if (file != NULL) {
        fprintf(file, "P6\n%d %d\n255\n", IMAGE_W, IMAGE_H);
        fwrite(canvas, 1, (size_t)IMAGE_W * IMAGE_H * 3, file);
        fclose(file);
        printf("  wrote gffx_panels.ppm (%dx%d, six panels)\n", IMAGE_W, IMAGE_H);
    }
    free(face_index); free(bary); free(distance);
    free(world); free(ndc); free(vertex_normals); free(face_normals);
    free(corner_smooth); free(corner_flat); free(pixel_normals);
    free(mesh.positions); free(mesh.faces); free(canvas);
    return 0;
}
