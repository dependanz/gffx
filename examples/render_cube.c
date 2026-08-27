/*
 * A visible end-to-end demonstration of the Phase 2 CPU kernels.
 *
 * This is a developer demo, not part of the library surface and not installed by any packaging
 * rule. It lives outside native/core so it may use stdio freely; the dependency-inspection gate
 * governs the runtime scaffold, not this file. No GFFX operation is advertised by its existence.
 *
 * It drives six real kernels in one pipeline:
 *
 *   mesh.vertex_normals      smooth normals over the rotated cube
 *   mesh.gather_faces        per-face-corner normals for interpolation
 *   transforms.transform_points  world -> clip through a camera matrix
 *   transforms.perspective_divide  clip -> NDC with the guarded divide
 *   render.rasterize         fragments with depth ordering
 *   render.interpolate       per-pixel normals from the corner normals
 *
 * and shades the result with a single directional light. Output is ASCII for the terminal plus
 * a binary PPM that any image viewer opens.
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

#define CUBE_VERTICES 8
#define CUBE_FACES 12

static gffx_execution_context cpu_context(void) {
    gffx_execution_context context = {0};
    context.struct_size = (uint32_t)sizeof(context);
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    context.device_index = 0;
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
    view.device_index = 0;
    view.flags = flags;
    return view;
}

static const double cube_positions[CUBE_VERTICES * 3] = {
    -0.5, -0.5, -0.5,    0.5, -0.5, -0.5,    0.5,  0.5, -0.5,   -0.5,  0.5, -0.5,
    -0.5, -0.5,  0.5,    0.5, -0.5,  0.5,    0.5,  0.5,  0.5,   -0.5,  0.5,  0.5
};

/* Twelve triangles, each wound counter-clockwise seen from outside the cube. */
static const int32_t cube_faces[CUBE_FACES * 3] = {
    4, 5, 6,   4, 6, 7,      /* +z */
    1, 0, 3,   1, 3, 2,      /* -z */
    5, 1, 2,   5, 2, 6,      /* +x */
    0, 4, 7,   0, 7, 3,      /* -x */
    3, 7, 6,   3, 6, 2,      /* +y */
    0, 1, 5,   0, 5, 4       /* -y */
};

static void rotate_cube(double angle_x, double angle_y, double *out) {
    double cx = cos(angle_x), sx = sin(angle_x);
    double cy = cos(angle_y), sy = sin(angle_y);
    int index;
    for (index = 0; index < CUBE_VERTICES; ++index) {
        double x = cube_positions[index * 3 + 0];
        double y = cube_positions[index * 3 + 1];
        double z = cube_positions[index * 3 + 2];
        double y1 = y * cx - z * sx;
        double z1 = y * sx + z * cx;
        double x2 = x * cy + z1 * sy;
        double z2 = -x * sy + z1 * cy;
        out[index * 3 + 0] = x2;
        out[index * 3 + 1] = y1;
        out[index * 3 + 2] = z2 - 3.0;   /* push in front of a camera looking down -z */
    }
}

/* Projection matrix from pinhole intrinsics, per the project camera contract. */
static void projection_matrix(
    double fx, double fy, double cx, double cy,
    double width, double height, double near_plane, double far_plane, double *m
) {
    int index;
    for (index = 0; index < 16; ++index) m[index] = 0.0;
    m[0] = 2.0 * fx / width;
    m[2] = 1.0 - 2.0 * cx / width;
    m[5] = 2.0 * fy / height;
    m[6] = 2.0 * cy / height - 1.0;
    m[10] = (far_plane + near_plane) / (near_plane - far_plane);
    m[11] = (2.0 * far_plane * near_plane) / (near_plane - far_plane);
    m[14] = -1.0;
}

/* Runs the full pipeline at one resolution and fills `shade` with per-pixel light intensity in
 * [0,1], or a negative value where nothing was covered. */
static int render_scene(
    const double *world, int64_t height, int64_t width, double focal, double *shade
) {
    static const int32_t vertex_offsets[2] = {0, CUBE_VERTICES};
    static const int32_t face_offsets[2] = {0, CUBE_FACES};
    static const int64_t pair_strides[2] = {3, 1};
    static const int64_t quad_strides[2] = {4, 1};
    static const int64_t scalar_strides[1] = {1};
    static const int64_t matrix_strides[3] = {16, 4, 1};
    static const int64_t triple_strides[3] = {9, 3, 1};
    const double light[3] = {0.40824829046386307, 0.40824829046386307, 0.816496580927726};

    double normals[CUBE_VERTICES * 3];
    double corner_normals[CUBE_FACES * 3 * 3];
    double matrix[16];
    double homogeneous[CUBE_VERTICES * 4];
    double ndc[CUBE_VERTICES * 3];
    uint8_t divide_valid[CUBE_VERTICES];
    double workspace[CUBE_VERTICES * 3];

    int64_t vertex_shape[2] = {CUBE_VERTICES, 3};
    int64_t face_shape[2] = {CUBE_FACES, 3};
    int64_t offset_shape[1] = {2};
    int64_t matrix_shape[3] = {1, 4, 4};
    int64_t homogeneous_shape[2] = {CUBE_VERTICES, 4};
    int64_t valid_shape[1] = {CUBE_VERTICES};
    int64_t corner_shape[3] = {CUBE_FACES, 3, 3};

    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace_buffer = {0};
    gffx_status status;

    int64_t fragment_count = height * width;
    int64_t fragment_shape[4];
    int64_t bary_shape[5];
    int64_t attribute_shape[5];
    int64_t fragment_strides[4];
    int64_t bary_strides[5];
    int64_t attribute_strides[5];
    int32_t *face_index = NULL;
    double *barycentric = NULL;
    double *depth = NULL;
    double *distance = NULL;
    double *pixel_normals = NULL;
    int64_t index;
    int result = 0;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    workspace_buffer.struct_size = (uint32_t)sizeof(workspace_buffer);
    workspace_buffer.abi_version = GFFX_ABI_VERSION;
    workspace_buffer.data = workspace;
    workspace_buffer.capacity_bytes = sizeof(workspace);
    workspace_buffer.alignment = UINT64_C(8);
    workspace_buffer.device_type = GFFX_DEVICE_CPU;

    /* The demo allocates its fragment buffers with malloc, which the library core never does. */
    face_index = (int32_t *)malloc((size_t)fragment_count * sizeof(int32_t));
    barycentric = (double *)malloc((size_t)fragment_count * 3 * sizeof(double));
    depth = (double *)malloc((size_t)fragment_count * sizeof(double));
    distance = (double *)malloc((size_t)fragment_count * sizeof(double));
    pixel_normals = (double *)malloc((size_t)fragment_count * 3 * sizeof(double));
    if (face_index == NULL || barycentric == NULL || depth == NULL || distance == NULL ||
        pixel_normals == NULL) {
        result = 1;
        goto cleanup;
    }

    fragment_shape[0] = 1; fragment_shape[1] = height;
    fragment_shape[2] = width; fragment_shape[3] = 1;
    bary_shape[0] = 1; bary_shape[1] = height; bary_shape[2] = width;
    bary_shape[3] = 1; bary_shape[4] = 3;
    attribute_shape[0] = 1; attribute_shape[1] = height; attribute_shape[2] = width;
    attribute_shape[3] = 1; attribute_shape[4] = 3;
    fragment_strides[3] = 1; fragment_strides[2] = 1;
    fragment_strides[1] = width; fragment_strides[0] = height * width;
    bary_strides[4] = 1; bary_strides[3] = 3; bary_strides[2] = 3;
    bary_strides[1] = width * 3; bary_strides[0] = height * width * 3;
    memcpy(attribute_strides, bary_strides, sizeof(bary_strides));

    {
        gffx_tensor_view world_view = view_of((void *)world, GFFX_DTYPE_FLOAT64, 2u,
                                              vertex_shape, pair_strides, GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view faces_view = view_of((void *)cube_faces, GFFX_DTYPE_INT32, 2u,
                                              face_shape, pair_strides, GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view normals_view = view_of(normals, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                                pair_strides, GFFX_TENSOR_OUTPUT);
        gffx_tensor_view corner_view = view_of(corner_normals, GFFX_DTYPE_FLOAT64, 3u,
                                               corner_shape, triple_strides,
                                               GFFX_TENSOR_OUTPUT);
        gffx_tensor_view matrix_view;
        gffx_tensor_view offsets_view = view_of((void *)vertex_offsets, GFFX_DTYPE_INT32, 1u,
                                                offset_shape, scalar_strides,
                                                GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view face_offsets_view = view_of((void *)face_offsets, GFFX_DTYPE_INT32, 1u,
                                                     offset_shape, scalar_strides,
                                                     GFFX_TENSOR_READ_ONLY);
        gffx_tensor_view homogeneous_view = view_of(homogeneous, GFFX_DTYPE_FLOAT64, 2u,
                                                    homogeneous_shape, quad_strides,
                                                    GFFX_TENSOR_OUTPUT);
        gffx_tensor_view ndc_view = view_of(ndc, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                            pair_strides, GFFX_TENSOR_OUTPUT);
        gffx_tensor_view divide_valid_view = view_of(divide_valid, GFFX_DTYPE_BOOL, 1u,
                                                     valid_shape, scalar_strides,
                                                     GFFX_TENSOR_OUTPUT);

        /* 1. Smooth vertex normals over the rotated cube, area weighted. */
        status = gffx_mesh_vertex_normals(&world_view, &faces_view, 1e-12,
                                          GFFX_MESH_WEIGHTING_AREA, &context, &normals_view,
                                          NULL, &diagnostic);
        if (status != GFFX_STATUS_OK) { printf("vertex_normals failed: %u\n", status); result = 1; goto cleanup; }

        /* 2. Per-face-corner normals, which is exactly what gather_faces produces. */
        {
            gffx_tensor_view normals_input = view_of(normals, GFFX_DTYPE_FLOAT64, 2u,
                                                     vertex_shape, pair_strides,
                                                     GFFX_TENSOR_READ_ONLY);
            status = gffx_mesh_gather_faces(&normals_input, &faces_view, &context, &corner_view,
                                            NULL, &diagnostic);
            if (status != GFFX_STATUS_OK) { printf("gather_faces failed: %u\n", status); result = 1; goto cleanup; }
        }

        /* 3. Camera matrix, then world -> clip. */
        projection_matrix(focal, focal, (double)width * 0.5, (double)height * 0.5,
                          (double)width, (double)height, 0.1, 10.0, matrix);
        matrix_view = view_of(matrix, GFFX_DTYPE_FLOAT64, 3u, matrix_shape, matrix_strides,
                              GFFX_TENSOR_READ_ONLY);
        status = gffx_transforms_transform_points(&world_view, &matrix_view, &offsets_view,
                                                  &context, &homogeneous_view, NULL,
                                                  &diagnostic);
        if (status != GFFX_STATUS_OK) { printf("transform_points failed: %u\n", status); result = 1; goto cleanup; }

        /* 4. Guarded perspective divide to NDC. */
        {
            gffx_tensor_view homogeneous_input = view_of(homogeneous, GFFX_DTYPE_FLOAT64, 2u,
                                                         homogeneous_shape, quad_strides,
                                                         GFFX_TENSOR_READ_ONLY);
            status = gffx_transforms_perspective_divide(&homogeneous_input, 1e-9, &context,
                                                        &ndc_view, &divide_valid_view, NULL,
                                                        &diagnostic);
            if (status != GFFX_STATUS_OK) { printf("perspective_divide failed: %u\n", status); result = 1; goto cleanup; }
        }

        /* 5. Rasterize, culling back faces so only the visible cube shell is drawn. */
        {
            gffx_tensor_view ndc_input = view_of(ndc, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                                 pair_strides, GFFX_TENSOR_READ_ONLY);
            gffx_tensor_view index_view = view_of(face_index, GFFX_DTYPE_INT32, 4u,
                                                  fragment_shape, fragment_strides,
                                                  GFFX_TENSOR_OUTPUT);
            gffx_tensor_view bary_view = view_of(barycentric, GFFX_DTYPE_FLOAT64, 5u,
                                                 bary_shape, bary_strides, GFFX_TENSOR_OUTPUT);
            gffx_tensor_view depth_view = view_of(depth, GFFX_DTYPE_FLOAT64, 4u, fragment_shape,
                                                  fragment_strides, GFFX_TENSOR_OUTPUT);
            gffx_tensor_view distance_view = view_of(distance, GFFX_DTYPE_FLOAT64, 4u,
                                                     fragment_shape, fragment_strides,
                                                     GFFX_TENSOR_OUTPUT);
            status = gffx_render_rasterize(&ndc_input, &faces_view, &offsets_view,
                                           &face_offsets_view, height, width, 1, 0.0,
                                           GFFX_CULL_BACK, 1e-12, &context, &index_view,
                                           &bary_view, &depth_view, &distance_view, NULL,
                                           &diagnostic);
            if (status != GFFX_STATUS_OK) { printf("rasterize failed: %u\n", status); result = 1; goto cleanup; }

            /* 6. Interpolate the corner normals across each covered pixel. */
            {
                gffx_tensor_view index_input = view_of(face_index, GFFX_DTYPE_INT32, 4u,
                                                       fragment_shape, fragment_strides,
                                                       GFFX_TENSOR_READ_ONLY);
                gffx_tensor_view bary_input = view_of(barycentric, GFFX_DTYPE_FLOAT64, 5u,
                                                      bary_shape, bary_strides,
                                                      GFFX_TENSOR_READ_ONLY);
                gffx_tensor_view corner_input = view_of(corner_normals, GFFX_DTYPE_FLOAT64, 3u,
                                                        corner_shape, triple_strides,
                                                        GFFX_TENSOR_READ_ONLY);
                gffx_tensor_view pixel_view = view_of(pixel_normals, GFFX_DTYPE_FLOAT64, 5u,
                                                      attribute_shape, attribute_strides,
                                                      GFFX_TENSOR_OUTPUT);
                status = gffx_render_interpolate(&index_input, &bary_input, &corner_input,
                                                 &context, &pixel_view, NULL, &diagnostic);
                if (status != GFFX_STATUS_OK) { printf("interpolate failed: %u\n", status); result = 1; goto cleanup; }
            }
        }
    }

    /* Shade: normalize the interpolated normal and take a clamped Lambert term. */
    for (index = 0; index < fragment_count; ++index) {
        double nx = pixel_normals[index * 3 + 0];
        double ny = pixel_normals[index * 3 + 1];
        double nz = pixel_normals[index * 3 + 2];
        double length = sqrt(nx * nx + ny * ny + nz * nz);
        double lambert;
        if (face_index[index] < 0 || length <= 0.0) {
            shade[index] = -1.0;
            continue;
        }
        nx /= length; ny /= length; nz /= length;
        lambert = nx * light[0] + ny * light[1] + nz * light[2];
        if (lambert < 0.0) lambert = 0.0;
        shade[index] = 0.15 + 0.85 * lambert;   /* ambient plus diffuse */
    }

cleanup:
    free(face_index);
    free(barycentric);
    free(depth);
    free(distance);
    free(pixel_normals);
    return result;
}

int main(void) {
    enum { ASCII_W = 62, ASCII_H = 30, IMAGE = 256 };
    static const char ramp[] = " .:-=+*#%@";
    double world[CUBE_VERTICES * 3];
    double ascii_shade[ASCII_H * ASCII_W];
    double *image_shade;
    int64_t row;
    int64_t column;
    FILE *file;

    rotate_cube(0.6, 0.9, world);

    /* Terminal view. The focal length is chosen so the cube fills the frame; character cells
     * are about twice as tall as wide, which the 62x30 grid roughly compensates for. */
    if (render_scene(world, ASCII_H, ASCII_W, (double)ASCII_W * 0.85, ascii_shade) != 0) {
        return 1;
    }
    printf("\n  gffx Phase 2 kernels: rotated cube, smooth normals, one directional light\n");
    printf("  vertex_normals -> gather_faces -> transform_points -> perspective_divide\n");
    printf("  -> rasterize (back-face culled) -> interpolate\n\n");
    for (row = 0; row < ASCII_H; ++row) {
        printf("  ");
        for (column = 0; column < ASCII_W; ++column) {
            double value = ascii_shade[row * ASCII_W + column];
            if (value < 0.0) {
                putchar(' ');
            } else {
                int level = (int)(value * 9.0);
                if (level < 0) level = 0;
                if (level > 9) level = 9;
                putchar(ramp[level]);
            }
        }
        putchar('\n');
    }
    putchar('\n');

    /* Higher-resolution image for an actual viewer. */
    image_shade = (double *)malloc((size_t)IMAGE * IMAGE * sizeof(double));
    if (image_shade == NULL) return 1;
    if (render_scene(world, IMAGE, IMAGE, (double)IMAGE * 1.1, image_shade) != 0) {
        free(image_shade);
        return 1;
    }
    file = fopen("gffx_cube.ppm", "wb");
    if (file != NULL) {
        fprintf(file, "P6\n%d %d\n255\n", (int)IMAGE, (int)IMAGE);
        for (row = 0; row < IMAGE; ++row) {
            for (column = 0; column < IMAGE; ++column) {
                double value = image_shade[row * IMAGE + column];
                unsigned char rgb[3];
                if (value < 0.0) {
                    rgb[0] = 24; rgb[1] = 26; rgb[2] = 32;
                } else {
                    int level = (int)(value * 255.0);
                    if (level < 0) level = 0;
                    if (level > 255) level = 255;
                    rgb[0] = (unsigned char)level;
                    rgb[1] = (unsigned char)((level * 240) / 255);
                    rgb[2] = (unsigned char)((level * 205) / 255);
                }
                fwrite(rgb, 1, 3, file);
            }
        }
        fclose(file);
        printf("  wrote gffx_cube.ppm (%dx%d)\n\n", (int)IMAGE, (int)IMAGE);
    }
    free(image_shade);
    return 0;
}
