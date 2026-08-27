/*
 * Emits the parity oracle consumed by tests/pytorch/test_face_geometry.py.
 *
 * The adapter fixtures must compare against the C reference rather than against transcribed
 * numbers, but from Python the only route to that reference is the adapter under test. This
 * program breaks the circle: it calls the C kernel directly and writes its results as JSON, which
 * the Python fixtures then read. The kernel stays the single source of truth and no expected value
 * is ever written by hand.
 *
 * The output is committed. Regenerate it by running this program rather than by editing the file,
 * exactly as the PLY fixtures are generated rather than checked in as assets. Values are written
 * with %.17g, which round-trips a double exactly, so the comparison in the fixtures can be
 * bit-exact rather than tolerance-based.
 *
 * Built through an EXCLUDE_FROM_ALL target. It registers no test, is installed by no packaging
 * rule, advertises no operation, and lives outside the inspected runtime scaffold, so it may use
 * stdio.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ORACLE_EPS 9.5367431640625e-7 /* 2^-20, the shared default */

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

static void emit_doubles(FILE *out, const char *key, const double *values, int64_t count) {
    int64_t index;
    fprintf(out, "      \"%s\": [", key);
    for (index = 0; index < count; ++index) {
        fprintf(out, "%s%.17g", index ? ", " : "", values[index]);
    }
    fprintf(out, "],\n");
}

static void emit_ints(FILE *out, const char *key, const int32_t *values, int64_t count) {
    int64_t index;
    fprintf(out, "      \"%s\": [", key);
    for (index = 0; index < count; ++index) {
        fprintf(out, "%s%d", index ? ", " : "", (int)values[index]);
    }
    fprintf(out, "],\n");
}

static void emit_bools(FILE *out, const char *key, const uint8_t *values, int64_t count,
                       int trailing_comma) {
    int64_t index;
    fprintf(out, "      \"%s\": [", key);
    for (index = 0; index < count; ++index) {
        fprintf(out, "%s%s", index ? ", " : "", values[index] ? "true" : "false");
    }
    fprintf(out, "]%s\n", trailing_comma ? "," : "");
}

/* Runs forward, and backward when cotangents are supplied, writing one JSON case. */
static int emit_case(
    FILE *out, const char *name,
    const double *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    const double *grad_unit_normals, const double *grad_areas,
    int last_case
) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    double normals[64 * 3];
    double areas[64];
    uint8_t valid[64];
    double grad_vertices[64 * 3];
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t area_shape[1];
    gffx_tensor_view vertices_view, faces_view, normals_view, areas_view, valid_view;
    gffx_status status;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    area_shape[0] = face_count;

    vertices_view = view_of((void *)vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                            GFFX_TENSOR_READ_ONLY);
    faces_view = view_of((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                         GFFX_TENSOR_READ_ONLY);
    normals_view = view_of(normals, GFFX_DTYPE_FLOAT64, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_OUTPUT);
    areas_view = view_of(areas, GFFX_DTYPE_FLOAT64, 1u, area_shape, scalar_strides,
                         GFFX_TENSOR_OUTPUT);
    valid_view = view_of(valid, GFFX_DTYPE_BOOL, 1u, area_shape, scalar_strides,
                         GFFX_TENSOR_OUTPUT);

    status = gffx_mesh_face_geometry(&vertices_view, &faces_view, ORACLE_EPS, &context,
                                     &normals_view, &areas_view, &valid_view, NULL, &diagnostic);
    if (status != GFFX_STATUS_OK) {
        printf("forward failed for %s: status %u\n", name, status);
        return 0;
    }

    fprintf(out, "    {\n      \"name\": \"%s\",\n", name);
    emit_doubles(out, "vertices", vertices, vertex_count * 3);
    emit_ints(out, "faces", faces, face_count * 3);
    emit_doubles(out, "unit_normals", normals, face_count * 3);
    emit_doubles(out, "areas", areas, face_count);
    emit_bools(out, "valid", valid, face_count, grad_unit_normals != NULL);

    if (grad_unit_normals != NULL) {
        gffx_tensor_view grad_normals_view, grad_areas_view, grad_vertices_view;
        grad_normals_view = view_of((void *)grad_unit_normals, GFFX_DTYPE_FLOAT64, 2u, face_shape,
                                    pair_strides, GFFX_TENSOR_READ_ONLY);
        grad_areas_view = view_of((void *)grad_areas, GFFX_DTYPE_FLOAT64, 1u, area_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);
        grad_vertices_view = view_of(grad_vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                                     pair_strides, GFFX_TENSOR_OUTPUT);
        status = gffx_mesh_face_geometry_backward(&vertices_view, &faces_view, ORACLE_EPS,
                                                  &grad_normals_view, &grad_areas_view, &context,
                                                  &grad_vertices_view, NULL, &diagnostic);
        if (status != GFFX_STATUS_OK) {
            printf("backward failed for %s: status %u\n", name, status);
            return 0;
        }
        emit_doubles(out, "grad_unit_normals", grad_unit_normals, face_count * 3);
        emit_doubles(out, "grad_areas", grad_areas, face_count);
        emit_doubles(out, "grad_vertices", grad_vertices, vertex_count * 3);
        /* Trim the trailing comma of the last member so the object stays valid JSON. */
        fseek(out, -2, SEEK_CUR);
        fprintf(out, "\n");
    }
    fprintf(out, "    }%s\n", last_case ? "" : ",");
    return 1;
}

int main(void) {
    /* A unit tetrahedron: exact coordinates, all four faces non-degenerate. */
    static const double tetra_vertices[12] = {
        0.0, 0.0, 0.0,   1.0, 0.0, 0.0,   0.0, 1.0, 0.0,   0.0, 0.0, 1.0
    };
    static const int32_t tetra_faces[12] = {0, 2, 1,  0, 1, 3,  0, 3, 2,  1, 2, 3};
    /* Cotangents chosen to be exactly representable and distinct per component, so a transposed
     * or misaligned gradient cannot coincidentally match. */
    static const double tetra_grad_normals[12] = {
        0.5, 0.25, -0.125,   -0.75, 0.5, 0.25,   0.125, -0.5, 0.75,   0.25, 0.125, -0.5
    };
    static const double tetra_grad_areas[4] = {1.0, -0.5, 0.25, 2.0};

    /* A mesh whose second face is exactly degenerate: three collinear vertices. */
    static const double degenerate_vertices[15] = {
        0.0, 0.0, 0.0,   1.0, 0.0, 0.0,   0.0, 1.0, 0.0,
        2.0, 0.0, 0.0,   3.0, 0.0, 0.0
    };
    static const int32_t degenerate_faces[6] = {0, 1, 2,  1, 3, 4};

    FILE *out = fopen("face_geometry_oracle.json", "wb");
    if (out == NULL) {
        printf("could not open face_geometry_oracle.json for writing\n");
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"note\": \"Generated by examples/emit_face_geometry_oracle.c. "
                 "Regenerate by running that program; never edit by hand.\",\n");
    fprintf(out, "  \"eps\": %.17g,\n", ORACLE_EPS);
    fprintf(out, "  \"cases\": [\n");
    if (!emit_case(out, "tetrahedron", tetra_vertices, 4, tetra_faces, 4,
                   tetra_grad_normals, tetra_grad_areas, 0)) {
        fclose(out);
        return 1;
    }
    if (!emit_case(out, "degenerate_face", degenerate_vertices, 5, degenerate_faces, 2,
                   NULL, NULL, 1)) {
        fclose(out);
        return 1;
    }
    fprintf(out, "  ]\n}\n");
    fclose(out);
    printf("wrote face_geometry_oracle.json\n");
    return 0;
}
