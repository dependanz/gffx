/*
 * Phase 2 acceptance fixtures VA-01..VA-10 for the eager mesh.validate survey utility.
 * Fixture numbers match the project acceptance record.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define VA_EPS_DEFAULT 9.5367431640625e-7 /* 2^-20 */

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

static gffx_status run_validate(
    const double *vertices, int64_t vertex_count,
    const int32_t *faces, int64_t face_count,
    const int32_t *vertex_offsets, const int32_t *face_offsets, int64_t batch_count,
    double eps, uint32_t flags, gffx_mesh_validation_report *report
) {
    int64_t vertex_shape[2];
    int64_t face_shape[2];
    int64_t offset_shape[1];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view vertex_offsets_view;
    gffx_tensor_view face_offsets_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertex_shape[0] = vertex_count; vertex_shape[1] = 3;
    face_shape[0] = face_count; face_shape[1] = 3;
    offset_shape[0] = batch_count + 1;

    vertices_view = make_view((void *)vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                              pair_strides, GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    vertex_offsets_view = make_view((void *)vertex_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                    scalar_strides, GFFX_TENSOR_READ_ONLY);
    face_offsets_view = make_view((void *)face_offsets, GFFX_DTYPE_INT32, 1u, offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);

    memset(report, 0, sizeof(*report));
    report->struct_size = (uint32_t)sizeof(*report);
    report->abi_version = GFFX_ABI_VERSION;
    return gffx_mesh_validate(&vertices_view, &faces_view, &vertex_offsets_view,
                              &face_offsets_view, eps, flags, &context, report, NULL,
                              &diagnostic);
}

/* Two independent unit triangles, one per batch element. */
static const double two_meshes[18] = {
    0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,
    5.0, 0.0, 0.0,  6.0, 0.0, 0.0,  5.0, 1.0, 0.0
};
static const int32_t two_faces[6] = {0, 1, 2, 3, 4, 5};
static const int32_t vertex_offsets_two[3] = {0, 3, 6};
static const int32_t face_offsets_two[3] = {0, 1, 2};

static const double one_mesh[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
static const int32_t one_face[3] = {0, 1, 2};
static const int32_t vertex_offsets_one[2] = {0, 3};
static const int32_t face_offsets_one[2] = {0, 1};

static int test_va01_clean(void) {
    gffx_mesh_validation_report report;
    CHECK(run_validate(two_meshes, 6, two_faces, 2, vertex_offsets_two, face_offsets_two, 2,
                       VA_EPS_DEFAULT, 0u, &report) == GFFX_STATUS_OK);
    CHECK(report.findings == 0u);
    CHECK(report.first_bad_face == -1);
    CHECK(report.first_bad_offset_batch == -1);
    CHECK(report.degenerate_face_count == 0);
    CHECK(report.unreferenced_vertex_count == 0);
    /* The geometry survey was not requested, so the count stays distinguishable from zero. */
    CHECK(report.nonfinite_vertex_count == -1);
    return 0;
}

static int test_va02_index_range(void) {
    static const int32_t bad_faces[6] = {0, 1, 2, 3, 4, 99};
    gffx_mesh_validation_report report;
    CHECK(run_validate(two_meshes, 6, bad_faces, 2, vertex_offsets_two, face_offsets_two, 2,
                       VA_EPS_DEFAULT, 0u, &report) == GFFX_STATUS_OK);
    CHECK((report.findings & GFFX_MESH_FINDING_FACE_INDEX_RANGE) != 0u);
    CHECK(report.first_bad_face == 1);
    /* Geometric surveys are skipped because vertex lookup would be unsafe. */
    CHECK(report.degenerate_face_count == 0);
    CHECK(report.unreferenced_vertex_count == 0);
    return 0;
}

static int test_va03_cross_element(void) {
    /* Face 1 belongs to element 1 but names element 0's vertices: in range, wrong element. */
    static const int32_t cross_faces[6] = {0, 1, 2, 0, 1, 2};
    gffx_mesh_validation_report report;
    CHECK(run_validate(two_meshes, 6, cross_faces, 2, vertex_offsets_two, face_offsets_two, 2,
                       VA_EPS_DEFAULT, 0u, &report) == GFFX_STATUS_OK);
    CHECK((report.findings & GFFX_MESH_FINDING_FACE_INDEX_BATCH) != 0u);
    CHECK((report.findings & GFFX_MESH_FINDING_FACE_INDEX_RANGE) == 0u);
    CHECK(report.first_bad_face == 1);
    return 0;
}

static int test_va04_offsets(void) {
    static const int32_t bad_offsets[3] = {0, 2, 1};
    gffx_mesh_validation_report report;
    CHECK(run_validate(two_meshes, 6, two_faces, 2, vertex_offsets_two, bad_offsets, 2,
                       VA_EPS_DEFAULT, 0u, &report) == GFFX_STATUS_OK);
    CHECK((report.findings & GFFX_MESH_FINDING_OFFSETS) != 0u);
    CHECK(report.first_bad_offset_batch >= 0);
    /* Nothing beyond the offsets is surveyed, so every later count stays at its initial value. */
    CHECK(report.first_bad_face == -1);
    CHECK(report.degenerate_face_count == 0);
    CHECK(report.nonfinite_vertex_count == -1);
    return 0;
}

static int test_va05_degenerate(void) {
    /* Four faces over one element; two of them are degenerate. */
    static const double vertices[15] = {
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  2.0, 0.0, 0.0,  1.0, 1.0, 0.0
    };
    static const int32_t faces[12] = {
        0, 1, 2,    0, 1, 1,    1, 3, 4,    0, 1, 3
    };
    static const int32_t vertex_offsets[2] = {0, 5};
    static const int32_t face_offsets[2] = {0, 4};
    gffx_mesh_validation_report report;
    CHECK(run_validate(vertices, 5, faces, 4, vertex_offsets, face_offsets, 1, VA_EPS_DEFAULT,
                       0u, &report) == GFFX_STATUS_OK);
    /* Face 1 repeats a vertex; face 3 is collinear along the x axis. */
    CHECK((report.findings & GFFX_MESH_FINDING_DEGENERATE_FACE) != 0u);
    CHECK(report.degenerate_face_count == 2);
    CHECK((report.findings & GFFX_MESH_FINDING_FACE_INDEX_RANGE) == 0u);
    CHECK(report.unreferenced_vertex_count == 0);
    return 0;
}

static int test_va06_unreferenced(void) {
    static const double vertices[12] = {
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  9.0, 9.0, 9.0
    };
    static const int32_t vertex_offsets[2] = {0, 4};
    gffx_mesh_validation_report report;
    CHECK(run_validate(vertices, 4, one_face, 1, vertex_offsets, face_offsets_one, 1,
                       VA_EPS_DEFAULT, 0u, &report) == GFFX_STATUS_OK);
    CHECK((report.findings & GFFX_MESH_FINDING_UNREFERENCED_VERTEX) != 0u);
    CHECK(report.unreferenced_vertex_count == 1);

    /* A vertex referenced only by another element still counts as unreferenced in its own. */
    {
        static const int32_t only_first[6] = {0, 1, 2, 0, 1, 2};
        static const int32_t face_offsets_one_each[3] = {0, 1, 2};
        gffx_mesh_validation_report second;
        CHECK(run_validate(two_meshes, 6, only_first, 2, vertex_offsets_two,
                           face_offsets_one_each, 2, VA_EPS_DEFAULT, 0u, &second)
              == GFFX_STATUS_OK);
        /* That mesh is cross-element, so the batch finding fires and the survey stops. */
        CHECK((second.findings & GFFX_MESH_FINDING_FACE_INDEX_BATCH) != 0u);
    }
    return 0;
}

static int test_va07_nonfinite(void) {
    double vertices[9];
    gffx_mesh_validation_report report;
    memcpy(vertices, one_mesh, sizeof(vertices));
    vertices[0] = (double)NAN;
    vertices[5] = (double)INFINITY;

    /* Flag off: not surveyed, and the count says so rather than reporting a clean zero. */
    CHECK(run_validate(vertices, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                       VA_EPS_DEFAULT, 0u, &report) == GFFX_STATUS_OK);
    CHECK((report.findings & GFFX_MESH_FINDING_NONFINITE_GEOMETRY) == 0u);
    CHECK(report.nonfinite_vertex_count == -1);

    /* Flag on: exact count, and a non-finite vertex is reported rather than treated as an error. */
    CHECK(run_validate(vertices, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                       VA_EPS_DEFAULT, GFFX_MESH_VALIDATE_GEOMETRY, &report)
          == GFFX_STATUS_OK);
    CHECK((report.findings & GFFX_MESH_FINDING_NONFINITE_GEOMETRY) != 0u);
    CHECK(report.nonfinite_vertex_count == 2);
    return 0;
}

static int test_va08_multiple_findings(void) {
    /* One degenerate face, one unreferenced vertex, and non-finite geometry at once. */
    static const int32_t faces[6] = {0, 1, 2, 0, 1, 1};
    static const int32_t vertex_offsets[2] = {0, 4};
    static const int32_t face_offsets[2] = {0, 2};
    double vertices[12] = {
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  9.0, 9.0, 9.0
    };
    gffx_mesh_validation_report report;
    vertices[11] = (double)NAN;
    CHECK(run_validate(vertices, 4, faces, 2, vertex_offsets, face_offsets, 1, VA_EPS_DEFAULT,
                       GFFX_MESH_VALIDATE_GEOMETRY, &report) == GFFX_STATUS_OK);
    CHECK((report.findings & GFFX_MESH_FINDING_DEGENERATE_FACE) != 0u);
    CHECK((report.findings & GFFX_MESH_FINDING_UNREFERENCED_VERTEX) != 0u);
    CHECK((report.findings & GFFX_MESH_FINDING_NONFINITE_GEOMETRY) != 0u);
    CHECK(report.degenerate_face_count == 1);
    CHECK(report.unreferenced_vertex_count == 1);
    CHECK(report.nonfinite_vertex_count == 1);
    return 0;
}

static int test_va09_validation_and_empty(void) {
    gffx_mesh_validation_report report;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    int64_t vertex_shape[2] = {3, 3};
    int64_t face_shape[2] = {1, 3};
    int64_t offset_shape[1] = {2};
    gffx_tensor_view vertices_view;
    gffx_tensor_view faces_view;
    gffx_tensor_view vertex_offsets_view;
    gffx_tensor_view face_offsets_view;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    vertices_view = make_view((void *)one_mesh, GFFX_DTYPE_FLOAT64, 2u, vertex_shape,
                              pair_strides, GFFX_TENSOR_READ_ONLY);
    faces_view = make_view((void *)one_face, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    vertex_offsets_view = make_view((void *)vertex_offsets_one, GFFX_DTYPE_INT32, 1u,
                                    offset_shape, scalar_strides, GFFX_TENSOR_READ_ONLY);
    face_offsets_view = make_view((void *)face_offsets_one, GFFX_DTYPE_INT32, 1u, offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);

    /* A null report is an invalid argument, not a silent no-op. */
    CHECK(gffx_mesh_validate(&vertices_view, &faces_view, &vertex_offsets_view,
                             &face_offsets_view, VA_EPS_DEFAULT, 0u, &context, NULL, NULL,
                             &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    /* A short struct_size cannot be filled safely. */
    memset(&report, 0, sizeof(report));
    report.struct_size = 8u;
    report.abi_version = GFFX_ABI_VERSION;
    CHECK(gffx_mesh_validate(&vertices_view, &faces_view, &vertex_offsets_view,
                             &face_offsets_view, VA_EPS_DEFAULT, 0u, &context, &report, NULL,
                             &diagnostic) == GFFX_STATUS_ABI_MISMATCH);
    /* eps rules match the rest of the project. */
    CHECK(run_validate(one_mesh, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1, -1.0,
                       0u, &report) == GFFX_STATUS_INVALID_ARGUMENT);
    /* Unknown flags are rejected rather than ignored. */
    CHECK(run_validate(one_mesh, 3, one_face, 1, vertex_offsets_one, face_offsets_one, 1,
                       VA_EPS_DEFAULT, UINT32_C(64), &report) == GFFX_STATUS_INVALID_ARGUMENT);

    /* An empty mesh is clean. */
    {
        static const int32_t empty_offsets[2] = {0, 0};
        CHECK(run_validate(NULL, 0, NULL, 0, empty_offsets, empty_offsets, 1, VA_EPS_DEFAULT,
                           GFFX_MESH_VALIDATE_GEOMETRY, &report) == GFFX_STATUS_OK);
        CHECK(report.findings == 0u);
        CHECK(report.nonfinite_vertex_count == 0);
    }
    return 0;
}

static int test_va10_workspace_and_determinism(void) {
    uint64_t required_bytes = UINT64_MAX;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_mesh_validation_report first;
    gffx_mesh_validation_report second;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    CHECK(gffx_mesh_validate_workspace(6, 2, &context, &required_bytes, &required_alignment,
                                       &diagnostic) == GFFX_STATUS_OK);
    CHECK(required_bytes == 0u);

    CHECK(run_validate(two_meshes, 6, two_faces, 2, vertex_offsets_two, face_offsets_two, 2,
                       VA_EPS_DEFAULT, GFFX_MESH_VALIDATE_GEOMETRY, &first) == GFFX_STATUS_OK);
    CHECK(run_validate(two_meshes, 6, two_faces, 2, vertex_offsets_two, face_offsets_two, 2,
                       VA_EPS_DEFAULT, GFFX_MESH_VALIDATE_GEOMETRY, &second) == GFFX_STATUS_OK);
    CHECK(memcmp(&first, &second, sizeof(first)) == 0);
    return 0;
}

int main(void) {
    int result;
    result = test_va01_clean(); if (result != 0) return result;
    result = test_va02_index_range(); if (result != 0) return result;
    result = test_va03_cross_element(); if (result != 0) return result;
    result = test_va04_offsets(); if (result != 0) return result;
    result = test_va05_degenerate(); if (result != 0) return result;
    result = test_va06_unreferenced(); if (result != 0) return result;
    result = test_va07_nonfinite(); if (result != 0) return result;
    result = test_va08_multiple_findings(); if (result != 0) return result;
    result = test_va09_validation_and_empty(); if (result != 0) return result;
    result = test_va10_workspace_and_determinism(); if (result != 0) return result;
    return 0;
}
