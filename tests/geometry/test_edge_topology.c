/*
 * Phase 2 acceptance fixtures ET-01..ET-10 for mesh.build_edge_topology.
 *
 * Fixture numbers match the project acceptance record. Failures return the source line. Every
 * output is integer topology, so all comparisons are exact and there is no gradient fixture.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define ET_MAX_F 8
#define ET_MAX_HALF (3 * ET_MAX_F)

static const int64_t pair_strides[2] = {3, 1};
static const int64_t edge_strides[2] = {2, 1};
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
    buffer.alignment = UINT64_C(4);
    buffer.device_type = GFFX_DEVICE_CPU;
    buffer.device_index = 0;
    return buffer;
}

/* Convenience wrapper: all four outputs at their exact contract capacities. */
static gffx_status run_build(
    const int32_t *faces, int64_t face_count,
    const int32_t *face_offsets, int64_t batch_count,
    int32_t *edges, int32_t *edge_face_offsets, int32_t *edge_faces,
    int32_t *mesh_edge_offsets, void *workspace_data, uint64_t workspace_capacity
) {
    int64_t face_shape[2];
    int64_t face_offset_shape[1];
    int64_t edge_shape[2];
    int64_t edge_offset_shape[1];
    int64_t edge_face_shape[1];
    int64_t mesh_offset_shape[1];
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view faces_view;
    gffx_tensor_view face_offsets_view;
    gffx_tensor_view edges_view;
    gffx_tensor_view edge_offsets_view;
    gffx_tensor_view edge_faces_view;
    gffx_tensor_view mesh_offsets_view;
    gffx_buffer workspace;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    face_shape[0] = face_count; face_shape[1] = 3;
    face_offset_shape[0] = batch_count + 1;
    edge_shape[0] = face_count * 3; edge_shape[1] = 2;
    edge_offset_shape[0] = face_count * 3 + 1;
    edge_face_shape[0] = face_count * 3;
    mesh_offset_shape[0] = batch_count + 1;

    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    face_offsets_view = make_view((void *)face_offsets, GFFX_DTYPE_INT32, 1u, face_offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);
    edges_view = make_view(edges, GFFX_DTYPE_INT32, 2u, edge_shape, edge_strides,
                           GFFX_TENSOR_OUTPUT);
    edge_offsets_view = make_view(edge_face_offsets, GFFX_DTYPE_INT32, 1u, edge_offset_shape,
                                  scalar_strides, GFFX_TENSOR_OUTPUT);
    edge_faces_view = make_view(edge_faces, GFFX_DTYPE_INT32, 1u, edge_face_shape,
                                scalar_strides, GFFX_TENSOR_OUTPUT);
    mesh_offsets_view = make_view(mesh_edge_offsets, GFFX_DTYPE_INT32, 1u, mesh_offset_shape,
                                  scalar_strides, GFFX_TENSOR_OUTPUT);
    workspace = make_workspace(workspace_data, workspace_capacity);

    return gffx_mesh_build_edge_topology(&faces_view, &face_offsets_view, &context, &edges_view,
                                         &edge_offsets_view, &edge_faces_view,
                                         &mesh_offsets_view,
                                         workspace_data != NULL ? &workspace : NULL,
                                         &diagnostic);
}

static int edge_is(const int32_t *edges, int64_t row, int32_t low, int32_t high) {
    return edges[row * 2 + 0] == low && edges[row * 2 + 1] == high;
}

static int test_et01_single_triangle(void) {
    static const int32_t faces[3] = {0, 1, 2};
    static const int32_t offsets[2] = {0, 1};
    int32_t edges[6];
    int32_t edge_face_offsets[4];
    int32_t edge_faces[3];
    int32_t mesh_edge_offsets[2];
    int32_t workspace[9];
    int64_t index;

    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(mesh_edge_offsets[0] == 0 && mesh_edge_offsets[1] == 3);
    CHECK(edge_is(edges, 0, 0, 1));
    CHECK(edge_is(edges, 1, 0, 2));
    CHECK(edge_is(edges, 2, 1, 2));
    for (index = 0; index < 3; ++index) CHECK(edge_faces[index] == 0);
    CHECK(edge_face_offsets[0] == 0 && edge_face_offsets[1] == 1);
    CHECK(edge_face_offsets[2] == 2 && edge_face_offsets[3] == 3);
    return 0;
}

static int test_et02_shared_edge(void) {
    static const int32_t faces[6] = {0, 1, 2, 1, 3, 2};
    static const int32_t offsets[2] = {0, 2};
    int32_t edges[12];
    int32_t edge_face_offsets[7];
    int32_t edge_faces[6];
    int32_t mesh_edge_offsets[2];
    int32_t workspace[18];

    CHECK(run_build(faces, 2, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(mesh_edge_offsets[1] == 5);
    CHECK(edge_is(edges, 0, 0, 1));
    CHECK(edge_is(edges, 1, 0, 2));
    CHECK(edge_is(edges, 2, 1, 2));
    CHECK(edge_is(edges, 3, 1, 3));
    CHECK(edge_is(edges, 4, 2, 3));
    /* The shared edge (1,2) lists both faces in ascending order. */
    CHECK(edge_face_offsets[2] == 2 && edge_face_offsets[3] == 4);
    CHECK(edge_faces[2] == 0 && edge_faces[3] == 1);
    /* Total incidence is exactly 3F. */
    CHECK(edge_face_offsets[5] == 6);
    return 0;
}

static int test_et03_non_manifold(void) {
    /* Three triangles all containing edge (0,1). */
    static const int32_t faces[9] = {0, 1, 2, 0, 1, 3, 0, 1, 4};
    static const int32_t offsets[2] = {0, 3};
    int32_t edges[18];
    int32_t edge_face_offsets[10];
    int32_t edge_faces[9];
    int32_t mesh_edge_offsets[2];
    int32_t workspace[27];

    CHECK(run_build(faces, 3, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    /* Edge (0,1) sorts first and keeps all three incident faces. */
    CHECK(edge_is(edges, 0, 0, 1));
    CHECK(edge_face_offsets[0] == 0 && edge_face_offsets[1] == 3);
    CHECK(edge_faces[0] == 0 && edge_faces[1] == 1 && edge_faces[2] == 2);
    return 0;
}

static int test_et04_degenerate_self_edge(void) {
    /* Face 0 repeats vertex 0, producing the self-edge (0,0) and a duplicated (0,1). */
    static const int32_t faces[3] = {0, 0, 1};
    static const int32_t offsets[2] = {0, 1};
    int32_t edges[6];
    int32_t edge_face_offsets[4];
    int32_t edge_faces[3];
    int32_t mesh_edge_offsets[2];
    int32_t workspace[9];

    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    /* Half-edges are (0,0), (0,1), (1,0)->(0,1): two unique edges. */
    CHECK(mesh_edge_offsets[1] == 2);
    CHECK(edge_is(edges, 0, 0, 0));
    CHECK(edge_is(edges, 1, 0, 1));
    CHECK(edge_face_offsets[0] == 0 && edge_face_offsets[1] == 1 && edge_face_offsets[2] == 3);
    /* The same face appears twice for the duplicated edge; incidence is not deduplicated. */
    CHECK(edge_faces[1] == 0 && edge_faces[2] == 0);
    return 0;
}

static int test_et05_sentinels(void) {
    static const int32_t faces[3] = {0, 1, 2};
    static const int32_t offsets[2] = {0, 1};
    int32_t edges[6];
    int32_t edge_face_offsets[4];
    int32_t edge_faces[3];
    int32_t mesh_edge_offsets[2];
    int32_t workspace[9];
    static const int32_t faces_two[6] = {0, 1, 2, 0, 1, 2};
    int32_t edges_two[12];
    int32_t edge_face_offsets_two[7];
    int32_t edge_faces_two[6];
    int32_t mesh_two[2];
    int32_t workspace_two[18];
    static const int32_t offsets_two[2] = {0, 2};
    int64_t row;

    /* One triangle exactly fills three of three edge rows: no sentinel region. */
    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace)) == GFFX_STATUS_OK);

    /* Two identical faces give three unique edges out of six rows, so rows 3..5 are sentinels. */
    CHECK(run_build(faces_two, 2, offsets_two, 1, edges_two, edge_face_offsets_two,
                    edge_faces_two, mesh_two, workspace_two, sizeof(workspace_two))
          == GFFX_STATUS_OK);
    CHECK(mesh_two[1] == 3);
    for (row = 3; row < 6; ++row) {
        CHECK(edge_is(edges_two, row, -1, -1));
    }
    /* Trailing offsets repeat the final incidence total, keeping the array nondecreasing. */
    CHECK(edge_face_offsets_two[3] == 6);
    for (row = 3; row < 7; ++row) CHECK(edge_face_offsets_two[row] == 6);
    /* Each edge carries both faces. */
    CHECK(edge_faces_two[0] == 0 && edge_faces_two[1] == 1);
    return 0;
}

static int test_et06_batching(void) {
    /* Two batch elements reusing the same vertex indices must not merge. */
    static const int32_t faces[6] = {0, 1, 2, 0, 1, 2};
    static const int32_t offsets[3] = {0, 1, 2};
    int32_t edges[12];
    int32_t edge_face_offsets[7];
    int32_t edge_faces[6];
    int32_t mesh_edge_offsets[3];
    int32_t workspace[18];
    int64_t row;

    CHECK(run_build(faces, 2, offsets, 2, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(mesh_edge_offsets[0] == 0);
    CHECK(mesh_edge_offsets[1] == 3);
    CHECK(mesh_edge_offsets[2] == 6);
    for (row = 0; row < 2; ++row) {
        CHECK(edge_is(edges, row * 3 + 0, 0, 1));
        CHECK(edge_is(edges, row * 3 + 1, 0, 2));
        CHECK(edge_is(edges, row * 3 + 2, 1, 2));
    }
    /* Element 0 lists only face 0; element 1 lists only face 1. */
    CHECK(edge_faces[0] == 0 && edge_faces[1] == 0 && edge_faces[2] == 0);
    CHECK(edge_faces[3] == 1 && edge_faces[4] == 1 && edge_faces[5] == 1);
    return 0;
}

static int test_et07_empty(void) {
    static const int32_t offsets_zero[2] = {0, 0};
    static const int32_t faces[6] = {0, 1, 2, 0, 1, 2};
    static const int32_t offsets_gap[4] = {0, 1, 1, 2};
    int32_t mesh_edge_offsets[4];
    int32_t edges[12];
    int32_t edge_face_offsets[7];
    int32_t edge_faces[6];
    int32_t workspace[18];

    /* F = 0 with one batch element. */
    CHECK(run_build(NULL, 0, offsets_zero, 1, NULL, edge_face_offsets, NULL,
                    mesh_edge_offsets, NULL, 0) == GFFX_STATUS_OK);
    CHECK(mesh_edge_offsets[0] == 0 && mesh_edge_offsets[1] == 0);
    CHECK(edge_face_offsets[0] == 0);

    /* An empty middle batch element contributes nothing. */
    CHECK(run_build(faces, 2, offsets_gap, 3, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(mesh_edge_offsets[0] == 0);
    CHECK(mesh_edge_offsets[1] == 3);
    CHECK(mesh_edge_offsets[2] == 3);
    CHECK(mesh_edge_offsets[3] == 6);
    return 0;
}

static int test_et08_validation(void) {
    static const int32_t faces[3] = {0, 1, 2};
    int32_t offsets[2] = {0, 1};
    int32_t negative_faces[3] = {0, -1, 2};
    int32_t edges[6];
    int32_t edge_face_offsets[4];
    int32_t edge_faces[3];
    int32_t mesh_edge_offsets[2];
    int32_t workspace[9];
    int64_t face_shape[2] = {1, 3};
    int64_t face_offset_shape[1] = {2};
    int64_t edge_shape[2] = {3, 2};
    static const int64_t wrong_edge_shape[2] = {2, 2};
    int64_t edge_offset_shape[1] = {4};
    int64_t edge_face_shape[1] = {3};
    int64_t mesh_offset_shape[1] = {2};
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_tensor_view faces_view;
    gffx_tensor_view face_offsets_view;
    gffx_tensor_view edges_view;
    gffx_tensor_view edge_offsets_view;
    gffx_tensor_view edge_faces_view;
    gffx_tensor_view mesh_offsets_view;
    gffx_buffer workspace_buffer = make_workspace(workspace, sizeof(workspace));
    gffx_tensor_view broken;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    faces_view = make_view((void *)faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                           GFFX_TENSOR_READ_ONLY);
    face_offsets_view = make_view(offsets, GFFX_DTYPE_INT32, 1u, face_offset_shape,
                                  scalar_strides, GFFX_TENSOR_READ_ONLY);
    edges_view = make_view(edges, GFFX_DTYPE_INT32, 2u, edge_shape, edge_strides,
                           GFFX_TENSOR_OUTPUT);
    edge_offsets_view = make_view(edge_face_offsets, GFFX_DTYPE_INT32, 1u, edge_offset_shape,
                                  scalar_strides, GFFX_TENSOR_OUTPUT);
    edge_faces_view = make_view(edge_faces, GFFX_DTYPE_INT32, 1u, edge_face_shape,
                                scalar_strides, GFFX_TENSOR_OUTPUT);
    mesh_offsets_view = make_view(mesh_edge_offsets, GFFX_DTYPE_INT32, 1u, mesh_offset_shape,
                                  scalar_strides, GFFX_TENSOR_OUTPUT);

    /* Negative face indices. */
    CHECK(run_build(negative_faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);

    /* Offset rules. */
    offsets[0] = 1; offsets[1] = 1;
    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);
    offsets[0] = 0; offsets[1] = 0;
    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, sizeof(workspace))
          == GFFX_STATUS_INVALID_ARGUMENT);
    offsets[0] = 0; offsets[1] = 1;

    /* Undersized output capacity. */
    broken = edges_view;
    broken.shape = wrong_edge_shape;
    CHECK(gffx_mesh_build_edge_topology(&faces_view, &face_offsets_view, &context, &broken,
                                        &edge_offsets_view, &edge_faces_view,
                                        &mesh_offsets_view, &workspace_buffer, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);

    /* Wrong output dtype. */
    broken = edge_faces_view;
    broken.dtype = GFFX_DTYPE_FLOAT32;
    CHECK(gffx_mesh_build_edge_topology(&faces_view, &face_offsets_view, &context, &edges_view,
                                        &edge_offsets_view, &broken, &mesh_offsets_view,
                                        &workspace_buffer, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);

    /* Output aliasing an input. */
    broken = edges_view;
    broken.data = (void *)faces;
    CHECK(gffx_mesh_build_edge_topology(&faces_view, &face_offsets_view, &context, &broken,
                                        &edge_offsets_view, &edge_faces_view,
                                        &mesh_offsets_view, &workspace_buffer, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);

    /* Null outputs. */
    CHECK(gffx_mesh_build_edge_topology(&faces_view, &face_offsets_view, &context, NULL,
                                        &edge_offsets_view, &edge_faces_view,
                                        &mesh_offsets_view, &workspace_buffer, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static int test_et09_determinism(void) {
    static const int32_t faces[9] = {0, 1, 2, 1, 3, 2, 2, 3, 4};
    static const int32_t offsets[2] = {0, 3};
    int32_t edges_a[18];
    int32_t edges_b[18];
    int32_t offsets_a[10];
    int32_t offsets_b[10];
    int32_t faces_a[9];
    int32_t faces_b[9];
    int32_t mesh_a[2];
    int32_t mesh_b[2];
    int32_t workspace[27];

    CHECK(run_build(faces, 3, offsets, 1, edges_a, offsets_a, faces_a, mesh_a, workspace,
                    sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(run_build(faces, 3, offsets, 1, edges_b, offsets_b, faces_b, mesh_b, workspace,
                    sizeof(workspace)) == GFFX_STATUS_OK);
    CHECK(memcmp(edges_a, edges_b, sizeof(edges_a)) == 0);
    CHECK(memcmp(offsets_a, offsets_b, sizeof(offsets_a)) == 0);
    CHECK(memcmp(faces_a, faces_b, sizeof(faces_a)) == 0);
    CHECK(memcmp(mesh_a, mesh_b, sizeof(mesh_a)) == 0);
    return 0;
}

static int test_et10_workspace(void) {
    static const int32_t faces[3] = {0, 1, 2};
    static const int32_t offsets[2] = {0, 1};
    uint64_t required_bytes = 0;
    uint64_t required_alignment = 0;
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    int32_t edges[6];
    int32_t edge_face_offsets[4];
    int32_t edge_faces[3];
    int32_t mesh_edge_offsets[2];
    int32_t workspace[9];

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    CHECK(gffx_mesh_build_edge_topology_workspace(1, 1, &context, &required_bytes,
                                                  &required_alignment, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(required_bytes == UINT64_C(36));
    CHECK(required_alignment == UINT64_C(4));
    CHECK(gffx_mesh_build_edge_topology_workspace(4, 1, &context, &required_bytes,
                                                  &required_alignment, &diagnostic)
          == GFFX_STATUS_OK);
    CHECK(required_bytes == UINT64_C(144));

    /* A null or undersized workspace fails when F > 0. */
    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, NULL, 0) == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, UINT64_C(35))
          == GFFX_STATUS_INSUFFICIENT_WORKSPACE);
    CHECK(run_build(faces, 1, offsets, 1, edges, edge_face_offsets, edge_faces,
                    mesh_edge_offsets, workspace, UINT64_C(36)) == GFFX_STATUS_OK);
    return 0;
}

int main(void) {
    int result;
    result = test_et01_single_triangle(); if (result != 0) return result;
    result = test_et02_shared_edge(); if (result != 0) return result;
    result = test_et03_non_manifold(); if (result != 0) return result;
    result = test_et04_degenerate_self_edge(); if (result != 0) return result;
    result = test_et05_sentinels(); if (result != 0) return result;
    result = test_et06_batching(); if (result != 0) return result;
    result = test_et07_empty(); if (result != 0) return result;
    result = test_et08_validation(); if (result != 0) return result;
    result = test_et09_determinism(); if (result != 0) return result;
    result = test_et10_workspace(); if (result != 0) return result;
    return 0;
}
