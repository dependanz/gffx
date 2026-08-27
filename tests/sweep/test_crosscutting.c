/*
 * Phase 2 Step 4 cross-cutting fixtures SW-01..SW-09.
 * Fixture numbers match CROSSCUTTING_ACCEPTANCE_V0_1.md in the project record.
 *
 * These assert properties that belong to no single operation, and are therefore invisible to the
 * per-operation suites: an assumption every operation shares is the one nobody checks. The
 * operations are driven through one table so that each property is applied by identical code
 * rather than by per-operation reasoning, which is what makes an omission detectable.
 *
 * The per-operation records remain authoritative for semantics. Nothing here re-asserts what an
 * operation computes; everything here asserts how it behaves as a citizen of the ABI.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/points.h>
#include <gffx/render.h>
#include <gffx/status.h>
#include <gffx/tensor.h>
#include <gffx/transforms.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (0)

#define SWEEP_EPS 9.5367431640625e-7 /* 2^-20, the shared default */
#define SWEEP_OUT_BYTES 65536
#define SWEEP_WS_BYTES 65536

static const int64_t stride_pair[2] = {3, 1};
static const int64_t stride_scalar[1] = {1};
static const int64_t stride_mat[3] = {16, 4, 1};
static const int64_t stride_two[2] = {2, 1};

/* ------------------------------------------------------------------- shared fixture geometry */

/* Two batch elements: a unit tetrahedron and a translated copy. Packed layout throughout, so the
 * batch-invariance fixture has something real to compare against. */
static const double VERTS[24] = {
    0.0, 0.0, 0.0,   1.0, 0.0, 0.0,   0.0, 1.0, 0.0,   0.0, 0.0, 1.0,
    2.0, 0.0, 0.0,   3.0, 0.0, 0.0,   2.0, 1.0, 0.0,   2.0, 0.0, 1.0
};
static const int32_t FACES[24] = {
    0, 1, 2,   0, 1, 3,   0, 2, 3,   1, 2, 3,
    4, 5, 6,   4, 5, 7,   4, 6, 7,   5, 6, 7
};
static const int32_t VERT_OFFSETS[3] = {0, 4, 8};
static const int32_t FACE_OFFSETS[3] = {0, 4, 8};
static const double QUERY[12] = {
    0.1, 0.1, 0.1,   0.9, 0.1, 0.1,
    2.1, 0.1, 0.1,   2.9, 0.1, 0.1
};
static const int32_t QUERY_OFFSETS[3] = {0, 2, 4};
static const double MATRICES[32] = {
    1.0, 0.0, 0.0, 0.5,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, -3.0,  0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.5,  0.0, 0.0, 1.0, -4.0,  0.0, 0.0, 0.0, 1.0
};
static const double HOMOGENEOUS[16] = {
     0.5,  0.25, -1.0, 2.0,   -0.5, 0.75, -1.5, 3.0,
     0.25, -0.5, -2.0, 4.0,    0.0, 0.0,  -0.5, 1.0
};
static const uint32_t RNG_KEY[2] = {0x12345678u, 0x9abcdef0u};
static const uint32_t RNG_COUNTER[2] = {0u, 0u};

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

static gffx_buffer buffer_of(void *data, uint64_t bytes) {
    gffx_buffer buffer = {0};
    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.abi_version = GFFX_ABI_VERSION;
    buffer.data = data;
    buffer.capacity_bytes = bytes;
    buffer.alignment = 8u;
    buffer.device_type = GFFX_DEVICE_CPU;
    buffer.device_index = 0;
    return buffer;
}

/* --------------------------------------------------------------------------- operation table */

typedef struct {
    unsigned char *out;        /* output arena, sized SWEEP_OUT_BYTES */
    unsigned char *workspace;  /* scratch arena, sized SWEEP_WS_BYTES */
    int64_t batch_count;       /* 2 for the packed run, 1 for a standalone element */
    int element;               /* which element a standalone run covers */
} sweep_ctx;

/*
 * A byte range of the output arena holding batch element 0's share of one output.
 *
 * Outputs are packed per output region, not interleaved, so element 0 occupies a prefix of every
 * region rather than the first half of the arena. Comparing a naive half-arena window compares
 * element 1's data against zero padding and fails for reasons that say nothing about batching,
 * which is exactly the mistake the first draft of this fixture made.
 */
typedef struct {
    int64_t offset;
    int64_t bytes;
} sweep_range_bytes;

typedef struct {
    const char *name;
    /* Bytes of the output arena this operation writes. Exact, so the overwrite fixture can
     * distinguish "written" from "left alone". Zero means the extent is not pinned. */
    int64_t out_bytes;
    int batched;               /* participates in the batch-invariance fixture */
    gffx_status (*invoke)(sweep_ctx *ctx);
    /* Element 0's share of each output. Terminated by a zero-byte entry. */
    sweep_range_bytes element0[6];
    /* Every byte the packed run writes. The adapters lay outputs out with gaps between regions,
     * so the arena span is not the written extent and a sentinel legitimately survives between
     * regions; only these ranges may be asserted overwritten. */
    sweep_range_bytes written[6];
} sweep_op;

/* Each adapter selects the packed range for ctx->element when running standalone, so the same
 * arithmetic on the same values is compared against itself. */
static void sweep_range(const sweep_ctx *ctx, const int32_t *offsets,
                        int64_t *first, int64_t *count) {
    if (ctx->batch_count == 2) {
        *first = 0;
        *count = offsets[2];
    } else {
        *first = offsets[ctx->element];
        *count = offsets[ctx->element + 1] - offsets[ctx->element];
    }
}

static gffx_status op_face_geometry(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_vertex, vertex_count, first_face, face_count;
    int64_t vshape[2], fshape[2], nshape[2], ashape[1];
    gffx_tensor_view vertices, faces, normals, areas, valid;
    double *normal_data = (double *)ctx->out;
    double *area_data = normal_data + 24;
    unsigned char *valid_data = (unsigned char *)(area_data + 8);
    int32_t local_faces[24];
    int64_t i;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, VERT_OFFSETS, &first_vertex, &vertex_count);
    sweep_range(ctx, FACE_OFFSETS, &first_face, &face_count);
    /* face_geometry takes no offsets: indices are relative to the vertices it is given, so a
     * standalone element must be rebased rather than sliced. */
    for (i = 0; i < face_count * 3; ++i) {
        local_faces[i] = FACES[first_face * 3 + i] - (int32_t)first_vertex;
    }
    vshape[0] = vertex_count; vshape[1] = 3;
    fshape[0] = face_count; fshape[1] = 3;
    nshape[0] = face_count; nshape[1] = 3;
    ashape[0] = face_count;
    vertices = view_of((void *)(VERTS + first_vertex * 3), GFFX_DTYPE_FLOAT64, 2u, vshape,
                       stride_pair, GFFX_TENSOR_READ_ONLY);
    faces = view_of(local_faces, GFFX_DTYPE_INT32, 2u, fshape, stride_pair,
                    GFFX_TENSOR_READ_ONLY);
    normals = view_of(normal_data, GFFX_DTYPE_FLOAT64, 2u, nshape, stride_pair,
                      GFFX_TENSOR_OUTPUT);
    areas = view_of(area_data, GFFX_DTYPE_FLOAT64, 1u, ashape, stride_scalar,
                    GFFX_TENSOR_OUTPUT);
    valid = view_of(valid_data, GFFX_DTYPE_BOOL, 1u, ashape, stride_scalar, GFFX_TENSOR_OUTPUT);
    return gffx_mesh_face_geometry(&vertices, &faces, SWEEP_EPS, &context, &normals, &areas,
                                   &valid, &workspace, &diagnostic);
}

static gffx_status op_vertex_normals(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_vertex, vertex_count, first_face, face_count;
    int64_t vshape[2], fshape[2];
    gffx_tensor_view vertices, faces, normals;
    int32_t local_faces[24];
    int64_t i;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, VERT_OFFSETS, &first_vertex, &vertex_count);
    sweep_range(ctx, FACE_OFFSETS, &first_face, &face_count);
    for (i = 0; i < face_count * 3; ++i) {
        local_faces[i] = FACES[first_face * 3 + i] - (int32_t)first_vertex;
    }
    vshape[0] = vertex_count; vshape[1] = 3;
    fshape[0] = face_count; fshape[1] = 3;
    vertices = view_of((void *)(VERTS + first_vertex * 3), GFFX_DTYPE_FLOAT64, 2u, vshape,
                       stride_pair, GFFX_TENSOR_READ_ONLY);
    faces = view_of(local_faces, GFFX_DTYPE_INT32, 2u, fshape, stride_pair,
                    GFFX_TENSOR_READ_ONLY);
    normals = view_of(ctx->out, GFFX_DTYPE_FLOAT64, 2u, vshape, stride_pair,
                      GFFX_TENSOR_OUTPUT);
    return gffx_mesh_vertex_normals(&vertices, &faces, SWEEP_EPS, GFFX_MESH_WEIGHTING_AREA,
                                    &context, &normals, &workspace, &diagnostic);
}

static gffx_status op_gather_faces(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_vertex, vertex_count, first_face, face_count;
    int64_t vshape[2], fshape[2], oshape[3];
    int64_t ostrides[3];
    gffx_tensor_view vertices, faces, gathered;
    int32_t local_faces[24];
    int64_t i;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, VERT_OFFSETS, &first_vertex, &vertex_count);
    sweep_range(ctx, FACE_OFFSETS, &first_face, &face_count);
    for (i = 0; i < face_count * 3; ++i) {
        local_faces[i] = FACES[first_face * 3 + i] - (int32_t)first_vertex;
    }
    vshape[0] = vertex_count; vshape[1] = 3;
    fshape[0] = face_count; fshape[1] = 3;
    oshape[0] = face_count; oshape[1] = 3; oshape[2] = 3;
    ostrides[0] = 9; ostrides[1] = 3; ostrides[2] = 1;
    vertices = view_of((void *)(VERTS + first_vertex * 3), GFFX_DTYPE_FLOAT64, 2u, vshape,
                       stride_pair, GFFX_TENSOR_READ_ONLY);
    faces = view_of(local_faces, GFFX_DTYPE_INT32, 2u, fshape, stride_pair,
                    GFFX_TENSOR_READ_ONLY);
    gathered = view_of(ctx->out, GFFX_DTYPE_FLOAT64, 3u, oshape, ostrides, GFFX_TENSOR_OUTPUT);
    return gffx_mesh_gather_faces(&vertices, &faces, &context, &gathered, &workspace,
                                  &diagnostic);
}

static gffx_status op_transform_points(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_point, point_count;
    int64_t pshape[2], mshape[3], oshape[1], hshape[2];
    int64_t hstrides[2];
    int32_t offsets[2];
    gffx_tensor_view points, matrices, point_offsets, homogeneous;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, QUERY_OFFSETS, &first_point, &point_count);
    offsets[0] = 0; offsets[1] = (int32_t)point_count;
    pshape[0] = point_count; pshape[1] = 3;
    mshape[0] = ctx->batch_count; mshape[1] = 4; mshape[2] = 4;
    oshape[0] = ctx->batch_count + 1;
    hshape[0] = point_count; hshape[1] = 4;
    hstrides[0] = 4; hstrides[1] = 1;
    points = view_of((void *)(QUERY + first_point * 3), GFFX_DTYPE_FLOAT64, 2u, pshape,
                     stride_pair, GFFX_TENSOR_READ_ONLY);
    matrices = view_of((void *)(MATRICES + (ctx->batch_count == 2 ? 0 : ctx->element * 16)),
                       GFFX_DTYPE_FLOAT64, 3u, mshape, stride_mat, GFFX_TENSOR_READ_ONLY);
    point_offsets = view_of((void *)(ctx->batch_count == 2 ? QUERY_OFFSETS : offsets),
                            GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    homogeneous = view_of(ctx->out, GFFX_DTYPE_FLOAT64, 2u, hshape, hstrides,
                          GFFX_TENSOR_OUTPUT);
    return gffx_transforms_transform_points(&points, &matrices, &point_offsets, &context,
                                            &homogeneous, &workspace, &diagnostic);
}

static gffx_status op_perspective_divide(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t hshape[2], nshape[2], vshape[1];
    int64_t hstrides[2];
    gffx_tensor_view homogeneous, ndc, valid;
    double *ndc_data = (double *)ctx->out;
    unsigned char *valid_data = (unsigned char *)(ndc_data + 12);

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    hshape[0] = 4; hshape[1] = 4;
    nshape[0] = 4; nshape[1] = 3;
    vshape[0] = 4;
    hstrides[0] = 4; hstrides[1] = 1;
    homogeneous = view_of((void *)HOMOGENEOUS, GFFX_DTYPE_FLOAT64, 2u, hshape, hstrides,
                          GFFX_TENSOR_READ_ONLY);
    ndc = view_of(ndc_data, GFFX_DTYPE_FLOAT64, 2u, nshape, stride_pair, GFFX_TENSOR_OUTPUT);
    valid = view_of(valid_data, GFFX_DTYPE_BOOL, 1u, vshape, stride_scalar, GFFX_TENSOR_OUTPUT);
    return gffx_transforms_perspective_divide(&homogeneous, SWEEP_EPS, &context, &ndc, &valid,
                                              &workspace, &diagnostic);
}

static gffx_status op_edge_topology(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_face, face_count;
    int64_t fshape[2], eshape[2], oshape[1], efoshape[1], efshape[1], meshape[1];
    int32_t offsets[2];
    int32_t local_faces[24];
    gffx_tensor_view faces, face_offsets, edges, edge_face_offsets, edge_faces,
        mesh_edge_offsets;
    int32_t *edge_data = (int32_t *)ctx->out;
    int32_t *edge_face_offset_data = edge_data + 64;
    int32_t *edge_face_data = edge_face_offset_data + 64;
    int32_t *mesh_edge_offset_data = edge_face_data + 64;
    int64_t first_vertex, vertex_count, i;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, FACE_OFFSETS, &first_face, &face_count);
    sweep_range(ctx, VERT_OFFSETS, &first_vertex, &vertex_count);
    for (i = 0; i < face_count * 3; ++i) {
        local_faces[i] = FACES[first_face * 3 + i] - (int32_t)first_vertex;
    }
    offsets[0] = 0; offsets[1] = (int32_t)face_count;
    fshape[0] = face_count; fshape[1] = 3;
    oshape[0] = ctx->batch_count + 1;
    eshape[0] = face_count * 3; eshape[1] = 2;
    efoshape[0] = face_count * 3 + 1;
    meshape[0] = ctx->batch_count + 1;
    faces = view_of(local_faces, GFFX_DTYPE_INT32, 2u, fshape, stride_pair,
                    GFFX_TENSOR_READ_ONLY);
    face_offsets = view_of((void *)(ctx->batch_count == 2 ? FACE_OFFSETS : offsets),
                           GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    edges = view_of(edge_data, GFFX_DTYPE_INT32, 2u, eshape, stride_two, GFFX_TENSOR_OUTPUT);
    /* A separate shape array: view_of retains the pointer, so reusing one array would
     * retroactively change the view already built from it. */
    edge_face_offsets = view_of(edge_face_offset_data, GFFX_DTYPE_INT32, 1u, efoshape,
                                stride_scalar, GFFX_TENSOR_OUTPUT);
    efshape[0] = face_count * 3;
    edge_faces = view_of(edge_face_data, GFFX_DTYPE_INT32, 1u, efshape, stride_scalar,
                         GFFX_TENSOR_OUTPUT);
    mesh_edge_offsets = view_of(mesh_edge_offset_data, GFFX_DTYPE_INT32, 1u, meshape,
                                stride_scalar, GFFX_TENSOR_OUTPUT);
    return gffx_mesh_build_edge_topology(&faces, &face_offsets, &context, &edges,
                                         &edge_face_offsets, &edge_faces, &mesh_edge_offsets,
                                         &workspace, &diagnostic);
}

static gffx_status op_knn(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_query, query_count, first_reference, reference_count;
    int64_t qshape[2], rshape[2], oshape[1], dshape[2];
    int64_t dstrides[2];
    int32_t query_offsets_local[2];
    int32_t reference_offsets_local[2];
    gffx_tensor_view query, reference, query_offsets, reference_offsets, distance, index, valid;
    double *distance_data = (double *)ctx->out;
    int32_t *index_data = (int32_t *)(distance_data + 16);
    unsigned char *valid_data = (unsigned char *)(index_data + 16);

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, QUERY_OFFSETS, &first_query, &query_count);
    sweep_range(ctx, VERT_OFFSETS, &first_reference, &reference_count);
    query_offsets_local[0] = 0; query_offsets_local[1] = (int32_t)query_count;
    reference_offsets_local[0] = 0; reference_offsets_local[1] = (int32_t)reference_count;
    qshape[0] = query_count; qshape[1] = 3;
    rshape[0] = reference_count; rshape[1] = 3;
    oshape[0] = ctx->batch_count + 1;
    dshape[0] = query_count; dshape[1] = 2;
    dstrides[0] = 2; dstrides[1] = 1;
    query = view_of((void *)(QUERY + first_query * 3), GFFX_DTYPE_FLOAT64, 2u, qshape,
                    stride_pair, GFFX_TENSOR_READ_ONLY);
    reference = view_of((void *)(VERTS + first_reference * 3), GFFX_DTYPE_FLOAT64, 2u, rshape,
                        stride_pair, GFFX_TENSOR_READ_ONLY);
    query_offsets = view_of((void *)(ctx->batch_count == 2 ? QUERY_OFFSETS : query_offsets_local),
                            GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    reference_offsets = view_of(
        (void *)(ctx->batch_count == 2 ? VERT_OFFSETS : reference_offsets_local),
        GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    distance = view_of(distance_data, GFFX_DTYPE_FLOAT64, 2u, dshape, dstrides,
                       GFFX_TENSOR_OUTPUT);
    index = view_of(index_data, GFFX_DTYPE_INT32, 2u, dshape, dstrides, GFFX_TENSOR_OUTPUT);
    valid = view_of(valid_data, GFFX_DTYPE_BOOL, 2u, dshape, dstrides, GFFX_TENSOR_OUTPUT);
    return gffx_points_knn(&query, &reference, &query_offsets, &reference_offsets, 2, &context,
                           &distance, &index, &valid, &workspace, &diagnostic);
}

static gffx_status op_closest_point(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_query, query_count, first_vertex, vertex_count, first_face, face_count;
    int64_t qshape[2], vshape[2], fshape[2], oshape[1], sshape[1];
    int32_t point_offsets_local[2], vertex_offsets_local[2], face_offsets_local[2];
    int32_t local_faces[24];
    gffx_tensor_view query, vertices, faces, point_offsets, vertex_offsets, face_offsets,
        distance, face_index, barycentric, closest, valid;
    double *distance_data = (double *)ctx->out;
    double *bary_data = distance_data + 8;
    double *closest_data = bary_data + 24;
    int32_t *face_data = (int32_t *)(closest_data + 24);
    unsigned char *valid_data = (unsigned char *)(face_data + 8);
    int64_t i;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, QUERY_OFFSETS, &first_query, &query_count);
    sweep_range(ctx, VERT_OFFSETS, &first_vertex, &vertex_count);
    sweep_range(ctx, FACE_OFFSETS, &first_face, &face_count);
    for (i = 0; i < face_count * 3; ++i) {
        local_faces[i] = FACES[first_face * 3 + i] - (int32_t)first_vertex;
    }
    point_offsets_local[0] = 0; point_offsets_local[1] = (int32_t)query_count;
    vertex_offsets_local[0] = 0; vertex_offsets_local[1] = (int32_t)vertex_count;
    face_offsets_local[0] = 0; face_offsets_local[1] = (int32_t)face_count;
    qshape[0] = query_count; qshape[1] = 3;
    vshape[0] = vertex_count; vshape[1] = 3;
    fshape[0] = face_count; fshape[1] = 3;
    oshape[0] = ctx->batch_count + 1;
    sshape[0] = query_count;
    query = view_of((void *)(QUERY + first_query * 3), GFFX_DTYPE_FLOAT64, 2u, qshape,
                    stride_pair, GFFX_TENSOR_READ_ONLY);
    vertices = view_of((void *)(VERTS + first_vertex * 3), GFFX_DTYPE_FLOAT64, 2u, vshape,
                       stride_pair, GFFX_TENSOR_READ_ONLY);
    faces = view_of(local_faces, GFFX_DTYPE_INT32, 2u, fshape, stride_pair,
                    GFFX_TENSOR_READ_ONLY);
    point_offsets = view_of(
        (void *)(ctx->batch_count == 2 ? QUERY_OFFSETS : point_offsets_local),
        GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    vertex_offsets = view_of(
        (void *)(ctx->batch_count == 2 ? VERT_OFFSETS : vertex_offsets_local),
        GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    face_offsets = view_of(
        (void *)(ctx->batch_count == 2 ? FACE_OFFSETS : face_offsets_local),
        GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    distance = view_of(distance_data, GFFX_DTYPE_FLOAT64, 1u, sshape, stride_scalar,
                       GFFX_TENSOR_OUTPUT);
    face_index = view_of(face_data, GFFX_DTYPE_INT32, 1u, sshape, stride_scalar,
                         GFFX_TENSOR_OUTPUT);
    barycentric = view_of(bary_data, GFFX_DTYPE_FLOAT64, 2u, qshape, stride_pair,
                          GFFX_TENSOR_OUTPUT);
    closest = view_of(closest_data, GFFX_DTYPE_FLOAT64, 2u, qshape, stride_pair,
                      GFFX_TENSOR_OUTPUT);
    valid = view_of(valid_data, GFFX_DTYPE_BOOL, 1u, sshape, stride_scalar, GFFX_TENSOR_OUTPUT);
    return gffx_points_closest_point_on_mesh(&query, &vertices, &faces, &point_offsets,
                                             &vertex_offsets, &face_offsets, SWEEP_EPS,
                                             &context, &distance, &face_index, &barycentric,
                                             &closest, &valid, &workspace, &diagnostic);
}

static gffx_status op_sample_surface(sweep_ctx *ctx) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    gffx_buffer workspace = buffer_of(ctx->workspace, SWEEP_WS_BYTES);
    int64_t first_vertex, vertex_count, first_face, face_count;
    int64_t vshape[2], fshape[2], oshape[1], pshape[3], sshape[2], kshape[1], cshape[1];
    int64_t pstrides[3], sstrides[2];
    int32_t vertex_offsets_local[2], face_offsets_local[2];
    int32_t local_faces[24];
    gffx_tensor_view vertices, faces, vertex_offsets, face_offsets, key, counter, points,
        face_index, barycentric, next_counter;
    const int64_t samples = 4;
    double *point_data = (double *)ctx->out;
    double *bary_data = point_data + 64;
    int32_t *face_data = (int32_t *)(bary_data + 64);
    uint32_t *counter_data = (uint32_t *)(face_data + 32);
    int64_t i;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    sweep_range(ctx, VERT_OFFSETS, &first_vertex, &vertex_count);
    sweep_range(ctx, FACE_OFFSETS, &first_face, &face_count);
    for (i = 0; i < face_count * 3; ++i) {
        local_faces[i] = FACES[first_face * 3 + i] - (int32_t)first_vertex;
    }
    vertex_offsets_local[0] = 0; vertex_offsets_local[1] = (int32_t)vertex_count;
    face_offsets_local[0] = 0; face_offsets_local[1] = (int32_t)face_count;
    vshape[0] = vertex_count; vshape[1] = 3;
    fshape[0] = face_count; fshape[1] = 3;
    oshape[0] = ctx->batch_count + 1;
    pshape[0] = ctx->batch_count; pshape[1] = samples; pshape[2] = 3;
    pstrides[0] = samples * 3; pstrides[1] = 3; pstrides[2] = 1;
    sshape[0] = ctx->batch_count; sshape[1] = samples;
    sstrides[0] = samples; sstrides[1] = 1;
    kshape[0] = 2;
    cshape[0] = 2;
    vertices = view_of((void *)(VERTS + first_vertex * 3), GFFX_DTYPE_FLOAT64, 2u, vshape,
                       stride_pair, GFFX_TENSOR_READ_ONLY);
    faces = view_of(local_faces, GFFX_DTYPE_INT32, 2u, fshape, stride_pair,
                    GFFX_TENSOR_READ_ONLY);
    vertex_offsets = view_of(
        (void *)(ctx->batch_count == 2 ? VERT_OFFSETS : vertex_offsets_local),
        GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    face_offsets = view_of(
        (void *)(ctx->batch_count == 2 ? FACE_OFFSETS : face_offsets_local),
        GFFX_DTYPE_INT32, 1u, oshape, stride_scalar, GFFX_TENSOR_READ_ONLY);
    key = view_of((void *)RNG_KEY, GFFX_DTYPE_UINT32, 1u, kshape, stride_scalar,
                  GFFX_TENSOR_READ_ONLY);
    counter = view_of((void *)RNG_COUNTER, GFFX_DTYPE_UINT32, 1u, cshape, stride_scalar,
                      GFFX_TENSOR_READ_ONLY);
    points = view_of(point_data, GFFX_DTYPE_FLOAT64, 3u, pshape, pstrides, GFFX_TENSOR_OUTPUT);
    face_index = view_of(face_data, GFFX_DTYPE_INT32, 2u, sshape, sstrides,
                         GFFX_TENSOR_OUTPUT);
    barycentric = view_of(bary_data, GFFX_DTYPE_FLOAT64, 3u, pshape, pstrides,
                          GFFX_TENSOR_OUTPUT);
    next_counter = view_of(counter_data, GFFX_DTYPE_UINT32, 1u, cshape, stride_scalar,
                           GFFX_TENSOR_OUTPUT);
    return gffx_mesh_sample_surface(&vertices, &faces, &vertex_offsets, &face_offsets, samples,
                                    &key, &counter, SWEEP_EPS, &context, &points, &face_index,
                                    &barycentric, &next_counter, &workspace, &diagnostic);
}

/*
 * Element-0 ranges are derived from each adapter's own arena layout above. Where an output holds
 * global packed indices, element 0's values are unaffected by packing because that element starts
 * at zero; element 1 legitimately differs and is therefore not compared.
 */
static const sweep_op SWEEP_OPS[] = {
    {"face_geometry",      0, 0, op_face_geometry, {{0, 0}},
        {{0, 264}, {0, 0}}},
    {"vertex_normals",     0, 0, op_vertex_normals, {{0, 0}},
        {{0, 192}, {0, 0}}},
    {"gather_faces",       0, 0, op_gather_faces, {{0, 0}},
        {{0, 576}, {0, 0}}},
    {"transform_points",   0, 1, op_transform_points,
        {{0, 64}, {0, 0}},
        {{0, 128}, {0, 0}}},
    {"perspective_divide", 0, 0, op_perspective_divide, {{0, 0}},
        {{0, 100}, {0, 0}}},
    {"edge_topology",      0, 1, op_edge_topology,
        {{0, 48}, {256, 28}, {512, 48}, {768, 8}, {0, 0}},
        {{0, 192}, {256, 100}, {512, 96}, {768, 12}, {0, 0}}},
    {"knn",                0, 1, op_knn,
        {{0, 32}, {128, 16}, {192, 4}, {0, 0}},
        {{0, 64}, {128, 32}, {192, 8}, {0, 0}}},
    {"closest_point",      0, 1, op_closest_point,
        {{0, 16}, {64, 48}, {256, 48}, {448, 8}, {480, 2}, {0, 0}},
        {{0, 32}, {64, 96}, {256, 96}, {448, 16}, {480, 4}, {0, 0}}},
    {"sample_surface",     0, 1, op_sample_surface,
        {{0, 96}, {512, 96}, {1024, 16}, {1152, 8}, {0, 0}},
        {{0, 192}, {512, 192}, {1024, 32}, {1152, 8}, {0, 0}}}
};
#define SWEEP_OP_COUNT ((int)(sizeof(SWEEP_OPS) / sizeof(SWEEP_OPS[0])))

/* ---------------------------------------------------------------------------------- helpers */

static void fill(unsigned char *data, int64_t bytes, unsigned char pattern) {
    memset(data, pattern, (size_t)bytes);
}

/* Runs one operation into a fresh arena and returns the arena, so two runs can be compared
 * byte-for-byte without either observing the other. */
static gffx_status run_into(
    const sweep_op *op, unsigned char *out, unsigned char workspace_pattern,
    int64_t batch_count, int element
) {
    sweep_ctx ctx;
    static unsigned char workspace[SWEEP_WS_BYTES];
    fill(workspace, SWEEP_WS_BYTES, workspace_pattern);
    fill(out, SWEEP_OUT_BYTES, 0x00);
    ctx.out = out;
    ctx.workspace = workspace;
    ctx.batch_count = batch_count;
    ctx.element = element;
    return op->invoke(&ctx);
}

/* -------------------------------------------------------------------------------- fixtures */

/* SW-02: every input buffer is byte-identical after every forward call. */
static int test_sw02_input_immutability(void) {
    static unsigned char out[SWEEP_OUT_BYTES];
    double verts_copy[24];
    int32_t faces_copy[24];
    double query_copy[12];
    double matrices_copy[32];
    double homogeneous_copy[16];
    int32_t vert_offsets_copy[3];
    int32_t face_offsets_copy[3];
    int32_t query_offsets_copy[3];
    uint32_t key_copy[2];
    uint32_t counter_copy[4];
    int index;

    memcpy(verts_copy, VERTS, sizeof(VERTS));
    memcpy(faces_copy, FACES, sizeof(FACES));
    memcpy(query_copy, QUERY, sizeof(QUERY));
    memcpy(matrices_copy, MATRICES, sizeof(MATRICES));
    memcpy(homogeneous_copy, HOMOGENEOUS, sizeof(HOMOGENEOUS));
    memcpy(vert_offsets_copy, VERT_OFFSETS, sizeof(VERT_OFFSETS));
    memcpy(face_offsets_copy, FACE_OFFSETS, sizeof(FACE_OFFSETS));
    memcpy(query_offsets_copy, QUERY_OFFSETS, sizeof(QUERY_OFFSETS));
    memcpy(key_copy, RNG_KEY, sizeof(RNG_KEY));
    memcpy(counter_copy, RNG_COUNTER, sizeof(RNG_COUNTER));

    for (index = 0; index < SWEEP_OP_COUNT; ++index) {
        CHECK(run_into(&SWEEP_OPS[index], out, 0x00, 2, 0) == GFFX_STATUS_OK);
        CHECK(memcmp(verts_copy, VERTS, sizeof(VERTS)) == 0);
        CHECK(memcmp(faces_copy, FACES, sizeof(FACES)) == 0);
        CHECK(memcmp(query_copy, QUERY, sizeof(QUERY)) == 0);
        CHECK(memcmp(matrices_copy, MATRICES, sizeof(MATRICES)) == 0);
        CHECK(memcmp(homogeneous_copy, HOMOGENEOUS, sizeof(HOMOGENEOUS)) == 0);
        CHECK(memcmp(vert_offsets_copy, VERT_OFFSETS, sizeof(VERT_OFFSETS)) == 0);
        CHECK(memcmp(face_offsets_copy, FACE_OFFSETS, sizeof(FACE_OFFSETS)) == 0);
        CHECK(memcmp(query_offsets_copy, QUERY_OFFSETS, sizeof(QUERY_OFFSETS)) == 0);
        CHECK(memcmp(key_copy, RNG_KEY, sizeof(RNG_KEY)) == 0);
        CHECK(memcmp(counter_copy, RNG_COUNTER, sizeof(RNG_COUNTER)) == 0);
    }
    return 0;
}

/* SW-04 and SW-05: results must not depend on what the workspace happened to contain. */
static int test_sw04_dirty_workspace(void) {
    static unsigned char zeroed[SWEEP_OUT_BYTES];
    static unsigned char dirty[SWEEP_OUT_BYTES];
    static unsigned char residue[SWEEP_OUT_BYTES];
    int index;

    for (index = 0; index < SWEEP_OP_COUNT; ++index) {
        const sweep_op *op = &SWEEP_OPS[index];
        int range;
        CHECK(run_into(op, zeroed, 0x00, 2, 0) == GFFX_STATUS_OK);
        CHECK(run_into(op, dirty, 0xA5, 2, 0) == GFFX_STATUS_OK);
        CHECK(run_into(op, residue, 0xFF, 2, 0) == GFFX_STATUS_OK);
        for (range = 0; op->written[range].bytes != 0; ++range) {
            const sweep_range_bytes *r = &op->written[range];
            CHECK(memcmp(zeroed + r->offset, dirty + r->offset, (size_t)r->bytes) == 0);
            CHECK(memcmp(zeroed + r->offset, residue + r->offset, (size_t)r->bytes) == 0);
        }
    }
    return 0;
}

/* SW-06: identical inputs give bit-identical outputs, including on reused buffers. */
static int test_sw06_repeated_call(void) {
    static unsigned char first[SWEEP_OUT_BYTES];
    static unsigned char second[SWEEP_OUT_BYTES];
    int index;

    for (index = 0; index < SWEEP_OP_COUNT; ++index) {
        const sweep_op *op = &SWEEP_OPS[index];
        int range;
        CHECK(run_into(op, first, 0x00, 2, 0) == GFFX_STATUS_OK);
        CHECK(run_into(op, second, 0x00, 2, 0) == GFFX_STATUS_OK);
        for (range = 0; op->written[range].bytes != 0; ++range) {
            const sweep_range_bytes *r = &op->written[range];
            CHECK(memcmp(first + r->offset, second + r->offset, (size_t)r->bytes) == 0);
        }
    }
    return 0;
}

/* SW-07: a forward output is overwritten, never accumulated into or left alone. */
static int test_sw07_output_overwrite(void) {
    static unsigned char sentinel_run[SWEEP_OUT_BYTES];
    static unsigned char clean_run[SWEEP_OUT_BYTES];
    sweep_ctx ctx;
    static unsigned char workspace[SWEEP_WS_BYTES];
    int index;

    for (index = 0; index < SWEEP_OP_COUNT; ++index) {
        const sweep_op *op = &SWEEP_OPS[index];
        int range;
        CHECK(run_into(op, clean_run, 0x00, 2, 0) == GFFX_STATUS_OK);
        /* Pre-fill with a pattern the operation would have to overwrite to match. */
        fill(sentinel_run, SWEEP_OUT_BYTES, 0x5A);
        fill(workspace, SWEEP_WS_BYTES, 0x00);
        ctx.out = sentinel_run;
        ctx.workspace = workspace;
        ctx.batch_count = 2;
        ctx.element = 0;
        CHECK(op->invoke(&ctx) == GFFX_STATUS_OK);
        for (range = 0; op->written[range].bytes != 0; ++range) {
            const sweep_range_bytes *r = &op->written[range];
            CHECK(memcmp(sentinel_run + r->offset, clean_run + r->offset,
                         (size_t)r->bytes) == 0);
        }
    }
    return 0;
}

/* SW-01: an element's packed result equals its standalone result, bit for bit. */
static int test_sw01_batch_invariance(void) {
    static unsigned char packed[SWEEP_OUT_BYTES];
    static unsigned char alone[SWEEP_OUT_BYTES];
    int index;

    for (index = 0; index < SWEEP_OP_COUNT; ++index) {
        const sweep_op *op = &SWEEP_OPS[index];
        if (!op->batched) continue;
        int range;
        int compared = 0;
        CHECK(run_into(op, packed, 0x00, 2, 0) == GFFX_STATUS_OK);
        CHECK(run_into(op, alone, 0x00, 1, 0) == GFFX_STATUS_OK);
        for (range = 0; op->element0[range].bytes != 0; ++range) {
            const sweep_range_bytes *r = &op->element0[range];
            int64_t byte;
            int nonzero = 0;
            CHECK(memcmp(packed + r->offset, alone + r->offset, (size_t)r->bytes) == 0);
            /* A range of zeros would compare equal without proving anything, so require that
             * the operation actually wrote something into every range being compared. */
            for (byte = 0; byte < r->bytes; ++byte) {
                if (packed[r->offset + byte] != 0) nonzero = 1;
            }
            CHECK(nonzero);
            ++compared;
        }
        CHECK(compared > 0);
    }
    return 0;
}

/* SW-09: a workspace query is a pure function of its arguments. */
static int test_sw09_workspace_query_stability(void) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    uint64_t bytes_a = 0u, bytes_b = 0u, align_a = 0u, align_b = 0u;
    int repeat;

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    for (repeat = 0; repeat < 3; ++repeat) {
        CHECK(gffx_mesh_face_geometry_workspace(8, 8, GFFX_DTYPE_FLOAT64, &context, &bytes_a,
                                                &align_a, &diagnostic) == GFFX_STATUS_OK);
        CHECK(gffx_mesh_face_geometry_workspace(8, 8, GFFX_DTYPE_FLOAT64, &context, &bytes_b,
                                                &align_b, &diagnostic) == GFFX_STATUS_OK);
        CHECK(bytes_a == bytes_b && align_a == align_b);
        CHECK(gffx_mesh_sample_surface_workspace(8, 8, 4, GFFX_DTYPE_FLOAT64, &context, &bytes_a,
                                                 &align_a, &diagnostic) == GFFX_STATUS_OK);
        CHECK(gffx_mesh_sample_surface_workspace(8, 8, 4, GFFX_DTYPE_FLOAT64, &context, &bytes_b,
                                                 &align_b, &diagnostic) == GFFX_STATUS_OK);
        CHECK(bytes_a == bytes_b && align_a == align_b);
        CHECK(gffx_points_knn_workspace(4, 8, 2, GFFX_DTYPE_FLOAT64, &context, &bytes_a,
                                        &align_a, &diagnostic) == GFFX_STATUS_OK);
        CHECK(gffx_points_knn_workspace(4, 8, 2, GFFX_DTYPE_FLOAT64, &context, &bytes_b,
                                        &align_b, &diagnostic) == GFFX_STATUS_OK);
        CHECK(bytes_a == bytes_b && align_a == align_b);
    }
    return 0;
}

/* SW-08: an input and an output sharing memory is rejected, not silently miscomputed. */
static int test_sw08_aliasing(void) {
    gffx_execution_context context = cpu_context();
    gffx_diagnostic_buffer diagnostic = {0};
    static unsigned char workspace[SWEEP_WS_BYTES];
    gffx_buffer workspace_buffer = buffer_of(workspace, SWEEP_WS_BYTES);
    double shared[24];
    int32_t faces_copy[24];
    int64_t vshape[2], fshape[2], ashape[1];
    gffx_tensor_view vertices, faces, normals, areas, valid;
    double areas_data[8];
    unsigned char valid_data[8];

    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    memcpy(shared, VERTS, sizeof(VERTS));
    memcpy(faces_copy, FACES, sizeof(FACES));
    vshape[0] = 8; vshape[1] = 3;
    fshape[0] = 8; fshape[1] = 3;
    ashape[0] = 8;
    /* unit_normals writes over the very buffer vertices reads from. */
    vertices = view_of(shared, GFFX_DTYPE_FLOAT64, 2u, vshape, stride_pair,
                       GFFX_TENSOR_READ_ONLY);
    faces = view_of(faces_copy, GFFX_DTYPE_INT32, 2u, fshape, stride_pair,
                    GFFX_TENSOR_READ_ONLY);
    normals = view_of(shared, GFFX_DTYPE_FLOAT64, 2u, fshape, stride_pair, GFFX_TENSOR_OUTPUT);
    areas = view_of(areas_data, GFFX_DTYPE_FLOAT64, 1u, ashape, stride_scalar,
                    GFFX_TENSOR_OUTPUT);
    valid = view_of(valid_data, GFFX_DTYPE_BOOL, 1u, ashape, stride_scalar, GFFX_TENSOR_OUTPUT);
    CHECK(gffx_mesh_face_geometry(&vertices, &faces, SWEEP_EPS, &context, &normals, &areas,
                                  &valid, &workspace_buffer, &diagnostic)
          == GFFX_STATUS_INVALID_ARGUMENT);
    return 0;
}

int main(void) {
    int result;
    result = test_sw01_batch_invariance(); if (result != 0) return result;
    result = test_sw02_input_immutability(); if (result != 0) return result;
    result = test_sw04_dirty_workspace(); if (result != 0) return result;
    result = test_sw06_repeated_call(); if (result != 0) return result;
    result = test_sw07_output_overwrite(); if (result != 0) return result;
    result = test_sw08_aliasing(); if (result != 0) return result;
    result = test_sw09_workspace_query_stability(); if (result != 0) return result;
    return 0;
}
