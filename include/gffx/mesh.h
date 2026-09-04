#ifndef GFFX_MESH_H
#define GFFX_MESH_H

#include <gffx/execution.h>
#include <gffx/tensor.h>

/*
 * mesh.face_geometry - per-face unit normals, areas, and validity.
 *
 * For face k with vertex indices (i0, i1, i2):
 *   e1 = vertices[i1] - vertices[i0]
 *   e2 = vertices[i2] - vertices[i0]
 *   c  = cross(e1, e2)              right-handed cross product
 *   d  = ||c||                      doubled triangle area
 *   valid[k] = (d > eps)            strict, evaluated in double precision
 *   valid:   unit_normals[k] = c / d,  areas[k] = d / 2
 *   invalid: unit_normals[k] = 0,      areas[k] = 0
 *
 * Contract summary (the project acceptance record is normative):
 * - vertices is [V,3] FLOAT32 or FLOAT64 and selects the computation dtype; faces is [F,3]
 *   INT32 with every index in [0, V). All views are dense, C-contiguous, CPU, and validated
 *   before any data dereference.
 * - unit_normals [F,3] and areas [F] use the vertices dtype; valid [F] is GFFX_DTYPE_BOOL with
 *   one byte per element. Outputs carry GFFX_TENSOR_OUTPUT and may not alias any input or each
 *   other.
 * - eps must be finite and >= 0. A NaN doubled area compares as not-valid and takes the
 *   invalid branch; the valid branch performs the stated arithmetic under IEEE semantics with
 *   no clamping or sanitization.
 * - Faces are processed independently in ascending index order; repeated calls on identical
 *   inputs are bitwise identical for the same build and configuration.
 * - The workspace pointer may be NULL when the workspace query reports zero required bytes.
 *   The scalar CPU reference requires zero bytes; callers must still use the query.
 *
 * Backward: accumulates d(sum(grad_unit_normals . unit_normals + grad_areas . areas))/d(vertices)
 * into grad_vertices [V,3], overwriting it. Either cotangent view pointer may be NULL, which is
 * treated as a zero cotangent, but not both. Contributions are accumulated per face in ascending
 * face order (deterministic). Invalid faces contribute exactly zero gradient.
 */

GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_mesh_face_geometry_workspace(
    int64_t vertex_count,
    int64_t face_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_face_geometry(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *unit_normals,
    gffx_tensor_view *areas,
    gffx_tensor_view *valid,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_face_geometry_backward(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    const gffx_tensor_view *grad_unit_normals,
    const gffx_tensor_view *grad_areas,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

/*
 * mesh.vertex_normals - accumulated, normalized per-vertex normals.
 *
 * Using the face quantities of mesh.face_geometry with the same eps: valid faces contribute
 * (d/2)*n (area mode) or n (uniform mode) to each of their three vertices; invalid faces
 * contribute nothing. Each accumulated sum s is normalized when ||s|| > eps (strict, compared
 * in double precision) and is otherwise the exact zero vector, which also covers isolated
 * vertices. Accumulation is per-face ascending, then per-vertex ascending; repeated calls are
 * bitwise identical.
 *
 * The workspace query reports the backward requirement: vertex_count * 3 * sizeof(dtype)
 * bytes at dtype alignment. The forward pass requires zero bytes and accepts a null workspace;
 * the backward pass with a nonzero vertex count requires at least the reported capacity and
 * fails with GFFX_STATUS_INSUFFICIENT_WORKSPACE otherwise. The backward cotangent
 * grad_unit_normals [V,3] is required, grad_vertices [V,3] is overwritten, and invalid faces
 * and zero-branch vertices contribute exactly zero gradient.
 */

#define GFFX_MESH_WEIGHTING_AREA UINT32_C(1)
#define GFFX_MESH_WEIGHTING_UNIFORM UINT32_C(2)

/*
 * mesh.gather_faces - per-face corner vertex positions.
 *
 * face_vertices[k][j][:] = vertices[faces[k][j]][:], preserving face and corner order exactly.
 * A pure gather: no arithmetic, no eps, no validity output, and values including NaN and
 * infinity are copied bit-for-bit. Backward scatter-adds the cotangent into grad_vertices [V,3]
 * in ascending face then corner order, overwriting it first. Both directions require zero
 * workspace bytes.
 */

GFFX_API gffx_status GFFX_CALL gffx_mesh_gather_faces_workspace(
    int64_t vertex_count,
    int64_t face_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_gather_faces(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_execution_context *context,
    gffx_tensor_view *face_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_gather_faces_backward(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *grad_face_vertices,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_vertex_normals_workspace(
    int64_t vertex_count,
    int64_t face_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_vertex_normals(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    uint32_t weighting,
    const gffx_execution_context *context,
    gffx_tensor_view *unit_normals,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_vertex_normals_backward(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    double eps,
    uint32_t weighting,
    const gffx_tensor_view *grad_unit_normals,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

/*
 * mesh.build_edge_topology - canonical undirected edges and their incident faces.
 *
 * Each face contributes three half-edges, canonicalized to (min, max), so the half-edge count is
 * exactly 3F and every output capacity is exact. Within a batch element, half-edges are ordered
 * by (min_vertex, max_vertex, face_index); equal canonical edges form one group that becomes one
 * row of edges, with its member faces written to edge_faces in ascending face order. Non-manifold
 * edges keep every incident face, degenerate self-edges (v, v) are retained like any other edge,
 * and batch elements never merge.
 *
 * With E the total unique edge count: rows [0, E) of edges are valid and every trailing row is
 * (-1, -1); edge_faces is fully written with exactly 3F incidences; entries [0, E] of
 * edge_face_offsets index edge_faces and trailing entries repeat the final value 3F so the array
 * stays nondecreasing; mesh_edge_offsets starts at 0 and ends at E.
 *
 * face_offsets is int32 [B+1] following the packed-offset rules. Face indices must be
 * nonnegative but are not range-checked, this operation receiving no vertices. Every output is
 * integer topology and therefore nondifferentiable, so there is no backward entry point. This is
 * setup-class work and is not permitted inside a claimed real-time frame path.
 *
 * The workspace query reports 3 * F * 3 * sizeof(int32) bytes at alignment 4, one
 * (min, max, face) triple per half-edge; sorting is an in-place heapsort over those triples.
 */

GFFX_API gffx_status GFFX_CALL gffx_mesh_build_edge_topology_workspace(
    int64_t face_count,
    int64_t batch_count,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_build_edge_topology(
    const gffx_tensor_view *faces,
    const gffx_tensor_view *face_offsets,
    const gffx_execution_context *context,
    gffx_tensor_view *edges,
    gffx_tensor_view *edge_face_offsets,
    gffx_tensor_view *edge_faces,
    gffx_tensor_view *mesh_edge_offsets,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

/*
 * mesh.sample_surface - area-weighted uniform sampling of triangle surfaces.
 *
 * Randomness is Philox4x32-10, counter-based and stateless; GFFX owns no random state and never
 * touches a framework global generator. For batch element b and sample s the 128-bit counter is
 * (rng_counter[0], rng_counter[1], (uint32)b, (uint32)s) under key (rng_key[0], rng_key[1]).
 * Embedding (b, s) in the counter rather than iterating a stream makes each sample independent
 * of evaluation order, so results are reproducible under any future parallelization. The four
 * output words are used as: u0 selects the face, u1 and u2 form the barycentric coordinates, and
 * u3 is reserved. A uniform in [0, 1) is word * 2^-32.
 *
 * A face is eligible when its doubled area exceeds eps, the same rule mesh.face_geometry uses.
 * Eligible faces are selected proportionally to true area through a cumulative table located by
 * binary search; that table is accumulated in double precision regardless of operand dtype,
 * because a float32 running sum loses monotonicity over many faces and would bias selection.
 * Requesting S > 0 samples from an element with no eligible face is INVALID_ARGUMENT; S = 0 is
 * always valid. Barycentrics use the square-root map b0 = 1 - sqrt(r1), b1 = sqrt(r1)*(1 - r2),
 * b2 = sqrt(r1)*r2, which is uniform over the triangle.
 *
 * next_counter is rng_counter incremented by one as a 64-bit little-endian value, low word
 * first, wrapping to zero past the maximum.
 *
 * Backward is a pure scatter needing neither the vertices nor the generator, points being linear
 * in the vertices once face and weights are fixed:
 * grad_vertices[faces[k][i]] += b_i * grad_points[b][s], overwritten then accumulated in
 * ascending (b, s, corner) order. Selection, probabilities, face_index, barycentric, S, the key,
 * and the counter are all nondifferentiable.
 *
 * The forward workspace query reports F * sizeof(double) bytes at alignment 8; the backward
 * requires zero.
 */

GFFX_API gffx_status GFFX_CALL gffx_mesh_sample_surface_workspace(
    int64_t vertex_count,
    int64_t face_count,
    int64_t sample_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_sample_surface(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    int64_t sample_count,
    const gffx_tensor_view *rng_key,
    const gffx_tensor_view *rng_counter,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *points,
    gffx_tensor_view *face_index,
    gffx_tensor_view *barycentric,
    gffx_tensor_view *next_counter,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_sample_surface_backward(
    const gffx_tensor_view *faces,
    const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,
    const gffx_tensor_view *grad_points,
    const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

/*
 * mesh.validate - eager setup-class survey of a mesh.
 *
 * This utility reports rather than gates. An operation kernel rejects the first problem it finds
 * and returns INVALID_ARGUMENT because it must not dereference bad memory; mesh.validate instead
 * surveys the whole mesh and returns OK with a populated report, so a caller sees everything at
 * once. It is not a substitute for the mandatory per-call validation: passing it exempts no later
 * call, and v0.1 offers no flag to skip that checking.
 *
 * Findings are established structurally before geometrically, and the survey stops short of work
 * that would be unsafe given what it already found. Malformed offsets return immediately, since
 * no element range can then be trusted. An out-of-range or cross-element face index makes vertex
 * lookup unsafe, so the degenerate-face and unreferenced-vertex surveys are skipped while the
 * recorded finding bits remain set.
 *
 * The non-finite geometry survey is opt-in through GFFX_MESH_VALIDATE_GEOMETRY because it costs
 * an extra O(V) pass. When it is not requested, nonfinite_vertex_count is -1 rather than 0, so
 * "not checked" stays distinguishable from "checked and clean". A non-finite vertex is not an
 * error here; reporting it is the job. Workspace is zero bytes and there is no backward entry
 * point, the report being a diagnostic rather than a differentiable value.
 */

#define GFFX_MESH_VALIDATE_GEOMETRY UINT32_C(1)

#define GFFX_MESH_FINDING_OFFSETS UINT32_C(1)
#define GFFX_MESH_FINDING_FACE_INDEX_RANGE UINT32_C(2)
#define GFFX_MESH_FINDING_FACE_INDEX_BATCH UINT32_C(4)
#define GFFX_MESH_FINDING_DEGENERATE_FACE UINT32_C(8)
#define GFFX_MESH_FINDING_NONFINITE_GEOMETRY UINT32_C(16)
#define GFFX_MESH_FINDING_UNREFERENCED_VERTEX UINT32_C(32)

typedef struct gffx_mesh_validation_report {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t findings;
    uint32_t reserved0;
    int64_t first_bad_face;
    int64_t first_bad_offset_batch;
    int64_t degenerate_face_count;
    int64_t nonfinite_vertex_count;
    int64_t unreferenced_vertex_count;
    uint64_t reserved[3];
} gffx_mesh_validation_report;

GFFX_API gffx_status GFFX_CALL gffx_mesh_validate_workspace(
    int64_t vertex_count,
    int64_t face_count,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_API gffx_status GFFX_CALL gffx_mesh_validate(
    const gffx_tensor_view *vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    double eps,
    uint32_t flags,
    const gffx_execution_context *context,
    gffx_mesh_validation_report *report,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif /* GFFX_MESH_H */
