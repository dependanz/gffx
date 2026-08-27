/*
 * mesh.build_edge_topology - Phase 2 CPU reference kernel.
 *
 * Each face contributes three half-edges canonicalized to (min, max), so the half-edge count is
 * exactly 3F and every output capacity is exact. Half-edges are sorted per batch element by the
 * triple (min, max, face) using an in-place heapsort over the caller workspace, then equal
 * canonical edges are grouped into one output row whose incident faces land in ascending face
 * order. Every output is integer topology, so this operation has no backward entry point.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/tensor.h>

#include "internal.h"
#include "mesh_common.h"

#include <stdint.h>

/* Workspace layout: one (min, max, face) triple per half-edge, interleaved. */
#define GFFX_EDGE_TRIPLE 3

static int gffx_edge_triple_less(const int32_t *a, const int32_t *b) {
    if (a[0] != b[0]) return a[0] < b[0];
    if (a[1] != b[1]) return a[1] < b[1];
    return a[2] < b[2];
}

static void gffx_edge_triple_swap(int32_t *a, int32_t *b) {
    int index;
    for (index = 0; index < GFFX_EDGE_TRIPLE; ++index) {
        int32_t hold = a[index];
        a[index] = b[index];
        b[index] = hold;
    }
}

/* Sifts the element at `root` down a heap of `count` triples based at `base`. */
static void gffx_edge_sift_down(int32_t *base, int64_t root, int64_t count) {
    for (;;) {
        int64_t child = root * 2 + 1;
        int64_t largest;
        if (child >= count) return;
        largest = child;
        if (child + 1 < count &&
            gffx_edge_triple_less(base + child * GFFX_EDGE_TRIPLE,
                                  base + (child + 1) * GFFX_EDGE_TRIPLE)) {
            largest = child + 1;
        }
        if (!gffx_edge_triple_less(base + root * GFFX_EDGE_TRIPLE,
                                   base + largest * GFFX_EDGE_TRIPLE)) {
            return;
        }
        gffx_edge_triple_swap(base + root * GFFX_EDGE_TRIPLE,
                              base + largest * GFFX_EDGE_TRIPLE);
        root = largest;
    }
}

/* In-place heapsort. Not stable, which is inconsequential: the ordering key is the whole triple,
 * so entries comparing equal are bit-identical and therefore interchangeable. */
static void gffx_edge_sort(int32_t *base, int64_t count) {
    int64_t index;
    if (count < 2) return;
    for (index = count / 2 - 1; index >= 0; --index) {
        gffx_edge_sift_down(base, index, count);
        if (index == 0) break;
    }
    for (index = count - 1; index > 0; --index) {
        gffx_edge_triple_swap(base, base + index * GFFX_EDGE_TRIPLE);
        gffx_edge_sift_down(base, 0, index);
    }
}

static gffx_status gffx_edge_check_int_output(
    const gffx_tensor_view *view,
    const char *role_message,
    uint32_t expected_rank,
    int64_t expected_rows,
    int64_t expected_cols,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_mesh_check_view(view, role_message, expected_rank, expected_rows,
                                              expected_cols, 1, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (view->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "topology outputs must use the int32 dtype"
        );
    }
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_build_edge_topology_workspace(
    int64_t face_count,
    int64_t batch_count,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (required_bytes == NULL || required_alignment == NULL) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "workspace query result pointers must not be null"
        );
    }
    if (face_count < INT64_C(0) || batch_count < INT64_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face and batch counts must be nonnegative"
        );
    }
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.build_edge_topology implements only the CPU backend in this phase"
        );
    }
    if ((uint64_t)face_count > UINT64_MAX / (UINT64_C(3) * GFFX_EDGE_TRIPLE * sizeof(int32_t))) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_OVERFLOW,
            "workspace byte requirement overflows 64-bit capacity"
        );
    }
    *required_bytes =
        (uint64_t)face_count * UINT64_C(3) * GFFX_EDGE_TRIPLE * (uint64_t)sizeof(int32_t);
    *required_alignment = (uint64_t)sizeof(int32_t);
    return GFFX_STATUS_OK;
}

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
) {
    gffx_status status = gffx_internal_prepare_diagnostic(diagnostic);
    int64_t face_count;
    int64_t batch_count;
    int64_t half_edge_count;
    uint64_t required_bytes;
    int64_t index;
    int64_t batch;
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_validate_execution_context(context, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (context->device_type != GFFX_DEVICE_CPU) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "mesh.build_edge_topology implements only the CPU backend in this phase"
        );
    }
    status = gffx_mesh_check_view(faces, "faces must be a [F,3] tensor view",
                                  2u, INT64_C(-1), INT64_C(3), 0, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (faces->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "faces must use the int32 dtype"
        );
    }
    face_count = faces->shape[0];
    half_edge_count = face_count * INT64_C(3);

    /* mesh_edge_offsets fixes B, and face_offsets must agree with it. */
    if (mesh_edge_offsets == NULL || mesh_edge_offsets->rank != 1u ||
        mesh_edge_offsets->shape == NULL || mesh_edge_offsets->shape[0] < INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "mesh edge offsets must be a [B+1] output view"
        );
    }
    batch_count = mesh_edge_offsets->shape[0] - INT64_C(1);
    status = gffx_edge_check_int_output(mesh_edge_offsets,
                                        "mesh edge offsets must be a [B+1] output view",
                                        1u, batch_count + INT64_C(1), INT64_C(0), diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    if (face_offsets == NULL || face_offsets->rank != 1u || face_offsets->shape == NULL ||
        face_offsets->shape[0] != batch_count + INT64_C(1)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "face offsets must have extent B+1"
        );
    }
    status = gffx_validate_tensor_view(face_offsets, diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    if (face_offsets->dtype != GFFX_DTYPE_INT32) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_UNSUPPORTED,
            "face offsets must use the int32 dtype"
        );
    }
    if ((face_offsets->flags & GFFX_TENSOR_OUTPUT) != UINT32_C(0)) {
        return gffx_internal_fail(
            diagnostic,
            GFFX_STATUS_INVALID_ARGUMENT,
            "operation inputs may not carry the output flag"
        );
    }

    status = gffx_edge_check_int_output(edges, "edges must be a [3F,2] output view",
                                        2u, half_edge_count, INT64_C(2), diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_edge_check_int_output(edge_face_offsets,
                                        "edge face offsets must be a [3F+1] output view",
                                        1u, half_edge_count + INT64_C(1), INT64_C(0),
                                        diagnostic);
    if (status != GFFX_STATUS_OK) return status;
    status = gffx_edge_check_int_output(edge_faces, "edge faces must be a [3F] output view",
                                        1u, half_edge_count, INT64_C(0), diagnostic);
    if (status != GFFX_STATUS_OK) return status;

    /* No output may alias an input or another output. */
    {
        const gffx_tensor_view *outputs[4];
        const gffx_tensor_view *inputs[2];
        int first;
        int second;
        outputs[0] = edges; outputs[1] = edge_face_offsets;
        outputs[2] = edge_faces; outputs[3] = mesh_edge_offsets;
        inputs[0] = faces; inputs[1] = face_offsets;
        for (first = 0; first < 4; ++first) {
            for (second = 0; second < 2; ++second) {
                if (gffx_mesh_views_overlap(outputs[first], inputs[second])) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_INVALID_ARGUMENT,
                        "outputs may not alias an input or another output"
                    );
                }
            }
            for (second = first + 1; second < 4; ++second) {
                if (gffx_mesh_views_overlap(outputs[first], outputs[second])) {
                    return gffx_internal_fail(
                        diagnostic,
                        GFFX_STATUS_INVALID_ARGUMENT,
                        "outputs may not alias an input or another output"
                    );
                }
            }
        }
    }

    /* Offsets and face indices are validated in full before any grouping work. */
    {
        const int32_t *offsets = (const int32_t *)gffx_mesh_elements_const(face_offsets);
        if (offsets[0] != INT32_C(0)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "the first face offset must be zero"
            );
        }
        for (index = 0; index < batch_count; ++index) {
            if (offsets[index + 1] < offsets[index]) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "face offsets must be nondecreasing"
                );
            }
        }
        if ((int64_t)offsets[batch_count] != face_count) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "the final face offset must equal the face count"
            );
        }
    }
    if (face_count > INT64_C(0)) {
        const int32_t *face_data = (const int32_t *)gffx_mesh_elements_const(faces);
        for (index = 0; index < half_edge_count; ++index) {
            if (face_data[index] < INT32_C(0)) {
                return gffx_internal_fail(
                    diagnostic,
                    GFFX_STATUS_INVALID_ARGUMENT,
                    "face indices must be nonnegative"
                );
            }
        }
    }

    required_bytes =
        (uint64_t)face_count * UINT64_C(3) * GFFX_EDGE_TRIPLE * (uint64_t)sizeof(int32_t);
    if (face_count > INT64_C(0)) {
        if (workspace == NULL || workspace->capacity_bytes < required_bytes) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INSUFFICIENT_WORKSPACE,
                "the forward pass requires the workspace capacity reported by the query"
            );
        }
        status = gffx_validate_buffer(workspace, diagnostic);
        if (status != GFFX_STATUS_OK) return status;
        if (workspace->device_type != GFFX_DEVICE_CPU) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_UNSUPPORTED,
                "mesh.build_edge_topology accepts only CPU workspace storage"
            );
        }
        if (((uintptr_t)workspace->data % (uintptr_t)sizeof(int32_t)) != 0u) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "the workspace data pointer must be aligned to int32"
            );
        }
        if (gffx_mesh_range_overlaps_view(workspace->data, required_bytes, faces) ||
            gffx_mesh_range_overlaps_view(workspace->data, required_bytes, face_offsets) ||
            gffx_mesh_range_overlaps_view(workspace->data, required_bytes, edges) ||
            gffx_mesh_range_overlaps_view(workspace->data, required_bytes, edge_face_offsets) ||
            gffx_mesh_range_overlaps_view(workspace->data, required_bytes, edge_faces) ||
            gffx_mesh_range_overlaps_view(workspace->data, required_bytes, mesh_edge_offsets)) {
            return gffx_internal_fail(
                diagnostic,
                GFFX_STATUS_INVALID_ARGUMENT,
                "the workspace may not alias an operand"
            );
        }
    }

    {
        const int32_t *face_data =
            face_count > INT64_C(0) ? (const int32_t *)gffx_mesh_elements_const(faces) : NULL;
        const int32_t *offsets = (const int32_t *)gffx_mesh_elements_const(face_offsets);
        int32_t *edge_data = (int32_t *)gffx_mesh_elements(edges);
        int32_t *edge_offset_data = (int32_t *)gffx_mesh_elements(edge_face_offsets);
        int32_t *edge_face_data = (int32_t *)gffx_mesh_elements(edge_faces);
        int32_t *mesh_offset_data = (int32_t *)gffx_mesh_elements(mesh_edge_offsets);
        int32_t *triples = face_count > INT64_C(0) ? (int32_t *)workspace->data : NULL;
        int64_t unique_total = 0;
        int64_t incidence_total = 0;

        mesh_offset_data[0] = 0;
        edge_offset_data[0] = 0;
        for (batch = 0; batch < batch_count; ++batch) {
            int64_t first_face = (int64_t)offsets[batch];
            int64_t last_face = (int64_t)offsets[batch + 1];
            int64_t local_count = (last_face - first_face) * INT64_C(3);
            int32_t *base = triples + first_face * INT64_C(3) * GFFX_EDGE_TRIPLE;
            int64_t cursor;

            for (index = first_face; index < last_face; ++index) {
                int corner;
                for (corner = 0; corner < 3; ++corner) {
                    int32_t a = face_data[index * 3 + corner];
                    int32_t b = face_data[index * 3 + ((corner + 1) % 3)];
                    int64_t slot = (index - first_face) * INT64_C(3) + corner;
                    int32_t *entry = base + slot * GFFX_EDGE_TRIPLE;
                    entry[0] = a < b ? a : b;
                    entry[1] = a < b ? b : a;
                    entry[2] = (int32_t)index;
                }
            }
            gffx_edge_sort(base, local_count);

            cursor = 0;
            while (cursor < local_count) {
                const int32_t *entry = base + cursor * GFFX_EDGE_TRIPLE;
                int64_t run = cursor;
                edge_data[unique_total * 2 + 0] = entry[0];
                edge_data[unique_total * 2 + 1] = entry[1];
                while (run < local_count) {
                    const int32_t *member = base + run * GFFX_EDGE_TRIPLE;
                    if (member[0] != entry[0] || member[1] != entry[1]) break;
                    edge_face_data[incidence_total] = member[2];
                    ++incidence_total;
                    ++run;
                }
                ++unique_total;
                edge_offset_data[unique_total] = (int32_t)incidence_total;
                cursor = run;
            }
            mesh_offset_data[batch + 1] = (int32_t)unique_total;
        }

        /* Trailing edge rows carry the unmistakable (-1, -1) sentinel. */
        for (index = unique_total; index < half_edge_count; ++index) {
            edge_data[index * 2 + 0] = -1;
            edge_data[index * 2 + 1] = -1;
        }
        /* Trailing offsets repeat the final incidence total so the array stays nondecreasing. */
        for (index = unique_total + INT64_C(1); index <= half_edge_count; ++index) {
            edge_offset_data[index] = (int32_t)incidence_total;
        }
    }
    return GFFX_STATUS_OK;
}
