/*
 * GFFX CUDA device kernels.
 *
 * Compiled offline to PTX at the architecture floor and embedded into the plugin; the driver
 * JIT-compiles the module for whatever GPU is present. Nothing here uses the CUDA Runtime, and the
 * plugin that carries it links only the driver library, so a user needs a driver and never a
 * toolkit.
 *
 * Each kernel mirrors its CPU reference operation for operation, in the same order, because
 * `FRAMEWORK_CONFORMANCE_V0_1.md` requires backends to agree and floating-point addition is not
 * associative. Two consequences are deliberate and easy to lose:
 *
 *   - The build passes `-fmad=false`. Left on, the compiler contracts `a*b + c` into a fused
 *     multiply-add, which is *more* accurate in isolation and therefore disagrees with the CPU,
 *     whose result the acceptance fixtures pin. Conformance beats local accuracy here: a backend
 *     that is closer to the true value but different from the reference still fails the contract.
 *   - The validity comparison is evaluated in double regardless of the operand dtype, matching
 *     `FACE_GEOMETRY_ACCEPTANCE_V0_1.md`, so a float32 mesh classifies faces identically on both
 *     backends rather than near the threshold only.
 *
 * Index range is checked by gffx_cuda_validate_faces below, launched before the operation on the
 * same stream, because host code cannot read device-resident indices and the contract's per-call
 * check is not skippable. The operation kernels below therefore assume validated indices, which is
 * the same assumption the CPU kernels make after the host has checked them.
 */

extern "C" __global__ void gffx_cuda_face_geometry_f32(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    double eps,
    long long face_count,
    float *__restrict__ unit_normals,
    float *__restrict__ areas,
    unsigned char *__restrict__ valid
) {
    long long face = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= face_count) return;

    const float *a = vertices + (long long)faces[face * 3 + 0] * 3;
    const float *b = vertices + (long long)faces[face * 3 + 1] * 3;
    const float *c = vertices + (long long)faces[face * 3 + 2] * 3;
    float e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
    float e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
    float cx = e1y * e2z - e1z * e2y;
    float cy = e1z * e2x - e1x * e2z;
    float cz = e1x * e2y - e1y * e2x;
    float doubled = sqrtf(cx * cx + cy * cy + cz * cz);
    if ((double)doubled > eps) {
        unit_normals[face * 3 + 0] = cx / doubled;
        unit_normals[face * 3 + 1] = cy / doubled;
        unit_normals[face * 3 + 2] = cz / doubled;
        areas[face] = doubled * 0.5f;
        valid[face] = 1u;
    } else {
        unit_normals[face * 3 + 0] = 0.0f;
        unit_normals[face * 3 + 1] = 0.0f;
        unit_normals[face * 3 + 2] = 0.0f;
        areas[face] = 0.0f;
        valid[face] = 0u;
    }
}

extern "C" __global__ void gffx_cuda_face_geometry_f64(
    const double *__restrict__ vertices,
    const int *__restrict__ faces,
    double eps,
    long long face_count,
    double *__restrict__ unit_normals,
    double *__restrict__ areas,
    unsigned char *__restrict__ valid
) {
    long long face = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= face_count) return;

    const double *a = vertices + (long long)faces[face * 3 + 0] * 3;
    const double *b = vertices + (long long)faces[face * 3 + 1] * 3;
    const double *c = vertices + (long long)faces[face * 3 + 2] * 3;
    double e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
    double e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
    double cx = e1y * e2z - e1z * e2y;
    double cy = e1z * e2x - e1x * e2z;
    double cz = e1x * e2y - e1y * e2x;
    double doubled = sqrt(cx * cx + cy * cy + cz * cz);
    if (doubled > eps) {
        unit_normals[face * 3 + 0] = cx / doubled;
        unit_normals[face * 3 + 1] = cy / doubled;
        unit_normals[face * 3 + 2] = cz / doubled;
        areas[face] = doubled * 0.5;
        valid[face] = 1u;
    } else {
        unit_normals[face * 3 + 0] = 0.0;
        unit_normals[face * 3 + 1] = 0.0;
        unit_normals[face * 3 + 2] = 0.0;
        areas[face] = 0.0;
        valid[face] = 0u;
    }
}

/*
 * Index validation for device-resident topology.
 *
 * EXECUTION_STATE_CONTRACT_V0_1.md makes the O(F) index check mandatory on every call and offers
 * no flag to skip it. The CPU path satisfies that by reading the face tensor directly; host code
 * cannot read device memory, so the check moves to the device rather than being dropped.
 *
 * The status word is a single int in the caller's workspace, which is why the CUDA workspace
 * requirement is nonzero where the scalar CPU reference needs none. Threads race to set it, and
 * the race is harmless: every writer stores the same value, so no atomic is needed for
 * correctness, only visibility, which the subsequent kernel launch on the same stream provides.
 */
extern "C" __global__ void gffx_cuda_validate_faces(
    const int *__restrict__ faces,
    long long index_count,
    int vertex_count,
    int *__restrict__ status
) {
    /* The bound is the number of indices, not the number of faces. The caller has already
     * multiplied by three, and multiplying again here made the trailing threads of the last block
     * read past the end of the face buffer. That read is out of bounds whatever it returns, and it
     * only failed visibly when the memory beyond happened to hold a value outside [0, V). */
    long long entry = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= index_count) return;
    int index = faces[entry];
    if (index < 0 || index >= vertex_count) {
        *status = 1;
    }
}

/* =============================================================================================
 * Order-independent operations.
 *
 * Each of these writes one output element per thread and accumulates nothing, so the result does
 * not depend on the order threads happen to run in. That is what makes them bit-identical to the
 * CPU reference rather than merely close, and it is also why they were implemented first: the
 * operations that accumulate into shared vertices cannot be done this way without breaking the
 * ordered-accumulation guarantee their contracts state, which is recorded as an open item.
 * ============================================================================================= */

/* Locates a point's batch element. Offsets are B+1 packed boundaries and B is small in practice,
 * so a linear scan costs less than the divergence a binary search would introduce. */
__device__ __forceinline__ int gffx_cuda_batch_of(
    const int *__restrict__ offsets, int batch_count, long long index
) {
    for (int batch = 0; batch < batch_count; ++batch) {
        if (index >= (long long)offsets[batch] && index < (long long)offsets[batch + 1]) {
            return batch;
        }
    }
    return -1;
}

#define GFFX_CUDA_TRANSFORM_POINTS(SUFFIX, SCALAR)                                             \
extern "C" __global__ void gffx_cuda_transform_points_##SUFFIX(                                \
    const SCALAR *__restrict__ points, const SCALAR *__restrict__ matrices,                    \
    const int *__restrict__ offsets, int batch_count, long long point_count,                   \
    SCALAR *__restrict__ homogeneous                                                           \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= point_count) return;                                                          \
    int batch = gffx_cuda_batch_of(offsets, batch_count, point);                               \
    if (batch < 0) return;                                                                     \
    const SCALAR *m = matrices + (long long)batch * 16;                                        \
    SCALAR x = points[point * 3 + 0];                                                          \
    SCALAR y = points[point * 3 + 1];                                                          \
    SCALAR z = points[point * 3 + 2];                                                          \
    for (int row = 0; row < 4; ++row) {                                                        \
        homogeneous[point * 4 + row] =                                                         \
            m[row * 4 + 0] * x + m[row * 4 + 1] * y + m[row * 4 + 2] * z + m[row * 4 + 3];     \
    }                                                                                          \
}

GFFX_CUDA_TRANSFORM_POINTS(f32, float)
GFFX_CUDA_TRANSFORM_POINTS(f64, double)

/* A point whose |w| does not exceed eps is invalid, and its ndc is exactly zero rather than an
 * infinity, matching TRANSFORMS_ACCEPTANCE_V0_1.md. The comparison is in double for both dtypes,
 * as on the CPU, so a float32 batch classifies identically. */
#define GFFX_CUDA_PERSPECTIVE_DIVIDE(SUFFIX, SCALAR)                                           \
extern "C" __global__ void gffx_cuda_perspective_divide_##SUFFIX(                              \
    const SCALAR *__restrict__ homogeneous, double eps, long long point_count,                 \
    SCALAR *__restrict__ ndc, unsigned char *__restrict__ valid                                \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= point_count) return;                                                          \
    SCALAR w = homogeneous[point * 4 + 3];                                                      \
    double magnitude = (double)w < 0.0 ? -(double)w : (double)w;                                \
    if (magnitude > eps) {                                                                      \
        ndc[point * 3 + 0] = homogeneous[point * 4 + 0] / w;                                    \
        ndc[point * 3 + 1] = homogeneous[point * 4 + 1] / w;                                    \
        ndc[point * 3 + 2] = homogeneous[point * 4 + 2] / w;                                    \
        valid[point] = 1u;                                                                      \
    } else {                                                                                    \
        ndc[point * 3 + 0] = (SCALAR)0;                                                          \
        ndc[point * 3 + 1] = (SCALAR)0;                                                          \
        ndc[point * 3 + 2] = (SCALAR)0;                                                          \
        valid[point] = 0u;                                                                      \
    }                                                                                            \
}

GFFX_CUDA_PERSPECTIVE_DIVIDE(f32, float)
GFFX_CUDA_PERSPECTIVE_DIVIDE(f64, double)

/* A pure gather: no arithmetic, so values including NaN and infinity are copied bit for bit. */
#define GFFX_CUDA_GATHER_FACES(SUFFIX, SCALAR)                                                 \
extern "C" __global__ void gffx_cuda_gather_faces_##SUFFIX(                                    \
    const SCALAR *__restrict__ vertices, const int *__restrict__ faces,                        \
    long long face_count, SCALAR *__restrict__ face_vertices                                   \
) {                                                                                            \
    long long entry = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (entry >= face_count * 3) return;                                                        \
    long long face = entry / 3;                                                                 \
    long long corner = entry % 3;                                                               \
    long long source = (long long)faces[face * 3 + corner] * 3;                                 \
    long long target = (face * 3 + corner) * 3;                                                 \
    face_vertices[target + 0] = vertices[source + 0];                                           \
    face_vertices[target + 1] = vertices[source + 1];                                           \
    face_vertices[target + 2] = vertices[source + 2];                                           \
}

GFFX_CUDA_GATHER_FACES(f32, float)
GFFX_CUDA_GATHER_FACES(f64, double)


/* -------------------------------------------------------------------------------- points.knn
 *
 * One thread per query, which is what keeps this deterministic: a query's K-list is private to its
 * thread, so no two threads touch the same output and the insertion order is the ascending
 * candidate scan the CPU performs. That ordering is not incidental. Scanning references upward and
 * inserting only on a strict improvement is what makes an exact distance tie keep the earlier,
 * lower-index entry, which PROXIMITY_ACCEPTANCE_V0_1.md requires.
 *
 * Selection uses squared distance with no square root, so the result carries no rounding from a
 * root, and a short batch element pads with +inf, -1 and false rather than with stale values.
 */
#define GFFX_CUDA_KNN(SUFFIX, SCALAR, INFINITY_VALUE)                                          \
extern "C" __global__ void gffx_cuda_knn_##SUFFIX(                                             \
    const SCALAR *__restrict__ query, const SCALAR *__restrict__ reference,                    \
    const int *__restrict__ query_offsets, const int *__restrict__ reference_offsets,          \
    int batch_count, long long neighbor_count, long long query_count,                          \
    SCALAR *__restrict__ distance_squared, int *__restrict__ reference_index,                  \
    unsigned char *__restrict__ valid                                                          \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= query_count) return;                                                          \
    int batch = gffx_cuda_batch_of(query_offsets, batch_count, point);                         \
    if (batch < 0) return;                                                                     \
    long long first = (long long)reference_offsets[batch];                                     \
    long long last = (long long)reference_offsets[batch + 1];                                  \
    SCALAR qx = query[point * 3 + 0];                                                          \
    SCALAR qy = query[point * 3 + 1];                                                          \
    SCALAR qz = query[point * 3 + 2];                                                          \
    for (long long slot = 0; slot < neighbor_count; ++slot) {                                  \
        distance_squared[point * neighbor_count + slot] = (INFINITY_VALUE);                    \
        reference_index[point * neighbor_count + slot] = -1;                                   \
        valid[point * neighbor_count + slot] = 0u;                                             \
    }                                                                                          \
    for (long long candidate = first; candidate < last; ++candidate) {                         \
        SCALAR dx = qx - reference[candidate * 3 + 0];                                         \
        SCALAR dy = qy - reference[candidate * 3 + 1];                                         \
        SCALAR dz = qz - reference[candidate * 3 + 2];                                         \
        SCALAR value = dx * dx + dy * dy + dz * dz;                                            \
        long long position = neighbor_count;                                                   \
        while (position > 0) {                                                                 \
            long long above = position - 1;                                                    \
            int is_padding = valid[point * neighbor_count + above] == 0u;                      \
            if (is_padding || value < distance_squared[point * neighbor_count + above]) {      \
                --position;                                                                    \
            } else {                                                                           \
                break;                                                                         \
            }                                                                                  \
        }                                                                                      \
        if (position >= neighbor_count) continue;                                              \
        for (long long slot = neighbor_count - 1; slot > position; --slot) {                   \
            distance_squared[point * neighbor_count + slot] =                                  \
                distance_squared[point * neighbor_count + slot - 1];                           \
            reference_index[point * neighbor_count + slot] =                                   \
                reference_index[point * neighbor_count + slot - 1];                            \
            valid[point * neighbor_count + slot] =                                             \
                valid[point * neighbor_count + slot - 1];                                      \
        }                                                                                      \
        distance_squared[point * neighbor_count + position] = value;                           \
        reference_index[point * neighbor_count + position] = (int)candidate;                   \
        valid[point * neighbor_count + position] = 1u;                                         \
    }                                                                                          \
}

GFFX_CUDA_KNN(f32, float, __int_as_float(0x7f800000))
GFFX_CUDA_KNN(f64, double, __longlong_as_double(0x7ff0000000000000LL))

/* ------------------------------------------------------------------------- render.interpolate
 *
 * One thread per fragment. The accumulation over three corners happens inside a single thread and
 * in the order the CPU uses, so it is deterministic; the sum is carried in double for both dtypes
 * exactly as on the host, with the narrowing to float happening once at the store.
 *
 * A background fragment, where face_index is negative, is exactly zero and never reads the
 * attribute array, which makes an out-of-range read impossible rather than merely unlikely.
 */
#define GFFX_CUDA_INTERPOLATE(SUFFIX, SCALAR)                                                  \
extern "C" __global__ void gffx_cuda_interpolate_##SUFFIX(                                     \
    const int *__restrict__ face_index, const SCALAR *__restrict__ barycentric,                \
    const SCALAR *__restrict__ face_attributes, long long fragment_count,                      \
    long long channel_count, long long face_count, SCALAR *__restrict__ attributes             \
) {                                                                                            \
    long long fragment = (long long)blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (fragment >= fragment_count) return;                                                    \
    long long face = (long long)face_index[fragment];                                          \
    if (face < 0 || face >= face_count) {                                                      \
        for (long long channel = 0; channel < channel_count; ++channel) {                      \
            attributes[fragment * channel_count + channel] = (SCALAR)0;                        \
        }                                                                                      \
        return;                                                                                \
    }                                                                                          \
    for (long long channel = 0; channel < channel_count; ++channel) {                          \
        double total = 0.0;                                                                    \
        for (int corner = 0; corner < 3; ++corner) {                                           \
            double weight = (double)barycentric[fragment * 3 + corner];                        \
            double value =                                                                     \
                (double)face_attributes[(face * 3 + corner) * channel_count + channel];        \
            total += weight * value;                                                           \
        }                                                                                      \
        attributes[fragment * channel_count + channel] = (SCALAR)total;                        \
    }                                                                                          \
}

GFFX_CUDA_INTERPOLATE(f32, float)
GFFX_CUDA_INTERPOLATE(f64, double)


/* --------------------------------------------------------- points.closest_point_on_mesh
 *
 * One thread per query point, scanning that element's faces in ascending order. The scan is what
 * makes this deterministic without any coordination between threads: each query owns its own
 * running best, and improving only on a strict decrease keeps the lower face index when two faces
 * are exactly equidistant, which PROXIMITY_ACCEPTANCE_V0_1.md requires.
 *
 * All arithmetic is in double regardless of the operand dtype, exactly as on the host, with the
 * narrowing to float happening once when the result is stored. That is why a float32 mesh selects
 * the same face on both backends rather than only away from ties.
 *
 * Complexity is O(P*F) per element. v0.1 ships no spatial acceleration structure, and the CUDA
 * backend does not quietly add one: a GPU that was fast for a different reason than the CPU would
 * make the two backends incomparable.
 */
__device__ __forceinline__ void gffx_cuda_closest_barycentric(
    double px, double py, double pz,
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double *b0, double *b1, double *b2
) {
    double abx = bx - ax, aby = by - ay, abz = bz - az;
    double acx = cx - ax, acy = cy - ay, acz = cz - az;
    double apx = px - ax, apy = py - ay, apz = pz - az;
    double d1 = abx * apx + aby * apy + abz * apz;
    double d2 = acx * apx + acy * apy + acz * apz;
    double bpx, bpy, bpz, d3, d4, vc;
    double cpx, cpy, cpz, d5, d6, vb, va;
    double denom, v, w;

    if (d1 <= 0.0 && d2 <= 0.0) { *b0 = 1.0; *b1 = 0.0; *b2 = 0.0; return; }

    bpx = px - bx; bpy = py - by; bpz = pz - bz;
    d3 = abx * bpx + aby * bpy + abz * bpz;
    d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0.0 && d4 <= d3) { *b0 = 0.0; *b1 = 1.0; *b2 = 0.0; return; }

    vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        v = d1 / (d1 - d3);
        *b0 = 1.0 - v; *b1 = v; *b2 = 0.0;
        return;
    }

    cpx = px - cx; cpy = py - cy; cpz = pz - cz;
    d5 = abx * cpx + aby * cpy + abz * cpz;
    d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0.0 && d5 <= d6) { *b0 = 0.0; *b1 = 0.0; *b2 = 1.0; return; }

    vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        w = d2 / (d2 - d6);
        *b0 = 1.0 - w; *b1 = 0.0; *b2 = w;
        return;
    }

    va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *b0 = 0.0; *b1 = 1.0 - w; *b2 = w;
        return;
    }

    denom = 1.0 / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    *b0 = 1.0 - v - w; *b1 = v; *b2 = w;
}

#define GFFX_CUDA_CLOSEST_POINT(SUFFIX, SCALAR, INFINITY_VALUE)                                \
extern "C" __global__ void gffx_cuda_closest_point_##SUFFIX(                                   \
    const SCALAR *__restrict__ points, const SCALAR *__restrict__ vertices,                    \
    const int *__restrict__ faces, const int *__restrict__ point_offsets,                      \
    const int *__restrict__ face_offsets, int batch_count, double eps,                         \
    long long point_count, SCALAR *__restrict__ distance_squared,                              \
    int *__restrict__ face_index, SCALAR *__restrict__ barycentric,                            \
    SCALAR *__restrict__ closest, unsigned char *__restrict__ valid                            \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= point_count) return;                                                          \
    int batch = gffx_cuda_batch_of(point_offsets, batch_count, point);                         \
    if (batch < 0) return;                                                                     \
    long long first_face = (long long)face_offsets[batch];                                     \
    long long last_face = (long long)face_offsets[batch + 1];                                  \
    double px = (double)points[point * 3 + 0];                                                 \
    double py = (double)points[point * 3 + 1];                                                 \
    double pz = (double)points[point * 3 + 2];                                                 \
    double best_distance = 1.0e308 * 10.0;                                                     \
    long long best_face = -1;                                                                  \
    double best_b0 = 0.0, best_b1 = 0.0, best_b2 = 0.0;                                        \
    double best_cx = 0.0, best_cy = 0.0, best_cz = 0.0;                                        \
    for (long long face = first_face; face < last_face; ++face) {                              \
        long long i0 = (long long)faces[face * 3 + 0];                                         \
        long long i1 = (long long)faces[face * 3 + 1];                                         \
        long long i2 = (long long)faces[face * 3 + 2];                                         \
        double ax = (double)vertices[i0 * 3 + 0];                                              \
        double ay = (double)vertices[i0 * 3 + 1];                                              \
        double az = (double)vertices[i0 * 3 + 2];                                              \
        double bx = (double)vertices[i1 * 3 + 0];                                              \
        double by = (double)vertices[i1 * 3 + 1];                                              \
        double bz = (double)vertices[i1 * 3 + 2];                                              \
        double cx = (double)vertices[i2 * 3 + 0];                                              \
        double cy = (double)vertices[i2 * 3 + 1];                                              \
        double cz = (double)vertices[i2 * 3 + 2];                                              \
        double e1x = bx - ax, e1y = by - ay, e1z = bz - az;                                    \
        double e2x = cx - ax, e2y = cy - ay, e2z = cz - az;                                    \
        double nx = e1y * e2z - e1z * e2y;                                                     \
        double ny = e1z * e2x - e1x * e2z;                                                     \
        double nz = e1x * e2y - e1y * e2x;                                                     \
        double doubled = sqrt(nx * nx + ny * ny + nz * nz);                                    \
        double b0, b1, b2, qx, qy, qz, dx, dy, dz, candidate;                                  \
        if (!(doubled > eps)) continue;                                                        \
        gffx_cuda_closest_barycentric(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz,          \
                                      &b0, &b1, &b2);                                          \
        qx = b0 * ax + b1 * bx + b2 * cx;                                                      \
        qy = b0 * ay + b1 * by + b2 * cy;                                                      \
        qz = b0 * az + b1 * bz + b2 * cz;                                                      \
        dx = px - qx; dy = py - qy; dz = pz - qz;                                              \
        candidate = dx * dx + dy * dy + dz * dz;                                               \
        if (candidate < best_distance) {                                                       \
            best_distance = candidate;                                                         \
            best_face = face;                                                                  \
            best_b0 = b0; best_b1 = b1; best_b2 = b2;                                          \
            best_cx = qx; best_cy = qy; best_cz = qz;                                          \
        }                                                                                      \
    }                                                                                          \
    if (best_face < 0) {                                                                       \
        distance_squared[point] = (INFINITY_VALUE);                                            \
        face_index[point] = -1;                                                                \
        valid[point] = 0u;                                                                     \
        for (int axis = 0; axis < 3; ++axis) {                                                 \
            barycentric[point * 3 + axis] = (SCALAR)0;                                         \
            closest[point * 3 + axis] = (SCALAR)0;                                             \
        }                                                                                      \
        return;                                                                                \
    }                                                                                          \
    distance_squared[point] = (SCALAR)best_distance;                                           \
    face_index[point] = (int)best_face;                                                        \
    valid[point] = 1u;                                                                         \
    barycentric[point * 3 + 0] = (SCALAR)best_b0;                                              \
    barycentric[point * 3 + 1] = (SCALAR)best_b1;                                              \
    barycentric[point * 3 + 2] = (SCALAR)best_b2;                                              \
    closest[point * 3 + 0] = (SCALAR)best_cx;                                                  \
    closest[point * 3 + 1] = (SCALAR)best_cy;                                                  \
    closest[point * 3 + 2] = (SCALAR)best_cz;                                                  \
}

GFFX_CUDA_CLOSEST_POINT(f32, float, __int_as_float(0x7f800000))
GFFX_CUDA_CLOSEST_POINT(f64, double, __longlong_as_double(0x7ff0000000000000LL))


/* --------------------------------------------------------------------------- render.rasterize
 *
 * One thread per pixel. Each pixel owns its own K-fragment list, so no two threads ever write the
 * same output and the insertion order is the ascending face scan the CPU performs. That is what
 * keeps an exact depth tie resolving to the lower face index without any coordination.
 *
 * All coverage, distance and barycentric arithmetic runs in pixel space, because signed_distance
 * is contractually in squared pixel units and a non-square image would scale the axes differently
 * in NDC. Culling reads the NDC orientation, which is the negation of the pixel-space area,
 * because the y-flip in the NDC-to-pixel mapping reverses handedness.
 *
 * The signed distance is measured to the triangle boundary rather than to the filled region, so it
 * keeps varying inside a large triangle and can drive an alpha ramp; a positive blur radius admits
 * fragments outside the triangle, which is what gives a silhouette a nonzero gradient at all.
 */
__device__ __forceinline__ double gffx_cuda_segment_distance_squared(
    double px, double py, double ux, double uy, double vx, double vy
) {
    double ex = vx - ux;
    double ey = vy - uy;
    double length_squared = ex * ex + ey * ey;
    double t = 0.0;
    double cx, cy, dx, dy;
    if (length_squared > 0.0) {
        t = ((px - ux) * ex + (py - uy) * ey) / length_squared;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
    }
    cx = ux + t * ex;
    cy = uy + t * ey;
    dx = px - cx;
    dy = py - cy;
    return dx * dx + dy * dy;
}

__device__ __forceinline__ double gffx_cuda_boundary_distance_squared(
    double px, double py,
    double ax, double ay, double bx, double by, double cx, double cy
) {
    double best = gffx_cuda_segment_distance_squared(px, py, ax, ay, bx, by);
    double candidate = gffx_cuda_segment_distance_squared(px, py, bx, by, cx, cy);
    if (candidate < best) best = candidate;
    candidate = gffx_cuda_segment_distance_squared(px, py, cx, cy, ax, ay);
    if (candidate < best) best = candidate;
    return best;
}

#define GFFX_CUDA_RASTERIZE(SUFFIX, SCALAR, INFINITY_VALUE)                                    \
extern "C" __global__ void gffx_cuda_rasterize_##SUFFIX(                                       \
    const SCALAR *__restrict__ ndc_vertices, const int *__restrict__ faces,                    \
    const int *__restrict__ face_offsets, long long image_height, long long image_width,       \
    long long faces_per_pixel, double blur_squared, unsigned int cull_mode, double eps,        \
    long long pixel_count, int *__restrict__ face_index, SCALAR *__restrict__ barycentric,     \
    SCALAR *__restrict__ depth, SCALAR *__restrict__ signed_distance                           \
) {                                                                                            \
    long long pixel = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (pixel >= pixel_count) return;                                                          \
    long long column = pixel % image_width;                                                    \
    long long row = (pixel / image_width) % image_height;                                      \
    long long batch = pixel / (image_width * image_height);                                    \
    long long pixel_base = pixel * faces_per_pixel;                                            \
    long long first_face = (long long)face_offsets[batch];                                     \
    long long last_face = (long long)face_offsets[batch + 1];                                  \
    double half_width = (double)image_width * 0.5;                                             \
    double half_height = (double)image_height * 0.5;                                           \
    double px = (double)column + 0.5;                                                          \
    double py = (double)row + 0.5;                                                             \
    for (long long slot = 0; slot < faces_per_pixel; ++slot) {                                 \
        long long entry = pixel_base + slot;                                                   \
        face_index[entry] = -1;                                                                \
        depth[entry] = (INFINITY_VALUE);                                                       \
        signed_distance[entry] = (INFINITY_VALUE);                                             \
        barycentric[entry * 3 + 0] = (SCALAR)0;                                                \
        barycentric[entry * 3 + 1] = (SCALAR)0;                                                \
        barycentric[entry * 3 + 2] = (SCALAR)0;                                                \
    }                                                                                          \
    for (long long face = first_face; face < last_face; ++face) {                              \
        long long i0 = (long long)faces[face * 3 + 0];                                         \
        long long i1 = (long long)faces[face * 3 + 1];                                         \
        long long i2 = (long long)faces[face * 3 + 2];                                         \
        double nx0 = (double)ndc_vertices[i0 * 3 + 0];                                         \
        double ny0 = (double)ndc_vertices[i0 * 3 + 1];                                         \
        double nz0 = (double)ndc_vertices[i0 * 3 + 2];                                         \
        double nx1 = (double)ndc_vertices[i1 * 3 + 0];                                         \
        double ny1 = (double)ndc_vertices[i1 * 3 + 1];                                         \
        double nz1 = (double)ndc_vertices[i1 * 3 + 2];                                         \
        double nx2 = (double)ndc_vertices[i2 * 3 + 0];                                         \
        double ny2 = (double)ndc_vertices[i2 * 3 + 1];                                         \
        double nz2 = (double)ndc_vertices[i2 * 3 + 2];                                         \
        double ax = (nx0 + 1.0) * half_width;                                                  \
        double ay = (1.0 - ny0) * half_height;                                                 \
        double bx = (nx1 + 1.0) * half_width;                                                  \
        double by = (1.0 - ny1) * half_height;                                                 \
        double cx = (nx2 + 1.0) * half_width;                                                  \
        double cy = (1.0 - ny2) * half_height;                                                 \
        double e0 = (bx - px) * (cy - py) - (by - py) * (cx - px);                             \
        double e1 = (cx - px) * (ay - py) - (cy - py) * (ax - px);                             \
        double e2 = (ax - px) * (by - py) - (ay - py) * (bx - px);                             \
        double area2 = e0 + e1 + e2;                                                           \
        double w0, w1, w2, fragment_depth, distance_squared, signed_value;                     \
        int inside;                                                                            \
        long long position;                                                                    \
        if (!(fabs(area2) > eps)) continue;                                                    \
        if (cull_mode == 2u && -area2 <= 0.0) continue;                                        \
        if (cull_mode == 3u && -area2 > 0.0) continue;                                         \
        w0 = e0 / area2;                                                                       \
        w1 = e1 / area2;                                                                       \
        w2 = e2 / area2;                                                                       \
        inside = (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0);                                        \
        distance_squared = gffx_cuda_boundary_distance_squared(px, py, ax, ay, bx, by, cx, cy);\
        if (!inside && distance_squared > blur_squared) continue;                              \
        fragment_depth = w0 * nz0 + w1 * nz1 + w2 * nz2;                                       \
        signed_value = inside ? -distance_squared : distance_squared;                          \
        position = faces_per_pixel;                                                            \
        while (position > 0) {                                                                 \
            long long above = pixel_base + position - 1;                                       \
            double previous = (double)depth[above];                                            \
            if (face_index[above] < 0 || fragment_depth < previous) {                          \
                --position;                                                                    \
            } else {                                                                           \
                break;                                                                         \
            }                                                                                  \
        }                                                                                      \
        if (position >= faces_per_pixel) continue;                                             \
        for (long long slot = faces_per_pixel - 1; slot > position; --slot) {                  \
            long long to = pixel_base + slot;                                                  \
            long long from = pixel_base + slot - 1;                                            \
            face_index[to] = face_index[from];                                                 \
            depth[to] = depth[from];                                                           \
            signed_distance[to] = signed_distance[from];                                       \
            barycentric[to * 3 + 0] = barycentric[from * 3 + 0];                               \
            barycentric[to * 3 + 1] = barycentric[from * 3 + 1];                               \
            barycentric[to * 3 + 2] = barycentric[from * 3 + 2];                               \
        }                                                                                      \
        {                                                                                      \
            long long entry = pixel_base + position;                                           \
            face_index[entry] = (int)face;                                                     \
            depth[entry] = (SCALAR)fragment_depth;                                             \
            signed_distance[entry] = (SCALAR)signed_value;                                     \
            barycentric[entry * 3 + 0] = (SCALAR)w0;                                           \
            barycentric[entry * 3 + 1] = (SCALAR)w1;                                           \
            barycentric[entry * 3 + 2] = (SCALAR)w2;                                           \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_RASTERIZE(f32, float, __int_as_float(0x7f800000))
GFFX_CUDA_RASTERIZE(f64, double, __longlong_as_double(0x7ff0000000000000LL))

/* ------------------------------------------------- render.texture_pyramid and render.texture
 *
 * The pyramid runs one kernel per level, one thread per output texel of that level. Levels are
 * inherently sequential because level l+1 reads level l, so the sequencing is in the launch order
 * rather than inside a kernel; within a level every texel is independent.
 *
 * Each thread sums its own two-by-two block in the fixed order the host uses, left to right then
 * top to bottom, and in the operand dtype rather than promoted to double. That differs from
 * render.interpolate, which accumulates in double on both backends. The difference is deliberate:
 * what bit-identity requires is that the two backends perform the same operations in the same
 * order, and the host reference for this operation was written to sum in the operand dtype. A
 * promotion here would make CUDA disagree with it, which is the failure the rule exists to prevent.
 *
 * The sampler is one thread per sample. Every output element depends only on its own coordinate
 * and the pyramid, so there is no cross-thread interaction to make deterministic in the first
 * place; that per-element independence is what the acceptance record's bit-identity claim rests on.
 */

__device__ __forceinline__ int gffx_cuda_texture_wrap(
    long long index, long long extent, unsigned int mode, long long *out
) {
    long long period;
    long long folded;
    if (mode == 1u) { /* repeat */
        *out = ((index % extent) + extent) % extent;
        return 1;
    }
    if (mode == 2u) { /* clamp */
        *out = index < 0 ? 0 : (index >= extent ? extent - 1 : index);
        return 1;
    }
    if (mode == 3u) { /* mirror */
        period = extent * 2;
        folded = ((index % period) + period) % period;
        *out = folded >= extent ? period - 1 - folded : folded;
        return 1;
    }
    if (index < 0 || index >= extent) return 0; /* border */
    *out = index;
    return 1;
}

__device__ __forceinline__ void gffx_cuda_texture_level_extent(
    long long height, long long width, long long level,
    long long *out_height, long long *out_width
) {
    long long h = height;
    long long w = width;
    for (long long i = 0; i < level; ++i) {
        h = h > 1 ? h / 2 : 1;
        w = w > 1 ? w / 2 : 1;
    }
    *out_height = h;
    *out_width = w;
}

#define GFFX_CUDA_TEXTURE_PYRAMID_LEVEL(SUFFIX, SCALAR)                                        \
extern "C" __global__ void gffx_cuda_texture_pyramid_##SUFFIX(                                 \
    const SCALAR *__restrict__ previous, SCALAR *__restrict__ current,                         \
    long long previous_height, long long previous_width,                                       \
    long long level_height, long long level_width, long long channels                          \
) {                                                                                            \
    long long texel = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (texel >= level_height * level_width) return;                                           \
    {                                                                                          \
        long long y = texel / level_width;                                                     \
        long long x = texel - y * level_width;                                                 \
        long long y0 = previous_height > 1 ? y * 2 : y;                                        \
        long long x0 = previous_width > 1 ? x * 2 : x;                                         \
        long long y1 = previous_height > 1 ? y0 + 1 : y0;                                      \
        long long x1 = previous_width > 1 ? x0 + 1 : x0;                                       \
        SCALAR divisor = (SCALAR)1;                                                            \
        if (previous_height > 1) divisor *= (SCALAR)2;                                         \
        if (previous_width > 1) divisor *= (SCALAR)2;                                          \
        for (long long c = 0; c < channels; ++c) {                                             \
            SCALAR sum = previous[(y0 * previous_width + x0) * channels + c];                  \
            if (x1 != x0) sum += previous[(y0 * previous_width + x1) * channels + c];          \
            if (y1 != y0) {                                                                    \
                sum += previous[(y1 * previous_width + x0) * channels + c];                    \
                if (x1 != x0) sum += previous[(y1 * previous_width + x1) * channels + c];      \
            }                                                                                  \
            current[(y * level_width + x) * channels + c] = sum / divisor;                     \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_TEXTURE_PYRAMID_LEVEL(f32, float)
GFFX_CUDA_TEXTURE_PYRAMID_LEVEL(f64, double)

#define GFFX_CUDA_TEXTURE_SAMPLE(SUFFIX, SCALAR, FLOOR_FN, SQRT_FN, LOG2_FN, NAN_EXPR)         \
__device__ __forceinline__ void gffx_cuda_texture_level_##SUFFIX(                              \
    const SCALAR *level, long long level_height, long long level_width, long long channels,    \
    SCALAR u, SCALAR v, unsigned int filter, unsigned int wrap_u, unsigned int wrap_v,         \
    const SCALAR *border, SCALAR *out                                                          \
) {                                                                                            \
    if (filter == 1u) { /* nearest */                                                          \
        long long x = (long long)FLOOR_FN(u * (SCALAR)level_width);                            \
        long long y = (long long)FLOOR_FN(v * (SCALAR)level_height);                           \
        long long xi, yi;                                                                      \
        int inside = gffx_cuda_texture_wrap(x, level_width, wrap_u, &xi) &                     \
                     gffx_cuda_texture_wrap(y, level_height, wrap_v, &yi);                     \
        for (long long c = 0; c < channels; ++c) {                                             \
            out[c] = inside ? level[(yi * level_width + xi) * channels + c] : border[c];        \
        }                                                                                      \
        return;                                                                                \
    }                                                                                          \
    {                                                                                          \
        SCALAR fx = u * (SCALAR)level_width - (SCALAR)0.5;                                     \
        SCALAR fy = v * (SCALAR)level_height - (SCALAR)0.5;                                    \
        long long x0 = (long long)FLOOR_FN(fx);                                                \
        long long y0 = (long long)FLOOR_FN(fy);                                                \
        SCALAR a = fx - (SCALAR)x0;                                                            \
        SCALAR b = fy - (SCALAR)y0;                                                            \
        long long xi0, xi1, yi0, yi1;                                                          \
        int ix0 = gffx_cuda_texture_wrap(x0, level_width, wrap_u, &xi0);                       \
        int ix1 = gffx_cuda_texture_wrap(x0 + 1, level_width, wrap_u, &xi1);                   \
        int iy0 = gffx_cuda_texture_wrap(y0, level_height, wrap_v, &yi0);                      \
        int iy1 = gffx_cuda_texture_wrap(y0 + 1, level_height, wrap_v, &yi1);                  \
        for (long long c = 0; c < channels; ++c) {                                             \
            SCALAR t00 = (ix0 && iy0) ? level[(yi0 * level_width + xi0) * channels + c]        \
                                      : border[c];                                             \
            SCALAR t10 = (ix1 && iy0) ? level[(yi0 * level_width + xi1) * channels + c]        \
                                      : border[c];                                             \
            SCALAR t01 = (ix0 && iy1) ? level[(yi1 * level_width + xi0) * channels + c]        \
                                      : border[c];                                             \
            SCALAR t11 = (ix1 && iy1) ? level[(yi1 * level_width + xi1) * channels + c]        \
                                      : border[c];                                             \
            SCALAR sum = ((SCALAR)1 - a) * ((SCALAR)1 - b) * t00;                              \
            sum += a * ((SCALAR)1 - b) * t10;                                                  \
            sum += ((SCALAR)1 - a) * b * t01;                                                  \
            sum += a * b * t11;                                                                \
            out[c] = sum;                                                                      \
        }                                                                                      \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_texture_##SUFFIX(                                         \
    const SCALAR *__restrict__ pyramid, const int *__restrict__ offsets,                       \
    long long level_count, long long height, long long width, long long channels,              \
    const SCALAR *__restrict__ coordinates, long long count,                                   \
    const SCALAR *__restrict__ derivatives, const SCALAR *__restrict__ lod_values,             \
    unsigned int filter, unsigned int mip_filter, unsigned int wrap_u, unsigned int wrap_v,    \
    const SCALAR *__restrict__ border, SCALAR *__restrict__ samples                            \
) {                                                                                            \
    long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (n >= count) return;                                                                    \
    {                                                                                          \
        SCALAR u = coordinates[n * 2];                                                         \
        SCALAR v = coordinates[n * 2 + 1];                                                     \
        SCALAR *out = samples + n * channels;                                                  \
        double lod = 0.0;                                                                      \
        long long first, second, h0, w0, h1, w1;                                               \
        double blend = 0.0;                                                                    \
        if (!(u == u) || !(v == v) || u * (SCALAR)0 != (SCALAR)0 ||                            \
            v * (SCALAR)0 != (SCALAR)0) {                                                      \
            for (long long c = 0; c < channels; ++c) out[c] = NAN_EXPR;                        \
            return;                                                                            \
        }                                                                                      \
        if (derivatives != 0) {                                                                \
            SCALAR ax = derivatives[n * 4] * (SCALAR)width;                                    \
            SCALAR bx = derivatives[n * 4 + 1] * (SCALAR)height;                               \
            SCALAR ay = derivatives[n * 4 + 2] * (SCALAR)width;                                \
            SCALAR by = derivatives[n * 4 + 3] * (SCALAR)height;                               \
            SCALAR rx = SQRT_FN(ax * ax + bx * bx);                                            \
            SCALAR ry = SQRT_FN(ay * ay + by * by);                                            \
            SCALAR rho = rx > ry ? rx : ry;                                                    \
            if (!(rho > (SCALAR)1.1754943508222875e-38))                                       \
                rho = (SCALAR)1.1754943508222875e-38;                                          \
            lod = (double)LOG2_FN(rho);                                                        \
        } else if (lod_values != 0) {                                                          \
            lod = (double)lod_values[n];                                                       \
        }                                                                                      \
        if (!(lod > 0.0)) lod = 0.0;                                                           \
        if (lod > (double)(level_count - 1)) lod = (double)(level_count - 1);                  \
        if (mip_filter == 1u) {                                                                \
            first = (long long)(lod + 0.5);                                                    \
            if (first > level_count - 1) first = level_count - 1;                              \
            second = first;                                                                    \
        } else {                                                                               \
            double base = floor(lod);                                                          \
            first = (long long)base;                                                           \
            second = first + 1;                                                                \
            if (second > level_count - 1) second = level_count - 1;                            \
            blend = lod - base;                                                                \
        }                                                                                      \
        gffx_cuda_texture_level_extent(height, width, first, &h0, &w0);                        \
        gffx_cuda_texture_level_##SUFFIX(pyramid + offsets[first], h0, w0, channels, u, v,     \
                                         filter, wrap_u, wrap_v, border, out);                 \
        if (blend > 0.0 && second != first) {                                                  \
            SCALAR coarse[4];                                                                  \
            gffx_cuda_texture_level_extent(height, width, second, &h1, &w1);                   \
            gffx_cuda_texture_level_##SUFFIX(pyramid + offsets[second], h1, w1, channels, u,   \
                                             v, filter, wrap_u, wrap_v, border, coarse);       \
            for (long long c = 0; c < channels; ++c) {                                         \
                out[c] = out[c] * (SCALAR)(1.0 - blend) + coarse[c] * (SCALAR)blend;           \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_TEXTURE_SAMPLE(f32, float, floorf, sqrtf, log2f, __int_as_float(0x7fc00000))
GFFX_CUDA_TEXTURE_SAMPLE(f64, double, floor, sqrt, log2,
                         __longlong_as_double(0x7ff8000000000000LL))

/* ------------------------------------------------------------------- backward kernels
 *
 * A forward kernel here is scatter-free: one thread owns one output. A backward usually is not,
 * because many outputs feed back into one input, and the obvious device answer - an atomic add -
 * completes in whatever order the hardware schedules. Floating-point addition is not associative,
 * so an atomic scatter gives a different answer run to run and would forfeit the determinism the
 * conformance contract promises.
 *
 * Every kernel below therefore inverts the scatter into a gather: the thread that owns an input
 * element walks the outputs that touch it, in the same ascending order the host reference uses.
 * That costs more work than an atomic scatter and is the point - it is what lets these results be
 * compared to the CPU with memcmp rather than a tolerance.
 */

/* transforms.perspective_divide backward. Elementwise: one thread per point, no interaction at
 * all, so determinism needs no argument beyond the arithmetic matching the host. */
#define GFFX_CUDA_DIVIDE_BACKWARD(SUFFIX, SCALAR, ABS_FN)                                      \
extern "C" __global__ void gffx_cuda_divide_backward_##SUFFIX(                                 \
    const SCALAR *__restrict__ homogeneous, double eps, const SCALAR *__restrict__ grad_ndc,   \
    long long point_count, SCALAR *__restrict__ grad_homogeneous                               \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= point_count) return;                                                          \
    {                                                                                          \
        SCALAR w = homogeneous[point * 4 + 3];                                                 \
        if ((double)ABS_FN(w) > eps) {                                                         \
            SCALAR gx = grad_ndc[point * 3 + 0];                                               \
            SCALAR gy = grad_ndc[point * 3 + 1];                                               \
            SCALAR gz = grad_ndc[point * 3 + 2];                                               \
            SCALAR nx = homogeneous[point * 4 + 0] / w;                                        \
            SCALAR ny = homogeneous[point * 4 + 1] / w;                                        \
            SCALAR nz = homogeneous[point * 4 + 2] / w;                                        \
            grad_homogeneous[point * 4 + 0] = gx / w;                                          \
            grad_homogeneous[point * 4 + 1] = gy / w;                                          \
            grad_homogeneous[point * 4 + 2] = gz / w;                                          \
            grad_homogeneous[point * 4 + 3] = -(gx * nx + gy * ny + gz * nz) / w;              \
        } else {                                                                               \
            grad_homogeneous[point * 4 + 0] = (SCALAR)0;                                       \
            grad_homogeneous[point * 4 + 1] = (SCALAR)0;                                       \
            grad_homogeneous[point * 4 + 2] = (SCALAR)0;                                       \
            grad_homogeneous[point * 4 + 3] = (SCALAR)0;                                       \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_DIVIDE_BACKWARD(f32, float, fabsf)
GFFX_CUDA_DIVIDE_BACKWARD(f64, double, fabs)

/*
 * transforms.transform_points backward, in two kernels because its two outputs have different
 * shapes of dependency.
 *
 * grad_points is elementwise: each point's gradient is a contraction against its own batch's
 * matrix. The thread finds its batch by binary search over the packed offsets rather than being
 * told, which keeps the launch one-dimensional over points.
 *
 * grad_matrices is the reduction. One thread owns one of the sixteen elements of one matrix and
 * walks that batch's points in ascending order, which is exactly the order the host loop
 * accumulates in. Threads never share an output, so no atomics and no ordering ambiguity.
 */
__device__ __forceinline__ long long gffx_cuda_batch_of_point(
    const int *__restrict__ offsets, long long batch_count, long long point
) {
    long long low = 0;
    long long high = batch_count - 1;
    while (low < high) {
        long long mid = (low + high + 1) / 2;
        if ((long long)offsets[mid] <= point) low = mid; else high = mid - 1;
    }
    return low;
}

#define GFFX_CUDA_TRANSFORM_BACKWARD(SUFFIX, SCALAR)                                           \
extern "C" __global__ void gffx_cuda_transform_backward_points_##SUFFIX(                       \
    const SCALAR *__restrict__ matrices, const int *__restrict__ offsets,                      \
    const SCALAR *__restrict__ grad_homogeneous, long long point_count, long long batch_count, \
    SCALAR *__restrict__ grad_points                                                           \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= point_count) return;                                                          \
    {                                                                                          \
        long long batch = gffx_cuda_batch_of_point(offsets, batch_count, point);               \
        const SCALAR *m = matrices + batch * 16;                                               \
        const SCALAR *g = grad_homogeneous + point * 4;                                        \
        for (int column = 0; column < 3; ++column) {                                           \
            grad_points[point * 3 + column] =                                                  \
                g[0] * m[0 * 4 + column] + g[1] * m[1 * 4 + column] +                          \
                g[2] * m[2 * 4 + column] + g[3] * m[3 * 4 + column];                           \
        }                                                                                      \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_transform_backward_matrices_##SUFFIX(                     \
    const SCALAR *__restrict__ points, const int *__restrict__ offsets,                        \
    const SCALAR *__restrict__ grad_homogeneous, long long batch_count,                        \
    SCALAR *__restrict__ grad_matrices                                                         \
) {                                                                                            \
    long long slot = (long long)blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (slot >= batch_count * 16) return;                                                      \
    {                                                                                          \
        long long batch = slot / 16;                                                           \
        long long element = slot - batch * 16;                                                 \
        long long row = element / 4;                                                           \
        long long column = element - row * 4;                                                  \
        SCALAR total = (SCALAR)0;                                                              \
        for (long long point = (long long)offsets[batch];                                      \
             point < (long long)offsets[batch + 1]; ++point) {                                 \
            SCALAR g = grad_homogeneous[point * 4 + row];                                      \
            /* Column 3 is the translation column, whose input is the implicit 1. */           \
            total += column < 3 ? g * points[point * 3 + column] : g;                          \
        }                                                                                      \
        grad_matrices[slot] = total;                                                           \
    }                                                                                          \
}

GFFX_CUDA_TRANSFORM_BACKWARD(f32, float)
GFFX_CUDA_TRANSFORM_BACKWARD(f64, double)

/*
 * mesh.gather_faces backward. The host scatters, adding each corner's cotangent onto its vertex in
 * ascending face then corner order. Inverted here into a gather: one thread owns one vertex and
 * scans every face corner in that same order, taking the ones that name it. Reproducing the host's
 * addition order is what makes the result comparable byte for byte.
 *
 * The cost is O(V*F) rather than O(F), which is the price of the ordering. A vertex-to-face
 * incidence structure would remove it, and mesh.build_edge_topology already computes one, but
 * consuming it here would make this backward depend on another operation's output; that is a
 * design change rather than a kernel, and it is not made silently.
 */
#define GFFX_CUDA_GATHER_FACES_BACKWARD(SUFFIX, SCALAR)                                        \
extern "C" __global__ void gffx_cuda_gather_faces_backward_##SUFFIX(                           \
    const int *__restrict__ faces, const SCALAR *__restrict__ grad_face_vertices,              \
    long long face_count, long long vertex_count, SCALAR *__restrict__ grad_vertices            \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR gx = (SCALAR)0;                                                                 \
        SCALAR gy = (SCALAR)0;                                                                 \
        SCALAR gz = (SCALAR)0;                                                                 \
        for (long long face = 0; face < face_count; ++face) {                                  \
            for (int corner = 0; corner < 3; ++corner) {                                       \
                if ((long long)faces[face * 3 + corner] != vertex) continue;                   \
                gx += grad_face_vertices[face * 9 + corner * 3 + 0];                           \
                gy += grad_face_vertices[face * 9 + corner * 3 + 1];                           \
                gz += grad_face_vertices[face * 9 + corner * 3 + 2];                           \
            }                                                                                  \
        }                                                                                      \
        grad_vertices[vertex * 3 + 0] = gx;                                                    \
        grad_vertices[vertex * 3 + 1] = gy;                                                    \
        grad_vertices[vertex * 3 + 2] = gz;                                                    \
    }                                                                                          \
}

GFFX_CUDA_GATHER_FACES_BACKWARD(f32, float)
GFFX_CUDA_GATHER_FACES_BACKWARD(f64, double)

/*
 * render.interpolate backward, in three kernels because its two outputs differ in kind and one of
 * them has two lawful implementations.
 *
 * grad_barycentric is per fragment: a thread owns one fragment and writes its own three corners.
 * No interaction, so it needs no determinism argument beyond matching the host arithmetic, which
 * accumulates the channel sum in double for both dtypes and narrows once at the store.
 *
 * grad_face_attributes is the scatter, and it is the case the standing policy of 2026-08-28
 * covers: deterministic by default, relaxed only when the caller sets the flag.
 *
 *   ordered - a thread owns one attribute entry and scans fragments ascending, taking the ones
 *   that name its face. Reproduces the host's addition order exactly, so it is comparable byte for
 *   byte. Costs O(F*3*C*fragments), which is the honest price of the ordering and is why the
 *   relaxed path exists at all.
 *
 *   atomic - a thread owns one fragment and adds into whatever entries it touches. Fast, and the
 *   completion order is the hardware's, so repeated runs differ in the last bits. Reached only
 *   through GFFX_EXECUTION_ALLOW_NONDETERMINISTIC; a caller who has not asked for that must never
 *   receive it.
 */

#define GFFX_CUDA_INTERPOLATE_BACKWARD(SUFFIX, SCALAR)                                         \
extern "C" __global__ void gffx_cuda_interpolate_backward_bary_##SUFFIX(                       \
    const int *__restrict__ face_index, const SCALAR *__restrict__ face_attributes,            \
    const SCALAR *__restrict__ grad_attributes, long long fragment_count,                      \
    long long channel_count, SCALAR *__restrict__ grad_barycentric                             \
) {                                                                                            \
    long long fragment = (long long)blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (fragment >= fragment_count) return;                                                    \
    {                                                                                          \
        long long face = (long long)face_index[fragment];                                      \
        for (int corner = 0; corner < 3; ++corner) {                                           \
            double weight_gradient = 0.0;                                                      \
            if (face >= 0) {                                                                   \
                for (long long channel = 0; channel < channel_count; ++channel) {              \
                    double cotangent =                                                         \
                        (double)grad_attributes[fragment * channel_count + channel];           \
                    double value = (double)face_attributes[                                    \
                        (face * 3 + corner) * channel_count + channel];                        \
                    weight_gradient += value * cotangent;                                      \
                }                                                                              \
            }                                                                                  \
            grad_barycentric[fragment * 3 + corner] = (SCALAR)weight_gradient;                 \
        }                                                                                      \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_interpolate_backward_attr_ordered_##SUFFIX(               \
    const int *__restrict__ face_index, const SCALAR *__restrict__ barycentric,                \
    const SCALAR *__restrict__ grad_attributes, long long fragment_count,                      \
    long long channel_count, long long face_count,                                             \
    SCALAR *__restrict__ grad_face_attributes                                                  \
) {                                                                                            \
    long long entry = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (entry >= face_count * 3 * channel_count) return;                                       \
    {                                                                                          \
        long long channel = entry % channel_count;                                             \
        long long corner = (entry / channel_count) % 3;                                        \
        long long face = entry / (channel_count * 3);                                          \
        SCALAR total = (SCALAR)0;                                                              \
        for (long long fragment = 0; fragment < fragment_count; ++fragment) {                  \
            if ((long long)face_index[fragment] != face) continue;                             \
            {                                                                                  \
                double weight = (double)barycentric[fragment * 3 + corner];                    \
                double cotangent =                                                             \
                    (double)grad_attributes[fragment * channel_count + channel];               \
                /* The host narrows each term before adding it to a running sum in the         \
                 * operand dtype; doing the sum in double here would not match. */             \
                total += (SCALAR)(weight * cotangent);                                         \
            }                                                                                  \
        }                                                                                      \
        grad_face_attributes[entry] = total;                                                   \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_interpolate_backward_attr_atomic_##SUFFIX(                \
    const int *__restrict__ face_index, const SCALAR *__restrict__ barycentric,                \
    const SCALAR *__restrict__ grad_attributes, long long fragment_count,                      \
    long long channel_count, long long face_count,                                             \
    SCALAR *__restrict__ grad_face_attributes                                                  \
) {                                                                                            \
    long long fragment = (long long)blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (fragment >= fragment_count) return;                                                    \
    {                                                                                          \
        long long face = (long long)face_index[fragment];                                      \
        if (face < 0 || face >= face_count) return;                                            \
        for (int corner = 0; corner < 3; ++corner) {                                           \
            double weight = (double)barycentric[fragment * 3 + corner];                        \
            for (long long channel = 0; channel < channel_count; ++channel) {                  \
                double cotangent =                                                             \
                    (double)grad_attributes[fragment * channel_count + channel];               \
                atomicAdd(&grad_face_attributes[(face * 3 + corner) * channel_count + channel], \
                          (SCALAR)(weight * cotangent));                                       \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_zero_##SUFFIX(                                            \
    SCALAR *__restrict__ data, long long count                                                 \
) {                                                                                            \
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (index >= count) return;                                                                \
    data[index] = (SCALAR)0;                                                                   \
}

GFFX_CUDA_INTERPOLATE_BACKWARD(f32, float)
GFFX_CUDA_INTERPOLATE_BACKWARD(f64, double)

/*
 * render.texture backward.
 *
 * grad_coordinates is per sample: a thread owns one sample and writes its own two components, so
 * it needs no ordering argument. grad_pyramid is the scatter, and gets the two paths the standing
 * policy requires - an ordered gather by default, an atomic scatter behind the flag.
 *
 * The ordered path here is a texel-owned gather: the thread owning one pyramid element scans every
 * sample and takes the ones whose footprint covers it. That is O(pyramid * samples), which is the
 * price of reproducing the host's addition order, and is exactly why the relaxed path exists.
 *
 * Both paths recompute the level selection and filter weights rather than caching them, because a
 * cache would be workspace the contract puts at zero bytes for this operation.
 */

/*
 * Tap resolution, generated per dtype rather than shared in double. The host computes the
 * bilinear fractions in the operand dtype, so computing them in double here and narrowing would
 * give a different float32 answer - which is precisely the divergence the bit-identity claim
 * forbids, and it would have been invisible in a float64-only test.
 */
#define GFFX_CUDA_TEXTURE_TAPS(SUFFIX, SCALAR, FLOOR_FN)                                       \
__device__ __forceinline__ void gffx_cuda_texture_taps_##SUFFIX(                               \
    SCALAR u, SCALAR v, long long level_width, long long level_height,                         \
    unsigned int wrap_u, unsigned int wrap_v,                                                  \
    long long *xi0, long long *xi1, long long *yi0, long long *yi1,                            \
    int *in_x0, int *in_x1, int *in_y0, int *in_y1, SCALAR *a, SCALAR *b                       \
) {                                                                                            \
    SCALAR fx = u * (SCALAR)level_width - (SCALAR)0.5;                                         \
    SCALAR fy = v * (SCALAR)level_height - (SCALAR)0.5;                                        \
    long long x0 = (long long)FLOOR_FN(fx);                                                    \
    long long y0 = (long long)FLOOR_FN(fy);                                                    \
    *a = fx - (SCALAR)x0;                                                                      \
    *b = fy - (SCALAR)y0;                                                                      \
    *in_x0 = gffx_cuda_texture_wrap(x0, level_width, wrap_u, xi0);                             \
    *in_x1 = gffx_cuda_texture_wrap(x0 + 1, level_width, wrap_u, xi1);                         \
    *in_y0 = gffx_cuda_texture_wrap(y0, level_height, wrap_v, yi0);                            \
    *in_y1 = gffx_cuda_texture_wrap(y0 + 1, level_height, wrap_v, yi1);                        \
}

GFFX_CUDA_TEXTURE_TAPS(f32, float, floorf)
GFFX_CUDA_TEXTURE_TAPS(f64, double, floor)

/* Level selection, shared by both grad_pyramid paths and by grad_coordinates so the three cannot
 * disagree about which level a sample read. */
__device__ __forceinline__ void gffx_cuda_texture_levels_for(
    const double *lod_in, long long level_count, unsigned int mip_filter,
    long long *first, long long *second, double *blend
) {
    double lod = *lod_in;
    if (!(lod > 0.0)) lod = 0.0;
    if (lod > (double)(level_count - 1)) lod = (double)(level_count - 1);
    if (mip_filter == 1u) {
        *first = (long long)(lod + 0.5);
        if (*first > level_count - 1) *first = level_count - 1;
        *second = *first;
        *blend = 0.0;
        return;
    }
    {
        double base = floor(lod);
        *first = (long long)base;
        *second = *first + 1;
        if (*second > level_count - 1) *second = level_count - 1;
        *blend = lod - base;
    }
}

#define GFFX_CUDA_TEXTURE_BACKWARD(SUFFIX, SCALAR, SQRT_FN, LOG2_FN, FLOOR_FN)                           \
__device__ __forceinline__ double gffx_cuda_texture_lod_##SUFFIX(                              \
    const SCALAR *derivatives, const SCALAR *lod_values, long long n,                          \
    long long height, long long width                                                          \
) {                                                                                            \
    if (derivatives != 0) {                                                                    \
        SCALAR ax = derivatives[n * 4] * (SCALAR)width;                                        \
        SCALAR bx = derivatives[n * 4 + 1] * (SCALAR)height;                                   \
        SCALAR ay = derivatives[n * 4 + 2] * (SCALAR)width;                                    \
        SCALAR by = derivatives[n * 4 + 3] * (SCALAR)height;                                   \
        SCALAR rx = SQRT_FN(ax * ax + bx * bx);                                                \
        SCALAR ry = SQRT_FN(ay * ay + by * by);                                                \
        SCALAR rho = rx > ry ? rx : ry;                                                        \
        if (!(rho > (SCALAR)1.1754943508222875e-38))                                           \
            rho = (SCALAR)1.1754943508222875e-38;                                              \
        return (double)LOG2_FN(rho);                                                           \
    }                                                                                          \
    if (lod_values != 0) return (double)lod_values[n];                                         \
    return 0.0;                                                                                \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_texture_backward_coords_##SUFFIX(                         \
    const SCALAR *__restrict__ pyramid, const int *__restrict__ offsets,                       \
    long long level_count, long long height, long long width, long long channels,              \
    const SCALAR *__restrict__ coordinates, long long count,                                   \
    const SCALAR *__restrict__ derivatives, const SCALAR *__restrict__ lod_values,             \
    unsigned int filter, unsigned int mip_filter, unsigned int wrap_u, unsigned int wrap_v,    \
    const SCALAR *__restrict__ grad_samples, SCALAR *__restrict__ grad_coordinates             \
) {                                                                                            \
    long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (n >= count) return;                                                                    \
    {                                                                                          \
        SCALAR u = coordinates[n * 2];                                                         \
        SCALAR v = coordinates[n * 2 + 1];                                                     \
        SCALAR gu = (SCALAR)0;                                                                 \
        SCALAR gv = (SCALAR)0;                                                                 \
        double lod;                                                                            \
        long long first, second;                                                               \
        double blend;                                                                          \
        int pass;                                                                              \
        /* NEAREST is piecewise constant, so its coordinate gradient is exactly zero rather     \
         * than a small approximation of one. A non-finite coordinate contributes nothing. */   \
        if (filter == 1u || !(u == u) || !(v == v) ||                                          \
            u * (SCALAR)0 != (SCALAR)0 || v * (SCALAR)0 != (SCALAR)0) {                        \
            grad_coordinates[n * 2] = (SCALAR)0;                                               \
            grad_coordinates[n * 2 + 1] = (SCALAR)0;                                           \
            return;                                                                            \
        }                                                                                      \
        lod = gffx_cuda_texture_lod_##SUFFIX(derivatives, lod_values, n, height, width);       \
        gffx_cuda_texture_levels_for(&lod, level_count, mip_filter, &first, &second, &blend);  \
        for (pass = 0; pass < 2; ++pass) {                                                     \
            long long level = pass == 0 ? first : second;                                      \
            SCALAR level_weight = pass == 0 ? (SCALAR)(1.0 - ((blend > 0.0 && second != first) \
                                                              ? blend : 0.0))                  \
                                            : (SCALAR)blend;                                   \
            long long lh, lw, xi0, xi1, yi0, yi1;                                              \
            int ix0, ix1, iy0, iy1;                                                            \
            SCALAR a, b;                                                                       \
            const SCALAR *data;                                                                \
            if (pass == 1 && !(blend > 0.0 && second != first)) break;                         \
            gffx_cuda_texture_level_extent(height, width, level, &lh, &lw);                    \
            gffx_cuda_texture_taps_##SUFFIX(u, v, lw, lh, wrap_u, wrap_v,               \
                                   &xi0, &xi1, &yi0, &yi1, &ix0, &ix1, &iy0, &iy1, &a, &b);    \
            data = pyramid + offsets[level];                                                   \
            for (long long c = 0; c < channels; ++c) {                                         \
                SCALAR g = level_weight * grad_samples[n * channels + c];                      \
                SCALAR t00 = (ix0 && iy0) ? data[(yi0 * lw + xi0) * channels + c] : (SCALAR)0; \
                SCALAR t10 = (ix1 && iy0) ? data[(yi0 * lw + xi1) * channels + c] : (SCALAR)0; \
                SCALAR t01 = (ix0 && iy1) ? data[(yi1 * lw + xi0) * channels + c] : (SCALAR)0; \
                SCALAR t11 = (ix1 && iy1) ? data[(yi1 * lw + xi1) * channels + c] : (SCALAR)0; \
                gu += g * (SCALAR)lw * (((SCALAR)1 - b) * (t10 - t00) +                \
                                        b * (t11 - t01));                              \
                gv += g * (SCALAR)lh * (((SCALAR)1 - a) * (t01 - t00) +                \
                                        a * (t11 - t10));                              \
            }                                                                                  \
        }                                                                                      \
        grad_coordinates[n * 2] = gu;                                                          \
        grad_coordinates[n * 2 + 1] = gv;                                                      \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_texture_backward_pyramid_atomic_##SUFFIX(                 \
    const int *__restrict__ offsets, long long level_count,                                    \
    long long height, long long width, long long channels,                                     \
    const SCALAR *__restrict__ coordinates, long long count,                                   \
    const SCALAR *__restrict__ derivatives, const SCALAR *__restrict__ lod_values,             \
    unsigned int filter, unsigned int mip_filter, unsigned int wrap_u, unsigned int wrap_v,    \
    const SCALAR *__restrict__ grad_samples, SCALAR *__restrict__ grad_pyramid                 \
) {                                                                                            \
    long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (n >= count) return;                                                                    \
    {                                                                                          \
        SCALAR u = coordinates[n * 2];                                                         \
        SCALAR v = coordinates[n * 2 + 1];                                                     \
        double lod;                                                                            \
        long long first, second;                                                               \
        double blend;                                                                          \
        int pass;                                                                              \
        if (!(u == u) || !(v == v) || u * (SCALAR)0 != (SCALAR)0 ||                            \
            v * (SCALAR)0 != (SCALAR)0) return;                                                \
        lod = gffx_cuda_texture_lod_##SUFFIX(derivatives, lod_values, n, height, width);       \
        gffx_cuda_texture_levels_for(&lod, level_count, mip_filter, &first, &second, &blend);  \
        for (pass = 0; pass < 2; ++pass) {                                                     \
            long long level = pass == 0 ? first : second;                                      \
            SCALAR level_weight = pass == 0 ? (SCALAR)(1.0 - ((blend > 0.0 && second != first) \
                                                              ? blend : 0.0))                  \
                                            : (SCALAR)blend;                                   \
            long long lh, lw, xi0, xi1, yi0, yi1;                                              \
            int ix0, ix1, iy0, iy1;                                                            \
            SCALAR a, b;                                                                       \
            SCALAR *data;                                                                      \
            if (pass == 1 && !(blend > 0.0 && second != first)) break;                         \
            gffx_cuda_texture_level_extent(height, width, level, &lh, &lw);                    \
            data = grad_pyramid + offsets[level];                                              \
            if (filter == 1u) {                                                                \
                long long x = (long long)FLOOR_FN(u * (SCALAR)lw);                        \
                long long y = (long long)FLOOR_FN(v * (SCALAR)lh);                        \
                long long xi, yi;                                                              \
                if (gffx_cuda_texture_wrap(x, lw, wrap_u, &xi) &                               \
                    gffx_cuda_texture_wrap(y, lh, wrap_v, &yi)) {                              \
                    for (long long c = 0; c < channels; ++c) {                                 \
                        atomicAdd(&data[(yi * lw + xi) * channels + c],                        \
                                  level_weight * grad_samples[n * channels + c]);              \
                    }                                                                          \
                }                                                                              \
                continue;                                                                      \
            }                                                                                  \
            gffx_cuda_texture_taps_##SUFFIX(u, v, lw, lh, wrap_u, wrap_v,               \
                                   &xi0, &xi1, &yi0, &yi1, &ix0, &ix1, &iy0, &iy1, &a, &b);    \
            for (long long c = 0; c < channels; ++c) {                                         \
                SCALAR g = level_weight * grad_samples[n * channels + c];                      \
                if (ix0 && iy0) atomicAdd(&data[(yi0 * lw + xi0) * channels + c],              \
                    ((SCALAR)1 - a) * ((SCALAR)1 - b) * g);                    \
                if (ix1 && iy0) atomicAdd(&data[(yi0 * lw + xi1) * channels + c],              \
                    a * ((SCALAR)1 - b) * g);                                  \
                if (ix0 && iy1) atomicAdd(&data[(yi1 * lw + xi0) * channels + c],              \
                    ((SCALAR)1 - a) * b * g);                                  \
                if (ix1 && iy1) atomicAdd(&data[(yi1 * lw + xi1) * channels + c],              \
                    a * b * g);                                                \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_TEXTURE_BACKWARD(f32, float, sqrtf, log2f, floorf)
GFFX_CUDA_TEXTURE_BACKWARD(f64, double, sqrt, log2, floor)

/*
 * The ordered default for grad_pyramid. One thread owns one pyramid element and scans every sample
 * in ascending order, taking the taps that land on it. Each sample's four taps are tested in the
 * host's own order - (x0,y0), (x1,y0), (x0,y1), (x1,y1) - because a wrap mode can map more than one
 * tap of a single sample onto the same texel, and those contributions are then separate additions
 * whose order matters.
 */
#define GFFX_CUDA_TEXTURE_BACKWARD_ORDERED(SUFFIX, SCALAR, FLOOR_FN)                                     \
extern "C" __global__ void gffx_cuda_texture_backward_pyramid_ordered_##SUFFIX(                \
    const int *__restrict__ offsets, long long level_count,                                    \
    long long height, long long width, long long channels,                                     \
    const SCALAR *__restrict__ coordinates, long long count,                                   \
    const SCALAR *__restrict__ derivatives, const SCALAR *__restrict__ lod_values,             \
    unsigned int filter, unsigned int mip_filter, unsigned int wrap_u, unsigned int wrap_v,    \
    const SCALAR *__restrict__ grad_samples, SCALAR *__restrict__ grad_pyramid,                \
    long long total_elements                                                                   \
) {                                                                                            \
    long long slot = (long long)blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (slot >= total_elements) return;                                                        \
    {                                                                                          \
        long long level = 0;                                                                   \
        long long lh, lw, local, texel, channel;                                               \
        SCALAR total = (SCALAR)0;                                                              \
        while (level + 1 < level_count && (long long)offsets[level + 1] <= slot) ++level;       \
        gffx_cuda_texture_level_extent(height, width, level, &lh, &lw);                        \
        local = slot - (long long)offsets[level];                                              \
        channel = local % channels;                                                            \
        texel = local / channels;                                                              \
        for (long long n = 0; n < count; ++n) {                                                \
            SCALAR u = coordinates[n * 2];                                                     \
            SCALAR v = coordinates[n * 2 + 1];                                                 \
            double lod;                                                                        \
            long long first, second;                                                           \
            double blend;                                                                      \
            int pass;                                                                          \
            if (!(u == u) || !(v == v) || u * (SCALAR)0 != (SCALAR)0 ||                        \
                v * (SCALAR)0 != (SCALAR)0) continue;                                          \
            lod = gffx_cuda_texture_lod_##SUFFIX(derivatives, lod_values, n, height, width);   \
            gffx_cuda_texture_levels_for(&lod, level_count, mip_filter, &first, &second,       \
                                         &blend);                                              \
            for (pass = 0; pass < 2; ++pass) {                                                 \
                long long used = pass == 0 ? first : second;                                   \
                SCALAR level_weight = pass == 0                                                \
                    ? (SCALAR)(1.0 - ((blend > 0.0 && second != first) ? blend : 0.0))          \
                    : (SCALAR)blend;                                                           \
                long long xi0, xi1, yi0, yi1;                                                  \
                int ix0, ix1, iy0, iy1;                                                        \
                SCALAR a, b;                                                                   \
                SCALAR g;                                                                      \
                if (pass == 1 && !(blend > 0.0 && second != first)) break;                     \
                if (used != level) continue;                                                   \
                g = level_weight * grad_samples[n * channels + channel];                       \
                if (filter == 1u) {                                                            \
                    long long x = (long long)FLOOR_FN(u * (SCALAR)lw);                    \
                    long long y = (long long)FLOOR_FN(v * (SCALAR)lh);                    \
                    long long xi, yi;                                                          \
                    if ((gffx_cuda_texture_wrap(x, lw, wrap_u, &xi) &                          \
                         gffx_cuda_texture_wrap(y, lh, wrap_v, &yi)) &&                        \
                        yi * lw + xi == texel) {                                               \
                        total += g;                                                            \
                    }                                                                          \
                    continue;                                                                  \
                }                                                                              \
                gffx_cuda_texture_taps_##SUFFIX(u, v, lw, lh, wrap_u, wrap_v,           \
                                       &xi0, &xi1, &yi0, &yi1, &ix0, &ix1, &iy0, &iy1, &a, &b); \
                if (ix0 && iy0 && yi0 * lw + xi0 == texel)                                     \
                    total += ((SCALAR)1 - a) * ((SCALAR)1 - b) * g;            \
                if (ix1 && iy0 && yi0 * lw + xi1 == texel)                                     \
                    total += a * ((SCALAR)1 - b) * g;                          \
                if (ix0 && iy1 && yi1 * lw + xi0 == texel)                                     \
                    total += ((SCALAR)1 - a) * b * g;                          \
                if (ix1 && iy1 && yi1 * lw + xi1 == texel)                                     \
                    total += a * b * g;                                        \
            }                                                                                  \
        }                                                                                      \
        grad_pyramid[slot] = total;                                                            \
    }                                                                                          \
}

GFFX_CUDA_TEXTURE_BACKWARD_ORDERED(f32, float, floorf)
GFFX_CUDA_TEXTURE_BACKWARD_ORDERED(f64, double, floor)

/*
 * mesh.face_geometry backward. The host walks faces ascending and, within each face, adds to
 * corner 1, then corner 2, then subtracts the sum of both from corner 0. Inverted here into a
 * per-vertex gather that repeats that inner order exactly, because a degenerate face can name the
 * same vertex twice and the three contributions are then separate additions whose order matters.
 *
 * The face's own arithmetic is recomputed per visiting vertex rather than cached; caching it would
 * be workspace, and this operation's contract puts that at zero bytes.
 */
#define GFFX_CUDA_FACE_GEOMETRY_BACKWARD(SUFFIX, SCALAR, SQRT_FN)                              \
extern "C" __global__ void gffx_cuda_face_geometry_backward_##SUFFIX(                          \
    const SCALAR *__restrict__ vertices, const int *__restrict__ faces,                        \
    long long face_count, long long vertex_count, double eps,                                  \
    const SCALAR *__restrict__ grad_unit_normals, const SCALAR *__restrict__ grad_areas,       \
    SCALAR *__restrict__ grad_vertices                                                         \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR tx = (SCALAR)0, ty = (SCALAR)0, tz = (SCALAR)0;                                 \
        for (long long face = 0; face < face_count; ++face) {                                  \
            long long i0 = (long long)faces[face * 3 + 0];                                     \
            long long i1 = (long long)faces[face * 3 + 1];                                     \
            long long i2 = (long long)faces[face * 3 + 2];                                     \
            const SCALAR *a; const SCALAR *b; const SCALAR *c;                                 \
            SCALAR e1x, e1y, e1z, e2x, e2y, e2z, cx, cy, cz, doubled;                          \
            SCALAR nx, ny, nz, gnx, gny, gnz, ga, dot, gcx, gcy, gcz;                          \
            SCALAR g1x, g1y, g1z, g2x, g2y, g2z;                                               \
            if (i0 != vertex && i1 != vertex && i2 != vertex) continue;                        \
            a = vertices + i0 * 3; b = vertices + i1 * 3; c = vertices + i2 * 3;               \
            e1x = b[0] - a[0]; e1y = b[1] - a[1]; e1z = b[2] - a[2];                           \
            e2x = c[0] - a[0]; e2y = c[1] - a[1]; e2z = c[2] - a[2];                           \
            cx = e1y * e2z - e1z * e2y;                                                        \
            cy = e1z * e2x - e1x * e2z;                                                        \
            cz = e1x * e2y - e1y * e2x;                                                        \
            doubled = SQRT_FN(cx * cx + cy * cy + cz * cz);                                    \
            if (!((double)doubled > eps)) continue;                                            \
            nx = cx / doubled; ny = cy / doubled; nz = cz / doubled;                           \
            gnx = grad_unit_normals != 0 ? grad_unit_normals[face * 3 + 0] : (SCALAR)0;        \
            gny = grad_unit_normals != 0 ? grad_unit_normals[face * 3 + 1] : (SCALAR)0;        \
            gnz = grad_unit_normals != 0 ? grad_unit_normals[face * 3 + 2] : (SCALAR)0;        \
            ga = grad_areas != 0 ? grad_areas[face] : (SCALAR)0;                               \
            dot = gnx * nx + gny * ny + gnz * nz;                                              \
            gcx = (gnx - dot * nx) / doubled + (SCALAR)0.5 * ga * nx;                          \
            gcy = (gny - dot * ny) / doubled + (SCALAR)0.5 * ga * ny;                          \
            gcz = (gnz - dot * nz) / doubled + (SCALAR)0.5 * ga * nz;                          \
            g1x = e2y * gcz - e2z * gcy;                                                       \
            g1y = e2z * gcx - e2x * gcz;                                                       \
            g1z = e2x * gcy - e2y * gcx;                                                       \
            g2x = gcy * e1z - gcz * e1y;                                                       \
            g2y = gcz * e1x - gcx * e1z;                                                       \
            g2z = gcx * e1y - gcy * e1x;                                                       \
            /* Corner 1, then corner 2, then corner 0, matching the host. */                   \
            if (i1 == vertex) { tx += g1x; ty += g1y; tz += g1z; }                             \
            if (i2 == vertex) { tx += g2x; ty += g2y; tz += g2z; }                             \
            if (i0 == vertex) {                                                                \
                tx -= g1x + g2x; ty -= g1y + g2y; tz -= g1z + g2z;                             \
            }                                                                                  \
        }                                                                                      \
        grad_vertices[vertex * 3 + 0] = tx;                                                    \
        grad_vertices[vertex * 3 + 1] = ty;                                                    \
        grad_vertices[vertex * 3 + 2] = tz;                                                    \
    }                                                                                          \
}

GFFX_CUDA_FACE_GEOMETRY_BACKWARD(f32, float, sqrtf)
GFFX_CUDA_FACE_GEOMETRY_BACKWARD(f64, double, sqrt)

/*
 * render.texture_pyramid backward. The host scatters each level's gradient down to the level-0
 * rectangle it covers, with the accumulated weight uniform over that footprint. Inverted here into
 * a per-texel gather: the thread owning one level-0 texel walks the levels in ascending order and
 * takes the one coarse texel above it at each, which is the host's own order.
 *
 * This one needs no relaxed path. Each level contributes exactly one term per level-0 texel, so
 * the reduction is as long as the level chain - four or five terms, not millions - and the ordered
 * form is already the fast one.
 */
#define GFFX_CUDA_TEXTURE_PYRAMID_BACKWARD(SUFFIX, SCALAR)                                     \
extern "C" __global__ void gffx_cuda_texture_pyramid_backward_##SUFFIX(                        \
    const int *__restrict__ offsets, long long level_count,                                    \
    long long height, long long width, long long channels,                                     \
    const SCALAR *__restrict__ grad_pyramid, SCALAR *__restrict__ grad_texture                 \
) {                                                                                            \
    long long slot = (long long)blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (slot >= height * width * channels) return;                                             \
    {                                                                                          \
        long long channel = slot % channels;                                                   \
        long long texel = slot / channels;                                                     \
        long long y0 = texel / width;                                                          \
        long long x0 = texel - y0 * width;                                                     \
        SCALAR total = grad_pyramid[slot];                                                     \
        for (long long level = 1; level < level_count; ++level) {                              \
            long long cy = y0;                                                                 \
            long long cx = x0;                                                                 \
            long long area = 1;                                                                \
            long long lh, lw, step;                                                            \
            /* Walk up the chain, halving the index wherever that extent halved, and            \
             * accumulating the footprint area so the weight is one over it. */                \
            for (step = 1; step <= level; ++step) {                                            \
                long long ph, pw;                                                              \
                gffx_cuda_texture_level_extent(height, width, step - 1, &ph, &pw);             \
                if (ph > 1) { cy /= 2; area *= 2; }                                            \
                if (pw > 1) { cx /= 2; area *= 2; }                                            \
            }                                                                                  \
            gffx_cuda_texture_level_extent(height, width, level, &lh, &lw);                    \
            /* A dropped odd row or column has no coarse parent and receives nothing. */        \
            if (cy >= lh || cx >= lw) continue;                                                \
            total += grad_pyramid[offsets[level] + (cy * lw + cx) * channels + channel] /       \
                     (SCALAR)area;                                                             \
        }                                                                                      \
        grad_texture[slot] = total;                                                            \
    }                                                                                          \
}

GFFX_CUDA_TEXTURE_PYRAMID_BACKWARD(f32, float)
GFFX_CUDA_TEXTURE_PYRAMID_BACKWARD(f64, double)

/*
 * points.knn and points.closest_point_on_mesh backwards.
 *
 * Both have one output that is per query and one that scatters onto the reference geometry. The
 * per-query output needs no ordering argument; the scattering one gets the ordered default and the
 * atomic path behind the flag, as the standing policy requires.
 *
 * Arithmetic follows the host per dtype rather than promoting to double, because the host's own
 * float32 branch computes in float and promoting here would disagree with it.
 */

#define GFFX_CUDA_KNN_BACKWARD(SUFFIX, SCALAR)                                                 \
extern "C" __global__ void gffx_cuda_knn_backward_query_##SUFFIX(                              \
    const SCALAR *__restrict__ query, const SCALAR *__restrict__ reference,                    \
    const int *__restrict__ index, const unsigned char *__restrict__ valid,                    \
    const SCALAR *__restrict__ cotangent, long long query_count, long long neighbor_count,     \
    SCALAR *__restrict__ grad_query                                                            \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= query_count) return;                                                          \
    {                                                                                          \
        SCALAR g0 = (SCALAR)0, g1 = (SCALAR)0, g2 = (SCALAR)0;                                 \
        for (long long slot = 0; slot < neighbor_count; ++slot) {                              \
            long long entry = point * neighbor_count + slot;                                   \
            long long neighbor;                                                                \
            SCALAR scale;                                                                      \
            if (valid[entry] == 0u) continue;                                                  \
            neighbor = (long long)index[entry];                                                \
            scale = (SCALAR)2 * cotangent[entry];                                              \
            g0 += scale * (query[point * 3 + 0] - reference[neighbor * 3 + 0]);                \
            g1 += scale * (query[point * 3 + 1] - reference[neighbor * 3 + 1]);                \
            g2 += scale * (query[point * 3 + 2] - reference[neighbor * 3 + 2]);                \
        }                                                                                      \
        grad_query[point * 3 + 0] = g0;                                                        \
        grad_query[point * 3 + 1] = g1;                                                        \
        grad_query[point * 3 + 2] = g2;                                                        \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_knn_backward_reference_ordered_##SUFFIX(                  \
    const SCALAR *__restrict__ query, const SCALAR *__restrict__ reference,                    \
    const int *__restrict__ index, const unsigned char *__restrict__ valid,                    \
    const SCALAR *__restrict__ cotangent, long long query_count, long long neighbor_count,     \
    long long reference_count, SCALAR *__restrict__ grad_reference                             \
) {                                                                                            \
    long long target = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (target >= reference_count) return;                                                     \
    {                                                                                          \
        SCALAR g0 = (SCALAR)0, g1 = (SCALAR)0, g2 = (SCALAR)0;                                 \
        for (long long point = 0; point < query_count; ++point) {                              \
            for (long long slot = 0; slot < neighbor_count; ++slot) {                          \
                long long entry = point * neighbor_count + slot;                               \
                SCALAR scale;                                                                  \
                if (valid[entry] == 0u) continue;                                              \
                if ((long long)index[entry] != target) continue;                               \
                scale = (SCALAR)2 * cotangent[entry];                                          \
                g0 -= scale * (query[point * 3 + 0] - reference[target * 3 + 0]);              \
                g1 -= scale * (query[point * 3 + 1] - reference[target * 3 + 1]);              \
                g2 -= scale * (query[point * 3 + 2] - reference[target * 3 + 2]);              \
            }                                                                                  \
        }                                                                                      \
        grad_reference[target * 3 + 0] = g0;                                                   \
        grad_reference[target * 3 + 1] = g1;                                                   \
        grad_reference[target * 3 + 2] = g2;                                                   \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_knn_backward_reference_atomic_##SUFFIX(                   \
    const SCALAR *__restrict__ query, const SCALAR *__restrict__ reference,                    \
    const int *__restrict__ index, const unsigned char *__restrict__ valid,                    \
    const SCALAR *__restrict__ cotangent, long long query_count, long long neighbor_count,     \
    long long reference_count, SCALAR *__restrict__ grad_reference                             \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= query_count) return;                                                          \
    for (long long slot = 0; slot < neighbor_count; ++slot) {                                  \
        long long entry = point * neighbor_count + slot;                                       \
        long long neighbor;                                                                    \
        SCALAR scale;                                                                          \
        if (valid[entry] == 0u) continue;                                                      \
        neighbor = (long long)index[entry];                                                    \
        if (neighbor < 0 || neighbor >= reference_count) continue;                             \
        scale = (SCALAR)2 * cotangent[entry];                                                  \
        for (int axis = 0; axis < 3; ++axis) {                                                 \
            atomicAdd(&grad_reference[neighbor * 3 + axis],                                    \
                      -(scale * (query[point * 3 + axis] - reference[neighbor * 3 + axis])));  \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_KNN_BACKWARD(f32, float)
GFFX_CUDA_KNN_BACKWARD(f64, double)

/*
 * points.closest_point_on_mesh backward. The host computes the residual in double for both dtypes
 * and narrows once per stored term, so these do the same: the term is formed in double and cast to
 * the operand type before it joins the running sum.
 */
#define GFFX_CUDA_CLOSEST_BACKWARD(SUFFIX, SCALAR)                                             \
extern "C" __global__ void gffx_cuda_closest_backward_points_##SUFFIX(                         \
    const SCALAR *__restrict__ points, const SCALAR *__restrict__ closest,                     \
    const unsigned char *__restrict__ valid, const SCALAR *__restrict__ cotangent,             \
    long long point_count, SCALAR *__restrict__ grad_points                                    \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= point_count) return;                                                          \
    {                                                                                          \
        if (valid[point] == 0u) {                                                              \
            grad_points[point * 3 + 0] = (SCALAR)0;                                            \
            grad_points[point * 3 + 1] = (SCALAR)0;                                            \
            grad_points[point * 3 + 2] = (SCALAR)0;                                            \
            return;                                                                            \
        }                                                                                      \
        {                                                                                      \
            double g = (double)cotangent[point];                                               \
            for (int axis = 0; axis < 3; ++axis) {                                             \
                double px = (double)points[point * 3 + axis];                                  \
                double qx = (double)closest[point * 3 + axis];                                 \
                /* Accumulated from zero rather than assigned, because the host zeroes          \
                 * the array and adds into it. Where the term is a negative zero the two        \
                 * differ: assignment keeps -0.0 while 0.0 + (-0.0) is +0.0. The two            \
                 * compare equal, so only a bitwise comparison sees it - which is the           \
                 * reason this fixture uses memcmp rather than equality. */                     \
                SCALAR accumulated = (SCALAR)0;                                                 \
                accumulated += (SCALAR)(2.0 * (px - qx) * g);                                   \
                grad_points[point * 3 + axis] = accumulated;                                    \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_closest_backward_vertices_ordered_##SUFFIX(               \
    const SCALAR *__restrict__ points, const SCALAR *__restrict__ closest,                     \
    const SCALAR *__restrict__ barycentric, const int *__restrict__ face_index,                \
    const int *__restrict__ faces, const unsigned char *__restrict__ valid,                    \
    const SCALAR *__restrict__ cotangent, long long point_count, long long vertex_count,       \
    SCALAR *__restrict__ grad_vertices                                                         \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR total[3];                                                                       \
        total[0] = (SCALAR)0; total[1] = (SCALAR)0; total[2] = (SCALAR)0;                      \
        for (long long point = 0; point < point_count; ++point) {                              \
            long long face;                                                                    \
            double g;                                                                          \
            if (valid[point] == 0u) continue;                                                  \
            face = (long long)face_index[point];                                               \
            g = (double)cotangent[point];                                                      \
            /* Axis outer, corner inner, matching the host, because a degenerate face can       \
             * name this vertex more than once and those are separate subtractions. */          \
            for (int axis = 0; axis < 3; ++axis) {                                             \
                double px = (double)points[point * 3 + axis];                                  \
                double qx = (double)closest[point * 3 + axis];                                 \
                double residual = 2.0 * (px - qx) * g;                                         \
                for (int corner = 0; corner < 3; ++corner) {                                   \
                    if ((long long)faces[face * 3 + corner] != vertex) continue;               \
                    total[axis] -= (SCALAR)((double)barycentric[point * 3 + corner] *          \
                                            residual);                                         \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        grad_vertices[vertex * 3 + 0] = total[0];                                              \
        grad_vertices[vertex * 3 + 1] = total[1];                                              \
        grad_vertices[vertex * 3 + 2] = total[2];                                              \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_closest_backward_vertices_atomic_##SUFFIX(                \
    const SCALAR *__restrict__ points, const SCALAR *__restrict__ closest,                     \
    const SCALAR *__restrict__ barycentric, const int *__restrict__ face_index,                \
    const int *__restrict__ faces, const unsigned char *__restrict__ valid,                    \
    const SCALAR *__restrict__ cotangent, long long point_count, long long vertex_count,       \
    SCALAR *__restrict__ grad_vertices                                                         \
) {                                                                                            \
    long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (point >= point_count) return;                                                          \
    if (valid[point] == 0u) return;                                                            \
    {                                                                                          \
        long long face = (long long)face_index[point];                                         \
        double g = (double)cotangent[point];                                                   \
        for (int axis = 0; axis < 3; ++axis) {                                                 \
            double px = (double)points[point * 3 + axis];                                      \
            double qx = (double)closest[point * 3 + axis];                                     \
            double residual = 2.0 * (px - qx) * g;                                             \
            for (int corner = 0; corner < 3; ++corner) {                                       \
                long long vertex = (long long)faces[face * 3 + corner];                        \
                if (vertex < 0 || vertex >= vertex_count) continue;                            \
                atomicAdd(&grad_vertices[vertex * 3 + axis],                                   \
                          -(SCALAR)((double)barycentric[point * 3 + corner] * residual));      \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_CLOSEST_BACKWARD(f32, float)
GFFX_CUDA_CLOSEST_BACKWARD(f64, double)

/* ---------------------------------------------------------------- render.rasterize backward
 *
 * The largest of the backwards. Each fragment contributes to the three vertices of the face it
 * named, through three separate routes: the barycentric weights, the depth, and the signed
 * distance. All arithmetic is in double for both dtypes, exactly as the host does it, with the
 * narrowing to float happening once at the store.
 *
 * The scatter is onto vertices from fragments, and there are far more fragments than vertices, so
 * this gets both paths the standing policy requires. The ordered gather costs O(V * fragments) and
 * is the default; the atomic scatter is one thread per fragment and is reached only through
 * GFFX_EXECUTION_ALLOW_NONDETERMINISTIC.
 *
 * A fragment's contribution is computed identically in both paths by a shared device function, so
 * the two cannot drift apart in the arithmetic and differ only in how the results are combined.
 */

__device__ __forceinline__ double gffx_cuda_segment_distance_t(
    double px, double py, double ux, double uy, double vx, double vy, double *out_t
) {
    double ex = vx - ux;
    double ey = vy - uy;
    double length_squared = ex * ex + ey * ey;
    double t = 0.0;
    double cx, cy, dx, dy;
    if (length_squared > 0.0) {
        t = ((px - ux) * ex + (py - uy) * ey) / length_squared;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
    }
    cx = ux + t * ex;
    cy = uy + t * ey;
    dx = px - cx;
    dy = py - cy;
    *out_t = t;
    return dx * dx + dy * dy;
}

/* Strict improvement only, so an exact tie keeps the lower edge index, matching the host. */
__device__ __forceinline__ void gffx_cuda_boundary_edge(
    double px, double py, double ax, double ay, double bx, double by, double cx, double cy,
    int *out_edge, double *out_t
) {
    double best_t;
    double best = gffx_cuda_segment_distance_t(px, py, ax, ay, bx, by, &best_t);
    int best_edge = 0;
    double t;
    double candidate = gffx_cuda_segment_distance_t(px, py, bx, by, cx, cy, &t);
    if (candidate < best) { best = candidate; best_edge = 1; best_t = t; }
    candidate = gffx_cuda_segment_distance_t(px, py, cx, cy, ax, ay, &t);
    if (candidate < best) { best = candidate; best_edge = 2; best_t = t; }
    *out_edge = best_edge;
    *out_t = best_t;
}

/*
 * One fragment's contribution to its three corners, in pixel space and in NDC z. Returns 0 when
 * the fragment contributes nothing, so both callers skip identically.
 */
#define GFFX_CUDA_RASTERIZE_BACKWARD_TERM(SUFFIX, SCALAR)                                      \
__device__ __forceinline__ int gffx_cuda_rasterize_backward_term_##SUFFIX(                     \
    const SCALAR *__restrict__ ndc_vertices, const int *__restrict__ faces,                    \
    long long face, long long entry, long long image_height, long long image_width,            \
    long long row, long long column,                                                           \
    const SCALAR *__restrict__ grad_barycentric, const SCALAR *__restrict__ grad_depth,        \
    const SCALAR *__restrict__ grad_signed_distance,                                           \
    long long *ids, double *out_gx, double *out_gy, double *out_gz                             \
) {                                                                                            \
    double half_width = (double)image_width * 0.5;                                             \
    double half_height = (double)image_height * 0.5;                                           \
    double px = (double)column + 0.5;                                                          \
    double py = (double)row + 0.5;                                                             \
    double corner_x[3], corner_y[3], corner_z[3];                                              \
    double pixel_grad_x[3], pixel_grad_y[3], gz[3];                                            \
    double e[3], w[3], area2;                                                                  \
    double de_dx[3][3], de_dy[3][3], darea_dx[3], darea_dy[3];                                 \
    int corner, component;                                                                     \
    ids[0] = (long long)faces[face * 3 + 0];                                                   \
    ids[1] = (long long)faces[face * 3 + 1];                                                   \
    ids[2] = (long long)faces[face * 3 + 2];                                                   \
    for (corner = 0; corner < 3; ++corner) {                                                   \
        double nx = (double)ndc_vertices[ids[corner] * 3 + 0];                                 \
        double ny = (double)ndc_vertices[ids[corner] * 3 + 1];                                 \
        double nz = (double)ndc_vertices[ids[corner] * 3 + 2];                                 \
        corner_x[corner] = (nx + 1.0) * half_width;                                            \
        corner_y[corner] = (1.0 - ny) * half_height;                                           \
        corner_z[corner] = nz;                                                                 \
        pixel_grad_x[corner] = 0.0;                                                            \
        pixel_grad_y[corner] = 0.0;                                                            \
        gz[corner] = 0.0;                                                                      \
    }                                                                                          \
    e[0] = (corner_x[1] - px) * (corner_y[2] - py) - (corner_y[1] - py) * (corner_x[2] - px);  \
    e[1] = (corner_x[2] - px) * (corner_y[0] - py) - (corner_y[2] - py) * (corner_x[0] - px);  \
    e[2] = (corner_x[0] - px) * (corner_y[1] - py) - (corner_y[0] - py) * (corner_x[1] - px);  \
    area2 = e[0] + e[1] + e[2];                                                                \
    if (area2 == 0.0) return 0;                                                                \
    w[0] = e[0] / area2; w[1] = e[1] / area2; w[2] = e[2] / area2;                             \
    de_dx[0][0] = 0.0;                  de_dy[0][0] = 0.0;                                     \
    de_dx[0][1] = corner_y[2] - py;     de_dy[0][1] = -(corner_x[2] - px);                     \
    de_dx[0][2] = -(corner_y[1] - py);  de_dy[0][2] = corner_x[1] - px;                        \
    de_dx[1][0] = -(corner_y[2] - py);  de_dy[1][0] = corner_x[2] - px;                        \
    de_dx[1][1] = 0.0;                  de_dy[1][1] = 0.0;                                     \
    de_dx[1][2] = corner_y[0] - py;     de_dy[1][2] = -(corner_x[0] - px);                     \
    de_dx[2][0] = corner_y[1] - py;     de_dy[2][0] = -(corner_x[1] - px);                     \
    de_dx[2][1] = -(corner_y[0] - py);  de_dy[2][1] = corner_x[0] - px;                        \
    de_dx[2][2] = 0.0;                  de_dy[2][2] = 0.0;                                     \
    for (corner = 0; corner < 3; ++corner) {                                                   \
        darea_dx[corner] = de_dx[0][corner] + de_dx[1][corner] + de_dx[2][corner];             \
        darea_dy[corner] = de_dy[0][corner] + de_dy[1][corner] + de_dy[2][corner];             \
    }                                                                                          \
    for (component = 0; component < 3; ++component) {                                          \
        double weight_cotangent = 0.0;                                                         \
        if (grad_barycentric != 0)                                                             \
            weight_cotangent += (double)grad_barycentric[entry * 3 + component];               \
        if (grad_depth != 0)                                                                   \
            weight_cotangent += (double)grad_depth[entry] * corner_z[component];               \
        if (weight_cotangent == 0.0) continue;                                                 \
        for (corner = 0; corner < 3; ++corner) {                                               \
            double dw_dx = (de_dx[component][corner] -                                         \
                            w[component] * darea_dx[corner]) / area2;                          \
            double dw_dy = (de_dy[component][corner] -                                         \
                            w[component] * darea_dy[corner]) / area2;                          \
            pixel_grad_x[corner] += weight_cotangent * dw_dx;                                  \
            pixel_grad_y[corner] += weight_cotangent * dw_dy;                                  \
        }                                                                                      \
    }                                                                                          \
    if (grad_depth != 0) {                                                                     \
        double depth_cotangent = (double)grad_depth[entry];                                    \
        for (corner = 0; corner < 3; ++corner) gz[corner] += depth_cotangent * w[corner];      \
    }                                                                                          \
    if (grad_signed_distance != 0) {                                                           \
        double distance_cotangent = (double)grad_signed_distance[entry];                       \
        if (distance_cotangent != 0.0) {                                                       \
            int edge, first, second;                                                           \
            double edge_t, ux, uy, vx, vy, ccx, ccy, residual_x, residual_y;                   \
            int inside = (w[0] >= 0.0 && w[1] >= 0.0 && w[2] >= 0.0);                          \
            double sign = inside ? -1.0 : 1.0;                                                 \
            gffx_cuda_boundary_edge(px, py, corner_x[0], corner_y[0], corner_x[1], corner_y[1], \
                                    corner_x[2], corner_y[2], &edge, &edge_t);                 \
            first = edge;                                                                      \
            second = (edge + 1) % 3;                                                           \
            ux = corner_x[first]; uy = corner_y[first];                                        \
            vx = corner_x[second]; vy = corner_y[second];                                      \
            ccx = ux + edge_t * (vx - ux);                                                     \
            ccy = uy + edge_t * (vy - uy);                                                     \
            residual_x = -2.0 * (px - ccx) * sign * distance_cotangent;                        \
            residual_y = -2.0 * (py - ccy) * sign * distance_cotangent;                        \
            pixel_grad_x[first] += residual_x * (1.0 - edge_t);                                \
            pixel_grad_y[first] += residual_y * (1.0 - edge_t);                                \
            pixel_grad_x[second] += residual_x * edge_t;                                       \
            pixel_grad_y[second] += residual_y * edge_t;                                       \
        }                                                                                      \
    }                                                                                          \
    for (corner = 0; corner < 3; ++corner) {                                                   \
        out_gx[corner] = pixel_grad_x[corner] * half_width;                                    \
        out_gy[corner] = pixel_grad_y[corner] * (-half_height);                                \
        out_gz[corner] = gz[corner];                                                           \
    }                                                                                          \
    return 1;                                                                                  \
}

#define GFFX_CUDA_RASTERIZE_BACKWARD(SUFFIX, SCALAR)                                           \
extern "C" __global__ void gffx_cuda_rasterize_backward_ordered_##SUFFIX(                      \
    const SCALAR *__restrict__ ndc_vertices, const int *__restrict__ faces,                    \
    const int *__restrict__ face_index, long long image_height, long long image_width,         \
    long long faces_per_pixel, long long batch_count, long long vertex_count,                  \
    const SCALAR *__restrict__ grad_barycentric, const SCALAR *__restrict__ grad_depth,        \
    const SCALAR *__restrict__ grad_signed_distance, SCALAR *__restrict__ grad_ndc             \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR tx = (SCALAR)0, ty = (SCALAR)0, tz = (SCALAR)0;                                 \
        long long fragments = batch_count * image_height * image_width * faces_per_pixel;      \
        for (long long entry = 0; entry < fragments; ++entry) {                                \
            long long face = (long long)face_index[entry];                                     \
            long long slot_span = image_width * faces_per_pixel;                               \
            long long row = (entry / slot_span) % image_height;                                \
            long long column = (entry / faces_per_pixel) % image_width;                        \
            long long ids[3];                                                                  \
            double gx[3], gy[3], gz[3];                                                        \
            int corner;                                                                        \
            if (face < 0) continue;                                                            \
            if (!gffx_cuda_rasterize_backward_term_##SUFFIX(                                   \
                    ndc_vertices, faces, face, entry, image_height, image_width, row, column,  \
                    grad_barycentric, grad_depth, grad_signed_distance, ids, gx, gy, gz))      \
                continue;                                                                      \
            /* Corner order within a fragment, as the host writes it, because a degenerate      \
             * face can name this vertex more than once. */                                     \
            for (corner = 0; corner < 3; ++corner) {                                           \
                if (ids[corner] != vertex) continue;                                           \
                tx += (SCALAR)gx[corner];                                                      \
                ty += (SCALAR)gy[corner];                                                      \
                tz += (SCALAR)gz[corner];                                                      \
            }                                                                                  \
        }                                                                                      \
        grad_ndc[vertex * 3 + 0] = tx;                                                         \
        grad_ndc[vertex * 3 + 1] = ty;                                                         \
        grad_ndc[vertex * 3 + 2] = tz;                                                         \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_rasterize_backward_atomic_##SUFFIX(                       \
    const SCALAR *__restrict__ ndc_vertices, const int *__restrict__ faces,                    \
    const int *__restrict__ face_index, long long image_height, long long image_width,         \
    long long faces_per_pixel, long long fragment_count, long long vertex_count,               \
    const SCALAR *__restrict__ grad_barycentric, const SCALAR *__restrict__ grad_depth,        \
    const SCALAR *__restrict__ grad_signed_distance, SCALAR *__restrict__ grad_ndc             \
) {                                                                                            \
    long long entry = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (entry >= fragment_count) return;                                                       \
    {                                                                                          \
        long long face = (long long)face_index[entry];                                         \
        long long slot_span = image_width * faces_per_pixel;                                   \
        long long row = (entry / slot_span) % image_height;                                    \
        long long column = (entry / faces_per_pixel) % image_width;                            \
        long long ids[3];                                                                      \
        double gx[3], gy[3], gz[3];                                                            \
        int corner;                                                                            \
        if (face < 0) return;                                                                  \
        if (!gffx_cuda_rasterize_backward_term_##SUFFIX(                                       \
                ndc_vertices, faces, face, entry, image_height, image_width, row, column,      \
                grad_barycentric, grad_depth, grad_signed_distance, ids, gx, gy, gz))          \
            return;                                                                            \
        for (corner = 0; corner < 3; ++corner) {                                               \
            if (ids[corner] < 0 || ids[corner] >= vertex_count) continue;                      \
            atomicAdd(&grad_ndc[ids[corner] * 3 + 0], (SCALAR)gx[corner]);                     \
            atomicAdd(&grad_ndc[ids[corner] * 3 + 1], (SCALAR)gy[corner]);                     \
            atomicAdd(&grad_ndc[ids[corner] * 3 + 2], (SCALAR)gz[corner]);                     \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_RASTERIZE_BACKWARD_TERM(f32, float)
GFFX_CUDA_RASTERIZE_BACKWARD_TERM(f64, double)
GFFX_CUDA_RASTERIZE_BACKWARD(f32, float)
GFFX_CUDA_RASTERIZE_BACKWARD(f64, double)

/* --------------------------------------------------------------------- mesh.vertex_normals
 *
 * The last operation on the render critical path with no CUDA forward, which meant a device render
 * had to round-trip through the host for it.
 *
 * The host accumulates a per-face weight onto each of the face's three vertices in ascending face
 * order, then normalises each vertex sum. Both steps fit one thread per vertex: the thread scans
 * every face in that same order and takes the ones naming it, then normalises its own sum. That
 * needs no atomics and no second launch, and it reproduces the host's addition order exactly.
 *
 * The cost is O(V*F) rather than O(F), the same trade the other gathers make. All three corners of
 * a face receive the identical weight, so a degenerate face naming a vertex twice contributes
 * twice - which the corner loop below preserves rather than deduplicating.
 */
#define GFFX_CUDA_VERTEX_NORMALS(SUFFIX, SCALAR, SQRT_FN)                                      \
extern "C" __global__ void gffx_cuda_vertex_normals_##SUFFIX(                                  \
    const SCALAR *__restrict__ vertices, const int *__restrict__ faces,                        \
    long long face_count, long long vertex_count, double eps, unsigned int weighting,          \
    SCALAR *__restrict__ unit_normals                                                          \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR sx = (SCALAR)0, sy = (SCALAR)0, sz = (SCALAR)0;                                 \
        SCALAR magnitude;                                                                      \
        for (long long face = 0; face < face_count; ++face) {                                  \
            long long ids[3];                                                                  \
            const SCALAR *a; const SCALAR *b; const SCALAR *c;                                 \
            SCALAR e1x, e1y, e1z, e2x, e2y, e2z, cx, cy, cz, doubled, wx, wy, wz;              \
            int corner;                                                                        \
            ids[0] = (long long)faces[face * 3 + 0];                                           \
            ids[1] = (long long)faces[face * 3 + 1];                                           \
            ids[2] = (long long)faces[face * 3 + 2];                                           \
            if (ids[0] != vertex && ids[1] != vertex && ids[2] != vertex) continue;            \
            a = vertices + ids[0] * 3;                                                         \
            b = vertices + ids[1] * 3;                                                         \
            c = vertices + ids[2] * 3;                                                         \
            e1x = b[0] - a[0]; e1y = b[1] - a[1]; e1z = b[2] - a[2];                           \
            e2x = c[0] - a[0]; e2y = c[1] - a[1]; e2z = c[2] - a[2];                           \
            cx = e1y * e2z - e1z * e2y;                                                        \
            cy = e1z * e2x - e1x * e2z;                                                        \
            cz = e1x * e2y - e1y * e2x;                                                        \
            doubled = SQRT_FN(cx * cx + cy * cy + cz * cz);                                    \
            if (!((double)doubled > eps)) continue;                                            \
            if (weighting == 1u) { /* area */                                                  \
                wx = cx * (SCALAR)0.5; wy = cy * (SCALAR)0.5; wz = cz * (SCALAR)0.5;           \
            } else {                                                                           \
                wx = cx / doubled; wy = cy / doubled; wz = cz / doubled;                       \
            }                                                                                  \
            for (corner = 0; corner < 3; ++corner) {                                           \
                if (ids[corner] != vertex) continue;                                           \
                sx += wx; sy += wy; sz += wz;                                                  \
            }                                                                                  \
        }                                                                                      \
        magnitude = SQRT_FN(sx * sx + sy * sy + sz * sz);                                      \
        if ((double)magnitude > eps) {                                                         \
            unit_normals[vertex * 3 + 0] = sx / magnitude;                                     \
            unit_normals[vertex * 3 + 1] = sy / magnitude;                                     \
            unit_normals[vertex * 3 + 2] = sz / magnitude;                                     \
        } else {                                                                               \
            unit_normals[vertex * 3 + 0] = (SCALAR)0;                                          \
            unit_normals[vertex * 3 + 1] = (SCALAR)0;                                          \
            unit_normals[vertex * 3 + 2] = (SCALAR)0;                                          \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_VERTEX_NORMALS(f32, float, sqrtf)
GFFX_CUDA_VERTEX_NORMALS(f64, double, sqrt)

/* ------------------------------------------------------ mesh.vertex_normals backward
 *
 * Three launches, because the host algorithm has three phases and the middle one depends on the
 * first being complete for every vertex.
 *
 *   sums   - recompute the raw per-vertex accumulation, the forward's first phase without the
 *            normalisation. One thread per vertex, ordered gather over faces.
 *   q      - convert those sums in place to dL/ds, the gradient through the normalisation. Purely
 *            per vertex.
 *   scatter- for each face, read q from its three corners, form the face-weight gradient, and
 *            distribute it. This is the phase that scatters, and it takes the same ordered-gather
 *            treatment as mesh.face_geometry: corner 1, then corner 2, then corner 0 subtracting
 *            both, in ascending face order.
 *
 * The intermediate sums live in the caller's workspace, which is why this operation reports a
 * nonzero workspace requirement where most report none. The host does the same and for the same
 * reason.
 */
#define GFFX_CUDA_VERTEX_NORMALS_BACKWARD(SUFFIX, SCALAR, SQRT_FN)                             \
extern "C" __global__ void gffx_cuda_vertex_normals_sums_##SUFFIX(                             \
    const SCALAR *__restrict__ vertices, const int *__restrict__ faces,                        \
    long long face_count, long long vertex_count, double eps, unsigned int weighting,          \
    SCALAR *__restrict__ sums                                                                  \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR sx = (SCALAR)0, sy = (SCALAR)0, sz = (SCALAR)0;                                 \
        for (long long face = 0; face < face_count; ++face) {                                  \
            long long ids[3];                                                                  \
            const SCALAR *a; const SCALAR *b; const SCALAR *c;                                 \
            SCALAR e1x, e1y, e1z, e2x, e2y, e2z, cx, cy, cz, doubled, wx, wy, wz;              \
            int corner;                                                                        \
            ids[0] = (long long)faces[face * 3 + 0];                                           \
            ids[1] = (long long)faces[face * 3 + 1];                                           \
            ids[2] = (long long)faces[face * 3 + 2];                                           \
            if (ids[0] != vertex && ids[1] != vertex && ids[2] != vertex) continue;            \
            a = vertices + ids[0] * 3; b = vertices + ids[1] * 3; c = vertices + ids[2] * 3;   \
            e1x = b[0] - a[0]; e1y = b[1] - a[1]; e1z = b[2] - a[2];                           \
            e2x = c[0] - a[0]; e2y = c[1] - a[1]; e2z = c[2] - a[2];                           \
            cx = e1y * e2z - e1z * e2y;                                                        \
            cy = e1z * e2x - e1x * e2z;                                                        \
            cz = e1x * e2y - e1y * e2x;                                                        \
            doubled = SQRT_FN(cx * cx + cy * cy + cz * cz);                                    \
            if (!((double)doubled > eps)) continue;                                            \
            if (weighting == 1u) {                                                             \
                wx = cx * (SCALAR)0.5; wy = cy * (SCALAR)0.5; wz = cz * (SCALAR)0.5;           \
            } else {                                                                           \
                wx = cx / doubled; wy = cy / doubled; wz = cz / doubled;                       \
            }                                                                                  \
            for (corner = 0; corner < 3; ++corner) {                                           \
                if (ids[corner] != vertex) continue;                                           \
                sx += wx; sy += wy; sz += wz;                                                  \
            }                                                                                  \
        }                                                                                      \
        sums[vertex * 3 + 0] = sx;                                                             \
        sums[vertex * 3 + 1] = sy;                                                             \
        sums[vertex * 3 + 2] = sz;                                                             \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_vertex_normals_q_##SUFFIX(                                \
    const SCALAR *__restrict__ cotangent, long long vertex_count, double eps,                  \
    SCALAR *__restrict__ sums                                                                  \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR sx = sums[vertex * 3 + 0];                                                      \
        SCALAR sy = sums[vertex * 3 + 1];                                                      \
        SCALAR sz = sums[vertex * 3 + 2];                                                      \
        SCALAR magnitude = SQRT_FN(sx * sx + sy * sy + sz * sz);                               \
        if ((double)magnitude > eps) {                                                         \
            SCALAR nx = sx / magnitude;                                                        \
            SCALAR ny = sy / magnitude;                                                        \
            SCALAR nz = sz / magnitude;                                                        \
            SCALAR gx = cotangent[vertex * 3 + 0];                                             \
            SCALAR gy = cotangent[vertex * 3 + 1];                                             \
            SCALAR gz = cotangent[vertex * 3 + 2];                                             \
            SCALAR dot = gx * nx + gy * ny + gz * nz;                                          \
            sums[vertex * 3 + 0] = (gx - dot * nx) / magnitude;                                \
            sums[vertex * 3 + 1] = (gy - dot * ny) / magnitude;                                \
            sums[vertex * 3 + 2] = (gz - dot * nz) / magnitude;                                \
        } else {                                                                               \
            sums[vertex * 3 + 0] = (SCALAR)0;                                                  \
            sums[vertex * 3 + 1] = (SCALAR)0;                                                  \
            sums[vertex * 3 + 2] = (SCALAR)0;                                                  \
        }                                                                                      \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_vertex_normals_backward_##SUFFIX(                         \
    const SCALAR *__restrict__ vertices, const int *__restrict__ faces,                        \
    long long face_count, long long vertex_count, double eps, unsigned int weighting,          \
    const SCALAR *__restrict__ sums, SCALAR *__restrict__ grad_vertices                        \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR tx = (SCALAR)0, ty = (SCALAR)0, tz = (SCALAR)0;                                 \
        for (long long face = 0; face < face_count; ++face) {                                  \
            long long i0 = (long long)faces[face * 3 + 0];                                     \
            long long i1 = (long long)faces[face * 3 + 1];                                     \
            long long i2 = (long long)faces[face * 3 + 2];                                     \
            const SCALAR *a; const SCALAR *b; const SCALAR *c;                                 \
            SCALAR e1x, e1y, e1z, e2x, e2y, e2z, cx, cy, cz, doubled;                          \
            SCALAR qx, qy, qz, gcx, gcy, gcz, g1x, g1y, g1z, g2x, g2y, g2z;                    \
            if (i0 != vertex && i1 != vertex && i2 != vertex) continue;                        \
            a = vertices + i0 * 3; b = vertices + i1 * 3; c = vertices + i2 * 3;              \
            e1x = b[0] - a[0]; e1y = b[1] - a[1]; e1z = b[2] - a[2];                           \
            e2x = c[0] - a[0]; e2y = c[1] - a[1]; e2z = c[2] - a[2];                           \
            cx = e1y * e2z - e1z * e2y;                                                        \
            cy = e1z * e2x - e1x * e2z;                                                        \
            cz = e1x * e2y - e1y * e2x;                                                        \
            doubled = SQRT_FN(cx * cx + cy * cy + cz * cz);                                    \
            if (!((double)doubled > eps)) continue;                                            \
            qx = sums[i0 * 3 + 0] + sums[i1 * 3 + 0] + sums[i2 * 3 + 0];                       \
            qy = sums[i0 * 3 + 1] + sums[i1 * 3 + 1] + sums[i2 * 3 + 1];                       \
            qz = sums[i0 * 3 + 2] + sums[i1 * 3 + 2] + sums[i2 * 3 + 2];                       \
            if (weighting == 1u) {                                                             \
                gcx = (SCALAR)0.5 * qx; gcy = (SCALAR)0.5 * qy; gcz = (SCALAR)0.5 * qz;        \
            } else {                                                                           \
                SCALAR nx = cx / doubled;                                                      \
                SCALAR ny = cy / doubled;                                                      \
                SCALAR nz = cz / doubled;                                                      \
                SCALAR dot = qx * nx + qy * ny + qz * nz;                                      \
                gcx = (qx - dot * nx) / doubled;                                               \
                gcy = (qy - dot * ny) / doubled;                                               \
                gcz = (qz - dot * nz) / doubled;                                               \
            }                                                                                  \
            g1x = e2y * gcz - e2z * gcy;                                                       \
            g1y = e2z * gcx - e2x * gcz;                                                       \
            g1z = e2x * gcy - e2y * gcx;                                                       \
            g2x = gcy * e1z - gcz * e1y;                                                       \
            g2y = gcz * e1x - gcx * e1z;                                                       \
            g2z = gcx * e1y - gcy * e1x;                                                       \
            /* Corner 1, then corner 2, then corner 0 subtracting both, as the host writes it. */\
            if (i1 == vertex) { tx += g1x; ty += g1y; tz += g1z; }                             \
            if (i2 == vertex) { tx += g2x; ty += g2y; tz += g2z; }                             \
            if (i0 == vertex) {                                                                \
                tx -= g1x + g2x; ty -= g1y + g2y; tz -= g1z + g2z;                             \
            }                                                                                  \
        }                                                                                      \
        grad_vertices[vertex * 3 + 0] = tx;                                                    \
        grad_vertices[vertex * 3 + 1] = ty;                                                    \
        grad_vertices[vertex * 3 + 2] = tz;                                                    \
    }                                                                                          \
}

GFFX_CUDA_VERTEX_NORMALS_BACKWARD(f32, float, sqrtf)
GFFX_CUDA_VERTEX_NORMALS_BACKWARD(f64, double, sqrt)

/* -------------------------------------------------------------------- mesh.sample_surface
 *
 * The last operation without a CUDA forward, and the one whose determinism story is already told
 * by its contract rather than by the backend: the Philox counter carries the batch and the sample
 * index, so every sample is independent and reproducible without any sequential generator state.
 * That is what makes it parallelisable at all, and it means the device path does not have to
 * invent an ordering argument the way the scattering backwards did.
 *
 * Two launches. The cumulative area table is a prefix sum, which is sequential by nature, so one
 * thread per batch element walks its own faces in ascending order exactly as the host does - and
 * the table stays in double for both dtypes, because the contract requires the running sum to be
 * monotone so the binary search cannot land on a collapsed interval. Then one thread per sample
 * draws its own words and writes its own outputs.
 *
 * The batch total is not stored separately: it is the last entry of that element's table, since
 * the running sum is written after every face including ineligible ones.
 */

#define GFFX_CUDA_PHILOX_M0 0xD2511F53u
#define GFFX_CUDA_PHILOX_M1 0xCD9E8D57u
#define GFFX_CUDA_PHILOX_W0 0x9E3779B9u
#define GFFX_CUDA_PHILOX_W1 0xBB67AE85u

__device__ __forceinline__ void gffx_cuda_philox4x32_10(
    const unsigned int counter[4], const unsigned int key[2], unsigned int output[4]
) {
    unsigned int state[4];
    unsigned int local_key[2];
    int round;
    state[0] = counter[0]; state[1] = counter[1];
    state[2] = counter[2]; state[3] = counter[3];
    local_key[0] = key[0]; local_key[1] = key[1];
    for (round = 0; round < 10; ++round) {
        unsigned long long product0, product1;
        unsigned int high0, low0, high1, low1;
        if (round > 0) {
            local_key[0] += GFFX_CUDA_PHILOX_W0;
            local_key[1] += GFFX_CUDA_PHILOX_W1;
        }
        product0 = (unsigned long long)GFFX_CUDA_PHILOX_M0 * (unsigned long long)state[0];
        product1 = (unsigned long long)GFFX_CUDA_PHILOX_M1 * (unsigned long long)state[2];
        high0 = (unsigned int)(product0 >> 32);
        low0 = (unsigned int)product0;
        high1 = (unsigned int)(product1 >> 32);
        low1 = (unsigned int)product1;
        {
            unsigned int next0 = high1 ^ state[1] ^ local_key[0];
            unsigned int next1 = low1;
            unsigned int next2 = high0 ^ state[3] ^ local_key[1];
            unsigned int next3 = low0;
            state[0] = next0; state[1] = next1; state[2] = next2; state[3] = next3;
        }
    }
    output[0] = state[0]; output[1] = state[1];
    output[2] = state[2]; output[3] = state[3];
}

/* 2^-32 exactly; the result lies in [0, 1), matching the host constant bit for bit. */
__device__ __forceinline__ double gffx_cuda_uniform_from_word(unsigned int word) {
    return (double)word * 2.3283064365386963e-10;
}

#define GFFX_CUDA_SAMPLE_SURFACE(SUFFIX, SCALAR)                                               \
extern "C" __global__ void gffx_cuda_sample_cumulative_##SUFFIX(                               \
    const SCALAR *__restrict__ vertices, const int *__restrict__ faces,                        \
    const int *__restrict__ face_offsets, long long batch_count, double eps,                   \
    double *__restrict__ cumulative, unsigned int *__restrict__ degenerate                     \
) {                                                                                            \
    long long batch = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (batch >= batch_count) return;                                                          \
    {                                                                                          \
        long long first_face = (long long)face_offsets[batch];                                 \
        long long last_face = (long long)face_offsets[batch + 1];                              \
        double running = 0.0;                                                                  \
        long long eligible = 0;                                                                \
        for (long long face = first_face; face < last_face; ++face) {                          \
            long long i0 = (long long)faces[face * 3 + 0];                                     \
            long long i1 = (long long)faces[face * 3 + 1];                                     \
            long long i2 = (long long)faces[face * 3 + 2];                                     \
            double ax = (double)vertices[i0 * 3 + 0];                                          \
            double ay = (double)vertices[i0 * 3 + 1];                                          \
            double az = (double)vertices[i0 * 3 + 2];                                          \
            double bx = (double)vertices[i1 * 3 + 0];                                          \
            double by = (double)vertices[i1 * 3 + 1];                                          \
            double bz = (double)vertices[i1 * 3 + 2];                                          \
            double cx = (double)vertices[i2 * 3 + 0];                                          \
            double cy = (double)vertices[i2 * 3 + 1];                                          \
            double cz = (double)vertices[i2 * 3 + 2];                                          \
            double e1x = bx - ax, e1y = by - ay, e1z = bz - az;                                \
            double e2x = cx - ax, e2y = cy - ay, e2z = cz - az;                                \
            double nx = e1y * e2z - e1z * e2y;                                                 \
            double ny = e1z * e2x - e1x * e2z;                                                 \
            double nz = e1x * e2y - e1y * e2x;                                                 \
            double doubled = sqrt(nx * nx + ny * ny + nz * nz);                                \
            if (doubled > eps) { running += doubled * 0.5; ++eligible; }                        \
            cumulative[face] = running;                                                        \
        }                                                                                      \
        /* The host refuses a batch element with no positive-area face. A kernel cannot return   \
         * a status, so it raises a flag the dispatch reads back and turns into that error. */   \
        if (eligible == 0) *degenerate = 1u;                                                   \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_sample_surface_##SUFFIX(                                  \
    const SCALAR *__restrict__ vertices, const int *__restrict__ faces,                        \
    const int *__restrict__ face_offsets, long long batch_count, long long sample_count,       \
    const unsigned int *__restrict__ rng_key, const unsigned int *__restrict__ rng_counter,    \
    const double *__restrict__ cumulative,                                                     \
    SCALAR *__restrict__ points, int *__restrict__ face_index, SCALAR *__restrict__ barycentric \
) {                                                                                            \
    long long slot = (long long)blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (slot >= batch_count * sample_count) return;                                            \
    {                                                                                          \
        long long batch = slot / sample_count;                                                 \
        long long sample = slot - batch * sample_count;                                        \
        long long first_face = (long long)face_offsets[batch];                                 \
        long long last_face = (long long)face_offsets[batch + 1];                              \
        double running = cumulative[last_face - 1];                                            \
        unsigned int counter_words[4];                                                         \
        unsigned int key_words[2];                                                             \
        unsigned int words[4];                                                                 \
        double target, b0, b1, b2, su;                                                         \
        long long low = first_face;                                                            \
        long long high = last_face - 1;                                                        \
        long long chosen, i0, i1, i2;                                                          \
        long long out_base = (batch * sample_count + sample) * 3;                              \
        int axis;                                                                              \
        counter_words[0] = rng_counter[0];                                                     \
        counter_words[1] = rng_counter[1];                                                     \
        counter_words[2] = (unsigned int)batch;                                                \
        counter_words[3] = (unsigned int)sample;                                               \
        key_words[0] = rng_key[0];                                                             \
        key_words[1] = rng_key[1];                                                             \
        gffx_cuda_philox4x32_10(counter_words, key_words, words);                              \
        target = gffx_cuda_uniform_from_word(words[0]) * running;                              \
        while (low < high) {                                                                   \
            long long middle = low + (high - low) / 2;                                         \
            if (cumulative[middle] > target) high = middle; else low = middle + 1;             \
        }                                                                                      \
        chosen = low;                                                                          \
        while (chosen > first_face && cumulative[chosen - 1] >= cumulative[chosen]) --chosen;   \
        if (chosen < first_face) chosen = first_face;                                          \
        su = sqrt(gffx_cuda_uniform_from_word(words[1]));                                      \
        b1 = su * (1.0 - gffx_cuda_uniform_from_word(words[2]));                               \
        b2 = su * gffx_cuda_uniform_from_word(words[2]);                                       \
        b0 = 1.0 - su;                                                                         \
        i0 = (long long)faces[chosen * 3 + 0];                                                 \
        i1 = (long long)faces[chosen * 3 + 1];                                                 \
        i2 = (long long)faces[chosen * 3 + 2];                                                 \
        face_index[batch * sample_count + sample] = (int)chosen;                               \
        barycentric[out_base + 0] = (SCALAR)b0;                                                \
        barycentric[out_base + 1] = (SCALAR)b1;                                                \
        barycentric[out_base + 2] = (SCALAR)b2;                                                \
        /* The host narrows the weights to the operand dtype before combining for float32 and    \
         * keeps them in double for float64; mirrored here by casting the weight, not the sum. */\
        for (axis = 0; axis < 3; ++axis) {                                                     \
            points[out_base + axis] = (SCALAR)b0 * vertices[i0 * 3 + axis] +                   \
                                      (SCALAR)b1 * vertices[i1 * 3 + axis] +                   \
                                      (SCALAR)b2 * vertices[i2 * 3 + axis];                   \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_SAMPLE_SURFACE(f32, float)
GFFX_CUDA_SAMPLE_SURFACE(f64, double)

/* ------------------------------------------------------- mesh.sample_surface backward
 *
 * The last of the twelve. Structurally the simplest: a sampled point is a barycentric combination
 * of its face's three vertices, so the gradient is that combination transposed - each corner
 * receives its weight times the point's cotangent. No geometry is recomputed and nothing is
 * differentiated through the sampling itself, because the face choice and the barycentric weights
 * are draws from the counter rather than functions of the vertices.
 *
 * Both paths, as the policy requires: the ordered gather walks samples then corners in the host's
 * order, and the atomic scatter is one thread per sample behind the flag.
 */
#define GFFX_CUDA_SAMPLE_SURFACE_BACKWARD(SUFFIX, SCALAR)                                      \
extern "C" __global__ void gffx_cuda_sample_backward_ordered_##SUFFIX(                         \
    const int *__restrict__ faces, const int *__restrict__ face_index,                        \
    const SCALAR *__restrict__ barycentric, const SCALAR *__restrict__ grad_points,           \
    long long entry_count, long long vertex_count, SCALAR *__restrict__ grad_vertices          \
) {                                                                                            \
    long long vertex = (long long)blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (vertex >= vertex_count) return;                                                        \
    {                                                                                          \
        SCALAR t0 = (SCALAR)0, t1 = (SCALAR)0, t2 = (SCALAR)0;                                 \
        for (long long entry = 0; entry < entry_count; ++entry) {                              \
            long long face = (long long)face_index[entry];                                     \
            long long base = entry * 3;                                                        \
            int corner;                                                                        \
            if (face < 0) continue;                                                            \
            /* Corner order within a sample, as the host writes it, because a degenerate face   \
             * can name this vertex more than once. */                                          \
            for (corner = 0; corner < 3; ++corner) {                                           \
                if ((long long)faces[face * 3 + corner] != vertex) continue;                   \
                t0 += barycentric[base + corner] * grad_points[base + 0];                      \
                t1 += barycentric[base + corner] * grad_points[base + 1];                      \
                t2 += barycentric[base + corner] * grad_points[base + 2];                      \
            }                                                                                  \
        }                                                                                      \
        grad_vertices[vertex * 3 + 0] = t0;                                                    \
        grad_vertices[vertex * 3 + 1] = t1;                                                    \
        grad_vertices[vertex * 3 + 2] = t2;                                                    \
    }                                                                                          \
}                                                                                              \
                                                                                               \
extern "C" __global__ void gffx_cuda_sample_backward_atomic_##SUFFIX(                          \
    const int *__restrict__ faces, const int *__restrict__ face_index,                        \
    const SCALAR *__restrict__ barycentric, const SCALAR *__restrict__ grad_points,           \
    long long entry_count, long long vertex_count, SCALAR *__restrict__ grad_vertices          \
) {                                                                                            \
    long long entry = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (entry >= entry_count) return;                                                          \
    {                                                                                          \
        long long face = (long long)face_index[entry];                                         \
        long long base = entry * 3;                                                            \
        int corner;                                                                            \
        if (face < 0) return;                                                                  \
        for (corner = 0; corner < 3; ++corner) {                                               \
            long long vertex = (long long)faces[face * 3 + corner];                            \
            int axis;                                                                          \
            if (vertex < 0 || vertex >= vertex_count) continue;                                \
            for (axis = 0; axis < 3; ++axis) {                                                 \
                atomicAdd(&grad_vertices[vertex * 3 + axis],                                   \
                          barycentric[base + corner] * grad_points[base + axis]);              \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
}

GFFX_CUDA_SAMPLE_SURFACE_BACKWARD(f32, float)
GFFX_CUDA_SAMPLE_SURFACE_BACKWARD(f64, double)
