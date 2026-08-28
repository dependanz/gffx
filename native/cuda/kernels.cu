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
