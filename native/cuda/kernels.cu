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
    long long face_count,
    int vertex_count,
    int *__restrict__ status
) {
    long long entry = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= face_count * 3) return;
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
