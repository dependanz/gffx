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
