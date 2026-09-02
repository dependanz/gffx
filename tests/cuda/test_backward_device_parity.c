/*
 * CPU/CUDA parity for the implemented backward kernels, run against a real device through the
 * driver API.
 *
 * A backward is where a device implementation most easily stops being deterministic: many outputs
 * feed one input, and the obvious device answer is an atomic add whose completion order the
 * hardware chooses. Floating-point addition is not associative, so that ordering changes the last
 * bits from run to run.
 *
 * Two kinds of claim are checked here, and they are deliberately not checked the same way.
 * A default path must be bit-identical to the host, so it is compared with memcmp; a tolerance
 * there would admit exactly the divergence the conformance contract exists to catch. A path
 * reached only through GFFX_EXECUTION_ALLOW_NONDETERMINISTIC must be close but is explicitly not
 * required to be identical, so it is compared against a tolerance; asserting equality there would
 * either pass by luck or fail for the intended reason, and would be a meaningless test either way.
 *
 * The cases are sized so the reductions are long. A sparse case passes whether or not the ordering
 * is right, and proves nothing.
 */

#include <gffx/execution.h>
#include <gffx/mesh.h>
#include <gffx/render.h>
#include <gffx/status.h>
#include <gffx/tensor.h>
#include <gffx/transforms.h>

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
static char message[512];
static gffx_diagnostic_buffer diagnostic;
static gffx_execution_context host;
static gffx_execution_context device_context;
static gffx_execution_context device_relaxed;

static void setup_contexts(void) {
    memset(&diagnostic, 0, sizeof(diagnostic));
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    diagnostic.data = message;
    diagnostic.capacity_bytes = sizeof(message);
    message[0] = 0;

    memset(&host, 0, sizeof(host));
    host.struct_size = (uint32_t)sizeof(host);
    host.abi_version = GFFX_ABI_VERSION;
    host.device_type = GFFX_DEVICE_CPU;
    device_context = host;
    device_context.device_type = GFFX_DEVICE_CUDA;
    device_relaxed = device_context;
    device_relaxed.flags = GFFX_EXECUTION_ALLOW_NONDETERMINISTIC;
}

static gffx_tensor_view mk(
    void *data, gffx_dtype dtype, uint32_t rank, const int64_t *shape, const int64_t *strides,
    uint32_t flags, gffx_device_type device
) {
    gffx_tensor_view v;
    memset(&v, 0, sizeof(v));
    v.struct_size = (uint32_t)sizeof(v);
    v.abi_version = GFFX_ABI_VERSION;
    v.data = data;
    v.rank = rank;
    v.shape = shape;
    v.strides = strides;
    v.dtype = dtype;
    v.device_type = device;
    v.flags = flags;
    return v;
}

static void report_exact(const char *label, const void *a, const void *b, size_t bytes) {
    int same = memcmp(a, b, bytes) == 0;
    printf("  %-46s %s\n", label, same ? "bit-identical" : "DIFFERS");
    if (!same) ++failures;
}

static void report_close(const char *label, const double *reference, const double *actual,
                         size_t count, double tolerance) {
    double worst = 0.0;
    size_t i;
    for (i = 0; i < count; ++i) {
        double scale = fabs(reference[i]) > 1.0 ? fabs(reference[i]) : 1.0;
        double relative = fabs(actual[i] - reference[i]) / scale;
        if (relative > worst) worst = relative;
    }
    printf("  %-46s worst relative %.3g\n", label, worst);
    if (worst > tolerance) {
        printf("      exceeds the %.3g tolerance the relaxed path is held to\n", tolerance);
        ++failures;
    }
}

/* ------------------------------------------------------- transforms.perspective_divide */

#define DIVIDE_POINTS 257

static void run_perspective_divide(void) {
    CUdeviceptr d_hom, d_grad_ndc, d_out;
    double homogeneous[DIVIDE_POINTS * 4];
    double grad_ndc[DIVIDE_POINTS * 3];
    double cpu_out[DIVIDE_POINTS * 4];
    double gpu_out[DIVIDE_POINTS * 4];
    int64_t h_shape[2] = {DIVIDE_POINTS, 4};
    int64_t h_stride[2] = {4, 1};
    int64_t n_shape[2] = {DIVIDE_POINTS, 3};
    int64_t n_stride[2] = {3, 1};
    gffx_status st;
    int i;

    for (i = 0; i < DIVIDE_POINTS; ++i) {
        homogeneous[i * 4 + 0] = 0.7 + 0.013 * (double)i;
        homogeneous[i * 4 + 1] = -1.1 + 0.021 * (double)i;
        homogeneous[i * 4 + 2] = 0.4 - 0.007 * (double)i;
        /* Every 64th point sits exactly on w = 0 so the degenerate branch runs on both
         * backends rather than only on whichever one happens to be tested first. */
        homogeneous[i * 4 + 3] = (i % 64 == 0) ? 0.0 : (1.3 + 0.005 * (double)i);
        grad_ndc[i * 3 + 0] = 0.31 * (double)((i % 7) - 3);
        grad_ndc[i * 3 + 1] = 0.17 * (double)((i % 5) - 2);
        grad_ndc[i * 3 + 2] = 0.23 * (double)((i % 3) - 1);
    }
    cuMemAlloc(&d_hom, sizeof(homogeneous));
    cuMemAlloc(&d_grad_ndc, sizeof(grad_ndc));
    cuMemAlloc(&d_out, sizeof(cpu_out));
    cuMemcpyHtoD(d_hom, homogeneous, sizeof(homogeneous));
    cuMemcpyHtoD(d_grad_ndc, grad_ndc, sizeof(grad_ndc));

    {
        gffx_tensor_view hv = mk(homogeneous, GFFX_DTYPE_FLOAT64, 2u, h_shape, h_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gv = mk(grad_ndc, GFFX_DTYPE_FLOAT64, 2u, n_shape, n_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = mk(cpu_out, GFFX_DTYPE_FLOAT64, 2u, h_shape, h_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        st = gffx_transforms_perspective_divide_backward(&hv, 1e-8, &gv, &host, &ov, NULL,
                                                         &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view hv = mk((void *)(uintptr_t)d_hom, GFFX_DTYPE_FLOAT64, 2u, h_shape,
                                 h_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gv = mk((void *)(uintptr_t)d_grad_ndc, GFFX_DTYPE_FLOAT64, 2u, n_shape,
                                 n_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view ov = mk((void *)(uintptr_t)d_out, GFFX_DTYPE_FLOAT64, 2u, h_shape,
                                 h_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_transforms_perspective_divide_backward(&hv, 1e-8, &gv, &device_context, &ov,
                                                         NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  device failed: %s\n", message); ++failures; return; }
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_out, d_out, sizeof(gpu_out));
    report_exact("grad_homogeneous, degenerate branch included", cpu_out, gpu_out,
                 sizeof(cpu_out));
}

/* --------------------------------------------------------- transforms.transform_points */

#define TP_POINTS 257
#define TP_BATCHES 3

static void run_transform_points(void) {
    CUdeviceptr d_pts, d_mat, d_off, d_grad, d_gpts, d_gmat;
    double points[TP_POINTS * 3];
    double matrices[TP_BATCHES * 16];
    int32_t offsets[TP_BATCHES + 1];
    double grad_homogeneous[TP_POINTS * 4];
    double cpu_gpts[TP_POINTS * 3];
    double gpu_gpts[TP_POINTS * 3];
    double cpu_gmat[TP_BATCHES * 16];
    double gpu_gmat[TP_BATCHES * 16];
    int64_t p_shape[2] = {TP_POINTS, 3};
    int64_t p_stride[2] = {3, 1};
    int64_t m_shape[3] = {TP_BATCHES, 4, 4};
    int64_t m_stride[3] = {16, 4, 1};
    int64_t o_shape[1] = {TP_BATCHES + 1};
    int64_t o_stride[1] = {1};
    int64_t g_shape[2] = {TP_POINTS, 4};
    int64_t g_stride[2] = {4, 1};
    gffx_status st;
    int i;

    for (i = 0; i < TP_POINTS; ++i) {
        points[i * 3 + 0] = 0.9 - 0.011 * (double)i;
        points[i * 3 + 1] = 0.2 + 0.019 * (double)i;
        points[i * 3 + 2] = -0.5 + 0.003 * (double)i;
        grad_homogeneous[i * 4 + 0] = 0.13 * (double)((i % 11) - 5);
        grad_homogeneous[i * 4 + 1] = 0.29 * (double)((i % 9) - 4);
        grad_homogeneous[i * 4 + 2] = 0.07 * (double)((i % 13) - 6);
        grad_homogeneous[i * 4 + 3] = 0.19 * (double)((i % 6) - 3);
    }
    for (i = 0; i < TP_BATCHES * 16; ++i) matrices[i] = 0.5 + 0.037 * (double)i;
    /* Deliberately uneven: 100, 1 and 156 points. Equal batches would pass under a fixed
     * stride that is wrong, and the single-point batch catches an empty-or-one edge case. */
    offsets[0] = 0; offsets[1] = 100; offsets[2] = 101; offsets[3] = TP_POINTS;

    cuMemAlloc(&d_pts, sizeof(points));
    cuMemAlloc(&d_mat, sizeof(matrices));
    cuMemAlloc(&d_off, sizeof(offsets));
    cuMemAlloc(&d_grad, sizeof(grad_homogeneous));
    cuMemAlloc(&d_gpts, sizeof(cpu_gpts));
    cuMemAlloc(&d_gmat, sizeof(cpu_gmat));
    cuMemcpyHtoD(d_pts, points, sizeof(points));
    cuMemcpyHtoD(d_mat, matrices, sizeof(matrices));
    cuMemcpyHtoD(d_off, offsets, sizeof(offsets));
    cuMemcpyHtoD(d_grad, grad_homogeneous, sizeof(grad_homogeneous));

    {
        gffx_tensor_view pv = mk(points, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view mv = mk(matrices, GFFX_DTYPE_FLOAT64, 3u, m_shape, m_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = mk(offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gv = mk(grad_homogeneous, GFFX_DTYPE_FLOAT64, 2u, g_shape, g_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gp = mk(cpu_gpts, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view gm = mk(cpu_gmat, GFFX_DTYPE_FLOAT64, 3u, m_shape, m_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_transforms_transform_points_backward(&pv, &mv, &ov, &gv, &host, &gp, &gm, NULL,
                                                       &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view pv = mk((void *)(uintptr_t)d_pts, GFFX_DTYPE_FLOAT64, 2u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view mv = mk((void *)(uintptr_t)d_mat, GFFX_DTYPE_FLOAT64, 3u, m_shape,
                                 m_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view ov = mk((void *)(uintptr_t)d_off, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gv = mk((void *)(uintptr_t)d_grad, GFFX_DTYPE_FLOAT64, 2u, g_shape,
                                 g_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gp = mk((void *)(uintptr_t)d_gpts, GFFX_DTYPE_FLOAT64, 2u, p_shape,
                                 p_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view gm = mk((void *)(uintptr_t)d_gmat, GFFX_DTYPE_FLOAT64, 3u, m_shape,
                                 m_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_transforms_transform_points_backward(&pv, &mv, &ov, &gv, &device_context, &gp,
                                                       &gm, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  device failed: %s\n", message); ++failures; return; }
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_gpts, d_gpts, sizeof(gpu_gpts));
    cuMemcpyDtoH(gpu_gmat, d_gmat, sizeof(gpu_gmat));
    report_exact("grad_points, elementwise", cpu_gpts, gpu_gpts, sizeof(cpu_gpts));
    report_exact("grad_matrices, uneven 100/1/156 reduction", cpu_gmat, gpu_gmat,
                 sizeof(cpu_gmat));
}

/* --------------------------------------------------------------- mesh.gather_faces */

#define GF_VERTICES 64
#define GF_FACES 200

static void run_gather_faces(void) {
    CUdeviceptr d_vert, d_faces, d_cot, d_grad;
    double vertices[GF_VERTICES * 3];
    int32_t faces[GF_FACES * 3];
    double cotangent[GF_FACES * 9];
    double cpu_grad[GF_VERTICES * 3];
    double gpu_grad[GF_VERTICES * 3];
    int64_t v_shape[2] = {GF_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    int64_t f_shape[2] = {GF_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    int64_t c_shape[3] = {GF_FACES, 3, 3};
    int64_t c_stride[3] = {9, 3, 1};
    gffx_status st;
    int i;

    for (i = 0; i < GF_VERTICES * 3; ++i) vertices[i] = 0.1 * (double)i;
    for (i = 0; i < GF_FACES; ++i) {
        /* Heavy reuse: each vertex receives roughly nine contributions, so the accumulation
         * order actually matters. Distinct vertices per face would test nothing. */
        faces[i * 3 + 0] = (int32_t)(i % GF_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 7 + 3) % GF_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 13 + 11) % GF_VERTICES);
    }
    for (i = 0; i < GF_FACES * 9; ++i) cotangent[i] = 0.037 * (double)((i % 23) - 11);

    cuMemAlloc(&d_vert, sizeof(vertices));
    cuMemAlloc(&d_faces, sizeof(faces));
    cuMemAlloc(&d_cot, sizeof(cotangent));
    cuMemAlloc(&d_grad, sizeof(cpu_grad));
    cuMemcpyHtoD(d_vert, vertices, sizeof(vertices));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));
    cuMemcpyHtoD(d_cot, cotangent, sizeof(cotangent));

    {
        gffx_tensor_view vv = mk(vertices, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view cv = mk(cotangent, GFFX_DTYPE_FLOAT64, 3u, c_shape, c_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gv = mk(cpu_grad, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_mesh_gather_faces_backward(&vv, &fv, &cv, &host, &gv, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view vv = mk((void *)(uintptr_t)d_vert, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                 f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view cv = mk((void *)(uintptr_t)d_cot, GFFX_DTYPE_FLOAT64, 3u, c_shape,
                                 c_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gv = mk((void *)(uintptr_t)d_grad, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_mesh_gather_faces_backward(&vv, &fv, &cv, &device_context, &gv, NULL,
                                             &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  device failed: %s\n", message); ++failures; return; }
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_grad, d_grad, sizeof(gpu_grad));
    report_exact("grad_vertices, 64 vertices shared by 200 faces", cpu_grad, gpu_grad,
                 sizeof(cpu_grad));
}

/* ------------------------------------------------------------- render.interpolate */

#define IN_FRAGMENTS 4096
#define IN_FACES 16
#define IN_CHANNELS 3

static void run_interpolate(void) {
    CUdeviceptr d_index, d_bary, d_attr, d_cot, d_gbary, d_gattr;
    int32_t face_index[IN_FRAGMENTS];
    double barycentric[IN_FRAGMENTS * 3];
    double attributes[IN_FACES * 3 * IN_CHANNELS];
    double cotangent[IN_FRAGMENTS * IN_CHANNELS];
    double cpu_gbary[IN_FRAGMENTS * 3];
    double gpu_gbary[IN_FRAGMENTS * 3];
    double cpu_gattr[IN_FACES * 3 * IN_CHANNELS];
    double gpu_gattr[IN_FACES * 3 * IN_CHANNELS];
    double relaxed_gattr[IN_FACES * 3 * IN_CHANNELS];
    /* The fragment grid keeps the [B,H,W,K] rank the rasterizer produces. */
    int64_t i_shape[4] = {1, 64, 64, 1};
    int64_t i_stride[4] = {4096, 64, 1, 1};
    int64_t b_shape[5] = {1, 64, 64, 1, 3};
    int64_t b_stride[5] = {4096 * 3, 64 * 3, 3, 3, 1};
    int64_t a_shape[3] = {IN_FACES, 3, IN_CHANNELS};
    int64_t a_stride[3] = {3 * IN_CHANNELS, IN_CHANNELS, 1};
    int64_t c_shape[5] = {1, 64, 64, 1, IN_CHANNELS};
    int64_t c_stride[5] = {4096 * IN_CHANNELS, 64 * IN_CHANNELS, IN_CHANNELS, IN_CHANNELS, 1};
    gffx_status st;
    int i;

    for (i = 0; i < IN_FRAGMENTS; ++i) {
        /* 4096 fragments over 16 faces, so each attribute entry receives hundreds of
         * contributions; every 37th fragment is background and must contribute nothing. */
        face_index[i] = (i % 37 == 0) ? -1 : (int32_t)(i % IN_FACES);
        barycentric[i * 3 + 0] = 0.2 + 0.0001 * (double)(i % 91);
        barycentric[i * 3 + 1] = 0.3 + 0.0001 * (double)(i % 53);
        barycentric[i * 3 + 2] = 0.5 - 0.0001 * (double)((i % 91) + (i % 53));
        cotangent[i * IN_CHANNELS + 0] = 0.011 * (double)((i % 17) - 8);
        cotangent[i * IN_CHANNELS + 1] = 0.013 * (double)((i % 19) - 9);
        cotangent[i * IN_CHANNELS + 2] = 0.007 * (double)((i % 23) - 11);
    }
    for (i = 0; i < IN_FACES * 3 * IN_CHANNELS; ++i) attributes[i] = 0.4 + 0.021 * (double)i;

    cuMemAlloc(&d_index, sizeof(face_index));
    cuMemAlloc(&d_bary, sizeof(barycentric));
    cuMemAlloc(&d_attr, sizeof(attributes));
    cuMemAlloc(&d_cot, sizeof(cotangent));
    cuMemAlloc(&d_gbary, sizeof(cpu_gbary));
    cuMemAlloc(&d_gattr, sizeof(cpu_gattr));
    cuMemcpyHtoD(d_index, face_index, sizeof(face_index));
    cuMemcpyHtoD(d_bary, barycentric, sizeof(barycentric));
    cuMemcpyHtoD(d_attr, attributes, sizeof(attributes));
    cuMemcpyHtoD(d_cot, cotangent, sizeof(cotangent));

    {
        gffx_tensor_view iv = mk(face_index, GFFX_DTYPE_INT32, 4u, i_shape, i_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view bv = mk(barycentric, GFFX_DTYPE_FLOAT64, 5u, b_shape, b_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view av = mk(attributes, GFFX_DTYPE_FLOAT64, 3u, a_shape, a_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view cv = mk(cotangent, GFFX_DTYPE_FLOAT64, 5u, c_shape, c_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gb = mk(cpu_gbary, GFFX_DTYPE_FLOAT64, 5u, b_shape, b_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view ga = mk(cpu_gattr, GFFX_DTYPE_FLOAT64, 3u, a_shape, a_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_render_interpolate_backward(&iv, &bv, &av, &cv, &host, &gb, &ga, NULL,
                                              &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view iv = mk((void *)(uintptr_t)d_index, GFFX_DTYPE_INT32, 4u, i_shape,
                                 i_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view bv = mk((void *)(uintptr_t)d_bary, GFFX_DTYPE_FLOAT64, 5u, b_shape,
                                 b_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view av = mk((void *)(uintptr_t)d_attr, GFFX_DTYPE_FLOAT64, 3u, a_shape,
                                 a_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view cv = mk((void *)(uintptr_t)d_cot, GFFX_DTYPE_FLOAT64, 5u, c_shape,
                                 c_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gb = mk((void *)(uintptr_t)d_gbary, GFFX_DTYPE_FLOAT64, 5u, b_shape,
                                 b_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view ga = mk((void *)(uintptr_t)d_gattr, GFFX_DTYPE_FLOAT64, 3u, a_shape,
                                 a_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_render_interpolate_backward(&iv, &bv, &av, &cv, &device_context, &gb, &ga, NULL,
                                              &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device ordered failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_gbary, d_gbary, sizeof(gpu_gbary));
        cuMemcpyDtoH(gpu_gattr, d_gattr, sizeof(gpu_gattr));

        message[0] = 0;
        st = gffx_render_interpolate_backward(&iv, &bv, &av, &cv, &device_relaxed, &gb, &ga, NULL,
                                              &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device relaxed failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(relaxed_gattr, d_gattr, sizeof(relaxed_gattr));
    }
    report_exact("grad_barycentric, per fragment", cpu_gbary, gpu_gbary, sizeof(cpu_gbary));
    report_exact("grad_face_attributes, ordered default", cpu_gattr, gpu_gattr,
                 sizeof(cpu_gattr));
    /* The relaxed path trades bit-identity for speed, so it is held to a tolerance. Asserting
     * equality here would pass by luck or fail for the intended reason. */
    report_close("grad_face_attributes, relaxed atomic", cpu_gattr, relaxed_gattr,
                 (size_t)(IN_FACES * 3 * IN_CHANNELS), 1e-12);
}

int main(void) {
    CUdevice device;
    CUcontext cu;

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&cu, 0, device) != CUDA_SUCCESS) {
        printf("no usable CUDA device\n");
        return 2;
    }
    setup_contexts();

    printf("transforms.perspective_divide backward:\n");
    run_perspective_divide();
    printf("transforms.transform_points backward:\n");
    run_transform_points();
    printf("mesh.gather_faces backward:\n");
    run_gather_faces();
    printf("render.interpolate backward:\n");
    run_interpolate();

    printf("\n%d failing comparison(s)\n", failures);
    return failures != 0;
}
