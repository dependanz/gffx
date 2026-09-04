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
#include <gffx/points.h>
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


/* ----------------------------------------------------------------- render.texture */

#define TX_H 16
#define TX_W 8
#define TX_C 3
#define TX_LEVELS 5
#define TX_TOTAL 1024
#define TX_SAMPLES 512

static void run_texture(void) {
    CUdeviceptr d_pyr, d_off, d_coord, d_deriv, d_border, d_gsamp, d_gpyr, d_gcoord;
    double texture[TX_H * TX_W * TX_C];
    double pyramid[TX_TOTAL];
    int32_t offsets[TX_LEVELS + 1];
    double coordinates[TX_SAMPLES * 2];
    double derivatives[TX_SAMPLES * 4];
    double border[TX_C] = {-1.0, 0.25, 7.0};
    double grad_samples[TX_SAMPLES * TX_C];
    double cpu_gpyr[TX_TOTAL];
    double gpu_gpyr[TX_TOTAL];
    double relaxed_gpyr[TX_TOTAL];
    double cpu_gcoord[TX_SAMPLES * 2];
    double gpu_gcoord[TX_SAMPLES * 2];
    int64_t t_shape[3] = {TX_H, TX_W, TX_C};
    int64_t t_stride[3] = {TX_W * TX_C, TX_C, 1};
    int64_t p_shape[1] = {TX_TOTAL};
    int64_t p_stride[1] = {1};
    int64_t o_shape[1] = {TX_LEVELS + 1};
    int64_t o_stride[1] = {1};
    int64_t c_shape[2] = {TX_SAMPLES, 2};
    int64_t c_stride[2] = {2, 1};
    int64_t d_shape[2] = {TX_SAMPLES, 4};
    int64_t d_stride[2] = {4, 1};
    int64_t b_shape[1] = {TX_C};
    int64_t b_stride[1] = {1};
    int64_t g_shape[2] = {TX_SAMPLES, TX_C};
    int64_t g_stride[2] = {TX_C, 1};
    gffx_status st;
    int i;

    for (i = 0; i < TX_H * TX_W * TX_C; ++i) texture[i] = 0.3 + 0.017 * (double)i;
    for (i = 0; i < TX_SAMPLES; ++i) {
        /* Outside [0,1] at both ends so wrapping runs, and 512 samples into a 513-element
         * pyramid so most texels receive several contributions. */
        coordinates[i * 2] = -0.4 + 0.0031 * (double)i;
        coordinates[i * 2 + 1] = 1.3 - 0.0028 * (double)i;
        derivatives[i * 4 + 0] = 0.004 * (double)(i % 7);
        derivatives[i * 4 + 1] = 0.001 * (double)(i % 3);
        derivatives[i * 4 + 2] = 0.002 * (double)(i % 5);
        derivatives[i * 4 + 3] = 0.006 * (double)(i % 11);
        grad_samples[i * TX_C + 0] = 0.019 * (double)((i % 13) - 6);
        grad_samples[i * TX_C + 1] = 0.023 * (double)((i % 17) - 8);
        grad_samples[i * TX_C + 2] = 0.011 * (double)((i % 11) - 5);
    }
    {
        gffx_tensor_view tv = mk(texture, GFFX_DTYPE_FLOAT64, 3u, t_shape, t_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view pv = mk(pyramid, GFFX_DTYPE_FLOAT64, 1u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = mk(offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_render_texture_pyramid(&tv, 0, &host, &pv, &ov, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  pyramid failed: %s\n", message); ++failures; return; }
    }

    cuMemAlloc(&d_pyr, sizeof(pyramid));
    cuMemAlloc(&d_off, sizeof(offsets));
    cuMemAlloc(&d_coord, sizeof(coordinates));
    cuMemAlloc(&d_deriv, sizeof(derivatives));
    cuMemAlloc(&d_border, sizeof(border));
    cuMemAlloc(&d_gsamp, sizeof(grad_samples));
    cuMemAlloc(&d_gpyr, sizeof(cpu_gpyr));
    cuMemAlloc(&d_gcoord, sizeof(cpu_gcoord));
    cuMemcpyHtoD(d_pyr, pyramid, sizeof(pyramid));
    cuMemcpyHtoD(d_off, offsets, sizeof(offsets));
    cuMemcpyHtoD(d_coord, coordinates, sizeof(coordinates));
    cuMemcpyHtoD(d_deriv, derivatives, sizeof(derivatives));
    cuMemcpyHtoD(d_border, border, sizeof(border));
    cuMemcpyHtoD(d_gsamp, grad_samples, sizeof(grad_samples));

    {
        gffx_tensor_view pv = mk(pyramid, GFFX_DTYPE_FLOAT64, 1u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = mk(offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view cv = mk(coordinates, GFFX_DTYPE_FLOAT64, 2u, c_shape, c_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view dv = mk(derivatives, GFFX_DTYPE_FLOAT64, 2u, d_shape, d_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view bv = mk(border, GFFX_DTYPE_FLOAT64, 1u, b_shape, b_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gs = mk(grad_samples, GFFX_DTYPE_FLOAT64, 2u, g_shape, g_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gp = mk(cpu_gpyr, GFFX_DTYPE_FLOAT64, 1u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view gc = mk(cpu_gcoord, GFFX_DTYPE_FLOAT64, 2u, c_shape, c_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_render_texture_backward(&pv, &ov, TX_H, TX_W, &cv, &dv, NULL,
                                          GFFX_FILTER_BILINEAR, GFFX_MIP_LINEAR,
                                          GFFX_WRAP_REPEAT, GFFX_WRAP_MIRROR, &bv, &gs,
                                          &host, &gp, &gc, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view pv = mk((void *)(uintptr_t)d_pyr, GFFX_DTYPE_FLOAT64, 1u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view ov = mk((void *)(uintptr_t)d_off, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view cv = mk((void *)(uintptr_t)d_coord, GFFX_DTYPE_FLOAT64, 2u, c_shape,
                                 c_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view dv = mk((void *)(uintptr_t)d_deriv, GFFX_DTYPE_FLOAT64, 2u, d_shape,
                                 d_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view bv = mk((void *)(uintptr_t)d_border, GFFX_DTYPE_FLOAT64, 1u, b_shape,
                                 b_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gs = mk((void *)(uintptr_t)d_gsamp, GFFX_DTYPE_FLOAT64, 2u, g_shape,
                                 g_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gp = mk((void *)(uintptr_t)d_gpyr, GFFX_DTYPE_FLOAT64, 1u, p_shape,
                                 p_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view gc = mk((void *)(uintptr_t)d_gcoord, GFFX_DTYPE_FLOAT64, 2u, c_shape,
                                 c_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_render_texture_backward(&pv, &ov, TX_H, TX_W, &cv, &dv, NULL,
                                          GFFX_FILTER_BILINEAR, GFFX_MIP_LINEAR,
                                          GFFX_WRAP_REPEAT, GFFX_WRAP_MIRROR, &bv, &gs,
                                          &device_context, &gp, &gc, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device ordered failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_gpyr, d_gpyr, sizeof(gpu_gpyr));
        cuMemcpyDtoH(gpu_gcoord, d_gcoord, sizeof(gpu_gcoord));

        message[0] = 0;
        st = gffx_render_texture_backward(&pv, &ov, TX_H, TX_W, &cv, &dv, NULL,
                                          GFFX_FILTER_BILINEAR, GFFX_MIP_LINEAR,
                                          GFFX_WRAP_REPEAT, GFFX_WRAP_MIRROR, &bv, &gs,
                                          &device_relaxed, &gp, &gc, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device relaxed failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(relaxed_gpyr, d_gpyr, sizeof(relaxed_gpyr));
    }
    report_exact("grad_coordinates, per sample", cpu_gcoord, gpu_gcoord, sizeof(cpu_gcoord));
    report_exact("grad_pyramid, ordered default",
                 cpu_gpyr, gpu_gpyr, (size_t)offsets[TX_LEVELS] * sizeof(double));
    report_close("grad_pyramid, relaxed atomic", cpu_gpyr, relaxed_gpyr,
                 (size_t)offsets[TX_LEVELS], 1e-12);
}


/* --------------------------------------------------------------- mesh.face_geometry */

#define FG_VERTICES 96
#define FG_FACES 300

static void run_face_geometry(void) {
    CUdeviceptr d_vert, d_faces, d_gnorm, d_garea, d_grad;
    double vertices[FG_VERTICES * 3];
    int32_t faces[FG_FACES * 3];
    double grad_normals[FG_FACES * 3];
    double grad_areas[FG_FACES];
    double cpu_grad[FG_VERTICES * 3];
    double gpu_grad[FG_VERTICES * 3];
    int64_t v_shape[2] = {FG_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    int64_t f_shape[2] = {FG_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    int64_t a_shape[1] = {FG_FACES};
    int64_t a_stride[1] = {1};
    gffx_status st;
    int i;

    /* Positions spread so no face is degenerate; a degenerate face would take the eps branch
     * and skip, which is worth covering but must not be the whole test. */
    for (i = 0; i < FG_VERTICES; ++i) {
        vertices[i * 3 + 0] = 0.37 * (double)(i % 11) - 1.4;
        vertices[i * 3 + 1] = 0.29 * (double)(i % 7) + 0.3;
        vertices[i * 3 + 2] = 0.43 * (double)(i % 13) - 2.1;
    }
    for (i = 0; i < FG_FACES; ++i) {
        /* Heavy sharing, so each vertex accumulates from roughly nine faces and the order of
         * those additions is what the comparison is actually testing. */
        faces[i * 3 + 0] = (int32_t)(i % FG_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 5 + 17) % FG_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 11 + 41) % FG_VERTICES);
        grad_normals[i * 3 + 0] = 0.017 * (double)((i % 19) - 9);
        grad_normals[i * 3 + 1] = 0.023 * (double)((i % 23) - 11);
        grad_normals[i * 3 + 2] = 0.013 * (double)((i % 17) - 8);
        grad_areas[i] = 0.031 * (double)((i % 13) - 6);
    }

    cuMemAlloc(&d_vert, sizeof(vertices));
    cuMemAlloc(&d_faces, sizeof(faces));
    cuMemAlloc(&d_gnorm, sizeof(grad_normals));
    cuMemAlloc(&d_garea, sizeof(grad_areas));
    cuMemAlloc(&d_grad, sizeof(cpu_grad));
    cuMemcpyHtoD(d_vert, vertices, sizeof(vertices));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));
    cuMemcpyHtoD(d_gnorm, grad_normals, sizeof(grad_normals));
    cuMemcpyHtoD(d_garea, grad_areas, sizeof(grad_areas));

    {
        gffx_tensor_view vv = mk(vertices, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view nv = mk(grad_normals, GFFX_DTYPE_FLOAT64, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view av = mk(grad_areas, GFFX_DTYPE_FLOAT64, 1u, a_shape, a_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gv = mk(cpu_grad, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_mesh_face_geometry_backward(&vv, &fv, 1e-12, &nv, &av, &host, &gv, NULL,
                                              &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view vv = mk((void *)(uintptr_t)d_vert, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                 f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view nv = mk((void *)(uintptr_t)d_gnorm, GFFX_DTYPE_FLOAT64, 2u, f_shape,
                                 f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view av = mk((void *)(uintptr_t)d_garea, GFFX_DTYPE_FLOAT64, 1u, a_shape,
                                 a_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gv = mk((void *)(uintptr_t)d_grad, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_mesh_face_geometry_backward(&vv, &fv, 1e-12, &nv, &av, &device_context, &gv,
                                              NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  device failed: %s\n", message); ++failures; return; }
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_grad, d_grad, sizeof(gpu_grad));
    report_exact("grad_vertices, 96 vertices across 300 faces", cpu_grad, gpu_grad,
                 sizeof(cpu_grad));
}


/* --------------------------------------------------------- render.texture_pyramid */

static void run_texture_pyramid(void) {
    CUdeviceptr d_off, d_gpyr, d_gtex;
    double texture[TX_H * TX_W * TX_C];
    double pyramid[TX_TOTAL];
    int32_t offsets[TX_LEVELS + 1];
    double grad_pyramid[TX_TOTAL];
    double cpu_gtex[TX_H * TX_W * TX_C];
    double gpu_gtex[TX_H * TX_W * TX_C];
    int64_t t_shape[3] = {TX_H, TX_W, TX_C};
    int64_t t_stride[3] = {TX_W * TX_C, TX_C, 1};
    int64_t p_shape[1] = {TX_TOTAL};
    int64_t p_stride[1] = {1};
    int64_t o_shape[1] = {TX_LEVELS + 1};
    int64_t o_stride[1] = {1};
    gffx_status st;
    int i;

    for (i = 0; i < TX_H * TX_W * TX_C; ++i) texture[i] = 0.3 + 0.017 * (double)i;
    {
        gffx_tensor_view tv = mk(texture, GFFX_DTYPE_FLOAT64, 3u, t_shape, t_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view pv = mk(pyramid, GFFX_DTYPE_FLOAT64, 1u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = mk(offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_render_texture_pyramid(&tv, 0, &host, &pv, &ov, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  pyramid failed: %s\n", message); ++failures; return; }
    }
    /* Seed every level, so the gather has to walk the whole chain rather than one step. */
    for (i = 0; i < offsets[TX_LEVELS]; ++i) grad_pyramid[i] = 0.013 * (double)((i % 29) - 14);

    cuMemAlloc(&d_off, sizeof(offsets));
    cuMemAlloc(&d_gpyr, sizeof(grad_pyramid));
    cuMemAlloc(&d_gtex, sizeof(cpu_gtex));
    cuMemcpyHtoD(d_off, offsets, sizeof(offsets));
    cuMemcpyHtoD(d_gpyr, grad_pyramid, sizeof(grad_pyramid));

    {
        gffx_tensor_view ov = mk(offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gv = mk(grad_pyramid, GFFX_DTYPE_FLOAT64, 1u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view tv = mk(cpu_gtex, GFFX_DTYPE_FLOAT64, 3u, t_shape, t_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_render_texture_pyramid_backward(&ov, TX_H, TX_W, TX_C, &gv, &host, &tv, NULL,
                                                  &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view ov = mk((void *)(uintptr_t)d_off, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gv = mk((void *)(uintptr_t)d_gpyr, GFFX_DTYPE_FLOAT64, 1u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view tv = mk((void *)(uintptr_t)d_gtex, GFFX_DTYPE_FLOAT64, 3u, t_shape,
                                 t_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_render_texture_pyramid_backward(&ov, TX_H, TX_W, TX_C, &gv, &device_context, &tv,
                                                  NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  device failed: %s\n", message); ++failures; return; }
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_gtex, d_gtex, sizeof(gpu_gtex));
    report_exact("grad_texture, full level chain gathered", cpu_gtex, gpu_gtex, sizeof(cpu_gtex));
}


/* ------------------------------------------------------------------------- points.knn */

#define KNN_QUERY 128
#define KNN_REFERENCE 24
#define KNN_K 4

static void run_knn(void) {
    CUdeviceptr d_q, d_r, d_idx, d_valid, d_cot, d_gq, d_gr;
    double query[KNN_QUERY * 3];
    double reference[KNN_REFERENCE * 3];
    double distance[KNN_QUERY * KNN_K];
    int32_t index[KNN_QUERY * KNN_K];
    unsigned char valid[KNN_QUERY * KNN_K];
    double cotangent[KNN_QUERY * KNN_K];
    int32_t q_offsets[2];
    int32_t r_offsets[2];
    double cpu_gq[KNN_QUERY * 3], gpu_gq[KNN_QUERY * 3];
    double cpu_gr[KNN_REFERENCE * 3], gpu_gr[KNN_REFERENCE * 3];
    double relaxed_gr[KNN_REFERENCE * 3];
    int64_t q_shape[2] = {KNN_QUERY, 3};
    int64_t q_stride[2] = {3, 1};
    int64_t r_shape[2] = {KNN_REFERENCE, 3};
    int64_t r_stride[2] = {3, 1};
    int64_t k_shape[2] = {KNN_QUERY, KNN_K};
    int64_t k_stride[2] = {KNN_K, 1};
    int64_t o_shape[1] = {2};
    int64_t o_stride[1] = {1};
    gffx_status st;
    int i;

    /* Only 24 reference points for 128 queries, so each reference is chosen by many queries and
     * the scatter onto it is long. A one-to-one mapping would test nothing about ordering. */
    for (i = 0; i < KNN_QUERY; ++i) {
        query[i * 3 + 0] = 0.31 * (double)(i % 17) - 2.0;
        query[i * 3 + 1] = 0.27 * (double)(i % 11) + 0.5;
        query[i * 3 + 2] = 0.19 * (double)(i % 13) - 1.0;
    }
    for (i = 0; i < KNN_REFERENCE; ++i) {
        reference[i * 3 + 0] = 0.53 * (double)(i % 7) - 1.5;
        reference[i * 3 + 1] = 0.41 * (double)(i % 5) + 0.2;
        reference[i * 3 + 2] = 0.37 * (double)(i % 3) - 0.8;
    }
    for (i = 0; i < KNN_QUERY * KNN_K; ++i) cotangent[i] = 0.017 * (double)((i % 19) - 9);
    q_offsets[0] = 0; q_offsets[1] = KNN_QUERY;
    r_offsets[0] = 0; r_offsets[1] = KNN_REFERENCE;

    {
        gffx_tensor_view qv = mk(query, GFFX_DTYPE_FLOAT64, 2u, q_shape, q_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view rv = mk(reference, GFFX_DTYPE_FLOAT64, 2u, r_shape, r_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view qo = mk(q_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view ro = mk(r_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view dv = mk(distance, GFFX_DTYPE_FLOAT64, 2u, k_shape, k_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(index, GFFX_DTYPE_INT32, 2u, k_shape, k_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view vv = mk(valid, GFFX_DTYPE_BOOL, 2u, k_shape, k_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_points_knn(&qv, &rv, &qo, &ro, KNN_K, &host, &dv, &iv, &vv, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  forward failed: %s\n", message); ++failures; return; }
    }

    cuMemAlloc(&d_q, sizeof(query)); cuMemAlloc(&d_r, sizeof(reference));
    cuMemAlloc(&d_idx, sizeof(index)); cuMemAlloc(&d_valid, sizeof(valid));
    cuMemAlloc(&d_cot, sizeof(cotangent));
    cuMemAlloc(&d_gq, sizeof(cpu_gq)); cuMemAlloc(&d_gr, sizeof(cpu_gr));
    cuMemcpyHtoD(d_q, query, sizeof(query));
    cuMemcpyHtoD(d_r, reference, sizeof(reference));
    cuMemcpyHtoD(d_idx, index, sizeof(index));
    cuMemcpyHtoD(d_valid, valid, sizeof(valid));
    cuMemcpyHtoD(d_cot, cotangent, sizeof(cotangent));

    {
        gffx_tensor_view qv = mk(query, GFFX_DTYPE_FLOAT64, 2u, q_shape, q_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view rv = mk(reference, GFFX_DTYPE_FLOAT64, 2u, r_shape, r_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(index, GFFX_DTYPE_INT32, 2u, k_shape, k_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view vv = mk(valid, GFFX_DTYPE_BOOL, 2u, k_shape, k_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view cv = mk(cotangent, GFFX_DTYPE_FLOAT64, 2u, k_shape, k_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gq = mk(cpu_gq, GFFX_DTYPE_FLOAT64, 2u, q_shape, q_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view gr = mk(cpu_gr, GFFX_DTYPE_FLOAT64, 2u, r_shape, r_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_points_knn_backward(&qv, &rv, &iv, &vv, &cv, &host, &gq, &gr, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view qv = mk((void *)(uintptr_t)d_q, GFFX_DTYPE_FLOAT64, 2u, q_shape, q_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view rv = mk((void *)(uintptr_t)d_r, GFFX_DTYPE_FLOAT64, 2u, r_shape, r_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view iv = mk((void *)(uintptr_t)d_idx, GFFX_DTYPE_INT32, 2u, k_shape, k_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view vv = mk((void *)(uintptr_t)d_valid, GFFX_DTYPE_BOOL, 2u, k_shape,
                                 k_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view cv = mk((void *)(uintptr_t)d_cot, GFFX_DTYPE_FLOAT64, 2u, k_shape,
                                 k_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gq = mk((void *)(uintptr_t)d_gq, GFFX_DTYPE_FLOAT64, 2u, q_shape,
                                 q_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view gr = mk((void *)(uintptr_t)d_gr, GFFX_DTYPE_FLOAT64, 2u, r_shape,
                                 r_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_points_knn_backward(&qv, &rv, &iv, &vv, &cv, &device_context, &gq, &gr, NULL,
                                      &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device ordered failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_gq, d_gq, sizeof(gpu_gq));
        cuMemcpyDtoH(gpu_gr, d_gr, sizeof(gpu_gr));
        message[0] = 0;
        st = gffx_points_knn_backward(&qv, &rv, &iv, &vv, &cv, &device_relaxed, &gq, &gr, NULL,
                                      &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device relaxed failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(relaxed_gr, d_gr, sizeof(relaxed_gr));
    }
    report_exact("grad_query, per query point", cpu_gq, gpu_gq, sizeof(cpu_gq));
    report_exact("grad_reference, ordered default", cpu_gr, gpu_gr, sizeof(cpu_gr));
    report_close("grad_reference, relaxed atomic", cpu_gr, relaxed_gr,
                 (size_t)(KNN_REFERENCE * 3), 1e-12);
}


/* ------------------------------------------------- points.closest_point_on_mesh */

#define CP_POINTS 96
#define CP_VERTICES 40
#define CP_FACES 60

static void run_closest_point(void) {
    CUdeviceptr d_pts, d_vert, d_faces, d_fidx, d_bary, d_close, d_valid, d_cot, d_gp, d_gv;
    double points[CP_POINTS * 3];
    double vertices[CP_VERTICES * 3];
    int32_t faces[CP_FACES * 3];
    double distance[CP_POINTS];
    int32_t face_index[CP_POINTS];
    double barycentric[CP_POINTS * 3];
    double closest[CP_POINTS * 3];
    unsigned char valid[CP_POINTS];
    double cotangent[CP_POINTS];
    int32_t p_offsets[2], v_offsets[2], f_offsets[2];
    double cpu_gp[CP_POINTS * 3], gpu_gp[CP_POINTS * 3];
    double cpu_gv[CP_VERTICES * 3], gpu_gv[CP_VERTICES * 3], relaxed_gv[CP_VERTICES * 3];
    int64_t p_shape[2] = {CP_POINTS, 3};
    int64_t p_stride[2] = {3, 1};
    int64_t v_shape[2] = {CP_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    int64_t f_shape[2] = {CP_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    int64_t s_shape[1] = {CP_POINTS};
    int64_t s_stride[1] = {1};
    int64_t o_shape[1] = {2};
    int64_t o_stride[1] = {1};
    gffx_status st;
    int i;

    for (i = 0; i < CP_POINTS; ++i) {
        points[i * 3 + 0] = 0.41 * (double)(i % 13) - 2.2;
        points[i * 3 + 1] = 0.33 * (double)(i % 7) + 0.4;
        points[i * 3 + 2] = 0.29 * (double)(i % 11) - 1.3;
        cotangent[i] = 0.021 * (double)((i % 17) - 8);
    }
    for (i = 0; i < CP_VERTICES; ++i) {
        vertices[i * 3 + 0] = 0.53 * (double)(i % 5) - 1.1;
        vertices[i * 3 + 1] = 0.47 * (double)(i % 8) - 0.6;
        vertices[i * 3 + 2] = 0.61 * (double)(i % 3) - 0.9;
    }
    for (i = 0; i < CP_FACES; ++i) {
        /* Only 40 vertices behind 60 faces and 96 query points, so a vertex is named by several
         * faces and reached by several points; that is what makes the scatter long. */
        faces[i * 3 + 0] = (int32_t)(i % CP_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 7 + 5) % CP_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 11 + 13) % CP_VERTICES);
    }
    p_offsets[0] = 0; p_offsets[1] = CP_POINTS;
    v_offsets[0] = 0; v_offsets[1] = CP_VERTICES;
    f_offsets[0] = 0; f_offsets[1] = CP_FACES;

    {
        gffx_tensor_view pv = mk(points, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view vv = mk(vertices, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view po = mk(p_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view vo = mk(v_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fo = mk(f_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view dv = mk(distance, GFFX_DTYPE_FLOAT64, 1u, s_shape, s_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(face_index, GFFX_DTYPE_INT32, 1u, s_shape, s_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view bv = mk(barycentric, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view cv = mk(closest, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view av = mk(valid, GFFX_DTYPE_BOOL, 1u, s_shape, s_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_points_closest_point_on_mesh(&pv, &vv, &fv, &po, &vo, &fo, 1e-12, &host,
                                               &dv, &iv, &bv, &cv, &av, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  forward failed: %s\n", message); ++failures; return; }
    }

    cuMemAlloc(&d_pts, sizeof(points)); cuMemAlloc(&d_vert, sizeof(vertices));
    cuMemAlloc(&d_faces, sizeof(faces)); cuMemAlloc(&d_fidx, sizeof(face_index));
    cuMemAlloc(&d_bary, sizeof(barycentric)); cuMemAlloc(&d_close, sizeof(closest));
    cuMemAlloc(&d_valid, sizeof(valid)); cuMemAlloc(&d_cot, sizeof(cotangent));
    cuMemAlloc(&d_gp, sizeof(cpu_gp)); cuMemAlloc(&d_gv, sizeof(cpu_gv));
    cuMemcpyHtoD(d_pts, points, sizeof(points));
    cuMemcpyHtoD(d_vert, vertices, sizeof(vertices));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));
    cuMemcpyHtoD(d_fidx, face_index, sizeof(face_index));
    cuMemcpyHtoD(d_bary, barycentric, sizeof(barycentric));
    cuMemcpyHtoD(d_close, closest, sizeof(closest));
    cuMemcpyHtoD(d_valid, valid, sizeof(valid));
    cuMemcpyHtoD(d_cot, cotangent, sizeof(cotangent));

    {
        gffx_tensor_view pv = mk(points, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view vv = mk(vertices, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(face_index, GFFX_DTYPE_INT32, 1u, s_shape, s_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view bv = mk(barycentric, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view cv = mk(closest, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view av = mk(valid, GFFX_DTYPE_BOOL, 1u, s_shape, s_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gd = mk(cotangent, GFFX_DTYPE_FLOAT64, 1u, s_shape, s_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gp = mk(cpu_gp, GFFX_DTYPE_FLOAT64, 2u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view gv = mk(cpu_gv, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_points_closest_point_on_mesh_backward(&pv, &vv, &fv, &iv, &bv, &cv, &av, &gd,
                                                        &host, &gp, &gv, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view pv = mk((void *)(uintptr_t)d_pts, GFFX_DTYPE_FLOAT64, 2u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view vv = mk((void *)(uintptr_t)d_vert, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                 f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view iv = mk((void *)(uintptr_t)d_fidx, GFFX_DTYPE_INT32, 1u, s_shape,
                                 s_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view bv = mk((void *)(uintptr_t)d_bary, GFFX_DTYPE_FLOAT64, 2u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view cv = mk((void *)(uintptr_t)d_close, GFFX_DTYPE_FLOAT64, 2u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view av = mk((void *)(uintptr_t)d_valid, GFFX_DTYPE_BOOL, 1u, s_shape,
                                 s_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gd = mk((void *)(uintptr_t)d_cot, GFFX_DTYPE_FLOAT64, 1u, s_shape,
                                 s_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gp = mk((void *)(uintptr_t)d_gp, GFFX_DTYPE_FLOAT64, 2u, p_shape,
                                 p_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view gv = mk((void *)(uintptr_t)d_gv, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_points_closest_point_on_mesh_backward(&pv, &vv, &fv, &iv, &bv, &cv, &av, &gd,
                                                        &device_context, &gp, &gv, NULL,
                                                        &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device ordered failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_gp, d_gp, sizeof(gpu_gp));
        cuMemcpyDtoH(gpu_gv, d_gv, sizeof(gpu_gv));
        message[0] = 0;
        st = gffx_points_closest_point_on_mesh_backward(&pv, &vv, &fv, &iv, &bv, &cv, &av, &gd,
                                                        &device_relaxed, &gp, &gv, NULL,
                                                        &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device relaxed failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(relaxed_gv, d_gv, sizeof(relaxed_gv));
    }
    report_exact("grad_points, per query point", cpu_gp, gpu_gp, sizeof(cpu_gp));
    report_exact("grad_vertices, ordered default", cpu_gv, gpu_gv, sizeof(cpu_gv));
    report_close("grad_vertices, relaxed atomic", cpu_gv, relaxed_gv,
                 (size_t)(CP_VERTICES * 3), 1e-12);
}


/* --------------------------------------------------------------------- render.rasterize */

#define RS_VERTICES 12
#define RS_FACES 20
#define RS_SIZE 64
#define RS_SLOTS 2
#define RS_FRAGMENTS (RS_SIZE * RS_SIZE * RS_SLOTS)

static void run_rasterize(void) {
    CUdeviceptr d_ndc, d_faces, d_fidx, d_gb, d_gd, d_gs, d_gn;
    double ndc[RS_VERTICES * 3];
    int32_t faces[RS_FACES * 3];
    int32_t v_offsets[2], f_offsets[2];
    int32_t face_index[RS_FRAGMENTS];
    double barycentric[RS_FRAGMENTS * 3];
    double depth[RS_FRAGMENTS];
    double distance[RS_FRAGMENTS];
    double grad_bary[RS_FRAGMENTS * 3];
    double grad_depth[RS_FRAGMENTS];
    double grad_distance[RS_FRAGMENTS];
    double cpu_gn[RS_VERTICES * 3], gpu_gn[RS_VERTICES * 3], relaxed_gn[RS_VERTICES * 3];
    int64_t v_shape[2] = {RS_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    int64_t f_shape[2] = {RS_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    int64_t o_shape[1] = {2};
    int64_t o_stride[1] = {1};
    int64_t i_shape[4] = {1, RS_SIZE, RS_SIZE, RS_SLOTS};
    int64_t i_stride[4] = {RS_FRAGMENTS, RS_SIZE * RS_SLOTS, RS_SLOTS, 1};
    int64_t b_shape[5] = {1, RS_SIZE, RS_SIZE, RS_SLOTS, 3};
    int64_t b_stride[5] = {RS_FRAGMENTS * 3, RS_SIZE * RS_SLOTS * 3, RS_SLOTS * 3, 3, 1};
    gffx_status st;
    int i;

    /* Overlapping triangles across the whole viewport, so most pixels carry two fragments and a
     * vertex is reached from many of them. */
    for (i = 0; i < RS_VERTICES; ++i) {
        double a = 6.2831853 * (double)i / (double)RS_VERTICES;
        ndc[i * 3 + 0] = 0.75 * cos(a) * (0.6 + 0.4 * (double)(i % 3));
        ndc[i * 3 + 1] = 0.75 * sin(a) * (0.6 + 0.4 * (double)(i % 4));
        ndc[i * 3 + 2] = 0.05 * (double)(i % 5);
    }
    for (i = 0; i < RS_FACES; ++i) {
        faces[i * 3 + 0] = (int32_t)(i % RS_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 5 + 3) % RS_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 7 + 6) % RS_VERTICES);
    }
    v_offsets[0] = 0; v_offsets[1] = RS_VERTICES;
    f_offsets[0] = 0; f_offsets[1] = RS_FACES;
    for (i = 0; i < RS_FRAGMENTS; ++i) {
        grad_bary[i * 3 + 0] = 0.013 * (double)((i % 17) - 8);
        grad_bary[i * 3 + 1] = 0.011 * (double)((i % 13) - 6);
        grad_bary[i * 3 + 2] = 0.017 * (double)((i % 19) - 9);
        grad_depth[i] = 0.023 * (double)((i % 11) - 5);
        grad_distance[i] = 0.007 * (double)((i % 23) - 11);
    }

    {
        gffx_tensor_view nv = mk(ndc, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view vo = mk(v_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fo = mk(f_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(face_index, GFFX_DTYPE_INT32, 4u, i_shape, i_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view bv = mk(barycentric, GFFX_DTYPE_FLOAT64, 5u, b_shape, b_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view dv = mk(depth, GFFX_DTYPE_FLOAT64, 4u, i_shape, i_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view sv = mk(distance, GFFX_DTYPE_FLOAT64, 4u, i_shape, i_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_render_rasterize(&nv, &fv, &vo, &fo, RS_SIZE, RS_SIZE, RS_SLOTS, 3.0,
                                   GFFX_CULL_NONE, 1e-12, &host, &iv, &bv, &dv, &sv, NULL,
                                   &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  forward failed: %s\n", message); ++failures; return; }
    }

    cuMemAlloc(&d_ndc, sizeof(ndc)); cuMemAlloc(&d_faces, sizeof(faces));
    cuMemAlloc(&d_fidx, sizeof(face_index)); cuMemAlloc(&d_gb, sizeof(grad_bary));
    cuMemAlloc(&d_gd, sizeof(grad_depth)); cuMemAlloc(&d_gs, sizeof(grad_distance));
    cuMemAlloc(&d_gn, sizeof(cpu_gn));
    cuMemcpyHtoD(d_ndc, ndc, sizeof(ndc));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));
    cuMemcpyHtoD(d_fidx, face_index, sizeof(face_index));
    cuMemcpyHtoD(d_gb, grad_bary, sizeof(grad_bary));
    cuMemcpyHtoD(d_gd, grad_depth, sizeof(grad_depth));
    cuMemcpyHtoD(d_gs, grad_distance, sizeof(grad_distance));

    {
        gffx_tensor_view nv = mk(ndc, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(face_index, GFFX_DTYPE_INT32, 4u, i_shape, i_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gb = mk(grad_bary, GFFX_DTYPE_FLOAT64, 5u, b_shape, b_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gd = mk(grad_depth, GFFX_DTYPE_FLOAT64, 4u, i_shape, i_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gs = mk(grad_distance, GFFX_DTYPE_FLOAT64, 4u, i_shape, i_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gn = mk(cpu_gn, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_render_rasterize_backward(&nv, &fv, RS_SIZE, RS_SIZE, &iv, &gb, &gd, &gs,
                                            &host, &gn, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) { printf("  host failed: %s\n", message); ++failures; return; }
    }
    {
        gffx_tensor_view nv = mk((void *)(uintptr_t)d_ndc, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                 f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view iv = mk((void *)(uintptr_t)d_fidx, GFFX_DTYPE_INT32, 4u, i_shape,
                                 i_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gb = mk((void *)(uintptr_t)d_gb, GFFX_DTYPE_FLOAT64, 5u, b_shape,
                                 b_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gd = mk((void *)(uintptr_t)d_gd, GFFX_DTYPE_FLOAT64, 4u, i_shape,
                                 i_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gs = mk((void *)(uintptr_t)d_gs, GFFX_DTYPE_FLOAT64, 4u, i_shape,
                                 i_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gn = mk((void *)(uintptr_t)d_gn, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_render_rasterize_backward(&nv, &fv, RS_SIZE, RS_SIZE, &iv, &gb, &gd, &gs,
                                            &device_context, &gn, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device ordered failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_gn, d_gn, sizeof(gpu_gn));
        message[0] = 0;
        st = gffx_render_rasterize_backward(&nv, &fv, RS_SIZE, RS_SIZE, &iv, &gb, &gd, &gs,
                                            &device_relaxed, &gn, NULL, &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device relaxed failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(relaxed_gn, d_gn, sizeof(relaxed_gn));
    }
    report_exact("grad_ndc_vertices, ordered default", cpu_gn, gpu_gn, sizeof(cpu_gn));
    report_close("grad_ndc_vertices, relaxed atomic", cpu_gn, relaxed_gn,
                 (size_t)(RS_VERTICES * 3), 1e-12);
}


/* ------------------------------------------------------------------ mesh.vertex_normals */

#define VN_VERTICES 80
#define VN_FACES 160

static void run_vertex_normals(void) {
    CUdeviceptr d_vert, d_faces, d_out;
    double vertices[VN_VERTICES * 3];
    int32_t faces[VN_FACES * 3];
    double cpu_normals[VN_VERTICES * 3], gpu_normals[VN_VERTICES * 3];
    int64_t v_shape[2] = {VN_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    int64_t f_shape[2] = {VN_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    gffx_status st;
    int i, w;
    unsigned int weightings[2] = {GFFX_MESH_WEIGHTING_AREA, GFFX_MESH_WEIGHTING_UNIFORM};
    const char *names[2] = {"area weighting", "uniform weighting"};

    for (i = 0; i < VN_VERTICES; ++i) {
        vertices[i * 3 + 0] = 0.37 * (double)(i % 11) - 1.9;
        vertices[i * 3 + 1] = 0.29 * (double)(i % 7) + 0.4;
        vertices[i * 3 + 2] = 0.43 * (double)(i % 13) - 2.6;
    }
    for (i = 0; i < VN_FACES; ++i) {
        faces[i * 3 + 0] = (int32_t)(i % VN_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 5 + 11) % VN_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 13 + 7) % VN_VERTICES);
    }
    cuMemAlloc(&d_vert, sizeof(vertices));
    cuMemAlloc(&d_faces, sizeof(faces));
    cuMemAlloc(&d_out, sizeof(cpu_normals));
    cuMemcpyHtoD(d_vert, vertices, sizeof(vertices));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));

    for (w = 0; w < 2; ++w) {
        {
            gffx_tensor_view vv = mk(vertices, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                     GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
            gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                     GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
            gffx_tensor_view ov = mk(cpu_normals, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                     GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
            message[0] = 0;
            st = gffx_mesh_vertex_normals(&vv, &fv, 1e-12, weightings[w], &host, &ov, NULL,
                                          &diagnostic);
            if (st != GFFX_STATUS_OK) {
                printf("  host failed: %s\n", message); ++failures; return;
            }
        }
        {
            gffx_tensor_view vv = mk((void *)(uintptr_t)d_vert, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                     v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
            gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                     f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
            gffx_tensor_view ov = mk((void *)(uintptr_t)d_out, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                     v_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
            message[0] = 0;
            st = gffx_mesh_vertex_normals(&vv, &fv, 1e-12, weightings[w], &device_context, &ov,
                                          NULL, &diagnostic);
            if (st != GFFX_STATUS_OK) {
                printf("  device failed: %s\n", message); ++failures; return;
            }
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_normals, d_out, sizeof(gpu_normals));
        report_exact(names[w], cpu_normals, gpu_normals, sizeof(cpu_normals));
    }
}


/* --------------------------------------------------- mesh.vertex_normals backward */

static void run_vertex_normals_backward(void) {
    CUdeviceptr d_vert, d_faces, d_cot, d_out, d_scratch;
    double vertices[VN_VERTICES * 3];
    int32_t faces[VN_FACES * 3];
    double cotangent[VN_VERTICES * 3];
    double cpu_grad[VN_VERTICES * 3], gpu_grad[VN_VERTICES * 3];
    double host_scratch[VN_VERTICES * 3];
    int64_t v_shape[2] = {VN_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    int64_t f_shape[2] = {VN_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    gffx_buffer host_workspace;
    gffx_buffer device_workspace;
    uint64_t required = 0, alignment = 0;
    gffx_status st;
    int i, w;
    unsigned int weightings[2] = {GFFX_MESH_WEIGHTING_AREA, GFFX_MESH_WEIGHTING_UNIFORM};
    const char *names[2] = {"grad_vertices, area weighting", "grad_vertices, uniform weighting"};

    for (i = 0; i < VN_VERTICES; ++i) {
        vertices[i * 3 + 0] = 0.37 * (double)(i % 11) - 1.9;
        vertices[i * 3 + 1] = 0.29 * (double)(i % 7) + 0.4;
        vertices[i * 3 + 2] = 0.43 * (double)(i % 13) - 2.6;
        cotangent[i * 3 + 0] = 0.019 * (double)((i % 13) - 6);
        cotangent[i * 3 + 1] = 0.023 * (double)((i % 17) - 8);
        cotangent[i * 3 + 2] = 0.011 * (double)((i % 7) - 3);
    }
    for (i = 0; i < VN_FACES; ++i) {
        faces[i * 3 + 0] = (int32_t)(i % VN_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 5 + 11) % VN_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 13 + 7) % VN_VERTICES);
    }

    /* Ask the library for the requirement rather than assuming it, on the device context, which
     * is the path that was previously refused outright. */
    message[0] = 0;
    st = gffx_mesh_vertex_normals_workspace(VN_VERTICES, VN_FACES, GFFX_DTYPE_FLOAT64,
                                            &device_context, &required, &alignment,
                                            &diagnostic);
    if (st != GFFX_STATUS_OK) {
        printf("  device workspace query failed: %s\n", message); ++failures; return;
    }
    if (required != (uint64_t)VN_VERTICES * 3u * sizeof(double)) {
        printf("  unexpected workspace requirement %llu\n", (unsigned long long)required);
        ++failures; return;
    }

    cuMemAlloc(&d_vert, sizeof(vertices)); cuMemAlloc(&d_faces, sizeof(faces));
    cuMemAlloc(&d_cot, sizeof(cotangent)); cuMemAlloc(&d_out, sizeof(cpu_grad));
    cuMemAlloc(&d_scratch, (size_t)required);
    cuMemcpyHtoD(d_vert, vertices, sizeof(vertices));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));
    cuMemcpyHtoD(d_cot, cotangent, sizeof(cotangent));

    memset(&host_workspace, 0, sizeof(host_workspace));
    host_workspace.struct_size = (uint32_t)sizeof(host_workspace);
    host_workspace.abi_version = GFFX_ABI_VERSION;
    host_workspace.data = host_scratch;
    host_workspace.capacity_bytes = sizeof(host_scratch);
    host_workspace.alignment = alignment;
    /* The buffer carries its own device, and zero is not a valid one - a memset alone leaves it
     * rejected by the ABI validator rather than defaulted to the host. */
    host_workspace.device_type = GFFX_DEVICE_CPU;
    device_workspace = host_workspace;
    device_workspace.data = (void *)(uintptr_t)d_scratch;
    device_workspace.capacity_bytes = required;
    device_workspace.device_type = GFFX_DEVICE_CUDA;

    for (w = 0; w < 2; ++w) {
        {
            gffx_tensor_view vv = mk(vertices, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                     GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
            gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                     GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
            gffx_tensor_view cv = mk(cotangent, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                     GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
            gffx_tensor_view gv = mk(cpu_grad, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                     GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
            message[0] = 0;
            st = gffx_mesh_vertex_normals_backward(&vv, &fv, 1e-12, weightings[w], &cv, &host,
                                                   &gv, &host_workspace, &diagnostic);
            if (st != GFFX_STATUS_OK) {
                printf("  host failed: %s\n", message); ++failures; return;
            }
        }
        {
            gffx_tensor_view vv = mk((void *)(uintptr_t)d_vert, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                     v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
            gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                     f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
            gffx_tensor_view cv = mk((void *)(uintptr_t)d_cot, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                     v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
            gffx_tensor_view gv = mk((void *)(uintptr_t)d_out, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                     v_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
            message[0] = 0;
            st = gffx_mesh_vertex_normals_backward(&vv, &fv, 1e-12, weightings[w], &cv,
                                                   &device_context, &gv, &device_workspace,
                                                   &diagnostic);
            if (st != GFFX_STATUS_OK) {
                printf("  device failed: %s\n", message); ++failures; return;
            }
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_grad, d_out, sizeof(gpu_grad));
        report_exact(names[w], cpu_grad, gpu_grad, sizeof(cpu_grad));
    }
}


/* ------------------------------------------------------------------ mesh.sample_surface */

#define SS_VERTICES 40
#define SS_FACES 60
#define SS_SAMPLES 256

static void run_sample_surface(void) {
    CUdeviceptr d_vert, d_faces, d_voff, d_foff, d_key, d_ctr;
    CUdeviceptr d_points, d_index, d_bary, d_next, d_scratch;
    double vertices[SS_VERTICES * 3];
    int32_t faces[SS_FACES * 3];
    int32_t v_offsets[2], f_offsets[2];
    unsigned int key[2];
    unsigned int counter[2];
    double cpu_points[SS_SAMPLES * 3], gpu_points[SS_SAMPLES * 3];
    int32_t cpu_index[SS_SAMPLES], gpu_index[SS_SAMPLES];
    double cpu_bary[SS_SAMPLES * 3], gpu_bary[SS_SAMPLES * 3];
    unsigned int cpu_next[2], gpu_next[2];
    double host_scratch[SS_FACES + 2];
    int64_t v_shape[2] = {SS_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    int64_t f_shape[2] = {SS_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    int64_t o_shape[1] = {2};
    int64_t o_stride[1] = {1};
    int64_t p_shape[3] = {1, SS_SAMPLES, 3};
    int64_t p_stride[3] = {SS_SAMPLES * 3, 3, 1};
    int64_t i_shape[2] = {1, SS_SAMPLES};
    int64_t i_stride[2] = {SS_SAMPLES, 1};
    int64_t k_shape[1] = {2};
    int64_t k_stride[1] = {1};
    gffx_buffer host_workspace, device_workspace;
    uint64_t host_required = 0, device_required = 0, alignment = 0;
    gffx_status st;
    int i;

    key[0] = 0x1234abcdu; key[1] = 0x9876fedcu;
    /* Seeded so the advance carries into the high word. */
    counter[0] = 0xfffffffeu; counter[1] = 7u;

    for (i = 0; i < SS_VERTICES; ++i) {
        vertices[i * 3 + 0] = 0.53 * (double)(i % 7) - 1.4;
        vertices[i * 3 + 1] = 0.37 * (double)(i % 5) + 0.6;
        vertices[i * 3 + 2] = 0.61 * (double)(i % 11) - 2.0;
    }
    for (i = 0; i < SS_FACES; ++i) {
        faces[i * 3 + 0] = (int32_t)(i % SS_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 7 + 3) % SS_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 13 + 9) % SS_VERTICES);
    }
    v_offsets[0] = 0; v_offsets[1] = SS_VERTICES;
    f_offsets[0] = 0; f_offsets[1] = SS_FACES;

    /* Both requirements come from the library. They differ, because the device needs one extra
     * word for the degenerate-batch flag, and asking separately is the point of a per-device
     * query rather than a single number. */
    message[0] = 0;
    st = gffx_mesh_sample_surface_workspace(SS_VERTICES, SS_FACES, SS_SAMPLES,
                                            GFFX_DTYPE_FLOAT64, &host, &host_required,
                                            &alignment, &diagnostic);
    if (st != GFFX_STATUS_OK) {
        printf("  host workspace query failed: %s\n", message); ++failures; return;
    }
    message[0] = 0;
    st = gffx_mesh_sample_surface_workspace(SS_VERTICES, SS_FACES, SS_SAMPLES,
                                            GFFX_DTYPE_FLOAT64, &device_context,
                                            &device_required, &alignment, &diagnostic);
    if (st != GFFX_STATUS_OK) {
        printf("  device workspace query failed: %s\n", message); ++failures; return;
    }
    if (device_required <= host_required) {
        printf("  device requirement did not exceed host\n"); ++failures; return;
    }

    cuMemAlloc(&d_vert, sizeof(vertices)); cuMemAlloc(&d_faces, sizeof(faces));
    cuMemAlloc(&d_voff, sizeof(v_offsets)); cuMemAlloc(&d_foff, sizeof(f_offsets));
    cuMemAlloc(&d_key, sizeof(key)); cuMemAlloc(&d_ctr, sizeof(counter));
    cuMemAlloc(&d_points, sizeof(cpu_points)); cuMemAlloc(&d_index, sizeof(cpu_index));
    cuMemAlloc(&d_bary, sizeof(cpu_bary)); cuMemAlloc(&d_next, sizeof(cpu_next));
    cuMemAlloc(&d_scratch, (size_t)device_required);
    cuMemcpyHtoD(d_vert, vertices, sizeof(vertices));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));
    cuMemcpyHtoD(d_voff, v_offsets, sizeof(v_offsets));
    cuMemcpyHtoD(d_foff, f_offsets, sizeof(f_offsets));
    cuMemcpyHtoD(d_key, key, sizeof(key));
    cuMemcpyHtoD(d_ctr, counter, sizeof(counter));

    memset(&host_workspace, 0, sizeof(host_workspace));
    host_workspace.struct_size = (uint32_t)sizeof(host_workspace);
    host_workspace.abi_version = GFFX_ABI_VERSION;
    host_workspace.data = host_scratch;
    host_workspace.capacity_bytes = sizeof(host_scratch);
    host_workspace.alignment = alignment;
    host_workspace.device_type = GFFX_DEVICE_CPU;
    device_workspace = host_workspace;
    device_workspace.data = (void *)(uintptr_t)d_scratch;
    device_workspace.capacity_bytes = device_required;
    device_workspace.device_type = GFFX_DEVICE_CUDA;

    {
        gffx_tensor_view vv = mk(vertices, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view vo = mk(v_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view fo = mk(f_offsets, GFFX_DTYPE_INT32, 1u, o_shape, o_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view kv = mk(key, GFFX_DTYPE_UINT32, 1u, k_shape, k_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view cv = mk(counter, GFFX_DTYPE_UINT32, 1u, k_shape, k_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view pv = mk(cpu_points, GFFX_DTYPE_FLOAT64, 3u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(cpu_index, GFFX_DTYPE_INT32, 2u, i_shape, i_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view bv = mk(cpu_bary, GFFX_DTYPE_FLOAT64, 3u, p_shape, p_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view nv = mk(cpu_next, GFFX_DTYPE_UINT32, 1u, k_shape, k_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_mesh_sample_surface(&vv, &fv, &vo, &fo, SS_SAMPLES, &kv, &cv, 1e-12, &host,
                                      &pv, &iv, &bv, &nv, &host_workspace, &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  host failed: %s\n", message); ++failures; return;
        }
    }
    {
        gffx_tensor_view vv = mk((void *)(uintptr_t)d_vert, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                 f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view vo = mk((void *)(uintptr_t)d_voff, GFFX_DTYPE_INT32, 1u, o_shape,
                                 o_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view fo = mk((void *)(uintptr_t)d_foff, GFFX_DTYPE_INT32, 1u, o_shape,
                                 o_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view kv = mk((void *)(uintptr_t)d_key, GFFX_DTYPE_UINT32, 1u, k_shape,
                                 k_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view cv = mk((void *)(uintptr_t)d_ctr, GFFX_DTYPE_UINT32, 1u, k_shape,
                                 k_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view pv = mk((void *)(uintptr_t)d_points, GFFX_DTYPE_FLOAT64, 3u, p_shape,
                                 p_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view iv = mk((void *)(uintptr_t)d_index, GFFX_DTYPE_INT32, 2u, i_shape,
                                 i_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view bv = mk((void *)(uintptr_t)d_bary, GFFX_DTYPE_FLOAT64, 3u, p_shape,
                                 p_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        gffx_tensor_view nv = mk((void *)(uintptr_t)d_next, GFFX_DTYPE_UINT32, 1u, k_shape,
                                 k_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_mesh_sample_surface(&vv, &fv, &vo, &fo, SS_SAMPLES, &kv, &cv, 1e-12,
                                      &device_context, &pv, &iv, &bv, &nv, &device_workspace,
                                      &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device failed: %s\n", message); ++failures; return;
        }
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_points, d_points, sizeof(gpu_points));
    cuMemcpyDtoH(gpu_index, d_index, sizeof(gpu_index));
    cuMemcpyDtoH(gpu_bary, d_bary, sizeof(gpu_bary));
    cuMemcpyDtoH(gpu_next, d_next, sizeof(gpu_next));
    /* Face selection is the strongest of these: an integer chosen by binary search over the
     * cumulative table, so any drift in the table or in the Philox stream changes it outright
     * rather than by a rounding step. */
    report_exact("face_index, 256 samples", cpu_index, gpu_index, sizeof(cpu_index));
    report_exact("barycentric", cpu_bary, gpu_bary, sizeof(cpu_bary));
    report_exact("points", cpu_points, gpu_points, sizeof(cpu_points));
    report_exact("next_counter, carrying advance", cpu_next, gpu_next, sizeof(cpu_next));
}


/* --------------------------------------------------- mesh.sample_surface backward */

static void run_sample_surface_backward(void) {
    CUdeviceptr d_faces, d_index, d_bary, d_grad, d_out;
    int32_t faces[SS_FACES * 3];
    int32_t face_index[SS_SAMPLES];
    double barycentric[SS_SAMPLES * 3];
    double grad_points[SS_SAMPLES * 3];
    double cpu_grad[SS_VERTICES * 3], gpu_grad[SS_VERTICES * 3], relaxed_grad[SS_VERTICES * 3];
    int64_t f_shape[2] = {SS_FACES, 3};
    int64_t f_stride[2] = {3, 1};
    int64_t i_shape[2] = {1, SS_SAMPLES};
    int64_t i_stride[2] = {SS_SAMPLES, 1};
    int64_t p_shape[3] = {1, SS_SAMPLES, 3};
    int64_t p_stride[3] = {SS_SAMPLES * 3, 3, 1};
    int64_t v_shape[2] = {SS_VERTICES, 3};
    int64_t v_stride[2] = {3, 1};
    gffx_status st;
    int i;

    for (i = 0; i < SS_FACES; ++i) {
        faces[i * 3 + 0] = (int32_t)(i % SS_VERTICES);
        faces[i * 3 + 1] = (int32_t)((i * 7 + 3) % SS_VERTICES);
        faces[i * 3 + 2] = (int32_t)((i * 13 + 9) % SS_VERTICES);
    }
    /* 256 samples over 60 faces on 40 vertices, so each vertex receives many contributions. */
    for (i = 0; i < SS_SAMPLES; ++i) {
        face_index[i] = (int32_t)(i % SS_FACES);
        barycentric[i * 3 + 0] = 0.2 + 0.001 * (double)(i % 97);
        barycentric[i * 3 + 1] = 0.3 + 0.001 * (double)(i % 61);
        barycentric[i * 3 + 2] = 0.5 - 0.001 * (double)((i % 97) + (i % 61));
        grad_points[i * 3 + 0] = 0.017 * (double)((i % 13) - 6);
        grad_points[i * 3 + 1] = 0.023 * (double)((i % 17) - 8);
        grad_points[i * 3 + 2] = 0.011 * (double)((i % 11) - 5);
    }

    cuMemAlloc(&d_faces, sizeof(faces)); cuMemAlloc(&d_index, sizeof(face_index));
    cuMemAlloc(&d_bary, sizeof(barycentric)); cuMemAlloc(&d_grad, sizeof(grad_points));
    cuMemAlloc(&d_out, sizeof(cpu_grad));
    cuMemcpyHtoD(d_faces, faces, sizeof(faces));
    cuMemcpyHtoD(d_index, face_index, sizeof(face_index));
    cuMemcpyHtoD(d_bary, barycentric, sizeof(barycentric));
    cuMemcpyHtoD(d_grad, grad_points, sizeof(grad_points));

    {
        gffx_tensor_view fv = mk(faces, GFFX_DTYPE_INT32, 2u, f_shape, f_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view iv = mk(face_index, GFFX_DTYPE_INT32, 2u, i_shape, i_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view bv = mk(barycentric, GFFX_DTYPE_FLOAT64, 3u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view gv = mk(grad_points, GFFX_DTYPE_FLOAT64, 3u, p_shape, p_stride,
                                 GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = mk(cpu_grad, GFFX_DTYPE_FLOAT64, 2u, v_shape, v_stride,
                                 GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        message[0] = 0;
        st = gffx_mesh_sample_surface_backward(&fv, &iv, &bv, &gv, &host, &ov, NULL,
                                               &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  host failed: %s\n", message); ++failures; return;
        }
    }
    {
        gffx_tensor_view fv = mk((void *)(uintptr_t)d_faces, GFFX_DTYPE_INT32, 2u, f_shape,
                                 f_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view iv = mk((void *)(uintptr_t)d_index, GFFX_DTYPE_INT32, 2u, i_shape,
                                 i_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view bv = mk((void *)(uintptr_t)d_bary, GFFX_DTYPE_FLOAT64, 3u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view gv = mk((void *)(uintptr_t)d_grad, GFFX_DTYPE_FLOAT64, 3u, p_shape,
                                 p_stride, GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
        gffx_tensor_view ov = mk((void *)(uintptr_t)d_out, GFFX_DTYPE_FLOAT64, 2u, v_shape,
                                 v_stride, GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_mesh_sample_surface_backward(&fv, &iv, &bv, &gv, &device_context, &ov, NULL,
                                               &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device ordered failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(gpu_grad, d_out, sizeof(gpu_grad));
        message[0] = 0;
        st = gffx_mesh_sample_surface_backward(&fv, &iv, &bv, &gv, &device_relaxed, &ov, NULL,
                                               &diagnostic);
        if (st != GFFX_STATUS_OK) {
            printf("  device relaxed failed: %s\n", message); ++failures; return;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(relaxed_grad, d_out, sizeof(relaxed_grad));
    }
    report_exact("grad_vertices, ordered default", cpu_grad, gpu_grad, sizeof(cpu_grad));
    report_close("grad_vertices, relaxed atomic", cpu_grad, relaxed_grad,
                 (size_t)(SS_VERTICES * 3), 1e-12);
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
    printf("mesh.face_geometry backward:\n");
    run_face_geometry();
    printf("mesh.gather_faces backward:\n");
    run_gather_faces();
    printf("render.interpolate backward:\n");
    run_interpolate();
    printf("render.texture backward:\n");
    run_texture();
    printf("render.texture_pyramid backward:\n");
    run_texture_pyramid();
    printf("points.knn backward:\n");
    run_knn();
    printf("points.closest_point_on_mesh backward:\n");
    run_closest_point();
    printf("render.rasterize backward:\n");
    run_rasterize();
    printf("mesh.vertex_normals forward:\n");
    run_vertex_normals();
    printf("mesh.vertex_normals backward:\n");
    run_vertex_normals_backward();
    printf("mesh.sample_surface forward:\n");
    run_sample_surface();
    printf("mesh.sample_surface backward:\n");
    run_sample_surface_backward();

    printf("\n%d failing comparison(s)\n", failures);
    return failures != 0;
}
