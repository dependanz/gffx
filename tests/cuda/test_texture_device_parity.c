/*
 * CPU/CUDA bit-identity check for render.texture_pyramid and render.texture, run against the real
 * device through the driver API rather than through PyTorch, because the PyTorch installed on this
 * machine is a CPU-only build and its TB-14 CUDA cases skip.
 *
 * Comparison is memcmp on the raw bytes, not a tolerance: the acceptance record's section 2.6
 * claims bit-identity, and a tolerance would pass exactly the divergence the claim exists to rule
 * out.
 */

#include <gffx/execution.h>
#include <gffx/render.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define H 16
#define W 8
#define C 3
#define LEVELS 5
#define TOTAL 1024
#define N 64

static gffx_execution_context context_for(gffx_device_type type) {
    gffx_execution_context c;
    memset(&c, 0, sizeof(c));
    c.struct_size = (uint32_t)sizeof(c);
    c.abi_version = GFFX_ABI_VERSION;
    c.device_type = type;
    c.device_index = 0;
    return c;
}

static gffx_tensor_view view_of(
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
    v.device_index = 0;
    v.flags = flags;
    return v;
}

static int failures = 0;

static void report(const char *label, const void *a, const void *b, size_t bytes) {
    int same = memcmp(a, b, bytes) == 0;
    printf("  %-46s %s\n", label, same ? "bit-identical" : "DIFFERS");
    if (!same) {
        size_t i;
        const unsigned char *x = (const unsigned char *)a;
        const unsigned char *y = (const unsigned char *)b;
        for (i = 0; i < bytes; ++i) {
            if (x[i] != y[i]) { printf("      first differing byte at %u\n", (unsigned)i); break; }
        }
        ++failures;
    }
}

static int run_f64(void) {
    CUdevice device;
    CUcontext cu_context;
    CUdeviceptr d_texture, d_pyramid, d_offsets, d_coords, d_derivs, d_border, d_samples;
    double texture[H * W * C];
    double cpu_pyramid[TOTAL];
    double gpu_pyramid[TOTAL];
    int32_t cpu_offsets[LEVELS + 1];
    int32_t gpu_offsets[LEVELS + 1];
    double coords[N * 2];
    double derivs[N * 4];
    double border[C] = {-1.0, 0.25, 7.0};
    double cpu_samples[N * C];
    double gpu_samples[N * C];
    int64_t tex_shape[3] = {H, W, C};
    int64_t tex_stride[3] = {W * C, C, 1};
    int64_t pyr_shape[1] = {TOTAL};
    int64_t pyr_stride[1] = {1};
    int64_t off_shape[1] = {LEVELS + 1};
    int64_t off_stride[1] = {1};
    int64_t crd_shape[2] = {N, 2};
    int64_t crd_stride[2] = {2, 1};
    int64_t drv_shape[2] = {N, 4};
    int64_t drv_stride[2] = {4, 1};
    int64_t brd_shape[1] = {C};
    int64_t brd_stride[1] = {1};
    int64_t smp_shape[2] = {N, C};
    int64_t smp_stride[2] = {C, 1};
    gffx_execution_context cpu = context_for(GFFX_DEVICE_CPU);
    gffx_execution_context gpu = context_for(GFFX_DEVICE_CUDA);
    gffx_diagnostic_buffer diag;
    char message[512];
    gffx_status st;
    int i;
    unsigned int filters[2] = {GFFX_FILTER_NEAREST, GFFX_FILTER_BILINEAR};
    unsigned int mips[2] = {GFFX_MIP_NEAREST, GFFX_MIP_LINEAR};
    unsigned int wraps[4] = {GFFX_WRAP_REPEAT, GFFX_WRAP_CLAMP, GFFX_WRAP_MIRROR,
                             GFFX_WRAP_BORDER};
    int f, m, w;

    memset(&diag, 0, sizeof(diag));
    diag.struct_size = (uint32_t)sizeof(diag);
    diag.abi_version = GFFX_ABI_VERSION;
    diag.data = message;
    diag.capacity_bytes = sizeof(message);
    message[0] = 0;

    for (i = 0; i < H * W * C; ++i) texture[i] = 0.3 + 0.017 * (double)i;
    for (i = 0; i < N; ++i) {
        /* Deliberately outside [0,1] on both ends so every wrap mode is exercised. */
        coords[i * 2] = -0.4 + 0.031 * (double)i;
        coords[i * 2 + 1] = 1.3 - 0.028 * (double)i;
        derivs[i * 4] = 0.004 * (double)(i % 7);
        derivs[i * 4 + 1] = 0.001 * (double)(i % 3);
        derivs[i * 4 + 2] = 0.002 * (double)(i % 5);
        derivs[i * 4 + 3] = 0.006 * (double)(i % 11);
    }

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&cu_context, 0, device) != CUDA_SUCCESS) {
        printf("no usable CUDA device\n");
        return 2;
    }
    cuMemAlloc(&d_texture, sizeof(texture));
    cuMemAlloc(&d_pyramid, sizeof(cpu_pyramid));
    cuMemAlloc(&d_offsets, sizeof(cpu_offsets));
    cuMemAlloc(&d_coords, sizeof(coords));
    cuMemAlloc(&d_derivs, sizeof(derivs));
    cuMemAlloc(&d_border, sizeof(border));
    cuMemAlloc(&d_samples, sizeof(cpu_samples));
    cuMemcpyHtoD(d_texture, texture, sizeof(texture));
    cuMemcpyHtoD(d_coords, coords, sizeof(coords));
    cuMemcpyHtoD(d_derivs, derivs, sizeof(derivs));
    cuMemcpyHtoD(d_border, border, sizeof(border));

    {
        gffx_tensor_view tv = view_of(texture, GFFX_DTYPE_FLOAT64, 3u, tex_shape, tex_stride,
                                      GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view pv = view_of(cpu_pyramid, GFFX_DTYPE_FLOAT64, 1u, pyr_shape, pyr_stride,
                                      GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = view_of(cpu_offsets, GFFX_DTYPE_INT32, 1u, off_shape, off_stride,
                                      GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        st = gffx_render_texture_pyramid(&tv, 0, &cpu, &pv, &ov, NULL, &diag);
        printf("cpu pyramid status=%d %s\n", (int)st, message[0] ? message : "");
        if (st != GFFX_STATUS_OK) return 1;
    }
    {
        gffx_tensor_view tv = view_of((void *)(uintptr_t)d_texture, GFFX_DTYPE_FLOAT64, 3u,
                                      tex_shape, tex_stride, GFFX_TENSOR_READ_ONLY,
                                      GFFX_DEVICE_CUDA);
        gffx_tensor_view pv = view_of((void *)(uintptr_t)d_pyramid, GFFX_DTYPE_FLOAT64, 1u,
                                      pyr_shape, pyr_stride, GFFX_TENSOR_OUTPUT,
                                      GFFX_DEVICE_CUDA);
        gffx_tensor_view ov = view_of((void *)(uintptr_t)d_offsets, GFFX_DTYPE_INT32, 1u,
                                      off_shape, off_stride, GFFX_TENSOR_OUTPUT,
                                      GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_render_texture_pyramid(&tv, 0, &gpu, &pv, &ov, NULL, &diag);
        printf("gpu pyramid status=%d %s\n", (int)st, message[0] ? message : "");
        if (st != GFFX_STATUS_OK) return 1;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_pyramid, d_pyramid, sizeof(gpu_pyramid));
    cuMemcpyDtoH(gpu_offsets, d_offsets, sizeof(gpu_offsets));
    printf("\npyramid:\n");
    report("level offsets", cpu_offsets, gpu_offsets, sizeof(cpu_offsets));
    report("level data (all %d levels)", cpu_pyramid,
           gpu_pyramid, (size_t)cpu_offsets[LEVELS] * sizeof(double));

    printf("\nsampler, every filter/mip/wrap combination:\n");
    for (f = 0; f < 2; ++f) {
        for (m = 0; m < 2; ++m) {
            for (w = 0; w < 4; ++w) {
                char label[128];
                int use_derivs;
                for (use_derivs = 0; use_derivs < 2; ++use_derivs) {
                    gffx_tensor_view pv = view_of(cpu_pyramid, GFFX_DTYPE_FLOAT64, 1u, pyr_shape,
                                                  pyr_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view ov = view_of(cpu_offsets, GFFX_DTYPE_INT32, 1u, off_shape,
                                                  off_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view cv = view_of(coords, GFFX_DTYPE_FLOAT64, 2u, crd_shape,
                                                  crd_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view dv = view_of(derivs, GFFX_DTYPE_FLOAT64, 2u, drv_shape,
                                                  drv_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view bv = view_of(border, GFFX_DTYPE_FLOAT64, 1u, brd_shape,
                                                  brd_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view sv = view_of(cpu_samples, GFFX_DTYPE_FLOAT64, 2u, smp_shape,
                                                  smp_stride, GFFX_TENSOR_OUTPUT,
                                                  GFFX_DEVICE_CPU);
                    message[0] = 0;
                    st = gffx_render_texture(&pv, &ov, H, W, &cv, use_derivs ? &dv : NULL, NULL,
                                             filters[f], mips[m], wraps[w], wraps[w], &bv,
                                             &cpu, &sv, NULL, &diag);
                    if (st != GFFX_STATUS_OK) {
                        printf("  cpu sample failed status=%d %s\n", (int)st, message);
                        return 1;
                    }
                }
                {
                    gffx_tensor_view pv = view_of((void *)(uintptr_t)d_pyramid,
                                                  GFFX_DTYPE_FLOAT64, 1u, pyr_shape, pyr_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view ov = view_of((void *)(uintptr_t)d_offsets, GFFX_DTYPE_INT32,
                                                  1u, off_shape, off_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view cv = view_of((void *)(uintptr_t)d_coords, GFFX_DTYPE_FLOAT64,
                                                  2u, crd_shape, crd_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view dv = view_of((void *)(uintptr_t)d_derivs, GFFX_DTYPE_FLOAT64,
                                                  2u, drv_shape, drv_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view bv = view_of((void *)(uintptr_t)d_border, GFFX_DTYPE_FLOAT64,
                                                  1u, brd_shape, brd_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view sv = view_of((void *)(uintptr_t)d_samples,
                                                  GFFX_DTYPE_FLOAT64, 2u, smp_shape, smp_stride,
                                                  GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
                    message[0] = 0;
                    st = gffx_render_texture(&pv, &ov, H, W, &cv, &dv, NULL, filters[f], mips[m],
                                             wraps[w], wraps[w], &bv, &gpu, &sv, NULL, &diag);
                    if (st != GFFX_STATUS_OK) {
                        printf("  gpu sample failed status=%d %s\n", (int)st, message);
                        return 1;
                    }
                }
                cuCtxSynchronize();
                cuMemcpyDtoH(gpu_samples, d_samples, sizeof(gpu_samples));
                sprintf(label, "filter=%s mip=%s wrap=%s (derivative lod)",
                        f ? "bilinear" : "nearest", m ? "linear" : "nearest",
                        w == 0 ? "repeat" : (w == 1 ? "clamp" : (w == 2 ? "mirror" : "border")));
                report(label, cpu_samples, gpu_samples, sizeof(cpu_samples));
            }
        }
    }
    printf("\n%d differing cases\n", failures);
    return failures != 0;
}
static int run_f32(void) {
    CUdevice device;
    CUcontext cu_context;
    CUdeviceptr d_texture, d_pyramid, d_offsets, d_coords, d_derivs, d_border, d_samples;
    float texture[H * W * C];
    float cpu_pyramid[TOTAL];
    float gpu_pyramid[TOTAL];
    int32_t cpu_offsets[LEVELS + 1];
    int32_t gpu_offsets[LEVELS + 1];
    float coords[N * 2];
    float derivs[N * 4];
    float border[C] = {-1.0, 0.25, 7.0};
    float cpu_samples[N * C];
    float gpu_samples[N * C];
    int64_t tex_shape[3] = {H, W, C};
    int64_t tex_stride[3] = {W * C, C, 1};
    int64_t pyr_shape[1] = {TOTAL};
    int64_t pyr_stride[1] = {1};
    int64_t off_shape[1] = {LEVELS + 1};
    int64_t off_stride[1] = {1};
    int64_t crd_shape[2] = {N, 2};
    int64_t crd_stride[2] = {2, 1};
    int64_t drv_shape[2] = {N, 4};
    int64_t drv_stride[2] = {4, 1};
    int64_t brd_shape[1] = {C};
    int64_t brd_stride[1] = {1};
    int64_t smp_shape[2] = {N, C};
    int64_t smp_stride[2] = {C, 1};
    gffx_execution_context cpu = context_for(GFFX_DEVICE_CPU);
    gffx_execution_context gpu = context_for(GFFX_DEVICE_CUDA);
    gffx_diagnostic_buffer diag;
    char message[512];
    gffx_status st;
    int i;
    unsigned int filters[2] = {GFFX_FILTER_NEAREST, GFFX_FILTER_BILINEAR};
    unsigned int mips[2] = {GFFX_MIP_NEAREST, GFFX_MIP_LINEAR};
    unsigned int wraps[4] = {GFFX_WRAP_REPEAT, GFFX_WRAP_CLAMP, GFFX_WRAP_MIRROR,
                             GFFX_WRAP_BORDER};
    int f, m, w;

    memset(&diag, 0, sizeof(diag));
    diag.struct_size = (uint32_t)sizeof(diag);
    diag.abi_version = GFFX_ABI_VERSION;
    diag.data = message;
    diag.capacity_bytes = sizeof(message);
    message[0] = 0;

    for (i = 0; i < H * W * C; ++i) texture[i] = 0.3 + 0.017 * (float)i;
    for (i = 0; i < N; ++i) {
        /* Deliberately outside [0,1] on both ends so every wrap mode is exercised. */
        coords[i * 2] = -0.4 + 0.031 * (float)i;
        coords[i * 2 + 1] = 1.3 - 0.028 * (float)i;
        derivs[i * 4] = 0.004 * (float)(i % 7);
        derivs[i * 4 + 1] = 0.001 * (float)(i % 3);
        derivs[i * 4 + 2] = 0.002 * (float)(i % 5);
        derivs[i * 4 + 3] = 0.006 * (float)(i % 11);
    }

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&cu_context, 0, device) != CUDA_SUCCESS) {
        printf("no usable CUDA device\n");
        return 2;
    }
    cuMemAlloc(&d_texture, sizeof(texture));
    cuMemAlloc(&d_pyramid, sizeof(cpu_pyramid));
    cuMemAlloc(&d_offsets, sizeof(cpu_offsets));
    cuMemAlloc(&d_coords, sizeof(coords));
    cuMemAlloc(&d_derivs, sizeof(derivs));
    cuMemAlloc(&d_border, sizeof(border));
    cuMemAlloc(&d_samples, sizeof(cpu_samples));
    cuMemcpyHtoD(d_texture, texture, sizeof(texture));
    cuMemcpyHtoD(d_coords, coords, sizeof(coords));
    cuMemcpyHtoD(d_derivs, derivs, sizeof(derivs));
    cuMemcpyHtoD(d_border, border, sizeof(border));

    {
        gffx_tensor_view tv = view_of(texture, GFFX_DTYPE_FLOAT32, 3u, tex_shape, tex_stride,
                                      GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CPU);
        gffx_tensor_view pv = view_of(cpu_pyramid, GFFX_DTYPE_FLOAT32, 1u, pyr_shape, pyr_stride,
                                      GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        gffx_tensor_view ov = view_of(cpu_offsets, GFFX_DTYPE_INT32, 1u, off_shape, off_stride,
                                      GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CPU);
        st = gffx_render_texture_pyramid(&tv, 0, &cpu, &pv, &ov, NULL, &diag);
        printf("cpu pyramid status=%d %s\n", (int)st, message[0] ? message : "");
        if (st != GFFX_STATUS_OK) return 1;
    }
    {
        gffx_tensor_view tv = view_of((void *)(uintptr_t)d_texture, GFFX_DTYPE_FLOAT32, 3u,
                                      tex_shape, tex_stride, GFFX_TENSOR_READ_ONLY,
                                      GFFX_DEVICE_CUDA);
        gffx_tensor_view pv = view_of((void *)(uintptr_t)d_pyramid, GFFX_DTYPE_FLOAT32, 1u,
                                      pyr_shape, pyr_stride, GFFX_TENSOR_OUTPUT,
                                      GFFX_DEVICE_CUDA);
        gffx_tensor_view ov = view_of((void *)(uintptr_t)d_offsets, GFFX_DTYPE_INT32, 1u,
                                      off_shape, off_stride, GFFX_TENSOR_OUTPUT,
                                      GFFX_DEVICE_CUDA);
        message[0] = 0;
        st = gffx_render_texture_pyramid(&tv, 0, &gpu, &pv, &ov, NULL, &diag);
        printf("gpu pyramid status=%d %s\n", (int)st, message[0] ? message : "");
        if (st != GFFX_STATUS_OK) return 1;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(gpu_pyramid, d_pyramid, sizeof(gpu_pyramid));
    cuMemcpyDtoH(gpu_offsets, d_offsets, sizeof(gpu_offsets));
    printf("\npyramid:\n");
    report("level offsets", cpu_offsets, gpu_offsets, sizeof(cpu_offsets));
    report("level data (all %d levels)", cpu_pyramid,
           gpu_pyramid, (size_t)cpu_offsets[LEVELS] * sizeof(float));

    printf("\nsampler, every filter/mip/wrap combination:\n");
    for (f = 0; f < 2; ++f) {
        for (m = 0; m < 2; ++m) {
            for (w = 0; w < 4; ++w) {
                char label[128];
                int use_derivs;
                for (use_derivs = 0; use_derivs < 2; ++use_derivs) {
                    gffx_tensor_view pv = view_of(cpu_pyramid, GFFX_DTYPE_FLOAT32, 1u, pyr_shape,
                                                  pyr_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view ov = view_of(cpu_offsets, GFFX_DTYPE_INT32, 1u, off_shape,
                                                  off_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view cv = view_of(coords, GFFX_DTYPE_FLOAT32, 2u, crd_shape,
                                                  crd_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view dv = view_of(derivs, GFFX_DTYPE_FLOAT32, 2u, drv_shape,
                                                  drv_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view bv = view_of(border, GFFX_DTYPE_FLOAT32, 1u, brd_shape,
                                                  brd_stride, GFFX_TENSOR_READ_ONLY,
                                                  GFFX_DEVICE_CPU);
                    gffx_tensor_view sv = view_of(cpu_samples, GFFX_DTYPE_FLOAT32, 2u, smp_shape,
                                                  smp_stride, GFFX_TENSOR_OUTPUT,
                                                  GFFX_DEVICE_CPU);
                    message[0] = 0;
                    st = gffx_render_texture(&pv, &ov, H, W, &cv, use_derivs ? &dv : NULL, NULL,
                                             filters[f], mips[m], wraps[w], wraps[w], &bv,
                                             &cpu, &sv, NULL, &diag);
                    if (st != GFFX_STATUS_OK) {
                        printf("  cpu sample failed status=%d %s\n", (int)st, message);
                        return 1;
                    }
                }
                {
                    gffx_tensor_view pv = view_of((void *)(uintptr_t)d_pyramid,
                                                  GFFX_DTYPE_FLOAT32, 1u, pyr_shape, pyr_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view ov = view_of((void *)(uintptr_t)d_offsets, GFFX_DTYPE_INT32,
                                                  1u, off_shape, off_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view cv = view_of((void *)(uintptr_t)d_coords, GFFX_DTYPE_FLOAT32,
                                                  2u, crd_shape, crd_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view dv = view_of((void *)(uintptr_t)d_derivs, GFFX_DTYPE_FLOAT32,
                                                  2u, drv_shape, drv_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view bv = view_of((void *)(uintptr_t)d_border, GFFX_DTYPE_FLOAT32,
                                                  1u, brd_shape, brd_stride,
                                                  GFFX_TENSOR_READ_ONLY, GFFX_DEVICE_CUDA);
                    gffx_tensor_view sv = view_of((void *)(uintptr_t)d_samples,
                                                  GFFX_DTYPE_FLOAT32, 2u, smp_shape, smp_stride,
                                                  GFFX_TENSOR_OUTPUT, GFFX_DEVICE_CUDA);
                    message[0] = 0;
                    st = gffx_render_texture(&pv, &ov, H, W, &cv, &dv, NULL, filters[f], mips[m],
                                             wraps[w], wraps[w], &bv, &gpu, &sv, NULL, &diag);
                    if (st != GFFX_STATUS_OK) {
                        printf("  gpu sample failed status=%d %s\n", (int)st, message);
                        return 1;
                    }
                }
                cuCtxSynchronize();
                cuMemcpyDtoH(gpu_samples, d_samples, sizeof(gpu_samples));
                sprintf(label, "filter=%s mip=%s wrap=%s (derivative lod)",
                        f ? "bilinear" : "nearest", m ? "linear" : "nearest",
                        w == 0 ? "repeat" : (w == 1 ? "clamp" : (w == 2 ? "mirror" : "border")));
                report(label, cpu_samples, gpu_samples, sizeof(cpu_samples));
            }
        }
    }
    printf("\n%d differing cases\n", failures);
    return failures != 0;
}

int main(void) {
    int f64_failures;
    int f32_failures;
    printf("=== float64 ===\n");
    f64_failures = run_f64();
    failures = 0;
    printf("\n=== float32 ===\n");
    f32_failures = run_f32();
    printf("\nfloat64 differing: %d, float32 differing: %d\n", f64_failures, f32_failures);
    return (f64_failures != 0 || f32_failures != 0) ? 1 : 0;
}
