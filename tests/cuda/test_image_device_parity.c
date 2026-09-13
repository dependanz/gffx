/* Public C transfer parity on a real device, independent of any framework.
 * CPU/CUDA byte identity is not an independent accuracy oracle: Decimal tests
 * qualify the mathematical envelope separately. No default-stream assumption.
 */
#include <gffx/image.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef union samples { double f64[32]; float f32[32]; } samples;
typedef struct descriptor { gffx_tensor_view view; int64_t shape[3], strides[3]; } descriptor;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "image device parity line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static float from32(uint32_t bits) { float x; memcpy(&x, &bits, sizeof(x)); return x; }
static double from64(uint64_t bits) { double x; memcpy(&x, &bits, sizeof(x)); return x; }
static uint32_t bits32(float x) { uint32_t bits; memcpy(&bits, &x, sizeof(bits)); return bits; }
static uint64_t bits64(double x) { uint64_t bits; memcpy(&bits, &x, sizeof(bits)); return bits; }

static void populate(samples *x, samples *g, gffx_dtype dtype) {
    int i;
    memset(x, 0, sizeof(*x)); memset(g, 0, sizeof(*g));
    for (i = 0; i < 32; ++i) {
        double value = (i - 8) / 16., gradient = (i % 9 - 4) / 8.;
        if (dtype == GFFX_DTYPE_FLOAT32) { x->f32[i] = (float)value; g->f32[i] = (float)gradient; }
        else { x->f64[i] = value; g->f64[i] = gradient; }
    }
    if (dtype == GFFX_DTYPE_FLOAT32) {
        uint32_t decode = bits32(.04045f), encode = bits32(.0031308f);
        x->f32[0] = from32(UINT32_C(0x80000000)); x->f32[1] = 0;
        x->f32[2] = -1; x->f32[3] = -1000;
        x->f32[4] = from32(1); x->f32[5] = from32(UINT32_C(0x80000001));
        x->f32[6] = from32(UINT32_C(0x007fffff)); x->f32[7] = from32(UINT32_C(0x00800000));
        for (i = 0; i < 3; ++i) { x->f32[8+i] = from32(decode + (uint32_t)i - 1); x->f32[11+i] = from32(encode + (uint32_t)i - 1); }
        x->f32[14] = .18f; x->f32[15] = 1; x->f32[16] = 4; x->f32[17] = 1000;
    } else {
        uint64_t decode = bits64(.04045), encode = bits64(.0031308);
        x->f64[0] = from64(UINT64_C(0x8000000000000000)); x->f64[1] = 0;
        x->f64[2] = -1; x->f64[3] = -1000;
        x->f64[4] = from64(1); x->f64[5] = from64(UINT64_C(0x8000000000000001));
        x->f64[6] = from64(UINT64_C(0x000fffffffffffff)); x->f64[7] = from64(UINT64_C(0x0010000000000000));
        for (i = 0; i < 3; ++i) { x->f64[8+i] = from64(decode + (uint64_t)i - 1); x->f64[11+i] = from64(encode + (uint64_t)i - 1); }
        x->f64[14] = .18; x->f64[15] = 1; x->f64[16] = 4; x->f64[17] = 1000;
    }
}

static void init(descriptor *d, void *data, gffx_dtype dtype, int shape_kind) {
    memset(d, 0, sizeof(*d));
    d->view.struct_size = sizeof(d->view); d->view.abi_version = GFFX_ABI_VERSION;
    d->view.data = data; d->view.dtype = dtype; d->view.flags = GFFX_TENSOR_READ_ONLY;
    d->view.device_type = GFFX_DEVICE_CPU;
    if (shape_kind == 0) { d->view.rank = 1; d->shape[0] = 32; d->strides[0] = 1; }
    else if (shape_kind == 2) { d->view.rank = 3; d->shape[0] = 2; d->shape[1] = 0; d->shape[2] = 3;
        d->strides[0] = 3; d->strides[1] = 3; d->strides[2] = 1; }
    /* Scalar descriptors deliberately have NULL shape/stride pointers. */
    if (d->view.rank) { d->view.shape = d->shape; d->view.strides = d->strides; }
}

static gffx_status call(int backward, uint32_t transfer, const gffx_tensor_view *x,
    const gffx_tensor_view *g, const gffx_execution_context *context,
    gffx_tensor_view *out, const gffx_buffer *scratch) {
    return backward ? gffx_image_transfer_backward(x, g, transfer, context, out, scratch, NULL)
                    : gffx_image_transfer(x, transfer, context, out, scratch, NULL);
}

static int read_equal(CUdeviceptr ptr, const samples *expected, size_t bytes, CUstream stream) {
    samples observed;
    CHECK(cuStreamSynchronize(stream) == CUDA_SUCCESS);
    CHECK(cuMemcpyDtoH(&observed, ptr, bytes) == CUDA_SUCCESS);
    CHECK(memcmp(&observed, expected, bytes) == 0);
    return 0;
}

static int fixture(gffx_dtype dtype, uint32_t transfer, int shape_kind, CUstream stream) {
    samples input, gradient, expected, sentinel, saved_input, saved_gradient;
    descriptor x, g, out;
    gffx_tensor_view dx, dg, dout;
    gffx_execution_context cpu = {0}, gpu;
    gffx_buffer scratch = {0}, bad;
    CUdeviceptr px = 0, pg = 0, po = 0, ps = 0;
    uint64_t cpu_bytes, cpu_alignment, bytes, alignment;
    int backward, repetition, failure;
    size_t item = dtype == GFFX_DTYPE_FLOAT32 ? sizeof(float) : sizeof(double);
    int64_t count = shape_kind == 0 ? 32 : shape_kind == 1 ? 1 : 0;
    size_t extent = (size_t)count * item;
    populate(&input, &gradient, dtype);
    if (shape_kind == 1) {
        if (dtype == GFFX_DTYPE_FLOAT32) { input.f32[0] = .18f; gradient.f32[0] = -.75f; }
        else { input.f64[0] = .18; gradient.f64[0] = -.75; }
    }
    saved_input = input; saved_gradient = gradient;
    memset(&sentinel, 0x5a, sizeof(sentinel));
    init(&x, count ? &input : NULL, dtype, shape_kind);
    init(&g, count ? &gradient : NULL, dtype, shape_kind);
    init(&out, count ? &expected : NULL, dtype, shape_kind); out.view.flags = GFFX_TENSOR_OUTPUT;
    cpu.struct_size = sizeof(cpu); cpu.abi_version = GFFX_ABI_VERSION; cpu.device_type = GFFX_DEVICE_CPU;
    gpu = cpu; gpu.device_type = GFFX_DEVICE_CUDA; gpu.stream = (void *)stream;
    CHECK(stream != NULL);
    CHECK(gffx_image_transfer_workspace(count, dtype, &cpu, &cpu_bytes, &cpu_alignment, NULL) == GFFX_STATUS_OK);
    CHECK(cpu_bytes == 0 && cpu_alignment == 1);
    CHECK(gffx_image_transfer_workspace(count, dtype, &gpu, &bytes, &alignment, NULL) == GFFX_STATUS_OK);
    CHECK(bytes > 0 && bytes <= 4096 && alignment > 0 && alignment <= 256);
    CHECK(cuMemAlloc(&px, sizeof(samples)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&pg, sizeof(samples)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&po, sizeof(samples)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&ps, bytes) == CUDA_SUCCESS);
    CHECK(cuMemcpyHtoD(px, &input, sizeof(input)) == CUDA_SUCCESS);
    CHECK(cuMemcpyHtoD(pg, &gradient, sizeof(gradient)) == CUDA_SUCCESS);
    scratch.struct_size = sizeof(scratch); scratch.abi_version = GFFX_ABI_VERSION;
    scratch.data = (void *)(uintptr_t)ps; scratch.capacity_bytes = bytes; scratch.alignment = alignment;
    scratch.device_type = GFFX_DEVICE_CUDA;
    dx = x.view; dx.data = count ? (void *)(uintptr_t)px : NULL; dx.device_type = GFFX_DEVICE_CUDA;
    dg = g.view; dg.data = count ? (void *)(uintptr_t)pg : NULL; dg.device_type = GFFX_DEVICE_CUDA;
    dout = out.view; dout.data = count ? (void *)(uintptr_t)po : NULL; dout.device_type = GFFX_DEVICE_CUDA;
    for (backward = 0; backward < 2; ++backward) {
        expected = sentinel;
        CHECK(call(backward, transfer, &x.view, &g.view, &cpu, &out.view, NULL) == GFFX_STATUS_OK);
        for (repetition = 0; repetition < 3; ++repetition) {
            CHECK(cuMemcpyHtoD(po, &sentinel, sizeof(sentinel)) == CUDA_SUCCESS);
            CHECK(call(backward, transfer, &dx, &dg, &gpu, &dout, &scratch) == GFFX_STATUS_OK);
            CHECK(read_equal(po, &expected, extent ? extent : sizeof(sentinel), stream) == 0);
        }
    }
    if (shape_kind == 0) for (backward = 0; backward < 2; ++backward) {
        /* Input NaN/+inf/-inf, and each nonfinite cotangent for backward.
         * Last-element corruption catches implementations that write a prefix
         * before completing validation. CPU and GPU preserve sentinels. */
        for (failure = 0; failure < (backward ? 6 : 3); ++failure) {
            samples *changed = failure < 3 ? &input : &gradient;
            int kind = failure % 3;
            input = saved_input; gradient = saved_gradient;
            if (dtype == GFFX_DTYPE_FLOAT32) changed->f32[31] = from32(kind == 0 ? UINT32_C(0x7fc00001) : kind == 1 ? UINT32_C(0x7f800000) : UINT32_C(0xff800000));
            else changed->f64[31] = from64(kind == 0 ? UINT64_C(0x7ff8000000000001) : kind == 1 ? UINT64_C(0x7ff0000000000000) : UINT64_C(0xfff0000000000000));
            expected = sentinel;
            CHECK(call(backward, transfer, &x.view, &g.view, &cpu, &out.view, NULL) == GFFX_STATUS_INVALID_ARGUMENT);
            CHECK(memcmp(&expected, &sentinel, sizeof(expected)) == 0);
            CHECK(cuMemcpyHtoD(px, &input, sizeof(input)) == CUDA_SUCCESS);
            CHECK(cuMemcpyHtoD(pg, &gradient, sizeof(gradient)) == CUDA_SUCCESS);
            CHECK(cuMemcpyHtoD(po, &sentinel, sizeof(sentinel)) == CUDA_SUCCESS);
            CHECK(call(backward, transfer, &dx, &dg, &gpu, &dout, &scratch) == GFFX_STATUS_INVALID_ARGUMENT);
            CHECK(read_equal(po, &sentinel, sizeof(sentinel), stream) == 0);
        }
        input = saved_input; gradient = saved_gradient;
        CHECK(cuMemcpyHtoD(px, &input, sizeof(input)) == CUDA_SUCCESS);
        CHECK(cuMemcpyHtoD(pg, &gradient, sizeof(gradient)) == CUDA_SUCCESS);
        for (failure = 0; failure < 5; ++failure) {
            gffx_tensor_view alias = dout;
            bad = scratch;
            CHECK(cuMemcpyHtoD(po, &sentinel, sizeof(sentinel)) == CUDA_SUCCESS);
            if (failure == 0) bad.capacity_bytes = bytes - 1;
            else if (failure == 1) bad.data = (void *)(uintptr_t)px;
            else if (failure == 2) bad.data = (void *)(uintptr_t)po;
            else if (failure == 3) alias.data = (void *)(uintptr_t)px;
            else alias.data = (void *)(uintptr_t)pg;
            /* A forward output may overlap an unused cotangent, so only the
             * backward operation has the fifth alias restriction. */
            if (failure == 4 && !backward) continue;
            CHECK(call(backward, transfer, &dx, &dg, &gpu, &alias, &bad) ==
                  (failure == 0 ? GFFX_STATUS_INSUFFICIENT_WORKSPACE : GFFX_STATUS_INVALID_ARGUMENT));
            CHECK(read_equal(po, &sentinel, sizeof(sentinel), stream) == 0);
            CHECK(read_equal(px, &saved_input, sizeof(saved_input), stream) == 0);
            CHECK(read_equal(pg, &saved_gradient, sizeof(saved_gradient), stream) == 0);
        }
    }
    CHECK(cuMemFree(px) == CUDA_SUCCESS); CHECK(cuMemFree(pg) == CUDA_SUCCESS);
    CHECK(cuMemFree(po) == CUDA_SUCCESS); CHECK(cuMemFree(ps) == CUDA_SUCCESS);
    return 0;
}

int main(int argc, char **argv) {
    CUdevice device; CUcontext context; CUstream stream;
    int devices = 0, transfer, shape_kind;
    CHECK(argc == 2);
#if defined(_WIN32)
    CHECK(_putenv_s("GFFX_CUDA_PLUGIN_PATH", argv[1]) == 0);
#else
    CHECK(setenv("GFFX_CUDA_PLUGIN_PATH", argv[1], 1) == 0);
#endif
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&devices) != CUDA_SUCCESS || devices == 0) return 77;
    CHECK(cuDeviceGet(&device, 0) == CUDA_SUCCESS);
    CHECK(cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS);
    CHECK(cuCtxSetCurrent(context) == CUDA_SUCCESS);
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
    for (transfer = 0; transfer < 2; ++transfer) for (shape_kind = 0; shape_kind < 3; ++shape_kind) {
        CHECK(fixture(GFFX_DTYPE_FLOAT32, (uint32_t)transfer, shape_kind, stream) == 0);
        CHECK(fixture(GFFX_DTYPE_FLOAT64, (uint32_t)transfer, shape_kind, stream) == 0);
    }
    CHECK(cuStreamDestroy(stream) == CUDA_SUCCESS); CHECK(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);
    CHECK(cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS);
    return 0;
}
