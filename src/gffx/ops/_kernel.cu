#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__
void vec_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C"
int add_vectors_cuda(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    cudaError_t err;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    size_t num_bytes = static_cast<size_t>(n) * sizeof(float);

    printf("test invoke\n");

    err = cudaMalloc(&d_a, num_bytes);
    if (err) printf("CUDA malloc d_a failed\n"); return -1;
    err = cudaMalloc(&d_b, num_bytes);
    if (err) printf("CUDA malloc d_b failed\n"); return -1;
    err = cudaMalloc(&d_out, num_bytes);
    if (err) printf("CUDA malloc d_out failed\n"); return -1;

    cudaMemcpy(d_a, a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, num_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);

    cudaMemcpy(out, d_out, num_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}