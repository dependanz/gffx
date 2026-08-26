/*
 * CUDA device code is deliberately absent from Phase 1 Step 9.
 *
 * The isolated gffx_cuda12 host plugin is written in C and uses only the CUDA Driver API. Future
 * functional kernels will be compiled as CUDA device artifacts and loaded through that API; they
 * must not introduce a CUDA Runtime dependency. A load/enumeration probe is infrastructure
 * evidence only and advertises no graphics or geometry operation.
 */
