/*
 * Calls a CUDA operation through the plugin's operation table.
 *
 * This deliberately bypasses the public C entry points. Those still reject any non-CPU context in
 * mesh_common.h, and the host publishes no accessor for the negotiated table, both of which are
 * open items recorded in the project plan. Waiting for them would mean the kernel stayed untested
 * until two unrelated decisions were settled, so this exercises the layer that does exist: load the
 * plugin, negotiate the handshake, take the operation table, and launch.
 *
 * What that proves is the dispatch path end to end, which is what the ABI extension was for. What
 * it does not prove is that a user can reach it; that is what device routing and the host accessor
 * add, and this test will be joined by a conformance test through the public API once they land.
 *
 * The test is skipped rather than failed when no CUDA device or plugin is present, because a
 * machine without a GPU is a legitimate development environment for every other part of GFFX.
 */

#include <cuda.h>

#include "plugin_api.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE library_handle;
static library_handle open_library(const char *path) { return LoadLibraryA(path); }
static void *library_symbol(library_handle h, const char *n) {
    return (void *)(uintptr_t)GetProcAddress(h, n);
}
#else
#include <dlfcn.h>
typedef void *library_handle;
static library_handle open_library(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void *library_symbol(library_handle h, const char *n) { return dlsym(h, n); }
#endif

#define CHECK(condition) do { if (!(condition)) { \
    printf("FAILED: %s (line %d)\n", #condition, __LINE__); return 1; } } while (0)

/* A unit tetrahedron. Three faces have doubled area 1 and the fourth sqrt(3), so the areas are
 * 0.5, 0.5, 0.5 and sqrt(3)/2, which the CPU suite already pins. */
static const double VERTICES[12] = {
    0.0, 0.0, 0.0,   1.0, 0.0, 0.0,   0.0, 1.0, 0.0,   0.0, 0.0, 1.0
};
static const int FACES[12] = {0, 2, 1,  0, 1, 3,  0, 3, 2,  1, 2, 3};

static gffx_tensor_view device_view(
    CUdeviceptr pointer, gffx_dtype dtype, uint32_t rank, const int64_t *shape,
    const int64_t *strides, uint32_t flags
) {
    gffx_tensor_view view;
    memset(&view, 0, sizeof(view));
    view.struct_size = (uint32_t)sizeof(view);
    view.abi_version = GFFX_ABI_VERSION;
    view.data = (void *)(uintptr_t)pointer;
    view.rank = rank;
    view.shape = shape;
    view.strides = strides;
    view.dtype = dtype;
    view.device_type = GFFX_DEVICE_CUDA;
    view.device_index = 0;
    view.flags = flags;
    return view;
}

int main(int argc, char **argv) {
    static const int64_t pair_strides[2] = {3, 1};
    static const int64_t scalar_strides[1] = {1};
    int64_t vertex_shape[2] = {4, 3};
    int64_t face_shape[2] = {4, 3};
    int64_t scalar_shape[1] = {4};

    library_handle plugin;
    gffx_cuda_plugin_handshake_fn handshake;
    gffx_cuda_plugin_api api;
    gffx_diagnostic_buffer diagnostic;
    char message[512];
    gffx_execution_context context;
    gffx_tensor_view vertices_view, faces_view, normals_view, areas_view, valid_view;
    CUdevice device;
    CUcontext cuda_context;
    CUdeviceptr d_vertices, d_faces, d_normals, d_areas, d_valid, d_workspace;
    gffx_buffer workspace;
    uint64_t workspace_bytes = 0, workspace_alignment = 0;
    int bad_faces[12];
    double normals[12], areas[4];
    unsigned char valid[4];
    int index;
    int device_count = 0;

    if (argc != 2) {
        printf("usage: %s <plugin path>\n", argv[0]);
        return 2;
    }

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&device_count) != CUDA_SUCCESS ||
        device_count == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }

    plugin = open_library(argv[1]);
    if (plugin == NULL) {
        printf("SKIP: plugin not loadable at %s\n", argv[1]);
        return 0;
    }
    handshake = (gffx_cuda_plugin_handshake_fn)library_symbol(
        plugin, GFFX_CUDA_PLUGIN_HANDSHAKE_SYMBOL);
    CHECK(handshake != NULL);

    memset(&diagnostic, 0, sizeof(diagnostic));
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    diagnostic.abi_version = GFFX_ABI_VERSION;
    diagnostic.data = message;
    diagnostic.capacity_bytes = (uint64_t)sizeof(message);

    memset(&api, 0, sizeof(api));
    api.struct_size = (uint32_t)sizeof(api);
    CHECK(handshake(GFFX_CUDA_PLUGIN_ABI_VERSION, GFFX_ABI_VERSION, &api, &diagnostic)
          == GFFX_STATUS_OK);

    /* The dispatch path the ABI extension added. */
    CHECK((api.flags & GFFX_CUDA_PLUGIN_FLAG_OPERATION_PROVIDER) != 0u);
    CHECK(api.operations != NULL);
    CHECK(api.operations->struct_size >= sizeof(*api.operations));
    CHECK(api.operations->mesh_face_geometry != NULL);
    CHECK(api.operations->workspace_query != NULL);
    /* An unimplemented entry stays NULL rather than pointing at a stub that would report success. */
    CHECK(api.operations->mesh_face_geometry_backward == NULL);

    CHECK(cuDeviceGet(&device, 0) == CUDA_SUCCESS);
    CHECK(cuCtxCreate(&cuda_context, 0, device) == CUDA_SUCCESS);

    CHECK(cuMemAlloc(&d_vertices, sizeof(VERTICES)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&d_faces, sizeof(FACES)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&d_normals, sizeof(normals)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&d_areas, sizeof(areas)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&d_valid, sizeof(valid)) == CUDA_SUCCESS);
    CHECK(cuMemcpyHtoD(d_vertices, VERTICES, sizeof(VERTICES)) == CUDA_SUCCESS);
    CHECK(cuMemcpyHtoD(d_faces, FACES, sizeof(FACES)) == CUDA_SUCCESS);

    memset(&context, 0, sizeof(context));
    context.struct_size = (uint32_t)sizeof(context);
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CUDA;
    context.device_index = 0;
    context.stream = NULL;   /* the default stream, which is still the caller's choice */

    /* The device workspace requirement is nonzero where the CPU reference needs none, because the
     * per-call index check has to run on the device. Asking the plugin rather than assuming is the
     * reason the ABI carries a per-backend query at all. */
    CHECK(api.operations->workspace_query(
              GFFX_CUDA_OP_MESH_FACE_GEOMETRY, NULL, 0u, GFFX_DTYPE_FLOAT64, &context,
              &workspace_bytes, &workspace_alignment, &diagnostic) == GFFX_STATUS_OK);
    CHECK(workspace_bytes >= sizeof(int));
    CHECK(cuMemAlloc(&d_workspace, (size_t)workspace_bytes) == CUDA_SUCCESS);
    memset(&workspace, 0, sizeof(workspace));
    workspace.struct_size = (uint32_t)sizeof(workspace);
    workspace.abi_version = GFFX_ABI_VERSION;
    workspace.data = (void *)(uintptr_t)d_workspace;
    workspace.capacity_bytes = workspace_bytes;
    workspace.alignment = workspace_alignment;
    workspace.device_type = GFFX_DEVICE_CUDA;

    vertices_view = device_view(d_vertices, GFFX_DTYPE_FLOAT64, 2u, vertex_shape, pair_strides,
                                GFFX_TENSOR_READ_ONLY);
    faces_view = device_view(d_faces, GFFX_DTYPE_INT32, 2u, face_shape, pair_strides,
                             GFFX_TENSOR_READ_ONLY);
    normals_view = device_view(d_normals, GFFX_DTYPE_FLOAT64, 2u, face_shape, pair_strides,
                               GFFX_TENSOR_OUTPUT);
    areas_view = device_view(d_areas, GFFX_DTYPE_FLOAT64, 1u, scalar_shape, scalar_strides,
                             GFFX_TENSOR_OUTPUT);
    valid_view = device_view(d_valid, GFFX_DTYPE_BOOL, 1u, scalar_shape, scalar_strides,
                             GFFX_TENSOR_OUTPUT);

    message[0] = '\0';
    CHECK(api.operations->mesh_face_geometry(
              &vertices_view, &faces_view, 9.5367431640625e-7, &context, &normals_view,
              &areas_view, &valid_view, &workspace, &diagnostic) == GFFX_STATUS_OK);
    CHECK(cuCtxSynchronize() == CUDA_SUCCESS);

    CHECK(cuMemcpyDtoH(normals, d_normals, sizeof(normals)) == CUDA_SUCCESS);
    CHECK(cuMemcpyDtoH(areas, d_areas, sizeof(areas)) == CUDA_SUCCESS);
    CHECK(cuMemcpyDtoH(valid, d_valid, sizeof(valid)) == CUDA_SUCCESS);

    /* The same values the CPU acceptance fixtures pin for this mesh. Exact rather than approximate
     * for the three axis-aligned faces, whose areas are representable; the equilateral face is
     * compared against sqrt(3)/2 within a tolerance because its value is irrational. */
    for (index = 0; index < 3; ++index) {
        CHECK(areas[index] == 0.5);
        CHECK(valid[index] == 1u);
    }
    CHECK(fabs(areas[3] - 0.8660254037844386) < 1e-15);
    CHECK(valid[3] == 1u);
    for (index = 0; index < 12; ++index) {
        CHECK(normals[index] == normals[index]);   /* no NaN reached the output */
    }
    /* Face 0 lies in the z = 0 plane wound so its normal points along -z. */
    CHECK(normals[0] == 0.0 && normals[1] == 0.0 && normals[2] == -1.0);

    /* A face index outside [0, V) must be refused. The host cannot see device-resident indices,
     * so this is the device-side check doing the work the CPU path does by dereferencing, and it
     * is the evidence that moving the check rather than dropping it actually holds. */
    memcpy(bad_faces, FACES, sizeof(FACES));
    bad_faces[5] = 99;
    CHECK(cuMemcpyHtoD(d_faces, bad_faces, sizeof(bad_faces)) == CUDA_SUCCESS);
    CHECK(cuMemsetD8(d_areas, 0xA5, sizeof(areas)) == CUDA_SUCCESS);
    message[0] = '\0';
    CHECK(api.operations->mesh_face_geometry(
              &vertices_view, &faces_view, 9.5367431640625e-7, &context, &normals_view,
              &areas_view, &valid_view, &workspace, &diagnostic) == GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(message[0] != '\0');
    CHECK(cuCtxSynchronize() == CUDA_SUCCESS);
    /* And no output was written: the operation kernel never launched, so the sentinel survives. */
    CHECK(cuMemcpyDtoH(areas, d_areas, sizeof(areas)) == CUDA_SUCCESS);
    {
        const unsigned char *bytes = (const unsigned char *)areas;
        CHECK(bytes[0] == 0xA5u);
    }

    /* A workspace too small for the status word is refused rather than silently skipping the
     * check, which would turn the contract's guarantee into a suggestion. */
    workspace.capacity_bytes = 0;
    CHECK(cuMemcpyHtoD(d_faces, FACES, sizeof(FACES)) == CUDA_SUCCESS);
    CHECK(api.operations->mesh_face_geometry(
              &vertices_view, &faces_view, 9.5367431640625e-7, &context, &normals_view,
              &areas_view, &valid_view, &workspace, &diagnostic)
          == GFFX_STATUS_INSUFFICIENT_WORKSPACE);

    cuMemFree(d_vertices); cuMemFree(d_faces); cuMemFree(d_normals);
    cuMemFree(d_areas); cuMemFree(d_valid); cuMemFree(d_workspace);
    cuCtxDestroy(cuda_context);
    printf("CUDA face_geometry: dispatched, index-validated on device, and workspace-checked\n");
    return 0;
}
