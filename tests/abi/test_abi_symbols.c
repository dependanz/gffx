#include <stddef.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static const char *required_symbols[] = {
    "gffx_get_abi_version",
    "gffx_validate_tensor_view",
    "gffx_validate_execution_context",
    "gffx_validate_buffer",
    "gffx_capabilities_query",
    "gffx_capabilities_probe",
    /* Phase 2 operation exports: compatible additions per the ABI contract. */
    "gffx_mesh_face_geometry",
    "gffx_mesh_face_geometry_backward",
    "gffx_mesh_face_geometry_workspace",
    "gffx_mesh_vertex_normals",
    "gffx_mesh_vertex_normals_backward",
    "gffx_mesh_vertex_normals_workspace",
    "gffx_mesh_gather_faces",
    "gffx_mesh_gather_faces_backward",
    "gffx_mesh_gather_faces_workspace",
    "gffx_transforms_transform_points",
    "gffx_transforms_transform_points_backward",
    "gffx_transforms_transform_points_workspace",
    "gffx_transforms_perspective_divide",
    "gffx_transforms_perspective_divide_backward",
    "gffx_transforms_perspective_divide_workspace",
    /* Topology construction is nondifferentiable, so it exports no backward symbol. */
    "gffx_mesh_build_edge_topology",
    "gffx_mesh_build_edge_topology_workspace",
    "gffx_points_knn",
    "gffx_points_knn_backward",
    "gffx_points_knn_workspace",
    "gffx_points_closest_point_on_mesh",
    "gffx_points_closest_point_on_mesh_backward",
    "gffx_points_closest_point_on_mesh_workspace"
};

int main(int argc, char **argv) {
    size_t index;
    if (argc != 2) return 1;
#if defined(_WIN32)
    {
        HMODULE library = LoadLibraryA(argv[1]);
        if (library == NULL) return 2;
        for (index = 0u; index < sizeof(required_symbols) / sizeof(required_symbols[0]); ++index) {
            if (GetProcAddress(library, required_symbols[index]) == NULL) {
                FreeLibrary(library);
                return (int)(10u + index);
            }
        }
        FreeLibrary(library);
    }
#else
    {
        void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (library == NULL) return 2;
        for (index = 0u; index < sizeof(required_symbols) / sizeof(required_symbols[0]); ++index) {
            if (dlsym(library, required_symbols[index]) == NULL) {
                dlclose(library);
                return (int)(10u + index);
            }
        }
        dlclose(library);
    }
#endif
    return 0;
}
