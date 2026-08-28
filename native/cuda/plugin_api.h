#ifndef GFFX_CUDA_PLUGIN_API_H
#define GFFX_CUDA_PLUGIN_API_H

#include <gffx/capabilities.h>
#include <gffx/execution.h>
#include <gffx/tensor.h>

/* Private host/plugin ABI. This header is intentionally not installed as public API. */
#define GFFX_CUDA_PLUGIN_ABI_VERSION_MAJOR UINT32_C(1)
#define GFFX_CUDA_PLUGIN_ABI_VERSION_MINOR UINT32_C(0)
#define GFFX_CUDA_PLUGIN_ABI_VERSION \
    GFFX_ABI_VERSION_ENCODE( \
        GFFX_CUDA_PLUGIN_ABI_VERSION_MAJOR, \
        GFFX_CUDA_PLUGIN_ABI_VERSION_MINOR \
    )

#define GFFX_CUDA_PLUGIN_HANDSHAKE_SYMBOL "gffx_cuda_plugin_handshake_v1"
#define GFFX_CUDA_PLUGIN_FLAG_CAPABILITY_PROVIDER UINT64_C(1)
/* Set when the plugin publishes an operation table. Its presence says a table exists, never that
 * any particular operation in it is implemented; that is per-entry and is a NULL check. */
#define GFFX_CUDA_PLUGIN_FLAG_OPERATION_PROVIDER UINT64_C(2)

#if defined(_WIN32)
#if defined(GFFX_BUILDING_CUDA_PLUGIN)
#define GFFX_CUDA_PLUGIN_API __declspec(dllexport)
#else
#define GFFX_CUDA_PLUGIN_API __declspec(dllimport)
#endif
#else
#define GFFX_CUDA_PLUGIN_API __attribute__((visibility("default")))
#endif

typedef gffx_status (GFFX_CALL *gffx_cuda_plugin_capabilities_fn)(
    uint32_t probe_flags,
    gffx_capability_report *report,
    gffx_diagnostic_buffer *diagnostic
);

/*
 * Operation entry points.
 *
 * Each signature is identical to the public C entry point of the same name, so the host forwards a
 * call rather than translating it, and a CUDA implementation is the same function with a different
 * body. The caller's stream arrives inside gffx_execution_context, which already carries a stream
 * field, so operation dispatch introduces no new plumbing for stream ownership.
 *
 * A NULL entry means the plugin does not implement that operation, and the host reports
 * GFFX_STATUS_UNSUPPORTED rather than falling back to the CPU kernel. A silent fallback would turn
 * a missing GPU kernel into a device-to-host copy and a performance cliff with no diagnostic,
 * which is the class of surprise the execution-state contract exists to prevent. A caller who
 * wants the CPU path can ask for the CPU device.
 *
 * Parameter names are omitted deliberately: these mirror public signatures that are documented
 * once, in the public headers, and repeating the names here would create a second place for them
 * to drift.
 */
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_face_geometry_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, double,
    const gffx_execution_context *, gffx_tensor_view *, gffx_tensor_view *, gffx_tensor_view *,
    const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_face_geometry_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, double, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_execution_context *, gffx_tensor_view *,
    const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_vertex_normals_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, double, uint32_t,
    const gffx_execution_context *, gffx_tensor_view *, const gffx_buffer *,
    gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_vertex_normals_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, double, uint32_t,
    const gffx_tensor_view *, const gffx_execution_context *, gffx_tensor_view *,
    const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_gather_faces_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_execution_context *,
    gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_gather_faces_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_execution_context *, gffx_tensor_view *, const gffx_buffer *,
    gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_transforms_transform_points_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_execution_context *, gffx_tensor_view *, const gffx_buffer *,
    gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_transforms_transform_points_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_execution_context *, gffx_tensor_view *,
    gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_transforms_perspective_divide_fn)(
    const gffx_tensor_view *, double, const gffx_execution_context *, gffx_tensor_view *,
    gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_transforms_perspective_divide_backward_fn)(
    const gffx_tensor_view *, double, const gffx_tensor_view *, const gffx_execution_context *,
    gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_build_edge_topology_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_execution_context *,
    gffx_tensor_view *, gffx_tensor_view *, gffx_tensor_view *, gffx_tensor_view *,
    const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_points_knn_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, int64_t, const gffx_execution_context *, gffx_tensor_view *,
    gffx_tensor_view *, gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_points_knn_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_execution_context *,
    gffx_tensor_view *, gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_points_closest_point_on_mesh_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *, double,
    const gffx_execution_context *, gffx_tensor_view *, gffx_tensor_view *, gffx_tensor_view *,
    gffx_tensor_view *, gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_points_closest_point_on_mesh_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_execution_context *,
    gffx_tensor_view *, gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_sample_surface_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, int64_t, const gffx_tensor_view *, const gffx_tensor_view *,
    double, const gffx_execution_context *, gffx_tensor_view *, gffx_tensor_view *,
    gffx_tensor_view *, gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_mesh_sample_surface_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_execution_context *, gffx_tensor_view *,
    const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_render_rasterize_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, int64_t, int64_t, int64_t, double, uint32_t, double,
    const gffx_execution_context *, gffx_tensor_view *, gffx_tensor_view *, gffx_tensor_view *,
    gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_render_rasterize_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, int64_t, int64_t,
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_execution_context *, gffx_tensor_view *,
    const gffx_buffer *, gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_render_interpolate_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_execution_context *, gffx_tensor_view *, const gffx_buffer *,
    gffx_diagnostic_buffer *
);
typedef gffx_status (GFFX_CALL *gffx_cuda_render_interpolate_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_execution_context *, gffx_tensor_view *,
    gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);

/* Identifies an operation in the generic workspace query. Values are stable once assigned; a new
 * operation appends rather than renumbering, because a plugin and a host may be built apart. */
#define GFFX_CUDA_OP_MESH_FACE_GEOMETRY UINT32_C(1)
#define GFFX_CUDA_OP_MESH_VERTEX_NORMALS UINT32_C(2)
#define GFFX_CUDA_OP_MESH_GATHER_FACES UINT32_C(3)
#define GFFX_CUDA_OP_TRANSFORMS_TRANSFORM_POINTS UINT32_C(4)
#define GFFX_CUDA_OP_TRANSFORMS_PERSPECTIVE_DIVIDE UINT32_C(5)
#define GFFX_CUDA_OP_MESH_BUILD_EDGE_TOPOLOGY UINT32_C(6)
#define GFFX_CUDA_OP_POINTS_KNN UINT32_C(7)
#define GFFX_CUDA_OP_POINTS_CLOSEST_POINT_ON_MESH UINT32_C(8)
#define GFFX_CUDA_OP_MESH_SAMPLE_SURFACE UINT32_C(9)
#define GFFX_CUDA_OP_RENDER_RASTERIZE UINT32_C(10)
#define GFFX_CUDA_OP_RENDER_INTERPOLATE UINT32_C(11)

/*
 * Device-side workspace requirement.
 *
 * One query serves every operation, taking an identifier and the shape vector that operation's own
 * query takes, because the alternative is twenty-two near-identical function pointers whose only
 * difference is their argument list. The device answer is not generally the host answer: a CUDA
 * implementation may need per-block temporaries where the scalar CPU reference needs none, so the
 * host asks the plugin rather than reusing the CPU figure.
 *
 * A NULL query alongside an implemented operation is an error rather than an implied zero.
 */
typedef gffx_status (GFFX_CALL *gffx_cuda_workspace_fn)(
    uint32_t operation,
    const int64_t *shape,
    uint32_t shape_count,
    gffx_dtype dtype,
    const gffx_execution_context *context,
    uint64_t *required_bytes,
    uint64_t *required_alignment,
    gffx_diagnostic_buffer *diagnostic
);

/*
 * The operation table.
 *
 * Separately size-prefixed from gffx_cuda_plugin_api so operations can be added without disturbing
 * the outer handshake struct, and so a host reading it applies the same offsetof-based check it
 * already applies to the outer one. Entries are NULL until implemented.
 */
typedef struct gffx_cuda_operations {
    uint32_t struct_size;
    uint32_t reserved0;

    gffx_cuda_workspace_fn workspace_query;

    gffx_cuda_mesh_face_geometry_fn mesh_face_geometry;
    gffx_cuda_mesh_face_geometry_backward_fn mesh_face_geometry_backward;
    gffx_cuda_mesh_vertex_normals_fn mesh_vertex_normals;
    gffx_cuda_mesh_vertex_normals_backward_fn mesh_vertex_normals_backward;
    gffx_cuda_mesh_gather_faces_fn mesh_gather_faces;
    gffx_cuda_mesh_gather_faces_backward_fn mesh_gather_faces_backward;
    gffx_cuda_transforms_transform_points_fn transforms_transform_points;
    gffx_cuda_transforms_transform_points_backward_fn transforms_transform_points_backward;
    gffx_cuda_transforms_perspective_divide_fn transforms_perspective_divide;
    gffx_cuda_transforms_perspective_divide_backward_fn transforms_perspective_divide_backward;
    gffx_cuda_mesh_build_edge_topology_fn mesh_build_edge_topology;
    gffx_cuda_points_knn_fn points_knn;
    gffx_cuda_points_knn_backward_fn points_knn_backward;
    gffx_cuda_points_closest_point_on_mesh_fn points_closest_point_on_mesh;
    gffx_cuda_points_closest_point_on_mesh_backward_fn points_closest_point_on_mesh_backward;
    gffx_cuda_mesh_sample_surface_fn mesh_sample_surface;
    gffx_cuda_mesh_sample_surface_backward_fn mesh_sample_surface_backward;
    gffx_cuda_render_rasterize_fn render_rasterize;
    gffx_cuda_render_rasterize_backward_fn render_rasterize_backward;
    gffx_cuda_render_interpolate_fn render_interpolate;
    gffx_cuda_render_interpolate_backward_fn render_interpolate_backward;

    uint64_t reserved[4];
} gffx_cuda_operations;

/*
 * The handshake struct.
 *
 * operations joined plugin ABI v1 rather than prompting a v2. The plugin ABI is private, is not
 * installed as public API, and has never shipped, so there is no compatibility to preserve; the
 * reserved tail exists precisely so v1 can grow before release freezes it. The field consumes one
 * reserved slot rather than extending the struct, so sizeof is unchanged and every existing size
 * check keeps passing unaltered.
 */
typedef struct gffx_cuda_plugin_api {
    uint32_t struct_size;
    uint32_t plugin_abi_version;
    uint32_t core_abi_min;
    uint32_t core_abi_max;
    uint64_t flags;
    const char *build_identity;
    gffx_cuda_plugin_capabilities_fn capabilities_probe;
    const gffx_cuda_operations *operations;
    uint64_t reserved[5];
} gffx_cuda_plugin_api;

typedef gffx_status (GFFX_CALL *gffx_cuda_plugin_handshake_fn)(
    uint32_t requested_plugin_abi,
    uint32_t host_core_abi,
    gffx_cuda_plugin_api *api,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_BEGIN

GFFX_CUDA_PLUGIN_API gffx_status GFFX_CALL gffx_cuda_plugin_handshake_v1(
    uint32_t requested_plugin_abi,
    uint32_t host_core_abi,
    gffx_cuda_plugin_api *api,
    gffx_diagnostic_buffer *diagnostic
);

GFFX_EXTERN_C_END

#endif
