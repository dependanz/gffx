#ifndef GFFX_CUDA_PLUGIN_API_H
#define GFFX_CUDA_PLUGIN_API_H

#include <gffx/capabilities.h>
#include <gffx/execution.h>
#include <gffx/tensor.h>
#include <stddef.h>

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
typedef gffx_status (GFFX_CALL *gffx_cuda_render_texture_pyramid_fn)(
    const gffx_tensor_view *, int64_t, const gffx_execution_context *,
    gffx_tensor_view *, gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
);

typedef gffx_status (GFFX_CALL *gffx_cuda_render_texture_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, int64_t, int64_t,
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    uint32_t, uint32_t, uint32_t, uint32_t, const gffx_tensor_view *,
    const gffx_execution_context *, gffx_tensor_view *, const gffx_buffer *,
    gffx_diagnostic_buffer *
);

typedef gffx_status (GFFX_CALL *gffx_cuda_render_texture_pyramid_backward_fn)(
    const gffx_tensor_view *, int64_t, int64_t, int64_t, const gffx_tensor_view *,
    const gffx_execution_context *, gffx_tensor_view *, const gffx_buffer *,
    gffx_diagnostic_buffer *
);

typedef gffx_status (GFFX_CALL *gffx_cuda_render_texture_backward_fn)(
    const gffx_tensor_view *, const gffx_tensor_view *, int64_t, int64_t,
    const gffx_tensor_view *, const gffx_tensor_view *, const gffx_tensor_view *,
    uint32_t, uint32_t, uint32_t, uint32_t, const gffx_tensor_view *,
    const gffx_tensor_view *, const gffx_execution_context *, gffx_tensor_view *,
    gffx_tensor_view *, const gffx_buffer *, gffx_diagnostic_buffer *
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
#define GFFX_CUDA_OP_RENDER_TEXTURE_PYRAMID UINT32_C(12)
#define GFFX_CUDA_OP_RENDER_TEXTURE UINT32_C(13)
#define GFFX_CUDA_OP_MESH_BLEND_SHAPES UINT32_C(14)
#define GFFX_CUDA_OP_TRANSFORMS_RODRIGUES UINT32_C(15)
#define GFFX_CUDA_OP_SKELETON_FORWARD_KINEMATICS UINT32_C(16)
#define GFFX_CUDA_OP_MESH_SKIN_LBS UINT32_C(17)
#define GFFX_CUDA_OP_MESH_EVALUATE_BARYCENTRIC UINT32_C(18)
#define GFFX_CUDA_OP_IMAGE_TRANSFER UINT32_C(19)
#define GFFX_CUDA_OP_IMAGE_GAUSSIAN_WEIGHTS UINT32_C(20)
#define GFFX_CUDA_OP_SHADING_LOSS_CURVE UINT32_C(21)
#define GFFX_CUDA_OP_SHADING_LOSS_QUERY UINT32_C(22)
#define GFFX_CUDA_OP_SHADING_PRINCIPLED UINT32_C(23)
#define GFFX_CUDA_OP_SHADING_PRINCIPLED_GROUPS UINT32_C(24)
#define GFFX_CUDA_OP_RENDER_OPAQUE_QUADRATURE UINT32_C(25)

typedef gffx_status (GFFX_CALL *gffx_cuda_animation_three_backward_fn)(
    const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,
    const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,gffx_tensor_view *,
    const gffx_buffer *,gffx_diagnostic_buffer *);
typedef gffx_status (GFFX_CALL *gffx_cuda_animation_one_fn)(
    const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,
    const gffx_buffer *,gffx_diagnostic_buffer *);
typedef gffx_status (GFFX_CALL *gffx_cuda_animation_four_fn)(
    const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,
    const gffx_execution_context *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
typedef gffx_status (GFFX_CALL *gffx_cuda_animation_bary_backward_fn)(
    const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,
    const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);

typedef gffx_status (GFFX_CALL *gffx_cuda_image_transfer_fn)(
    const gffx_tensor_view *,uint32_t,const gffx_execution_context *,
    gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
typedef gffx_status (GFFX_CALL *gffx_cuda_image_transfer_backward_fn)(
    const gffx_tensor_view *,const gffx_tensor_view *,uint32_t,
    const gffx_execution_context *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);

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
typedef gffx_status (GFFX_CALL *gffx_cuda_loss_curve_fn)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
typedef gffx_status (GFFX_CALL *gffx_cuda_loss_curve_backward_fn)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
typedef gffx_status (GFFX_CALL *gffx_cuda_loss_query_fn)(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *weights,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *cosine,const gffx_tensor_view *owners,const gffx_execution_context *context,gffx_tensor_view *query,gffx_tensor_view *mean,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
typedef gffx_status (GFFX_CALL *gffx_cuda_loss_query_backward_fn)(const gffx_tensor_view *query,const gffx_tensor_view *mean,const gffx_tensor_view *owners,const gffx_tensor_view *gradient,const gffx_tensor_view *mean_gradient,const gffx_execution_context *context,gffx_tensor_view *grad_u,gffx_tensor_view *grad_cosine,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);

typedef gffx_status (GFFX_CALL *gffx_cuda_principled_pairs_fn)(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *normal,const gffx_tensor_view *view,const gffx_tensor_view *light,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,uint32_t options,const gffx_execution_context *context,gffx_tensor_view *value,gffx_tensor_view *jacobian,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
typedef gffx_status (GFFX_CALL *gffx_cuda_principled_pairs_backward_fn)(const gffx_tensor_view *jacobian,const gffx_tensor_view *gradient,const gffx_execution_context *context,gffx_tensor_view *point_gradient,gffx_tensor_view *light_gradient,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
typedef gffx_status (GFFX_CALL *gffx_cuda_render_rasterize_window_fn)(
    const gffx_tensor_view *ndc_vertices,
    const gffx_tensor_view *faces,
    const gffx_tensor_view *vertex_offsets,
    const gffx_tensor_view *face_offsets,
    int64_t global_height,
    int64_t global_width,
    int64_t row_offset,
    int64_t column_offset,
    int64_t image_height,
    int64_t image_width,
    int64_t faces_per_pixel,
    double blur_radius_px,
    uint32_t cull_mode,
    double eps,
    const gffx_execution_context *context,
    gffx_tensor_view *face_index,
    gffx_tensor_view *barycentric,
    gffx_tensor_view *depth,
    gffx_tensor_view *signed_distance,
    const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic
);
typedef gffx_status (GFFX_CALL *gffx_cuda_opaque_quadrature_fn)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,int64_t,int64_t,double,double,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
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
    gffx_cuda_render_texture_pyramid_fn render_texture_pyramid;
    gffx_cuda_render_texture_fn render_texture;
    gffx_cuda_render_texture_backward_fn render_texture_backward;
    gffx_cuda_render_texture_pyramid_backward_fn render_texture_pyramid_backward;

    uint64_t reserved[4];
    /* Append-only tail: older providers end immediately before these fields. */
    gffx_cuda_render_interpolate_fn mesh_blend_shapes;
    gffx_cuda_animation_three_backward_fn mesh_blend_shapes_backward;
    gffx_cuda_animation_one_fn transforms_rodrigues;
    gffx_cuda_mesh_gather_faces_fn transforms_rodrigues_backward;
    gffx_cuda_mesh_gather_faces_fn skeleton_forward_kinematics;
    gffx_cuda_render_interpolate_fn skeleton_forward_kinematics_backward;
    gffx_cuda_render_interpolate_fn mesh_skin_lbs;
    gffx_cuda_animation_three_backward_fn mesh_skin_lbs_backward;
    gffx_cuda_animation_four_fn mesh_evaluate_barycentric;
    gffx_cuda_animation_bary_backward_fn mesh_evaluate_barycentric_backward;
    gffx_cuda_image_transfer_fn image_transfer;
    gffx_cuda_image_transfer_backward_fn image_transfer_backward;
    /* Additive Gaussian tail, same signature shape but a distinct operation. */
    gffx_cuda_image_transfer_fn image_gaussian_weights;
    gffx_cuda_image_transfer_backward_fn image_gaussian_weights_backward;
    gffx_cuda_loss_curve_fn shading_loss_curve;
    gffx_cuda_loss_curve_backward_fn shading_loss_curve_backward;
    gffx_cuda_loss_query_fn shading_loss_query;
    gffx_cuda_loss_query_backward_fn shading_loss_query_backward;
    gffx_cuda_principled_pairs_fn shading_principled_pairs;
    gffx_cuda_principled_pairs_backward_fn shading_principled_pairs_backward;
    /* Append-only renderer grouping; original pair fields retain their bounds. */
    gffx_cuda_principled_pairs_fn shading_principled_groups;
    gffx_cuda_principled_pairs_backward_fn shading_principled_groups_backward;
    /* Append-only forward global-window extension. */
    gffx_cuda_render_rasterize_window_fn render_rasterize_window;
    gffx_cuda_opaque_quadrature_fn render_opaque_quadrature;
    /* Appended provider tail: old providers remain valid, boundary calls refuse. */
    gffx_status (GFFX_CALL *render_opaque_boundary)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,int64_t,int64_t,double,double,int32_t,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    /* Append-only explicit material-integral transport; raw quadrature unchanged. */
    gffx_cuda_opaque_quadrature_fn render_opaque_material;
    /* Append-only bounded single-pixel continuation. Metadata arrays are host-owned. */
    gffx_status (GFFX_CALL *render_opaque_batch)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    /* Append-only shared immutable curve row; legacy grouped shapes unchanged. */
    gffx_cuda_principled_pairs_fn shading_principled_shared_groups;
    /* Append-only ordered sample carry, no changed legacy operation. */
    gffx_status (GFFX_CALL *shading_ordered_rgba)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *shading_unit_newton)(const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *shading_unit_vectors)(const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *shading_delta_brdf)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *shading_opaque_samples)(const gffx_tensor_view *const *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *shading_opaque_albedo)(const gffx_tensor_view *const *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *render_opaque_batch_rule)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t);
    gffx_status (GFFX_CALL *opaque_partition)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,gffx_tensor_view *,int32_t *,int,int,gffx_tensor_view *,gffx_tensor_view *,const gffx_execution_context *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *opaque_prefix_backward)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t *);
    gffx_status (GFFX_CALL *opaque_partition64)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,gffx_tensor_view *,int32_t *,int,int,gffx_tensor_view *,gffx_tensor_view *,const gffx_execution_context *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *opaque_prefix_backward64)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t *);
    gffx_status (GFFX_CALL *render_opaque_batch_rule64)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t);
    gffx_status (GFFX_CALL *render_opaque_boundary_batch64)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t);
    gffx_status (GFFX_CALL *opaque_partition80)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,gffx_tensor_view *,int32_t *,int,int,gffx_tensor_view *,gffx_tensor_view *,const gffx_execution_context *,const gffx_buffer *,gffx_diagnostic_buffer *);
    gffx_status (GFFX_CALL *opaque_prefix_backward80)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t *);
    gffx_status (GFFX_CALL *render_opaque_batch_rule80)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t);
    gffx_status (GFFX_CALL *render_opaque_boundary_batch80)(const gffx_tensor_view *,const gffx_tensor_view *,const gffx_tensor_view *,double,double,const int32_t *,int32_t *,const gffx_tensor_view *,const gffx_execution_context *,gffx_tensor_view *,gffx_tensor_view *,const gffx_buffer *,gffx_diagnostic_buffer *,int32_t,int32_t);
} gffx_cuda_operations;

/* Short-circuit the size check before touching a potentially absent tail field. */
#define GFFX_CUDA_HAS_OPERATION(table, field) \
    ((table) != NULL && (table)->struct_size >= offsetof(gffx_cuda_operations, field) + sizeof((table)->field) && (table)->field != NULL)

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
