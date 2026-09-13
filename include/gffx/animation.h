#ifndef GFFX_ANIMATION_H
#define GFFX_ANIMATION_H

#include <gffx/execution.h>
#include <gffx/tensor.h>

/* Dense animation schemas specialize the original packed mesh primitives.
 * All inputs are dense same-device/same-floating-dtype; no implicit copies.
 * CPU implementation first; CUDA reports unsupported until separately qualified.
 */
GFFX_EXTERN_C_BEGIN

GFFX_API gffx_status GFFX_CALL gffx_skeleton_forward_kinematics_workspace(
    int64_t B,int64_t J,gffx_dtype dtype,const gffx_execution_context *context,
    uint64_t *required_bytes,uint64_t *required_alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_skeleton_forward_kinematics(
    const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_execution_context *context,
    gffx_tensor_view *world,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_skeleton_forward_kinematics_backward(
    const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_tensor_view *grad_world,
    const gffx_execution_context *context,gffx_tensor_view *grad_local,const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic);

GFFX_API gffx_status GFFX_CALL gffx_mesh_skin_lbs_workspace(
    int64_t B,int64_t V,int64_t J,gffx_dtype dtype,const gffx_execution_context *context,
    uint64_t *required_bytes,uint64_t *required_alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_mesh_skin_lbs(
    const gffx_tensor_view *vertices,const gffx_tensor_view *transforms,const gffx_tensor_view *weights,
    const gffx_execution_context *context,gffx_tensor_view *output,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_mesh_skin_lbs_backward(
    const gffx_tensor_view *vertices,const gffx_tensor_view *transforms,const gffx_tensor_view *weights,
    const gffx_tensor_view *grad_output,const gffx_execution_context *context,
    gffx_tensor_view *grad_vertices,gffx_tensor_view *grad_transforms,gffx_tensor_view *grad_weights,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_mesh_evaluate_barycentric_workspace(
    int64_t B,int64_t V,int64_t N,gffx_dtype dtype,const gffx_execution_context *context,
    uint64_t *required_bytes,uint64_t *required_alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_mesh_evaluate_barycentric(
    const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,const gffx_execution_context *context,gffx_tensor_view *output,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_mesh_evaluate_barycentric_backward(
    const gffx_tensor_view *vertices,const gffx_tensor_view *faces,const gffx_tensor_view *face_index,
    const gffx_tensor_view *barycentric,const gffx_tensor_view *grad_output,
    const gffx_execution_context *context,gffx_tensor_view *grad_vertices,gffx_tensor_view *grad_barycentric,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);

GFFX_API gffx_status GFFX_CALL gffx_transforms_rodrigues_workspace(
    int64_t point_count, gffx_dtype dtype, const gffx_execution_context *context,
    uint64_t *required_bytes, uint64_t *required_alignment, gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_transforms_rodrigues(
    const gffx_tensor_view *axis_angle, const gffx_execution_context *context,
    gffx_tensor_view *rotation, const gffx_buffer *workspace, gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_transforms_rodrigues_backward(
    const gffx_tensor_view *axis_angle, const gffx_tensor_view *grad_rotation,
    const gffx_execution_context *context, gffx_tensor_view *grad_axis_angle,
    const gffx_buffer *workspace, gffx_diagnostic_buffer *diagnostic);

GFFX_API gffx_status GFFX_CALL gffx_mesh_blend_shapes_workspace(
    int64_t batch_count, int64_t vertex_count, int64_t shape_count, gffx_dtype dtype,
    const gffx_execution_context *context, uint64_t *required_bytes,
    uint64_t *required_alignment, gffx_diagnostic_buffer *diagnostic);

GFFX_API gffx_status GFFX_CALL gffx_mesh_blend_shapes(
    const gffx_tensor_view *template_vertices, const gffx_tensor_view *deltas,
    const gffx_tensor_view *weights, const gffx_execution_context *context,
    gffx_tensor_view *vertices, const gffx_buffer *workspace,
    gffx_diagnostic_buffer *diagnostic);

/* Any nonempty subset of gradient outputs may be requested. Each is overwritten,
 * not accumulated into caller storage. Shared reductions have fixed index order. */
GFFX_API gffx_status GFFX_CALL gffx_mesh_blend_shapes_backward(
    const gffx_tensor_view *template_vertices, const gffx_tensor_view *deltas,
    const gffx_tensor_view *weights, const gffx_tensor_view *grad_vertices,
    const gffx_execution_context *context, gffx_tensor_view *grad_template,
    gffx_tensor_view *grad_deltas, gffx_tensor_view *grad_weights,
    const gffx_buffer *workspace, gffx_diagnostic_buffer *diagnostic);

GFFX_EXTERN_C_END
#endif
