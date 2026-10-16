#ifndef GFFX_IMAGE_H
#define GFFX_IMAGE_H
#include <gffx/execution.h>
#include <gffx/tensor.h>

#define GFFX_IMAGE_SRGB_TO_LINEAR UINT32_C(0)
#define GFFX_IMAGE_LINEAR_TO_SRGB UINT32_C(1)

/* Arbitrary dense shape (including scalar/empty), matching floating dtype/device.
 * Signed/HDR channel values are not clipped. Threshold equality uses the linear
 * branch in the input dtype. Outputs may not alias inputs or caller workspace.
 * Structural/domain errors preserve output; arithmetic-overflow output is invalid.
 * Workspace is caller-owned: query the selected device, never assume CPU's zero.
 * First VJP only; this family is not a generic power function or image I/O API. */
GFFX_EXTERN_C_BEGIN
GFFX_API gffx_status GFFX_CALL gffx_image_transfer_workspace(
    int64_t count,gffx_dtype dtype,const gffx_execution_context *context,
    uint64_t *required_bytes,uint64_t *required_alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_image_transfer(
    const gffx_tensor_view *input,uint32_t transfer,const gffx_execution_context *context,
    gffx_tensor_view *output,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_image_transfer_backward(
    const gffx_tensor_view *input,const gffx_tensor_view *grad_output,uint32_t transfer,
    const gffx_execution_context *context,gffx_tensor_view *grad_input,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
/* Normalized finite Gaussian taps: sigma scalar or [B], radius 0..32,
 * represented sigma in [.1,32], float32/float64. Output [K] or [B,K], K=2R+1.
 * First VJP only, checked fixed pair reductions. Structural/domain failures
 * preserve outputs; arithmetic overflow invalidates output. Query caller scratch. */
GFFX_API gffx_status GFFX_CALL gffx_image_gaussian_weights_workspace(
    int64_t batch_count,gffx_dtype dtype,uint32_t radius,const gffx_execution_context *context,
    uint64_t *required_bytes,uint64_t *required_alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_image_gaussian_weights(
    const gffx_tensor_view *sigma,uint32_t radius,const gffx_execution_context *context,
    gffx_tensor_view *weights,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_image_gaussian_weights_backward(
    const gffx_tensor_view *sigma,const gffx_tensor_view *grad_weights,uint32_t radius,
    const gffx_execution_context *context,gffx_tensor_view *grad_sigma,
    const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_EXTERN_C_END
#endif
