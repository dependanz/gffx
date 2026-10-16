#ifndef GFFX_SHADING_H
#define GFFX_SHADING_H
#include <gffx/execution.h>
#include <gffx/tensor.h>
/* Immutable float64 log axis[K] and roughness controls[B,2*K+1,4], u[B].
 * B=0..32, K=1..512. Output value/partial[B,10*K-6] = nodal pairs,
 * horizon pair, angular controls. Only u has a first derivative.
 * Axis finite strictly increasing <=0; controls [-745,0], u [0,1].
 * Dense, matching device/index. Outputs/workspace disjoint from all inputs
 * and one another; inputs may alias. Domain failures preserve outputs;
 * arithmetic overflow invalidates outputs. Caller owns scratch and stream.
 * VJP is fixed adjacent-pair/carry-odd sum(partial*gradient), first order only.
 * No material admission, resource identity or coefficient gradient is implied. */
GFFX_EXTERN_C_BEGIN
/* Dense float64 normal/cotangent/value[P,3], BRDF/direction/irradiance/gradient[P,Q,3].
 * P=0..128,Q=0..8. Finite nonzero normal/direction; finite nonnegative BRDF and
 * irradiance; finite signed cotangent. Exact unit normalization and positive
 * cosine integration; adjacent-pair/carry-odd sample sum and BRDF-only VJP.
 * No direction/normal/light VJP implied. Outputs/scratch disjoint from inputs
 * and one another. Domain errors preserve outputs; arithmetic errors invalidate.
 * Eight bytes caller scratch; matching device/index and explicit stream. */
/* Bounded opaque constant-material sample composition: <=16 pixels,800 slots,
  128-sample cohorts. Primal only; callers own scratch and retain replay VJPs. */
GFFX_API gffx_status GFFX_CALL gffx_shading_opaque_samples_workspace(int64_t pixels,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_opaque_samples(const gffx_tensor_view *owners,const gffx_tensor_view *q,const gffx_tensor_view *world,const gffx_tensor_view *faces,const gffx_tensor_view *clip,const gffx_tensor_view *normals,const gffx_tensor_view *camera,const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,const gffx_tensor_view *direction,const gffx_tensor_view *incoming,const gffx_execution_context *context,gffx_tensor_view *weighted,gffx_tensor_view *rgba,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
/* Bounded sample value and constant-RGB-albedo VJP; full material Jacobian. */
GFFX_API gffx_status GFFX_CALL gffx_shading_opaque_albedo_workspace(int64_t pixels,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_opaque_albedo(const gffx_tensor_view *owners,const gffx_tensor_view *q,const gffx_tensor_view *world,const gffx_tensor_view *faces,const gffx_tensor_view *clip,const gffx_tensor_view *normals,const gffx_tensor_view *camera,const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,const gffx_tensor_view *direction,const gffx_tensor_view *incoming,const gffx_tensor_view *cotangent,const gffx_execution_context *context,gffx_tensor_view *weighted,gffx_tensor_view *rgba,gffx_tensor_view *gradient,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_delta_brdf_workspace(int64_t points,int64_t samples,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_delta_brdf(const gffx_tensor_view *normal,const gffx_tensor_view *brdf,const gffx_tensor_view *direction,const gffx_tensor_view *irradiance,const gffx_tensor_view *cotangent,const gffx_execution_context *context,gffx_tensor_view *value,gffx_tensor_view *gradient,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);

/* Dense float64 input/output[N,3],N=0..1048576. Finite nonzero vectors.
 * Exact max-absolute scaling, rounded square/left-associated three-term sum,
 * six unit-Newton steps and final divisions. No norm or sqrt substitution.
 * Matching device/index;output and8-byte scratch disjoint;domain failure atomic.
 * Forward-only primitive. Gradient-bearing compositions retain their own VJP. */
GFFX_API gffx_status GFFX_CALL gffx_shading_unit_vectors_workspace(
 int64_t count,const gffx_execution_context *context,uint64_t *bytes,
 uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_unit_vectors(
 const gffx_tensor_view *input,const gffx_execution_context *context,
 gffx_tensor_view *output,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);

/* Float64 dense input/output[N], N=0..1048576. Finite input in (0,4].
 * Exactly six separately rounded y=(y+x/y)*0.5 steps from y=1.5.
 * Normalization error proof applies only to [1,3]; not general sqrt accuracy.
 * Matching device/index; output and 8-byte scratch disjoint from inputs.
 * All input/domain failures precede writes. Autograd belongs to the adapter. */
GFFX_API gffx_status GFFX_CALL gffx_shading_unit_newton_workspace(
 int64_t count,const gffx_execution_context *context,uint64_t *bytes,
 uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_unit_newton(
 const gffx_tensor_view *input,const gffx_execution_context *context,
 gffx_tensor_view *output,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);

/* Renderer-internal ascending RGBA carry. Float64 carry[4], values[N,4],
 * N=0..800, output[4]. Each channel starts from carry and adds every sample
 * in ascending order with separately rounded additions, including zeros.
 * No reassociation, clipping or small-value cutoff. Inputs finite; overflow
 * invalidates outputs. Dense matching device; output/scratch disjoint.
 * No automatic differentiation: the scene replay owns its original VJP. */
GFFX_API gffx_status GFFX_CALL gffx_shading_ordered_rgba_workspace(
 int64_t count,const gffx_execution_context *context,uint64_t *bytes,
 uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_ordered_rgba(
 const gffx_tensor_view *carry,const gffx_tensor_view *values,
 const gffx_execution_context *context,gffx_tensor_view *output,
 const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_curve_workspace(
 int64_t batch,int64_t nodes,const gffx_execution_context *context,
 uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_curve(
 const gffx_tensor_view *axis,const gffx_tensor_view *controls,const gffx_tensor_view *u,
 const gffx_execution_context *context,gffx_tensor_view *value,gffx_tensor_view *partial,
 const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_curve_backward(
 const gffx_tensor_view *partial,const gffx_tensor_view *gradient,
 const gffx_execution_context *context,gffx_tensor_view *grad_u,
 const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
/* Raw immutable-resource query bridge: K=1..512, B=0..32, Q=0..1056.
 * cn[K], logs[K], weights[K-1,4], curve/partial[B,10K-6], cosine[Q],
 * int32 owners[Q]. query[Q,6]=values/du/dcos, mean[B,4]=values/du.
 * Only coordinate/cosine first VJP. See source design for exact domain/order.
 * Same dense/device, alias, pre-write domain and overflow rules as above. */
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_query_workspace(int64_t batch,int64_t nodes,int64_t queries,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_query(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *weights,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *cosine,const gffx_tensor_view *owners,const gffx_execution_context *context,gffx_tensor_view *query,gffx_tensor_view *mean,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_query_backward(const gffx_tensor_view *query,const gffx_tensor_view *mean,const gffx_tensor_view *owners,const gffx_tensor_view *gradient,const gffx_tensor_view *mean_gradient,const gffx_execution_context *context,gffx_tensor_view *grad_u,gffx_tensor_view *grad_cosine,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
/* Bounded Principled pairs: options bit0 two-sided, bit1 first derivatives.
 * Points<=8 forward/4 with derivatives, points*lights<=1024. Every tensor is
 * dense float64 on the context device. Jacobian[P,Q,3,16], or empty[0] without
 * derivatives. VJP returns point[P,16] (light columns zero) and light[P,Q,3].
 * Raw curves/means are immutable caller-supplied resources, not admission.
 * Domain errors preserve outputs; overflow invalidates them. Disjoint outputs
 * and caller scratch; exact layouts and reduction order in native source design. */
GFFX_API gffx_status GFFX_CALL gffx_shading_principled_pairs_workspace(int64_t points,int64_t lights,uint32_t derivatives,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_principled_pairs(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *normal,const gffx_tensor_view *view,const gffx_tensor_view *light,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,uint32_t options,const gffx_execution_context *context,gffx_tensor_view *value,gffx_tensor_view *jacobian,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_principled_pairs_backward(const gffx_tensor_view *jacobian,const gffx_tensor_view *gradient,const gffx_execution_context *context,gffx_tensor_view *point_gradient,gffx_tensor_view *light_gradient,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
/* Renderer-internal grouped boundary: P<=128, ordered groups of four plus tail;
 * P*Q<=1024. Identical shapes, ownership, arithmetic and status rules above.
 * Does not change admission of the original pairs family. */
GFFX_API gffx_status GFFX_CALL gffx_shading_principled_groups_workspace(int64_t points,int64_t lights,uint32_t derivatives,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_principled_groups(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *normal,const gffx_tensor_view *view,const gffx_tensor_view *light,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,uint32_t options,const gffx_execution_context *context,gffx_tensor_view *value,gffx_tensor_view *jacobian,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_API gffx_status GFFX_CALL gffx_shading_principled_groups_backward(const gffx_tensor_view *jacobian,const gffx_tensor_view *gradient,const gffx_execution_context *context,gffx_tensor_view *point_gradient,gffx_tensor_view *light_gradient,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
/* Same finite grouped family,with exactly ONE dense curve/partial/mean/width row.
 * All point/light fields,full Jacobians,VJP order and limits remain unchanged.
 * Use groups_workspace and groups_backward; old CUDA providers explicitly refuse. */
GFFX_API gffx_status GFFX_CALL gffx_shading_principled_shared_groups(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *normal,const gffx_tensor_view *view,const gffx_tensor_view *light,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,uint32_t options,const gffx_execution_context *context,gffx_tensor_view *value,gffx_tensor_view *jacobian,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic);
GFFX_EXTERN_C_END
#endif
