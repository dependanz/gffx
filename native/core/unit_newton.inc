#include "unit_newton_scalar.h"
GFFX_API gffx_status GFFX_CALL gffx_shading_unit_newton_workspace(int64_t n,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
 gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 if(n<0||n>1048576||!bytes||!alignment)return pc_fail(d,"unit Newton count/results invalid");
 if(c->device_type==GFFX_DEVICE_CUDA&&!GFFX_CUDA_HAS_OPERATION(gffx_cuda_loader_operations(),shading_unit_newton))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks unit Newton");
 *bytes=8;*alignment=8;return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_unit_newton(const gffx_tensor_view *input,const gffx_execution_context *c,gffx_tensor_view *output,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 const gffx_tensor_view *views[2]={input,output};int64_t n,j;uint64_t bytes,alignment;const double *x;double *y;gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 s=gffx_validate_tensor_view(input,d);if(s!=GFFX_STATUS_OK)return s;if(input->rank!=1)return pc_fail(d,"unit Newton input rank invalid");n=input->shape[0];
 s=gffx_shading_unit_newton_workspace(n,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
 s=pc_view(input,1,&n,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(output,1,&n,1,c,d);if(s!=GFFX_STATUS_OK)return s;
 s=pc_storage(views,2,1,w,bytes,c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_unit_newton(input,c,output,w,d);
 x=gffx_mesh_elements_const(input);y=gffx_mesh_elements(output);
 for(j=0;j<n;++j)if(!isfinite(x[j])||x[j]<=0.||x[j]>4.)return pc_fail(d,"unit Newton input must be finite in (0,4]");
 for(j=0;j<n;++j)y[j]=gffx_unit_newton_scalar(x[j]);return GFFX_STATUS_OK;
}
