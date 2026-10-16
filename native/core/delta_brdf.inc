#include "delta_brdf_scalar.h"
GFFX_API gffx_status GFFX_CALL gffx_shading_delta_brdf_workspace(int64_t p,int64_t q,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
 gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 if(p<0||p>128||q<0||q>8||!bytes||!alignment)return pc_fail(d,"delta BRDF count/results invalid");
 if(c->device_type==GFFX_DEVICE_CUDA&&!GFFX_CUDA_HAS_OPERATION(gffx_cuda_loader_operations(),shading_delta_brdf))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks delta BRDF");
 *bytes=8;*alignment=8;return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_delta_brdf(const gffx_tensor_view *normal,const gffx_tensor_view *brdf,const gffx_tensor_view *direction,const gffx_tensor_view *incoming,const gffx_tensor_view *cotangent,const gffx_execution_context *c,gffx_tensor_view *value,gffx_tensor_view *gradient,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 const gffx_tensor_view *views[7]={normal,brdf,direction,incoming,cotangent,value,gradient};int64_t p,q,j,shape[3];uint64_t bytes,alignment;const double *n,*b,*l,*i,*g;double *v,*out;gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 s=gffx_validate_tensor_view(brdf,d);if(s!=GFFX_STATUS_OK)return s;if(brdf->rank!=3||brdf->shape[2]!=3)return pc_fail(d,"delta BRDF shape invalid");p=brdf->shape[0];q=brdf->shape[1];
 s=gffx_shading_delta_brdf_workspace(p,q,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=p;shape[1]=3;s=pc_view(normal,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(cotangent,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(value,2,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[1]=q;shape[2]=3;s=pc_view(brdf,3,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(direction,3,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(incoming,3,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(gradient,3,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;
 s=pc_storage(views,7,2,w,bytes,c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_delta_brdf(normal,brdf,direction,incoming,cotangent,c,value,gradient,w,d);
 n=gffx_mesh_elements_const(normal);b=gffx_mesh_elements_const(brdf);l=gffx_mesh_elements_const(direction);i=gffx_mesh_elements_const(incoming);g=gffx_mesh_elements_const(cotangent);v=gffx_mesh_elements(value);out=gffx_mesh_elements(gradient);
 for(j=0;j<p;++j)if(gffx_dl_domain(n+3*j,(q?b+3*q*j:b),(q?l+3*q*j:l),(q?i+3*q*j:i),g+3*j,(int)q))return pc_fail(d,"delta BRDF input domain");
 for(j=0;j<p;++j)if(gffx_dl_point(n+3*j,(q?b+3*q*j:b),(q?l+3*q*j:l),(q?i+3*q*j:i),g+3*j,(int)q,v+3*j,(q?out+3*q*j:out)))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"delta BRDF arithmetic overflow");
 return GFFX_STATUS_OK;
}
