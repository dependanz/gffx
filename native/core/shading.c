/* Framework-neutral, allocation-free bounded Principled curve family. */
#include <gffx/shading.h>
#include "internal.h"
#include "mesh_common.h"
#include "cuda_loader.h"
#include "principled_curve_scalar.h"
static gffx_status pc_fail(gffx_diagnostic_buffer *d,const char *m){return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,m);}
static gffx_status pc_context(const gffx_execution_context *c,gffx_diagnostic_buffer *d){
 gffx_status s=gffx_internal_prepare_diagnostic(d);if(s!=GFFX_STATUS_OK)return s;
 s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type!=GFFX_DEVICE_CPU&&c->device_type!=GFFX_DEVICE_CUDA)return pc_fail(d,"curve device unsupported");return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_curve_workspace(int64_t batch,int64_t nodes,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
 int64_t shape[2];const gffx_cuda_operations *ops;gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 if(batch<0||batch>32||nodes<1||nodes>512||!bytes||!alignment)return pc_fail(d,"curve batch/nodes/results invalid");
 if(c->device_type==GFFX_DEVICE_CUDA){ops=gffx_cuda_loader_operations();
  if(!GFFX_CUDA_HAS_OPERATION(ops,shading_loss_curve)||!GFFX_CUDA_HAS_OPERATION(ops,shading_loss_curve_backward)||!ops->workspace_query)
   return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks native loss curve family");
  shape[0]=batch;shape[1]=nodes;return ops->workspace_query(GFFX_CUDA_OP_SHADING_LOSS_CURVE,shape,2,GFFX_DTYPE_FLOAT64,c,bytes,alignment,d);}
 *bytes=8+8*(uint64_t)batch*(uint64_t)(10*nodes-6>4*nodes?10*nodes-6:4*nodes);*alignment=8;return GFFX_STATUS_OK;
}
static gffx_status pc_view(const gffx_tensor_view *v,uint32_t rank,const int64_t *shape,int output,const gffx_execution_context *c,gffx_diagnostic_buffer *d){
 uint32_t j;gffx_status s=gffx_validate_tensor_view(v,d);if(s!=GFFX_STATUS_OK)return s;
 if(v->dtype!=GFFX_DTYPE_FLOAT64||v->rank!=rank||v->device_type!=c->device_type||v->device_index!=c->device_index||((v->flags&GFFX_TENSOR_OUTPUT)!=0)!=(output!=0))return pc_fail(d,"curve tensor dtype/rank/device/flags invalid");
 for(j=0;j<rank;++j)if(v->shape[j]!=shape[j])return pc_fail(d,"curve tensor shape invalid");return GFFX_STATUS_OK;
}
static gffx_status pc_storage(const gffx_tensor_view **v,int count,int outputs,const gffx_buffer *w,uint64_t bytes,const gffx_execution_context *c,gffx_diagnostic_buffer *d){
 int i,j;gffx_status s;
 if(!w)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"curve requires caller scratch");
 s=gffx_validate_buffer(w,d);if(s!=GFFX_STATUS_OK)return s;
 if(w->device_type!=c->device_type||w->device_index!=c->device_index||w->alignment<8||(uintptr_t)w->data%8)return pc_fail(d,"curve scratch device/alignment invalid");
 if(w->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"curve scratch too small");
 for(i=0;i<count;++i){if(gffx_mesh_range_overlaps_view(w->data,w->capacity_bytes,v[i]))return pc_fail(d,"curve scratch aliases tensor");
  for(j=i+1;j<count;++j)if((i>=count-outputs||j>=count-outputs)&&gffx_mesh_views_overlap(v[i],v[j]))return pc_fail(d,"curve output aliases tensor");}
 return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_curve(const gffx_tensor_view *axis,const gffx_tensor_view *controls,const gffx_tensor_view *u,const gffx_execution_context *c,gffx_tensor_view *value,gffx_tensor_view *partial,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 gffx_status s=pc_context(c,d);int64_t n,batch,shape[3],j,i,size;uint64_t bytes,alignment;const gffx_tensor_view *views[5]={axis,controls,u,value,partial};const double *x,*b,*q;double *y,*dy,*scratch;
 if(s!=GFFX_STATUS_OK)return s;
 s=gffx_validate_tensor_view(axis,d);if(s!=GFFX_STATUS_OK)return s;s=gffx_validate_tensor_view(u,d);if(s!=GFFX_STATUS_OK)return s;
 if(axis->rank!=1||u->rank!=1)return pc_fail(d,"curve axis/u must be rank one");n=axis->shape[0];batch=u->shape[0];
 s=gffx_shading_loss_curve_workspace(batch,n,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;size=10*n-6;
 shape[0]=n;s=pc_view(axis,1,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;shape[0]=batch;s=pc_view(u,1,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[1]=2*n+1;shape[2]=4;s=pc_view(controls,3,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;shape[1]=size;
 s=pc_view(value,2,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(partial,2,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;
 s=pc_storage(views,5,2,w,bytes,c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_loss_curve(axis,controls,u,c,value,partial,w,d);
 x=gffx_mesh_elements_const(axis);b=gffx_mesh_elements_const(controls);q=gffx_mesh_elements_const(u);
 for(j=0;j<n;++j)if(!isfinite(x[j])||x[j]>0.||(j&&(!(x[j]>x[j-1])||!isfinite(x[j]-x[j-1]))))return pc_fail(d,"curve axis domain");
 for(j=0;j<batch*(2*n+1)*4;++j)if(!isfinite(b[j])||b[j]<-745.||b[j]>0.)return pc_fail(d,"curve control domain");
 for(j=0;j<batch;++j)if(!isfinite(q[j])||q[j]<0.||q[j]>1.)return pc_fail(d,"curve coordinate domain");
 y=gffx_mesh_elements(value);dy=gffx_mesh_elements(partial);scratch=(double *)w->data+1;
 for(i=0;i<batch;++i){gffx_pc_curve_build((int)n,x,b+i*(2*n+1)*4,q[i],y+i*size,dy+i*size,scratch);
  for(j=0;j<size;++j)if(!isfinite(y[i*size+j])||!isfinite(dy[i*size+j]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"curve arithmetic overflow");}
 return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_curve_backward(const gffx_tensor_view *partial,const gffx_tensor_view *gradient,const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 gffx_status s=pc_context(c,d);int64_t batch,n,size,shape[2],i,j;uint64_t bytes,alignment;const gffx_tensor_view *views[3]={partial,gradient,out};const double *p,*g;double *y,*scratch;
 if(s!=GFFX_STATUS_OK)return s;s=gffx_validate_tensor_view(partial,d);if(s!=GFFX_STATUS_OK)return s;
 if(partial->rank!=2||partial->shape[1]<4||partial->shape[1]>5114||(partial->shape[1]+6)%10)return pc_fail(d,"curve partial shape invalid");
 batch=partial->shape[0];size=partial->shape[1];n=(size+6)/10;s=gffx_shading_loss_curve_workspace(batch,n,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=batch;shape[1]=size;s=pc_view(partial,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(gradient,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;
 s=pc_view(out,1,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_storage(views,3,1,w,bytes,c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_loss_curve_backward(partial,gradient,c,out,w,d);
 p=gffx_mesh_elements_const(partial);g=gffx_mesh_elements_const(gradient);y=gffx_mesh_elements(out);scratch=(double *)w->data+1;
 for(j=0;j<batch*size;++j)if(!isfinite(p[j])||!isfinite(g[j]))return pc_fail(d,"curve VJP inputs must be finite");
 for(i=0;i<batch;++i){for(j=0;j<size;++j){scratch[j]=p[i*size+j]*g[i*size+j];if(!isfinite(scratch[j]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"curve VJP product overflow");}
  y[i]=gffx_pc_sum(scratch,(int)size);if(!isfinite(y[i]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"curve VJP sum overflow");}
 return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_shading_loss_query_workspace(int64_t batch,int64_t n,int64_t q,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
 const gffx_cuda_operations *ops;int64_t shape[3];gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 if(batch<0||batch>32||n<1||n>512||q<0||q>1056||(!batch&&q)||!bytes||!alignment)return pc_fail(d,"query extent invalid");
 if(c->device_type==GFFX_DEVICE_CUDA){ops=gffx_cuda_loader_operations();if(!GFFX_CUDA_HAS_OPERATION(ops,shading_loss_query)||!GFFX_CUDA_HAS_OPERATION(ops,shading_loss_query_backward)||!ops->workspace_query)return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks query family");shape[0]=batch;shape[1]=n;shape[2]=q;return ops->workspace_query(GFFX_CUDA_OP_SHADING_LOSS_QUERY,shape,3,GFFX_DTYPE_FLOAT64,c,bytes,alignment,d);}
 *bytes=8+8*(uint64_t)batch*(uint64_t)(4*n>q?4*n:q);*alignment=8;return GFFX_STATUS_OK;
}
static gffx_status pq_owners(const gffx_tensor_view *o,int64_t q,const gffx_execution_context *c,gffx_diagnostic_buffer *d){
 gffx_status s=gffx_validate_tensor_view(o,d);if(s!=GFFX_STATUS_OK)return s;
 if(o->dtype!=GFFX_DTYPE_INT32||o->rank!=1||o->shape[0]!=q||o->device_type!=c->device_type||o->device_index!=c->device_index||(o->flags&GFFX_TENSOR_OUTPUT))return pc_fail(d,"query owners dtype/shape/device/flags invalid");return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_query(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *weights,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *cosine,const gffx_tensor_view *owners,const gffx_execution_context *c,gffx_tensor_view *query,gffx_tensor_view *mean,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 const gffx_tensor_view *views[9]={cn,logs,weights,curve,partial,cosine,owners,query,mean};gffx_status s=pc_context(c,d);int64_t b,n,q,size,shape[2],i,j;uint64_t bytes,alignment;const double *x,*l,*a,*v,*p,*co;const int32_t *o;double *y,*m,*scratch;
 if(s!=GFFX_STATUS_OK)return s;for(i=0;i<9;++i){s=gffx_validate_tensor_view(views[i],d);if(s!=GFFX_STATUS_OK)return s;}
 if(cn->rank!=1||curve->rank!=2||cosine->rank!=1)return pc_fail(d,"query extent ranks invalid");n=cn->shape[0];b=curve->shape[0];q=cosine->shape[0];size=10*n-6;
 s=gffx_shading_loss_query_workspace(b,n,q,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=n;s=pc_view(cn,1,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(logs,1,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=n-1;shape[1]=4;s=pc_view(weights,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=b;shape[1]=size;s=pc_view(curve,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(partial,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=q;s=pc_view(cosine,1,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pq_owners(owners,q,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[1]=6;s=pc_view(query,2,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;shape[0]=b;shape[1]=4;s=pc_view(mean,2,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;
 s=pc_storage(views,9,2,w,bytes,c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_loss_query(cn,logs,weights,curve,partial,cosine,owners,c,query,mean,w,d);
 x=gffx_mesh_elements_const(cn);l=gffx_mesh_elements_const(logs);a=gffx_mesh_elements_const(weights);v=gffx_mesh_elements_const(curve);p=gffx_mesh_elements_const(partial);co=gffx_mesh_elements_const(cosine);o=(const int32_t *)gffx_mesh_elements_const(owners);
 for(i=0;i<n;++i)if(!isfinite(x[i])||x[i]<=0.||x[i]>1.||(i&&x[i]<=x[i-1])||!isfinite(l[i])||l[i]>0.||(i&&(!(l[i]>l[i-1])||!isfinite(l[i]-l[i-1]))))return pc_fail(d,"query axes domain");if(x[n-1]!=1.)return pc_fail(d,"query cosine axis must end at one");
 for(i=0;i<4*(n-1);++i)if(!isfinite(a[i])||a[i]<0.)return pc_fail(d,"query weights domain");
 for(i=0;i<b*size;++i)if(!isfinite(v[i])||!isfinite(p[i]))return pc_fail(d,"query curve must be finite");
 for(i=0;i<q;++i)if(!isfinite(co[i])||co[i]<0.||co[i]>1.||o[i]<0||o[i]>=b)return pc_fail(d,"query coordinate/owner domain");
 y=gffx_mesh_elements(query);m=gffx_mesh_elements(mean);scratch=(double *)w->data+1;
 for(i=0;i<q;++i)gffx_pq_direction((int)n,x,l,v+o[i]*size,p+o[i]*size,co[i],y+6*i);
 for(i=0;i<b;++i)gffx_pq_mean((int)n,x,l,a,v+i*size,p+i*size,m+4*i,scratch);
 for(j=0;j<6*q;++j)if(!isfinite(y[j]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"query arithmetic overflow");for(j=0;j<4*b;++j)if(!isfinite(m[j]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"mean arithmetic overflow");return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_loss_query_backward(const gffx_tensor_view *query,const gffx_tensor_view *mean,const gffx_tensor_view *owners,const gffx_tensor_view *gradient,const gffx_tensor_view *mean_gradient,const gffx_execution_context *c,gffx_tensor_view *grad_u,gffx_tensor_view *grad_cosine,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 const gffx_tensor_view *views[7]={query,mean,owners,gradient,mean_gradient,grad_u,grad_cosine};gffx_status s=pc_context(c,d);int64_t b,q,i,shape[2];uint64_t bytes,alignment;const double *v,*m,*g,*h;const int32_t *o;double *u,*co,*scratch;
 if(s!=GFFX_STATUS_OK)return s;for(i=0;i<7;++i){s=gffx_validate_tensor_view(views[i],d);if(s!=GFFX_STATUS_OK)return s;}
 if(query->rank!=2||mean->rank!=2)return pc_fail(d,"query backward ranks invalid");q=query->shape[0];b=mean->shape[0];s=gffx_shading_loss_query_workspace(b,1,q,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=q;shape[1]=6;s=pc_view(query,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;shape[1]=2;s=pc_view(gradient,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(grad_cosine,1,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;s=pq_owners(owners,q,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=b;shape[1]=4;s=pc_view(mean,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;shape[1]=2;s=pc_view(mean_gradient,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(grad_u,1,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_storage(views,7,2,w,bytes,c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_loss_query_backward(query,mean,owners,gradient,mean_gradient,c,grad_u,grad_cosine,w,d);
 v=gffx_mesh_elements_const(query);m=gffx_mesh_elements_const(mean);g=gffx_mesh_elements_const(gradient);h=gffx_mesh_elements_const(mean_gradient);o=(const int32_t *)gffx_mesh_elements_const(owners);u=gffx_mesh_elements(grad_u);co=gffx_mesh_elements(grad_cosine);scratch=(double *)w->data+1;
 for(i=0;i<q*6;++i)if(!isfinite(v[i]))return pc_fail(d,"query partial must be finite");for(i=0;i<b*4;++i)if(!isfinite(m[i]))return pc_fail(d,"mean partial must be finite");for(i=0;i<q*2;++i)if(!isfinite(g[i]))return pc_fail(d,"query gradient must be finite");for(i=0;i<b*2;++i)if(!isfinite(h[i]))return pc_fail(d,"mean gradient must be finite");for(i=0;i<q;++i)if(o[i]<0||o[i]>=b)return pc_fail(d,"query owner domain");
 for(i=0;i<q;++i){co[i]=v[6*i+4]*g[2*i]+v[6*i+5]*g[2*i+1];if(!isfinite(co[i]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"cosine VJP overflow");}
 for(i=0;i<b;++i){u[i]=gffx_pq_contract((int)i,(int)q,v,m,o,g,h,scratch);if(!isfinite(u[i]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"coordinate VJP overflow");}return GFFX_STATUS_OK;
}

#include "principled_pairs.inc"
#include "ordered_rgba.inc"
#include "unit_newton.inc"
#include "unit_vectors.inc"

#include "delta_brdf.inc"

#include "opaque_samples.inc"
#include "opaque_albedo.inc"
