#include "opaque_samples_scalar.h"
GFFX_API gffx_status GFFX_CALL gffx_shading_opaque_samples_workspace(int64_t p,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
 gffx_status s=pc_context(c,d);if(s)return s;if(p<1||p>16||!bytes||!alignment)return pc_fail(d,"opaque sample pixel count/results invalid");
 if(c->device_type==GFFX_DEVICE_CUDA&&!GFFX_CUDA_HAS_OPERATION(gffx_cuda_loader_operations(),shading_opaque_samples))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks opaque samples");
 *bytes=8+(uint64_t)p*800*4*8;*alignment=8;return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_opaque_samples(const gffx_tensor_view *owners,const gffx_tensor_view *q,const gffx_tensor_view *world,const gffx_tensor_view *faces,const gffx_tensor_view *clip,const gffx_tensor_view *normals,const gffx_tensor_view *camera,const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,const gffx_tensor_view *direction,const gffx_tensor_view *incoming,const gffx_execution_context *c,gffx_tensor_view *weighted,gffx_tensor_view *rgba,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 const gffx_tensor_view *views[20]={owners,q,world,faces,clip,normals,camera,cn,logs,curve,partial,mean,width,albedo,roughness,f0,direction,incoming,weighted,rgba};GffxOpaqueSamples x;int i,k;int64_t p,nv,nf,n,largest=1;uint64_t bytes,alignment;gffx_status s=pc_context(c,d);if(s)return s;
 for(i=0;i<20;++i){s=gffx_validate_tensor_view(views[i],d);if(s)return s;}
 if(q->rank!=3||world->rank!=2||faces->rank!=2||cn->rank!=1)return pc_fail(d,"opaque sample ranks invalid");p=q->shape[0];nv=world->shape[0];nf=faces->shape[0];n=cn->shape[0];
 s=gffx_shading_opaque_samples_workspace(p,c,&bytes,&alignment,d);if(s)return s;
 if(n<1||n>512||nv<0||nf<0||nv>INT32_MAX/9||nf>INT32_MAX/9)return pc_fail(d,"opaque sample geometry/axis extent invalid");
 {int ranks[20]={2,3,2,2,2,3,1,1,1,2,2,2,1,1,1,1,1,1,3,2};
 int64_t shape[20][3]={{p,800,0},{p,800,4},{nv,3,0},{nf,3,0},{nv,4,0},{nf,3,3},{3,0,0},{n,0,0},{n,0,0},{1,10*n-6,0},{1,10*n-6,0},{1,4,0},{1,0,0},{3,0,0},{1,0,0},{3,0,0},{3,0,0},{3,0,0},{p,800,4},{p,4,0}};
 for(i=0;i<20;++i){const gffx_tensor_view *v=views[i];int dtype=(i==0||i==3)?GFFX_DTYPE_INT32:GFFX_DTYPE_FLOAT64;
  if(v->dtype!=dtype||v->rank!=(uint32_t)ranks[i]||v->device_type!=c->device_type||v->device_index!=c->device_index||((v->flags&GFFX_TENSOR_OUTPUT)!=0)!=(i>=18))return pc_fail(d,"opaque sample dtype/rank/device/flags invalid");
  for(k=0;k<ranks[i];++k)if(v->shape[k]!=shape[i][k])return pc_fail(d,"opaque sample shape invalid");
 }}
 s=pc_storage(views,20,2,w,bytes,c,d);if(s)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_opaque_samples(views,c,weighted,rgba,w,d);
 for(i=0;i<18;++i){x.data[i]=gffx_mesh_elements_const(views[i]);x.count[i]=1;for(k=0;k<(int)views[i]->rank;++k)x.count[i]*=views[i]->shape[k];if(x.count[i]>largest)largest=x.count[i];}
 x.pixels=(int)p;x.nv=(int)nv;x.nf=(int)nf;x.nodes=(int)n;x.weighted=gffx_mesh_elements(weighted);x.rgba=gffx_mesh_elements(rgba);
 {int64_t j;for(i=1;i<18;++i)if(i!=3)for(j=0;j<x.count[i];++j)if(!isfinite(((const double*)x.data[i])[j]))return pc_fail(d,"opaque sample input domain");
 largest=3;for(i=0;i<18;++i)if((i==0||i==3||i==7||i==9)&&x.count[i]>largest)largest=x.count[i];
 for(j=0;j<largest;++j)if(oc_value_domain(&x,j))return pc_fail(d,"opaque sample input domain");}
 for(i=0;i<p;++i)if(oc_execute(&x,i,(double*)w->data+1+3200*i))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"opaque sample arithmetic overflow");
 return GFFX_STATUS_OK;
}
