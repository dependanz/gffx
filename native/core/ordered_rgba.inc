/* Fixed ascending carry: use the common checked dense float64 boundary. */
GFFX_API gffx_status GFFX_CALL gffx_shading_ordered_rgba_workspace(int64_t n,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
 gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 if(n<0||n>800||!bytes||!alignment)return pc_fail(d,"ordered RGBA count/results invalid");
 if(c->device_type==GFFX_DEVICE_CUDA&&!GFFX_CUDA_HAS_OPERATION(gffx_cuda_loader_operations(),shading_ordered_rgba))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks ordered RGBA");
 *bytes=8;*alignment=8;return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_shading_ordered_rgba(const gffx_tensor_view *carry,const gffx_tensor_view *values,const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 const gffx_tensor_view *views[3]={carry,values,out};int64_t shape[2],n,j,k;uint64_t bytes,alignment;const double *a,*v;double *y;gffx_status s=pc_context(c,d);if(s!=GFFX_STATUS_OK)return s;
 s=gffx_validate_tensor_view(values,d);if(s!=GFFX_STATUS_OK)return s;if(values->rank!=2)return pc_fail(d,"ordered RGBA values rank invalid");n=values->shape[0];
 s=gffx_shading_ordered_rgba_workspace(n,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=4;s=pc_view(carry,1,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_view(out,1,shape,1,c,d);if(s!=GFFX_STATUS_OK)return s;
 shape[0]=n;shape[1]=4;s=pc_view(values,2,shape,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=pc_storage(views,3,1,w,bytes,c,d);if(s!=GFFX_STATUS_OK)return s;
 if(c->device_type==GFFX_DEVICE_CUDA)return gffx_cuda_loader_operations()->shading_ordered_rgba(carry,values,c,out,w,d);
 a=gffx_mesh_elements_const(carry);v=gffx_mesh_elements_const(values);y=gffx_mesh_elements(out);
 for(k=0;k<4;++k)if(!isfinite(a[k]))return pc_fail(d,"ordered RGBA carry must be finite");for(j=0;j<4*n;++j)if(!isfinite(v[j]))return pc_fail(d,"ordered RGBA values must be finite");
 for(k=0;k<4;++k){volatile double total=a[k];for(j=0;j<n;++j)total=total+v[4*j+k];y[k]=total;if(!isfinite(y[k]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"ordered RGBA overflow");}
 return GFFX_STATUS_OK;
}
