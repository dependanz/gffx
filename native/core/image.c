/* Dense scalar-channel transfer family. No allocation, framework, file I/O or cache. */
#include <gffx/image.h>
#include "internal.h"
#include "mesh_common.h"
#include "cuda_loader.h"
#include "image_transfer_scalar.h"
#include <limits.h>

static gffx_status im_fail(gffx_diagnostic_buffer *d,const char *message){
    return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,message);
}
static gffx_status im_context(const gffx_execution_context *c,gffx_diagnostic_buffer *d){
    gffx_status s=gffx_internal_prepare_diagnostic(d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    if(c->device_type!=GFFX_DEVICE_CPU&&c->device_type!=GFFX_DEVICE_CUDA)
        return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"image transfer supports CPU or CUDA");
    return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_image_transfer_workspace(int64_t count,gffx_dtype dtype,
    const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
    gffx_status s=im_context(c,d);const gffx_cuda_operations *ops;if(s!=GFFX_STATUS_OK)return s;
    if(count<0||bytes==NULL||alignment==NULL)return im_fail(d,"image workspace count/results invalid");
    if(dtype!=GFFX_DTYPE_FLOAT32&&dtype!=GFFX_DTYPE_FLOAT64)
        return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"image transfer requires float32 or float64");
    if((uint64_t)count>UINT64_MAX/gffx_mesh_dtype_size(dtype))
        return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"image element byte count overflow");
    if(c->device_type==GFFX_DEVICE_CPU){*bytes=0;*alignment=1;return GFFX_STATUS_OK;}
    ops=gffx_cuda_loader_operations();
    if(!GFFX_CUDA_HAS_OPERATION(ops,image_transfer)||!GFFX_CUDA_HAS_OPERATION(ops,image_transfer_backward)||ops->workspace_query==NULL)
        return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks image transfer family");
    return ops->workspace_query(GFFX_CUDA_OP_IMAGE_TRANSFER,&count,1,dtype,c,bytes,alignment,d);
}
static gffx_status im_view(const gffx_tensor_view *v,const gffx_tensor_view *shape,int output,
    const gffx_execution_context *c,gffx_diagnostic_buffer *d){
    uint32_t i;gffx_status s=gffx_validate_tensor_view(v,d);if(s!=GFFX_STATUS_OK)return s;
    if(v->dtype!=GFFX_DTYPE_FLOAT32&&v->dtype!=GFFX_DTYPE_FLOAT64)return im_fail(d,"image channels must be float32 or float64");
    if(v->device_type!=c->device_type||v->device_index!=c->device_index)return im_fail(d,"image tensor device mismatch");
    if(((v->flags&GFFX_TENSOR_OUTPUT)!=0)!=(output!=0))return im_fail(d,"image input/output flags invalid");
    if(shape!=NULL){
        if(v->rank!=shape->rank||v->dtype!=shape->dtype)return im_fail(d,"image tensor rank or dtype mismatch");
        for(i=0;i<v->rank;++i)if(v->shape[i]!=shape->shape[i])return im_fail(d,"image tensor shape mismatch");
    }
    if(gffx_mesh_element_count(v)>(uint64_t)INT64_MAX)return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"image element count overflow");
    return GFFX_STATUS_OK;
}
static gffx_status im_finite(const gffx_tensor_view *v,gffx_diagnostic_buffer *d){
    uint64_t i,n=gffx_mesh_element_count(v);const void *data=gffx_mesh_elements_const(v);
    for(i=0;i<n;++i){double x=v->dtype==GFFX_DTYPE_FLOAT32?((const float *)data)[i]:((const double *)data)[i];
        if(!isfinite(x))return im_fail(d,"image floating inputs must be finite");}
    return GFFX_STATUS_OK;
}
static gffx_status im_call(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t transfer,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    gffx_status s=im_context(c,d);uint64_t bytes,alignment,i,n;const gffx_cuda_operations *ops;
    if(s!=GFFX_STATUS_OK)return s;
    if(transfer>GFFX_IMAGE_LINEAR_TO_SRGB)return im_fail(d,"unknown image transfer mode");
    s=im_view(x,NULL,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    s=im_view(out,x,1,c,d);if(s!=GFFX_STATUS_OK)return s;
    if(g!=NULL){s=im_view(g,x,0,c,d);if(s!=GFFX_STATUS_OK)return s;}
    n=gffx_mesh_element_count(x);
    s=gffx_image_transfer_workspace((int64_t)n,x->dtype,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    if(scratch!=NULL){
        s=gffx_validate_buffer(scratch,d);if(s!=GFFX_STATUS_OK)return s;
        if(scratch->device_type!=c->device_type||scratch->device_index!=c->device_index)return im_fail(d,"image workspace device mismatch");
        if(scratch->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"image workspace too small");
        if(scratch->alignment<alignment||(uintptr_t)scratch->data%alignment)return im_fail(d,"image workspace alignment invalid");
        if(gffx_mesh_range_overlaps_view(scratch->data,scratch->capacity_bytes,x)||
           gffx_mesh_range_overlaps_view(scratch->data,scratch->capacity_bytes,out)||
           (g!=NULL&&gffx_mesh_range_overlaps_view(scratch->data,scratch->capacity_bytes,g)))
            return im_fail(d,"image workspace aliases tensor storage");
    }else if(bytes!=0)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"image CUDA requires caller-owned workspace");
    if(gffx_mesh_views_overlap(x,out)||(g!=NULL&&gffx_mesh_views_overlap(g,out)))return im_fail(d,"image output aliases input");
    if(c->device_type==GFFX_DEVICE_CUDA){
        ops=gffx_cuda_loader_operations();
        if(g==NULL){if(!GFFX_CUDA_HAS_OPERATION(ops,image_transfer))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA image transfer unavailable");
            return ops->image_transfer(x,transfer,c,out,scratch,d);}
        if(!GFFX_CUDA_HAS_OPERATION(ops,image_transfer_backward))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA image transfer backward unavailable");
        return ops->image_transfer_backward(x,g,transfer,c,out,scratch,d);
    }
    s=im_finite(x,d);if(s!=GFFX_STATUS_OK)return s;
    if(g!=NULL){s=im_finite(g,d);if(s!=GFFX_STATUS_OK)return s;}
    if(x->dtype==GFFX_DTYPE_FLOAT32){
        const float *input=gffx_mesh_elements_const(x),*gradient=g==NULL?NULL:gffx_mesh_elements_const(g);float *output=gffx_mesh_elements(out);
        for(i=0;i<n;++i){float y,dy;int bad=gffx_im_transfer_f32(input[i],(int)transfer,&y,&dy);if(g!=NULL)y=gradient[i]*dy;
            if(bad||!isfinite(y))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"image transfer arithmetic is not representable");output[i]=y;}
    }else{
        const double *input=gffx_mesh_elements_const(x),*gradient=g==NULL?NULL:gffx_mesh_elements_const(g);double *output=gffx_mesh_elements(out);
        for(i=0;i<n;++i){double y,dy;int bad=gffx_im_transfer_f64(input[i],(int)transfer,&y,&dy);if(g!=NULL)y=gradient[i]*dy;
            if(bad||!isfinite(y))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"image transfer arithmetic is not representable");output[i]=y;}
    }
    return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_image_transfer(const gffx_tensor_view *x,uint32_t transfer,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    return im_call(x,NULL,transfer,c,out,scratch,d);
}
GFFX_API gffx_status GFFX_CALL gffx_image_transfer_backward(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t transfer,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    if(g==NULL){gffx_status s=gffx_internal_prepare_diagnostic(d);if(s!=GFFX_STATUS_OK)return s;return im_fail(d,"image backward requires gradient descriptor");}
    return im_call(x,g,transfer,c,out,scratch,d);
}
