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


/* Gaussian family shares only descriptor/workspace validation with transfer. */
#include "gaussian_scalar.h"
GFFX_API gffx_status GFFX_CALL gffx_image_gaussian_weights_workspace(int64_t batch,gffx_dtype dtype,
    uint32_t radius,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
    uint64_t count,size;const gffx_cuda_operations *ops;int64_t shape[2];gffx_status s=im_context(c,d);
    if(s!=GFFX_STATUS_OK)return s;
    if(batch<0||radius>32||!bytes||!alignment)return im_fail(d,"Gaussian workspace batch/radius/results invalid");
    if(dtype!=GFFX_DTYPE_FLOAT32&&dtype!=GFFX_DTYPE_FLOAT64)return im_fail(d,"Gaussian requires float32 or float64");
    if((uint64_t)batch>(uint64_t)INT64_MAX/(2*radius+1))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"Gaussian tap count overflow");
    count=(uint64_t)batch*(2*radius+1);size=gffx_mesh_dtype_size(dtype);
    if(count>SIZE_MAX/size)return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"Gaussian byte count overflow");
    if(c->device_type==GFFX_DEVICE_CPU){*bytes=0;*alignment=1;return GFFX_STATUS_OK;}
    ops=gffx_cuda_loader_operations();
    if(!GFFX_CUDA_HAS_OPERATION(ops,image_gaussian_weights)||!GFFX_CUDA_HAS_OPERATION(ops,image_gaussian_weights_backward)||!ops->workspace_query)
        return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks Gaussian weights family");
    shape[0]=batch;shape[1]=radius;
    return ops->workspace_query(GFFX_CUDA_OP_IMAGE_GAUSSIAN_WEIGHTS,shape,2,dtype,c,bytes,alignment,d);
}
static gffx_status gw_shape(const gffx_tensor_view *v,const gffx_tensor_view *sigma,uint32_t radius,
    int output,const gffx_execution_context *c,gffx_diagnostic_buffer *d){
    gffx_status s=im_view(v,NULL,output,c,d);if(s!=GFFX_STATUS_OK)return s;
    if(v->dtype!=sigma->dtype||v->rank!=sigma->rank+1||v->shape[v->rank-1]!=(int64_t)(2*radius+1)||
       (sigma->rank==1&&v->shape[0]!=sigma->shape[0]))return im_fail(d,"Gaussian tap tensor shape or dtype mismatch");
    return GFFX_STATUS_OK;
}
static gffx_status gw_call(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t radius,int backward,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    uint64_t batch,bytes,alignment,i;uint32_t j,count;gffx_status s=im_context(c,d);const gffx_cuda_operations *ops;
    if(s!=GFFX_STATUS_OK)return s;
    if(radius>32|| (backward&&!g))return im_fail(d,"Gaussian radius or gradient descriptor invalid");
    s=im_view(x,NULL,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    if(x->rank>1)return im_fail(d,"Gaussian sigma must be scalar or rank one");
    batch=gffx_mesh_element_count(x);count=2*radius+1;
    if(backward){s=gw_shape(g,x,radius,0,c,d);if(s!=GFFX_STATUS_OK)return s;s=im_view(out,x,1,c,d);}
    else s=gw_shape(out,x,radius,1,c,d);
    if(s!=GFFX_STATUS_OK)return s;
    s=gffx_image_gaussian_weights_workspace((int64_t)batch,x->dtype,radius,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    if(scratch){s=gffx_validate_buffer(scratch,d);if(s!=GFFX_STATUS_OK)return s;
        if(scratch->device_type!=c->device_type||scratch->device_index!=c->device_index)return im_fail(d,"Gaussian workspace device mismatch");
        if(scratch->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"Gaussian workspace too small");
        if(scratch->alignment<alignment||(uintptr_t)scratch->data%alignment)return im_fail(d,"Gaussian workspace alignment invalid");
        if(gffx_mesh_range_overlaps_view(scratch->data,scratch->capacity_bytes,x)||gffx_mesh_range_overlaps_view(scratch->data,scratch->capacity_bytes,out)||
            (g&&gffx_mesh_range_overlaps_view(scratch->data,scratch->capacity_bytes,g)))return im_fail(d,"Gaussian workspace aliases tensor storage");
    }else if(bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"Gaussian requires caller-owned workspace");
    if(gffx_mesh_views_overlap(x,out)||(g&&gffx_mesh_views_overlap(g,out)))return im_fail(d,"Gaussian output aliases input");
    if(c->device_type==GFFX_DEVICE_CUDA){ops=gffx_cuda_loader_operations();
        return backward?ops->image_gaussian_weights_backward(x,g,radius,c,out,scratch,d):ops->image_gaussian_weights(x,radius,c,out,scratch,d);}
    /* All domains across all batches are checked before any output write. */
    for(i=0;i<batch;++i){double sigma=x->dtype==GFFX_DTYPE_FLOAT32?((const float *)gffx_mesh_elements_const(x))[i]:((const double *)gffx_mesh_elements_const(x))[i];
        double lower=x->dtype==GFFX_DTYPE_FLOAT32?(double).1f:.1;
        if(!isfinite(sigma)||sigma<lower||sigma>32.)return im_fail(d,"Gaussian sigma outside represented [.1,32]");}
    if(g){s=im_finite(g,d);if(s!=GFFX_STATUS_OK)return s;}
    for(i=0;i<batch;++i){double w[65],derivative[65],sigma=x->dtype==GFFX_DTYPE_FLOAT32?((const float *)gffx_mesh_elements_const(x))[i]:((const double *)gffx_mesh_elements_const(x))[i];
        gffx_gw_evaluate(sigma,(int)radius,w,derivative);
        if(!backward){for(j=0;j<count;++j){if(x->dtype==GFFX_DTYPE_FLOAT32)((float *)gffx_mesh_elements(out))[i*count+j]=gffx_gw_float(w[j]);
            else ((double *)gffx_mesh_elements(out))[i*count+j]=w[j];}}
        else if(x->dtype==GFFX_DTYPE_FLOAT32){float values[65],sum=0.;const float *gradient=gffx_mesh_elements_const(g);
            if(radius){for(j=0;j<count;++j){values[j]=gffx_gw_float(derivative[j])*gradient[i*count+j];
                if(!isfinite(values[j]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"Gaussian VJP product overflow");}
                if(gffx_gw_sum32(values,(int)count,&sum))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"Gaussian VJP sum overflow");}
            ((float *)gffx_mesh_elements(out))[i]=sum;
        }else{double sum=0.;const double *gradient=gffx_mesh_elements_const(g);
            if(radius){for(j=0;j<count;++j){derivative[j]*=gradient[i*count+j];
                if(!isfinite(derivative[j]))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"Gaussian VJP product overflow");}
                if(gffx_gw_sum64(derivative,(int)count,&sum))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"Gaussian VJP sum overflow");}
            ((double *)gffx_mesh_elements(out))[i]=sum;}
    }
    return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_image_gaussian_weights(const gffx_tensor_view *x,uint32_t radius,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    return gw_call(x,NULL,radius,0,c,out,scratch,d);
}
GFFX_API gffx_status GFFX_CALL gffx_image_gaussian_weights_backward(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t radius,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    return gw_call(x,g,radius,1,c,out,scratch,d);
}
