/* Animation CPU reference. Arithmetic and differentiation contracts live in
 * ANIMATION_ACCEPTANCE_V0_1.md. No framework/allocator dependency. */
#include <gffx/animation.h>
#include "internal.h"
#include "mesh_common.h"
#include "cuda_loader.h"
#include <math.h>
#include <stdint.h>
#include "animation_scalar.h"

static gffx_status an_fail(gffx_diagnostic_buffer *d, const char *message) {
    return gffx_internal_fail(d, GFFX_STATUS_INVALID_ARGUMENT, message);
}

static gffx_status an_context(const gffx_execution_context *context,
                              const gffx_buffer *workspace, gffx_diagnostic_buffer *d) {
    gffx_status s = gffx_internal_prepare_diagnostic(d);
    if (s != GFFX_STATUS_OK) return s;
    s = gffx_validate_execution_context(context, d);
    if (s != GFFX_STATUS_OK) return s;
    if (context->device_type != GFFX_DEVICE_CPU)
        return gffx_internal_fail(d, GFFX_STATUS_UNSUPPORTED, "animation CPU slice does not support this device");
    if (workspace != NULL) {
        s = gffx_validate_buffer(workspace, d);
        if (s != GFFX_STATUS_OK) return s;
        if (workspace->device_type != context->device_type || workspace->device_index != context->device_index)
            return an_fail(d, "animation workspace must match the execution device");
    }
    return GFFX_STATUS_OK;
}

static gffx_status an_view(const gffx_tensor_view *v, uint32_t rank, const int64_t *shape,
                          gffx_dtype dtype, int output, gffx_diagnostic_buffer *d) {
    uint32_t i;
    gffx_status s;
    if (v == NULL || v->rank != rank || v->shape == NULL)
        return an_fail(d, "animation tensor has wrong rank");
    for (i = 0; i < rank; ++i)
        if (shape[i] >= 0 && v->shape[i] != shape[i])
            return an_fail(d, "animation tensor has wrong shape");
    s = gffx_validate_tensor_view(v, d);
    if (s != GFFX_STATUS_OK) return s;
    if (v->device_type != GFFX_DEVICE_CPU || v->device_index != 0)
        return an_fail(d, "animation tensors must match CPU execution device");
    if (v->dtype != dtype) return an_fail(d, "animation tensor dtype mismatch");
    if (((v->flags & GFFX_TENSOR_OUTPUT) != 0) != (output != 0))
        return an_fail(d, "animation input/output flags are invalid");
    if (!output && (dtype == GFFX_DTYPE_FLOAT32 || dtype == GFFX_DTYPE_FLOAT64)) {
        uint64_t n = gffx_mesh_element_count(v), j;
        const void *p = gffx_mesh_elements_const(v);
        for (j = 0; j < n; ++j) {
            double x = dtype == GFFX_DTYPE_FLOAT32 ? (double)((const float *)p)[j] : ((const double *)p)[j];
            if (!isfinite(x)) return an_fail(d, "animation floating inputs must be finite");
        }
    }
    return GFFX_STATUS_OK;
}

static gffx_status an_noalias(const gffx_tensor_view **inputs, unsigned ni,
                            gffx_tensor_view **outputs, unsigned no,
                            const gffx_buffer *workspace, gffx_diagnostic_buffer *d) {
    unsigned i, j;
    for (i = 0; i < no; ++i) {
        if (outputs[i] == NULL) continue;
        for (j = 0; j < ni; ++j)
            if (gffx_mesh_views_overlap(outputs[i], inputs[j])) return an_fail(d, "animation output aliases an input");
        for (j = 0; j < i; ++j)
            if (outputs[j] != NULL && gffx_mesh_views_overlap(outputs[i], outputs[j])) return an_fail(d, "animation outputs overlap");
        if (workspace != NULL && gffx_mesh_range_overlaps_view(workspace->data, workspace->capacity_bytes, outputs[i]))
            return an_fail(d, "animation workspace overlaps output");
    }
    if (workspace != NULL)
        for (j = 0; j < ni; ++j)
            if (gffx_mesh_range_overlaps_view(workspace->data, workspace->capacity_bytes, inputs[j]))
                return an_fail(d, "animation workspace overlaps input");
    return GFFX_STATUS_OK;
}

static gffx_status an_result(gffx_tensor_view *v, gffx_diagnostic_buffer *d) {
    uint64_t j, n;
    const void *p;
    if (v == NULL) return GFFX_STATUS_OK;
    n = gffx_mesh_element_count(v); p = gffx_mesh_elements_const(v);
    for (j = 0; j < n; ++j) {
        double x = v->dtype == GFFX_DTYPE_FLOAT32 ? (double)((const float *)p)[j] : ((const double *)p)[j];
        if (!isfinite(x)) return gffx_internal_fail(d, GFFX_STATUS_OVERFLOW, "animation arithmetic is not representable");
    }
    return GFFX_STATUS_OK;
}

#include "animation_cuda_dispatch.inc"

/* Validate requested output views against already-validated primal shapes. */
static gffx_status an_gradient_views(const gffx_tensor_view **primal,gffx_tensor_view **out,
                                    unsigned count,gffx_diagnostic_buffer *d) {
    unsigned i; int any=0; gffx_status s;
    for(i=0;i<count;++i) if(out[i]!=NULL) {
        any=1; s=an_view(out[i],primal[i]->rank,primal[i]->shape,primal[i]->dtype,1,d);
        if(s!=GFFX_STATUS_OK) return s;
    }
    return any?GFFX_STATUS_OK:an_fail(d,"at least one animation gradient is required");
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_blend_shapes_workspace(
    int64_t B, int64_t V, int64_t K, gffx_dtype dtype, const gffx_execution_context *context,
    uint64_t *bytes, uint64_t *alignment, gffx_diagnostic_buffer *d) {
    if(context!=NULL&&context->device_type==GFFX_DEVICE_CUDA)
        return an_cuda_blend_workspace(B,V,K,dtype,context,bytes,alignment,d);
    gffx_status s = an_context(context, NULL, d);
    if (s != GFFX_STATUS_OK) return s;
    if (bytes == NULL || alignment == NULL || B < 0 || V < 0 || K < 0)
        return an_fail(d, "animation workspace counts/results invalid");
    if (dtype != GFFX_DTYPE_FLOAT32 && dtype != GFFX_DTYPE_FLOAT64)
        return gffx_internal_fail(d, GFFX_STATUS_UNSUPPORTED, "animation supports float32 and float64");
    if (B > INT32_MAX || V > INT32_MAX || K > INT32_MAX ||
        (V != 0 && (B > INT64_MAX / 3 / V || K > INT64_MAX / 3 / V)) ||
        (K != 0 && B > INT64_MAX / K))
        return gffx_internal_fail(d, GFFX_STATUS_OVERFLOW, "animation counts overflow");
    *bytes = 0; *alignment = 1;
    return GFFX_STATUS_OK;
}

static gffx_status an_blend_check(const gffx_tensor_view *t, const gffx_tensor_view *delta,
                                const gffx_tensor_view *w, const gffx_execution_context *context,
                                const gffx_buffer *workspace, gffx_diagnostic_buffer *d) {
    int64_t ts[2] = {-1, 3}, ds[3], ws[2];
    uint64_t bytes, alignment;
    gffx_status s = an_context(context, workspace, d);
    if (s != GFFX_STATUS_OK) return s;
    if (t == NULL || (t->dtype != GFFX_DTYPE_FLOAT32 && t->dtype != GFFX_DTYPE_FLOAT64))
        return an_fail(d, "template must be float32 or float64");
    s = an_view(t, 2, ts, t->dtype, 0, d); if (s != GFFX_STATUS_OK) return s;
    ds[0] = -1; ds[1] = t->shape[0]; ds[2] = 3;
    s = an_view(delta, 3, ds, t->dtype, 0, d); if (s != GFFX_STATUS_OK) return s;
    ws[0] = -1; ws[1] = delta->shape[0];
    s = an_view(w, 2, ws, t->dtype, 0, d); if (s != GFFX_STATUS_OK) return s;
    return gffx_mesh_blend_shapes_workspace(w->shape[0], t->shape[0], delta->shape[0], t->dtype,
                                           context, &bytes, &alignment, d);
}

#define AN_BLEND_KERNEL(T, suffix) \
static void an_blend_##suffix(const T *t, const T *d, const T *w, T *y, int64_t B, int64_t V, int64_t K) { \
    int64_t b, v, k, c; \
    for (b=0;b<B;++b) for(v=0;v<V;++v) for(c=0;c<3;++c) { \
        T a=t[v*3+c]; for(k=0;k<K;++k) a=a+w[b*K+k]*d[(k*V+v)*3+c]; y[(b*V+v)*3+c]=a; \
    } \
} \
static void an_blend_backward_##suffix(const T *d, const T *w, const T *g, T *gt, T *gd, T *gw, \
                                       int64_t B, int64_t V, int64_t K) { \
    int64_t b,v,k,c; \
    if(gt) for(v=0;v<V;++v) for(c=0;c<3;++c) { T a=0; for(b=0;b<B;++b) a=a+g[(b*V+v)*3+c]; gt[v*3+c]=a; } \
    if(gd) for(k=0;k<K;++k) for(v=0;v<V;++v) for(c=0;c<3;++c) { T a=0; for(b=0;b<B;++b) a=a+w[b*K+k]*g[(b*V+v)*3+c]; gd[(k*V+v)*3+c]=a; } \
    if(gw) for(b=0;b<B;++b) for(k=0;k<K;++k) { T a=0; for(v=0;v<V;++v) for(c=0;c<3;++c) a=a+d[(k*V+v)*3+c]*g[(b*V+v)*3+c]; gw[b*K+k]=a; } \
}
AN_BLEND_KERNEL(float, f32)
AN_BLEND_KERNEL(double, f64)

/* Rodrigues uses K^2 = rr^T - qI; backward differentiates the chosen branch. */
#define AN_RODRIGUES_WRAPPER(T,S) \
static gffx_status an_rodrigues_##S(const T *r,const T *g,T *y,int64_t count,gffx_diagnostic_buffer *d) { \
    if(gffx_an_rodrigues_##S(r,g,y,count))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"Rodrigues squared angle overflow"); \
    return GFFX_STATUS_OK; \
}
AN_RODRIGUES_WRAPPER(float,f32)
AN_RODRIGUES_WRAPPER(double,f64)
#undef AN_RODRIGUES_WRAPPER

GFFX_API gffx_status GFFX_CALL gffx_transforms_rodrigues_workspace(
    int64_t count,gffx_dtype dtype,const gffx_execution_context *context,
    uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d) {
    if(context!=NULL&&context->device_type==GFFX_DEVICE_CUDA)return an_cuda_pose_workspace(15,1,count,dtype,context,bytes,alignment,d);
    return gffx_mesh_blend_shapes_workspace(1,count,0,dtype,context,bytes,alignment,d);
}

static gffx_status an_rodrigues_call(const gffx_tensor_view *r,const gffx_tensor_view *g,
    const gffx_execution_context *context,gffx_tensor_view *out,const gffx_buffer *workspace,gffx_diagnostic_buffer *d) {
    if(context!=NULL&&context->device_type==GFFX_DEVICE_CUDA)return an_cuda_rodrigues(r,g,context,out,workspace,d);
    int64_t shape[GFFX_MAX_RANK]; uint32_t i; int64_t count; gffx_status s;
    const gffx_tensor_view *inputs[2]={r,g}; gffx_tensor_view *outputs[1]={out};
    s=an_context(context,workspace,d); if(s != GFFX_STATUS_OK) return s;
    if(r==NULL || r->rank<1 || r->rank>=GFFX_MAX_RANK || r->shape==NULL || r->shape[r->rank-1]!=3)
        return an_fail(d,"axis_angle must have shape [...,3] within the native rank limit");
    if(r->dtype!=GFFX_DTYPE_FLOAT32 && r->dtype!=GFFX_DTYPE_FLOAT64) return an_fail(d,"axis_angle must be float32 or float64");
    s=an_view(r,r->rank,r->shape,r->dtype,0,d); if(s!=GFFX_STATUS_OK) return s;
    for(i=0;i<r->rank;++i) shape[i]=r->shape[i]; shape[r->rank]=3;
    if(g!=NULL) { s=an_view(g,r->rank+1,shape,r->dtype,0,d); if(s!=GFFX_STATUS_OK) return s; }
    s=an_view(out,g==NULL?r->rank+1:r->rank,g==NULL?shape:r->shape,r->dtype,1,d); if(s!=GFFX_STATUS_OK) return s;
    s=an_noalias(inputs,g==NULL?1:2,outputs,1,workspace,d); if(s!=GFFX_STATUS_OK) return s;
    count=(int64_t)(gffx_mesh_element_count(r)/3);
    if(r->dtype==GFFX_DTYPE_FLOAT32) s=an_rodrigues_f32(gffx_mesh_elements_const(r),g==NULL?NULL:gffx_mesh_elements_const(g),gffx_mesh_elements(out),count,d);
    else s=an_rodrigues_f64(gffx_mesh_elements_const(r),g==NULL?NULL:gffx_mesh_elements_const(g),gffx_mesh_elements(out),count,d);
    if(s!=GFFX_STATUS_OK) return s;
    return an_result(out,d);
}
GFFX_API gffx_status GFFX_CALL gffx_transforms_rodrigues(const gffx_tensor_view *r,
    const gffx_execution_context *context,gffx_tensor_view *out,const gffx_buffer *workspace,gffx_diagnostic_buffer *d) {
    return an_rodrigues_call(r,NULL,context,out,workspace,d);
}
GFFX_API gffx_status GFFX_CALL gffx_transforms_rodrigues_backward(const gffx_tensor_view *r,const gffx_tensor_view *g,
    const gffx_execution_context *context,gffx_tensor_view *out,const gffx_buffer *workspace,gffx_diagnostic_buffer *d) {
    if(g==NULL) return an_fail(d,"Rodrigues backward requires a cotangent");
    return an_rodrigues_call(r,g,context,out,workspace,d);
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_blend_shapes(
    const gffx_tensor_view *t, const gffx_tensor_view *delta, const gffx_tensor_view *w,
    const gffx_execution_context *context, gffx_tensor_view *y, const gffx_buffer *workspace,
    gffx_diagnostic_buffer *d) {
    if(context!=NULL&&context->device_type==GFFX_DEVICE_CUDA)
        return an_cuda_blend(t,delta,w,NULL,context,y,NULL,NULL,NULL,workspace,d);
    int64_t shape[3];
    const gffx_tensor_view *inputs[3] = {t,delta,w};
    gffx_tensor_view *outputs[1] = {y};
    gffx_status s = an_blend_check(t,delta,w,context,workspace,d);
    if(s != GFFX_STATUS_OK) return s;
    shape[0]=w->shape[0]; shape[1]=t->shape[0]; shape[2]=3;
    s=an_view(y,3,shape,t->dtype,1,d); if(s != GFFX_STATUS_OK) return s;
    s=an_noalias(inputs,3,outputs,1,workspace,d); if(s != GFFX_STATUS_OK) return s;
    if(t->dtype == GFFX_DTYPE_FLOAT32) an_blend_f32(gffx_mesh_elements_const(t),gffx_mesh_elements_const(delta),gffx_mesh_elements_const(w),gffx_mesh_elements(y),shape[0],shape[1],delta->shape[0]);
    else an_blend_f64(gffx_mesh_elements_const(t),gffx_mesh_elements_const(delta),gffx_mesh_elements_const(w),gffx_mesh_elements(y),shape[0],shape[1],delta->shape[0]);
    return an_result(y,d);
}

GFFX_API gffx_status GFFX_CALL gffx_mesh_blend_shapes_backward(
    const gffx_tensor_view *t, const gffx_tensor_view *delta, const gffx_tensor_view *w,
    const gffx_tensor_view *g, const gffx_execution_context *context,
    gffx_tensor_view *gt, gffx_tensor_view *gd, gffx_tensor_view *gw,
    const gffx_buffer *workspace, gffx_diagnostic_buffer *d) {
    if(context!=NULL&&context->device_type==GFFX_DEVICE_CUDA){
        if(g==NULL)return an_fail(d,"blend backward requires cotangent");
        return an_cuda_blend(t,delta,w,g,context,NULL,gt,gd,gw,workspace,d);
    }
    int64_t shape[3]; unsigned i;
    const gffx_tensor_view *inputs[4] = {t,delta,w,g};
    const gffx_tensor_view *primal[3] = {t,delta,w};
    gffx_tensor_view *outputs[3] = {gt,gd,gw};
    gffx_status s = an_blend_check(t,delta,w,context,workspace,d);
    if(s != GFFX_STATUS_OK) return s;
    if(gt == NULL && gd == NULL && gw == NULL) return an_fail(d,"at least one animation gradient is required");
    shape[0]=w->shape[0]; shape[1]=t->shape[0]; shape[2]=3;
    s=an_view(g,3,shape,t->dtype,0,d); if(s != GFFX_STATUS_OK) return s;
    for(i=0;i<3;++i) if(outputs[i] != NULL) { s=an_view(outputs[i],primal[i]->rank,primal[i]->shape,t->dtype,1,d); if(s != GFFX_STATUS_OK) return s; }
    s=an_noalias(inputs,4,outputs,3,workspace,d); if(s != GFFX_STATUS_OK) return s;
#define AN_GRAD_PTR(v) ((v) == NULL ? NULL : gffx_mesh_elements(v))
    if(t->dtype == GFFX_DTYPE_FLOAT32) an_blend_backward_f32(gffx_mesh_elements_const(delta),gffx_mesh_elements_const(w),gffx_mesh_elements_const(g),AN_GRAD_PTR(gt),AN_GRAD_PTR(gd),AN_GRAD_PTR(gw),shape[0],shape[1],delta->shape[0]);
    else an_blend_backward_f64(gffx_mesh_elements_const(delta),gffx_mesh_elements_const(w),gffx_mesh_elements_const(g),AN_GRAD_PTR(gt),AN_GRAD_PTR(gd),AN_GRAD_PTR(gw),shape[0],shape[1],delta->shape[0]);
#undef AN_GRAD_PTR
    for(i=0;i<3;++i) { s=an_result(outputs[i],d); if(s != GFFX_STATUS_OK) return s; }
    return GFFX_STATUS_OK;
}

/* Empty cotangents may have null data: operation mode comes from the descriptor,
 * never from the data pointer, so shared gradients are still written to zero. */
#define AN_SKIN(T,suffix) \
static void an_skin_##suffix(const T *v,const T *t,const T *w,const T *g,T *y,T *gv,T *gt,T *gw,int64_t B,int64_t V,int64_t J,int backward) { \
    int64_t b,n,j; int c,a; \
    if(!backward) { \
        for(b=0;b<B;++b) for(n=0;n<V;++n) for(c=0;c<3;++c) { T sum=0; \
            for(j=0;j<J;++j) { const T *m=t+(b*J+j)*16; T p=m[c*4]*v[(b*V+n)*3]; \
                p=p+m[c*4+1]*v[(b*V+n)*3+1]; p=p+m[c*4+2]*v[(b*V+n)*3+2]; p=p+m[c*4+3]; sum=sum+w[n*J+j]*p; \
            } y[(b*V+n)*3+c]=sum; \
        } return; \
    } \
    if(gv) for(b=0;b<B;++b) for(n=0;n<V;++n) for(a=0;a<3;++a) { T sum=0; \
        for(j=0;j<J;++j) for(c=0;c<3;++c) sum=sum+w[n*J+j]*t[(b*J+j)*16+c*4+a]*g[(b*V+n)*3+c]; gv[(b*V+n)*3+a]=sum; } \
    if(gt) for(b=0;b<B;++b) for(j=0;j<J;++j) for(c=0;c<4;++c) for(a=0;a<4;++a) { T sum=0; \
        if(c<3) for(n=0;n<V;++n) sum=sum+w[n*J+j]*(a<3?v[(b*V+n)*3+a]:(T)1)*g[(b*V+n)*3+c]; gt[(b*J+j)*16+c*4+a]=sum; } \
    if(gw) for(n=0;n<V;++n) for(j=0;j<J;++j) { T sum=0; \
        for(b=0;b<B;++b) for(c=0;c<3;++c) { const T *m=t+(b*J+j)*16; T p=m[c*4]*v[(b*V+n)*3]; \
            p=p+m[c*4+1]*v[(b*V+n)*3+1]; p=p+m[c*4+2]*v[(b*V+n)*3+2]; p=p+m[c*4+3]; sum=sum+g[(b*V+n)*3+c]*p; } gw[n*J+j]=sum; } \
}
AN_SKIN(float,f32)
AN_SKIN(double,f64)

GFFX_API gffx_status GFFX_CALL gffx_mesh_skin_lbs_workspace(int64_t B,int64_t V,int64_t J,gffx_dtype dtype,
    const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d) {
    if(context!=NULL&&context->device_type==GFFX_DEVICE_CUDA)return an_cuda_deformation_workspace(17,B,V,J,dtype,context,bytes,alignment,d);
    if(J==0 && V!=0) return an_fail(d,"nonempty skinning vertices require at least one joint");
    return gffx_mesh_blend_shapes_workspace(B,V,J,dtype,context,bytes,alignment,d);
}

static gffx_status an_skin_call(const gffx_tensor_view *v,const gffx_tensor_view *t,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *context,gffx_tensor_view *y,
    gffx_tensor_view *gv,gffx_tensor_view *gt,gffx_tensor_view *gw,const gffx_buffer *workspace,gffx_diagnostic_buffer *d) {
    if(context!=NULL&&context->device_type==GFFX_DEVICE_CUDA)return an_cuda_skin(v,t,w,g,context,y,gv,gt,gw,workspace,d);
    int64_t vs[3]={-1,-1,3},ts[4],ws[2]; uint64_t bytes,alignment; unsigned i;
    const gffx_tensor_view *inputs[4]={v,t,w,g},*primal[3]={v,t,w};
    gffx_tensor_view *out[3]={gv,gt,gw}; gffx_status s=an_context(context,workspace,d);
    if(s!=GFFX_STATUS_OK) return s;
    if(v==NULL || (v->dtype!=GFFX_DTYPE_FLOAT32 && v->dtype!=GFFX_DTYPE_FLOAT64)) return an_fail(d,"vertices require float32 or float64");
    s=an_view(v,3,vs,v->dtype,0,d); if(s!=GFFX_STATUS_OK) return s;
    ts[0]=v->shape[0];ts[1]=-1;ts[2]=4;ts[3]=4;
    s=an_view(t,4,ts,v->dtype,0,d); if(s!=GFFX_STATUS_OK) return s;
    ws[0]=v->shape[1];ws[1]=t->shape[1];
    s=an_view(w,2,ws,v->dtype,0,d); if(s!=GFFX_STATUS_OK) return s;
    s=gffx_mesh_skin_lbs_workspace(v->shape[0],v->shape[1],t->shape[1],v->dtype,context,&bytes,&alignment,d); if(s!=GFFX_STATUS_OK) return s;
    if(g==NULL) { out[0]=y; s=an_view(y,3,v->shape,v->dtype,1,d); }
    else { s=an_view(g,3,v->shape,v->dtype,0,d); if(s==GFFX_STATUS_OK) s=an_gradient_views(primal,out,3,d); }
    if(s!=GFFX_STATUS_OK) return s;
    s=an_noalias(inputs,g==NULL?3:4,out,g==NULL?1:3,workspace,d); if(s!=GFFX_STATUS_OK) return s;
#define AN_P(vv) ((vv)==NULL?NULL:gffx_mesh_elements(vv))
    if(v->dtype==GFFX_DTYPE_FLOAT32) an_skin_f32(gffx_mesh_elements_const(v),gffx_mesh_elements_const(t),gffx_mesh_elements_const(w),g==NULL?NULL:gffx_mesh_elements_const(g),AN_P(y),AN_P(gv),AN_P(gt),AN_P(gw),v->shape[0],v->shape[1],t->shape[1],g!=NULL);
    else an_skin_f64(gffx_mesh_elements_const(v),gffx_mesh_elements_const(t),gffx_mesh_elements_const(w),g==NULL?NULL:gffx_mesh_elements_const(g),AN_P(y),AN_P(gv),AN_P(gt),AN_P(gw),v->shape[0],v->shape[1],t->shape[1],g!=NULL);
#undef AN_P
    for(i=0;i<(g==NULL?1u:3u);++i) { s=an_result(out[i],d); if(s!=GFFX_STATUS_OK) return s; }
    return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_mesh_skin_lbs(const gffx_tensor_view *v,const gffx_tensor_view *t,const gffx_tensor_view *w,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    return an_skin_call(v,t,w,NULL,c,out,NULL,NULL,NULL,scratch,d);
}
GFFX_API gffx_status GFFX_CALL gffx_mesh_skin_lbs_backward(const gffx_tensor_view *v,const gffx_tensor_view *t,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *gv,gffx_tensor_view *gt,gffx_tensor_view *gw,
    const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    if(g==NULL) return an_fail(d,"skinning backward requires cotangent");
    return an_skin_call(v,t,w,g,c,NULL,gv,gt,gw,scratch,d);
}

#define AN_BARY(T,suffix) \
static void an_bary_##suffix(const T *v,const int32_t *f,const int32_t *idx,const T *w,const T *g, \
    T *y,T *gv,T *gw,int64_t B,int64_t V,int64_t N,int backward) { \
    int64_t b,n,p;int c,k; \
    if(!backward) { for(b=0;b<B;++b) for(n=0;n<N;++n) for(c=0;c<3;++c) { T sum=0; \
        for(k=0;k<3;++k) sum=sum+w[n*3+k]*v[(b*V+f[(int64_t)idx[n]*3+k])*3+c]; y[(b*N+n)*3+c]=sum; } return; } \
    if(gv) for(b=0;b<B;++b) for(p=0;p<V;++p) for(c=0;c<3;++c) { T sum=0; \
        for(n=0;n<N;++n) for(k=0;k<3;++k) if(f[(int64_t)idx[n]*3+k]==p) sum=sum+w[n*3+k]*g[(b*N+n)*3+c]; gv[(b*V+p)*3+c]=sum; } \
    if(gw) for(n=0;n<N;++n) for(k=0;k<3;++k) { T sum=0; \
        for(b=0;b<B;++b) for(c=0;c<3;++c) sum=sum+v[(b*V+f[(int64_t)idx[n]*3+k])*3+c]*g[(b*N+n)*3+c]; gw[n*3+k]=sum; } \
}
AN_BARY(float,f32)
AN_BARY(double,f64)

GFFX_API gffx_status GFFX_CALL gffx_mesh_evaluate_barycentric_workspace(int64_t B,int64_t V,int64_t N,gffx_dtype dtype,
    const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d) {
    if(c!=NULL&&c->device_type==GFFX_DEVICE_CUDA)return an_cuda_deformation_workspace(18,B,V,N,dtype,c,bytes,alignment,d);
    return gffx_mesh_blend_shapes_workspace(B,V,N,dtype,c,bytes,alignment,d);
}
static gffx_status an_bary_call(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *idx,
    const gffx_tensor_view *w,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,
    gffx_tensor_view *gv,gffx_tensor_view *gw,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    if(c!=NULL&&c->device_type==GFFX_DEVICE_CUDA)return an_cuda_bary(v,f,idx,w,g,c,y,gv,gw,scratch,d);
    int64_t vs[3]={-1,-1,3},fs[2]={-1,3},is[1]={-1},ws[2],ys[3],i; uint64_t bytes,alignment;
    const int32_t *fd,*id; const gffx_tensor_view *inputs[5]={v,f,idx,w,g},*primal[2]={v,w};
    gffx_tensor_view *outputs[2]={gv,gw}; gffx_status s=an_context(c,scratch,d);
    if(s!=GFFX_STATUS_OK) return s;
    if(v==NULL || (v->dtype!=GFFX_DTYPE_FLOAT32 && v->dtype!=GFFX_DTYPE_FLOAT64)) return an_fail(d,"vertices require float32 or float64");
    s=an_view(v,3,vs,v->dtype,0,d); if(s!=GFFX_STATUS_OK) return s;
    s=an_view(f,2,fs,GFFX_DTYPE_INT32,0,d); if(s!=GFFX_STATUS_OK) return s;
    s=an_view(idx,1,is,GFFX_DTYPE_INT32,0,d); if(s!=GFFX_STATUS_OK) return s;
    ws[0]=idx->shape[0];ws[1]=3;
    s=an_view(w,2,ws,v->dtype,0,d); if(s!=GFFX_STATUS_OK) return s;
    s=gffx_mesh_evaluate_barycentric_workspace(v->shape[0],v->shape[1],idx->shape[0],v->dtype,c,&bytes,&alignment,d); if(s!=GFFX_STATUS_OK) return s;
    fd=gffx_mesh_elements_const(f);id=gffx_mesh_elements_const(idx);
    for(i=0;i<f->shape[0]*3;++i) if(fd[i]<0||fd[i]>=v->shape[1]) return an_fail(d,"faces contain an out-of-range vertex");
    for(i=0;i<idx->shape[0];++i) if(id[i]<0||id[i]>=f->shape[0]) return an_fail(d,"landmark face_index is out of range");
    ys[0]=v->shape[0];ys[1]=idx->shape[0];ys[2]=3;
    if(g==NULL) { outputs[0]=y;s=an_view(y,3,ys,v->dtype,1,d); }
    else { s=an_view(g,3,ys,v->dtype,0,d);if(s==GFFX_STATUS_OK)s=an_gradient_views(primal,outputs,2,d); }
    if(s!=GFFX_STATUS_OK) return s;
    s=an_noalias(inputs,g==NULL?4:5,outputs,g==NULL?1:2,scratch,d);if(s!=GFFX_STATUS_OK)return s;
#define AN_P(vv) ((vv)==NULL?NULL:gffx_mesh_elements(vv))
    if(v->dtype==GFFX_DTYPE_FLOAT32)an_bary_f32(gffx_mesh_elements_const(v),fd,id,gffx_mesh_elements_const(w),g==NULL?NULL:gffx_mesh_elements_const(g),AN_P(y),AN_P(gv),AN_P(gw),v->shape[0],v->shape[1],idx->shape[0],g!=NULL);
    else an_bary_f64(gffx_mesh_elements_const(v),fd,id,gffx_mesh_elements_const(w),g==NULL?NULL:gffx_mesh_elements_const(g),AN_P(y),AN_P(gv),AN_P(gw),v->shape[0],v->shape[1],idx->shape[0],g!=NULL);
#undef AN_P
    for(i=0;i<(g==NULL?1:2);++i){s=an_result(outputs[i],d);if(s!=GFFX_STATUS_OK)return s;} return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_mesh_evaluate_barycentric(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *idx,const gffx_tensor_view *w,
    const gffx_execution_context *c,gffx_tensor_view *y,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return an_bary_call(v,f,idx,w,NULL,c,y,NULL,NULL,scratch,d);}
GFFX_API gffx_status GFFX_CALL gffx_mesh_evaluate_barycentric_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *idx,const gffx_tensor_view *w,const gffx_tensor_view *g,
    const gffx_execution_context *c,gffx_tensor_view *gv,gffx_tensor_view *gw,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    if(g==NULL)return an_fail(d,"landmark backward requires cotangent");return an_bary_call(v,f,idx,w,g,c,NULL,gv,gw,scratch,d);}

GFFX_API gffx_status GFFX_CALL gffx_skeleton_forward_kinematics_workspace(int64_t B,int64_t J,gffx_dtype dtype,
    const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d) {
    if(c!=NULL&&c->device_type==GFFX_DEVICE_CUDA)return an_cuda_pose_workspace(16,B,J,dtype,c,bytes,alignment,d);
    uint64_t unused,unit;
    gffx_status s=gffx_mesh_blend_shapes_workspace(B,J,0,dtype,c,&unused,alignment,d);
    if(s!=GFFX_STATUS_OK)return s;
    if(bytes==NULL)return an_fail(d,"FK workspace size pointer is null");
    unit=dtype==GFFX_DTYPE_FLOAT32?4:8;
    if((uint64_t)B!=0 && (uint64_t)J>(UINT64_MAX-(uint64_t)J*8)/((uint64_t)B*32*unit))
        return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"FK workspace extent overflow");
    *bytes=(uint64_t)J*8+(uint64_t)B*(uint64_t)J*32*unit;*alignment=8;
    return GFFX_STATUS_OK;
}

/* Fixed tie ordering; no recursion, and complete cycle validation before output writes. */
static gffx_status an_order(const int32_t *parents,int64_t J,int32_t *order,int32_t *done,gffx_diagnostic_buffer *d) {
    int64_t j,step;int found;
    for(j=0;j<J;++j){done[j]=0;if(parents[j]<-1||parents[j]>=J||parents[j]==j)return an_fail(d,"invalid skeleton parent index");}
    for(step=0;step<J;++step){found=0;for(j=0;j<J;++j)if(!done[j]&&(parents[j]==-1||done[parents[j]])){
        order[step]=(int32_t)j;done[j]=1;found=1;break;}
        if(!found)return an_fail(d,"skeleton hierarchy contains a cycle");}
    return GFFX_STATUS_OK;
}

#define AN_FK_WRAPPER(T,S) \
static gffx_status an_fk_##S(const T *local,const int32_t *parents,const int32_t *order,const T *g, \
    T *out,T *world,T *adj,int64_t B,int64_t J,gffx_diagnostic_buffer *d) { \
    if(gffx_an_fk_##S(local,parents,order,g,out,world,adj,B,J))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"FK matrix product overflow"); \
    return GFFX_STATUS_OK; \
}
AN_FK_WRAPPER(float,f32)
AN_FK_WRAPPER(double,f64)
#undef AN_FK_WRAPPER

static gffx_status an_fk_call(const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_tensor_view *g,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    if(c!=NULL&&c->device_type==GFFX_DEVICE_CUDA)return an_cuda_fk(local,parents,g,c,out,scratch,d);
    int64_t ls[4]={-1,-1,4,4},ps[1];uint64_t bytes,alignment;int32_t *order,*done;char *world,*adj;
    const gffx_tensor_view *inputs[3]={local,parents,g};gffx_tensor_view *outputs[1]={out};
    gffx_status s=an_context(c,scratch,d);if(s!=GFFX_STATUS_OK)return s;
    if(local==NULL||(local->dtype!=GFFX_DTYPE_FLOAT32&&local->dtype!=GFFX_DTYPE_FLOAT64))return an_fail(d,"FK local requires float32 or float64");
    s=an_view(local,4,ls,local->dtype,0,d);if(s!=GFFX_STATUS_OK)return s;
    ps[0]=local->shape[1];s=an_view(parents,1,ps,GFFX_DTYPE_INT32,0,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_view(out,4,local->shape,local->dtype,1,d);if(s!=GFFX_STATUS_OK)return s;
    if(g!=NULL){s=an_view(g,4,local->shape,local->dtype,0,d);if(s!=GFFX_STATUS_OK)return s;}
    s=gffx_skeleton_forward_kinematics_workspace(local->shape[0],local->shape[1],local->dtype,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    if(bytes>0&&(scratch==NULL||scratch->capacity_bytes<bytes))return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"FK workspace is too small");
    if(bytes>0&&(scratch->alignment<alignment||(uintptr_t)scratch->data%alignment!=0))return an_fail(d,"FK workspace requires 8-byte alignment");
    s=an_noalias(inputs,g==NULL?2:3,outputs,1,scratch,d);if(s!=GFFX_STATUS_OK)return s;
    if(local->shape[1]==0)return GFFX_STATUS_OK;
    order=(int32_t *)scratch->data;done=order+local->shape[1];world=(char *)(done+local->shape[1]);
    adj=world+local->shape[0]*local->shape[1]*16*(local->dtype==GFFX_DTYPE_FLOAT32?4:8);
    s=an_order(gffx_mesh_elements_const(parents),local->shape[1],order,done,d);if(s!=GFFX_STATUS_OK)return s;
    if(local->dtype==GFFX_DTYPE_FLOAT32)s=an_fk_f32(gffx_mesh_elements_const(local),gffx_mesh_elements_const(parents),order,g==NULL?NULL:gffx_mesh_elements_const(g),gffx_mesh_elements(out),(float *)world,(float *)adj,local->shape[0],local->shape[1],d);
    else s=an_fk_f64(gffx_mesh_elements_const(local),gffx_mesh_elements_const(parents),order,g==NULL?NULL:gffx_mesh_elements_const(g),gffx_mesh_elements(out),(double *)world,(double *)adj,local->shape[0],local->shape[1],d);
    if(s!=GFFX_STATUS_OK)return s;return an_result(out,d);
}
GFFX_API gffx_status GFFX_CALL gffx_skeleton_forward_kinematics(const gffx_tensor_view *local,const gffx_tensor_view *parents,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return an_fk_call(local,parents,NULL,c,out,scratch,d);}
GFFX_API gffx_status GFFX_CALL gffx_skeleton_forward_kinematics_backward(const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_tensor_view *g,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){if(g==NULL)return an_fail(d,"FK backward requires cotangent");return an_fk_call(local,parents,g,c,out,scratch,d);}
