/* Opaque alpha is the union of projected triangles, independent of depth.
 * Constant scratch, no hidden allocation. This CPU reference does not qualify
 * colour filtering or CUDA; callers receive explicit unsupported status there. */
#include <gffx/render.h>
#include "internal.h"
#include "mesh_common.h"
#include <string.h>
#include <limits.h>
#include "opaque_polygon.h"
#define CANDIDATES 32
typedef struct { Poly work[2*MAX_PARTS]; Tri triangles[CANDIDATES]; int64_t ids[CANDIDATES]; } Scratch;
static double read_real(const gffx_tensor_view *v,int64_t i) {
    return v->dtype==GFFX_DTYPE_FLOAT64?((const double*)gffx_mesh_elements_const(v))[i]:((const float*)gffx_mesh_elements_const(v))[i];
}
static void write_real(gffx_tensor_view *v,int64_t i,double a) {
    if(v->dtype==GFFX_DTYPE_FLOAT64)((double*)gffx_mesh_elements(v))[i]=a;
    else ((float*)gffx_mesh_elements(v))[i]=(float)a;
}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_coverage_workspace(
    gffx_dtype dtype,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic) {
    gffx_status s=gffx_internal_prepare_diagnostic(diagnostic);if(s)return s;
    if(!bytes || !alignment)return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque workspace result pointers required");
    s=gffx_validate_execution_context(context,diagnostic);if(s)return s;
    if(context->device_type!=GFFX_DEVICE_CPU || (dtype!=GFFX_DTYPE_FLOAT32 && dtype!=GFFX_DTYPE_FLOAT64))
        return gffx_internal_fail(diagnostic,GFFX_STATUS_UNSUPPORTED,"opaque coverage currently supports CPU float32/float64 only");
    *bytes=(uint64_t)sizeof(Scratch);*alignment=8;return GFFX_STATUS_OK;
}
static Tri make_triangle(const gffx_tensor_view *v,const int32_t *f,int64_t seed,double ox,double oy) {
    P xy[3];Tri t;D det;int i,k;memset(&t,0,sizeof(t));
    for(i=0;i<3;++i){xy[i].x.v=read_real(v,3*(int64_t)f[i])-ox;xy[i].y.v=read_real(v,3*(int64_t)f[i]+1)-oy;
        xy[i].x.d=seed==3*(int64_t)f[i]?1:0;xy[i].y.d=seed==3*(int64_t)f[i]+1?1:0;}
    det=sub(mul(sub(xy[1].x,xy[0].x),sub(xy[2].y,xy[0].y)),mul(sub(xy[1].y,xy[0].y),sub(xy[2].x,xy[0].x)));
    if(det.v==0)return t;
    for(i=0;i<3;++i){P a=xy[i],b=xy[(i+1)%3];D dx=sub(b.x,a.x),dy=sub(b.y,a.y);
        t.edge[i].a=neg(dy);t.edge[i].b=dx;t.edge[i].c=sub(mul(dy,a.x),mul(dx,a.y));
        if(det.v<0)t.edge[i]=negative(t.edge[i]);}
    t.valid=1;return t;
}
static D union_area(Scratch *scratch,int n,double x,double y,Ctx *ctx) {
    Poly p;D total=val(0);int i,j,k;p.n=4;p.p[0]=point(x,y);p.p[1]=point(x+1,y);p.p[2]=point(x+1,y+1);p.p[3]=point(x,y+1);
    for(i=0;i<n && !ctx->failed;++i)if(scratch->triangles[i].valid) {
        Poly subject=p,*a=scratch->work,*b=scratch->work+MAX_PARTS;int count=1;
        for(k=0;k<3;++k)subject=clip(&subject,scratch->triangles[i].edge[k],ctx);
        if(!subject.n || area(&subject).v==0)continue;a[0]=subject;
        for(j=0;j<i && count && !ctx->failed;++j)if(scratch->triangles[j].valid) {
            int m=0;for(k=0;k<count && !ctx->failed;++k)subtract_triangle(a+k,scratch->triangles+j,b,&m,ctx);
            {Poly *temp=a;a=b;b=temp;}count=m;
        }
        for(k=0;k<count;++k)total=add(total,area(a+k));
    }
    if(!isfinite(total.v) || !isfinite(total.d))ctx->failed=1;
    return total;
}
static gffx_status check(const gffx_tensor_view *v,const gffx_tensor_view *f,int64_t h,int64_t w,
    double ox,double oy,const gffx_tensor_view *gradient,const gffx_execution_context *context,
    gffx_tensor_view *out,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic) {
    uint64_t bytes,alignment;gffx_status s;int64_t i;const gffx_tensor_view *inputs[3]={v,f,gradient};int k;
    s=gffx_internal_prepare_diagnostic(diagnostic);if(s)return s;
    if(h<1 || w<1 || h>INT64_MAX/w || !isfinite(ox) || !isfinite(oy) || fabs(ox)>1048576 || fabs(oy)>1048576 || h>1048576 || w>1048576)
        return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque image/origin dimensions exceed the supported finite range");
    s=gffx_mesh_check_common(v,f,0,context,workspace,diagnostic);if(s)return s;
    s=gffx_render_opaque_coverage_workspace(v->dtype,context,&bytes,&alignment,diagnostic);if(s)return s;
    s=gffx_mesh_check_view(out,"opaque output shape",2,gradient?v->shape[0]:h,gradient?3:w,1,diagnostic);if(s)return s;
    if(out->dtype!=v->dtype)return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque output dtype must match vertices");
    if(gradient){s=gffx_mesh_check_view(gradient,"opaque gradient must be [H,W]",2,h,w,0,diagnostic);if(s)return s;
        if(gradient->dtype!=v->dtype)return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque gradient dtype mismatch");
        for(i=0;i<h*w;++i)if(!isfinite(read_real(gradient,i)))return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque gradient must be finite");}
    for(i=0;i<v->shape[0]*3;++i)if(!isfinite(read_real(v,i)) || fabs(read_real(v,i))>1048576)
        return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque coordinates must be finite within +/-1048576");
    if(!workspace || !workspace->data || workspace->capacity_bytes<bytes)
        return gffx_internal_fail(diagnostic,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"opaque workspace capacity insufficient");
    if((uintptr_t)workspace->data%alignment)return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque workspace must be 8-byte aligned");
    for(k=0;k<3;++k)if(inputs[k] && (gffx_mesh_views_overlap(inputs[k],out) || gffx_mesh_range_overlaps_view(workspace->data,bytes,inputs[k])))
        return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque input/output/workspace overlap");
    if(gffx_mesh_range_overlaps_view(workspace->data,bytes,out))return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque output/workspace overlap");
    return GFFX_STATUS_OK;
}
static gffx_status run(const gffx_tensor_view *v,const gffx_tensor_view *faces,int64_t h,int64_t w,
    double ox,double oy,const gffx_tensor_view *gradient,const gffx_execution_context *context,
    gffx_tensor_view *out,const gffx_buffer *workspace,gffx_diagnostic_buffer *diagnostic) {
    gffx_status s=check(v,faces,h,w,ox,oy,gradient,context,out,workspace,diagnostic);
    Scratch *scratch;const int32_t *f;int64_t y,x,fi;uint64_t out_n;int i,k,n;
    if(s)return s;scratch=(Scratch*)workspace->data;f=(const int32_t*)gffx_mesh_elements_const(faces);
    out_n=gffx_mesh_element_count(out);for(uint64_t q=0;q<out_n;++q)write_real(out,(int64_t)q,0);
    for(y=0;y<h;++y)for(x=0;x<w;++x) {
        double px=ox+(double)x,py=oy+(double)y;int64_t vertices[CANDIDATES*3];int nv=0;
        n=0;
        for(fi=0;fi<faces->shape[0];++fi) {
            double lo_x=read_real(v,3*(int64_t)f[3*fi]),hi_x=lo_x,lo_y=read_real(v,3*(int64_t)f[3*fi]+1),hi_y=lo_y;
            for(k=1;k<3;++k){double a=read_real(v,3*(int64_t)f[3*fi+k]),b=read_real(v,3*(int64_t)f[3*fi+k]+1);
                if(a<lo_x)lo_x=a;if(a>hi_x)hi_x=a;if(b<lo_y)lo_y=b;if(b>hi_y)hi_y=b;}
            /* Conservative closed bounds: never use sample hits to discover geometry. */
            if(hi_x<px || lo_x>px+1 || hi_y<py || lo_y>py+1)continue;
            if(n==CANDIDATES)return gffx_internal_fail(diagnostic,GFFX_STATUS_OVERFLOW,"opaque candidate capacity 32 exceeded; no partial output is valid");
            scratch->ids[n++]=fi;
        }
        /* Pixel-local arithmetic avoids cancellation of large global areas.
         * The origin is constant, so geometry seeds retain their derivatives. */
        for(i=0;i<n;++i)scratch->triangles[i]=make_triangle(v,f+3*scratch->ids[i],-1,px,py);
        {Ctx ctx={0,0};D total=union_area(scratch,n,0,0,&ctx);
            if(ctx.failed)return gffx_internal_fail(diagnostic,GFFX_STATUS_OVERFLOW,"opaque polygon/work capacity exceeded; no partial output is valid");
            if(!gradient){write_real(out,y*w+x,total.v);continue;}}
        /* Ordered transpose of exact scalar duals. Reference path, not a scale claim. */
        for(i=0;i<n;++i)for(k=0;k<3;++k){int64_t id=f[3*scratch->ids[i]+k];int j;for(j=0;j<nv;++j)if(vertices[j]==id)break;if(j==nv)vertices[nv++]=id;}
        for(i=0;i<nv;++i)for(k=0;k<2;++k) {
            int j;int64_t seed=3*vertices[i]+k;Ctx ctx={0,0};D total;
            for(j=0;j<n;++j)scratch->triangles[j]=make_triangle(v,f+3*scratch->ids[j],seed,px,py);
            total=union_area(scratch,n,0,0,&ctx);
            if(ctx.failed)return gffx_internal_fail(diagnostic,GFFX_STATUS_OVERFLOW,"opaque derivative work capacity exceeded; no partial output is valid");
            write_real(out,seed,read_real(out,seed)+total.d*read_real(gradient,y*w+x));
        }
    }
    return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_coverage(const gffx_tensor_view *v,const gffx_tensor_view *f,int64_t h,int64_t w,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *diag) {return run(v,f,h,w,ox,oy,NULL,ctx,out,scratch,diag);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_coverage_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *gradient,const gffx_execution_context *ctx,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *diag) {return run(v,f,h,w,ox,oy,gradient,ctx,out,scratch,diag);}
