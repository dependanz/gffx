/* Exact integration of piecewise screen-affine radiance over visible polygons.
 * CPU reference, explicit caller scratch; no hidden state or nonlinear shader. */
#include <gffx/render.h>
#include "internal.h"
#include "mesh_common.h"
#include <string.h>
#include <limits.h>
#include "cuda_loader.h"
#include "opaque_quadrature_scalar.h"
#include "opaque_boundary_scalar.h"
#include "opaque_batch_scalar.h"
#include "opaque_partition_scalar.h"
#include "opaque_pair_dispatch.h"
_Static_assert(sizeof(pair_Scratch)<=sizeof(Scratch),"paired scratch exceeds scalar allocation");
static void moments(const Poly *p,D *a,D *x,D *y) {
    int k;*a=val(0);*x=val(0);*y=val(0);
    for(k=0;k<p->n;++k){P u=p->p[k],v=p->p[(k+1)%p->n];D c=sub(mul(u.x,v.y),mul(u.y,v.x));
        *a=add(*a,c);*x=add(*x,mul(add(u.x,v.x),c));*y=add(*y,mul(add(u.y,v.y),c));}
    if(a->v<0){*a=neg(*a);*x=neg(*x);*y=neg(*y);}
    *a=mul(*a,val(.5));*x=divd(*x,val(6));*y=divd(*y,val(6));
}
static void pixel(Scratch *s,int n,const gffx_tensor_view *colour,D rgba[4],Ctx *ctx) {
    Poly p;int i,j,k,c;p.n=4;p.p[0]=point(0,0);p.p[1]=point(1,0);p.p[2]=point(1,1);p.p[3]=point(0,1);
    for(c=0;c<4;++c)rgba[c]=val(0);
    for(i=0;i<n;++i)for(k=0;k<3;++k)s->weights[i][k]=val(0);
    for(i=0;i<n && !ctx->failed;++i)if(s->triangles[i].tri.valid){
        Poly subject=p,*a=s->work,*b=s->work+MAX_PARTS;int count=1;
        for(k=0;k<3;++k)subject=clip(&subject,s->triangles[i].tri.edge[k],ctx);
        if(!subject.n || area(&subject).v==0)continue;a[0]=subject;
        for(j=0;j<n && count && !ctx->failed;++j)if(i!=j && s->triangles[j].tri.valid){
            Line d=difference(s->triangles[i].depth,s->triangles[j].depth);int m=0,tie=d.a.v==0 && d.b.v==0 && d.c.v==0;
            if(tie && j>i)continue;
            for(k=0;k<count && !ctx->failed;++k)subtract_visible(a+k,s->triangles+j,d,tie,b,&m,ctx,NULL,NULL,NULL,NULL,0,0);
            {Poly *tmp=a;a=b;b=tmp;}count=m;
        }
        for(k=0;k<count;++k){D ar,mx,my;int corner;moments(a+k,&ar,&mx,&my);rgba[3]=add(rgba[3],ar);
            for(corner=0;corner<3;++corner){Line q=s->triangles[i].bary[corner];D weight=add(add(mul(q.a,mx),mul(q.b,my)),mul(q.c,ar));
                s->weights[i][corner]=add(s->weights[i][corner],weight);}}
        for(k=0;k<3;++k)for(c=0;c<3;++c)rgba[c]=add(rgba[c],mul(s->weights[i][k],val(read_real(colour,9*s->ids[i]+3*k+c))));
    }
    for(c=0;c<4;++c)if(!isfinite(rgba[c].v)||!isfinite(rgba[c].d))ctx->failed=1;
}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_integrate_workspace(gffx_dtype dtype,const gffx_execution_context *context,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *diagnostic) {
    gffx_status s=gffx_internal_prepare_diagnostic(diagnostic);if(s)return s;
    if(!bytes || !alignment)return gffx_internal_fail(diagnostic,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour workspace result pointers required");
    s=gffx_validate_execution_context(context,diagnostic);if(s)return s;
    if(context->device_type!=GFFX_DEVICE_CPU || (dtype!=GFFX_DTYPE_FLOAT32 && dtype!=GFFX_DTYPE_FLOAT64))return gffx_internal_fail(diagnostic,GFFX_STATUS_UNSUPPORTED,"opaque colour currently supports CPU float32/float64 only");
    *bytes=sizeof(Scratch);*alignment=8;return GFFX_STATUS_OK;
}
static gffx_status view3(const gffx_tensor_view *v,int64_t a,int64_t b,int64_t c,int output,gffx_dtype dtype,gffx_diagnostic_buffer *d) {
    gffx_status s=gffx_mesh_check_view(v,"opaque colour tensor shape",3,a,0,output,d);if(s)return s;
    if(v->shape[1]!=b || v->shape[2]!=c || v->dtype!=dtype)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour tensor shape/dtype mismatch");return GFFX_STATUS_OK;
}
static gffx_status run(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *colour,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *gradient,const gffx_execution_context *context,gffx_tensor_view *out,gffx_tensor_view *gc,const gffx_buffer *workspace,gffx_diagnostic_buffer *d) {
    gffx_status status;uint64_t bytes,alignment,q;int64_t i,x,y;int j,k,n;Scratch *s;const int32_t *faces;
    const gffx_tensor_view *inputs[4]={v,f,colour,gradient};gffx_tensor_view *outputs[2]={out,gc};
    status=gffx_internal_prepare_diagnostic(d);if(status)return status;
    if(h<1||w<1||h>1048576||w>1048576||!isfinite(ox)||!isfinite(oy)||fabs(ox)>1048576||fabs(oy)>1048576)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour image/origin range");
    status=gffx_mesh_check_common(v,f,0,context,workspace,d);if(status)return status;
    status=gffx_render_opaque_integrate_workspace(v->dtype,context,&bytes,&alignment,d);if(status)return status;
    status=view3(colour,f->shape[0],3,3,0,v->dtype,d);if(status)return status;
    if(gradient){
        status=view3(gradient,h,w,4,0,v->dtype,d);if(status)return status;
        status=gffx_mesh_check_view(out,"opaque vertex gradient shape",2,v->shape[0],3,1,d);if(status)return status;
        if(out->dtype!=v->dtype)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque vertex gradient dtype");
        status=view3(gc,f->shape[0],3,3,1,v->dtype,d);if(status)return status;
    }else{status=view3(out,h,w,4,1,v->dtype,d);if(status)return status;}
    for(j=0;j<4;++j)if(inputs[j] && j!=1)for(q=0;q<gffx_mesh_element_count(inputs[j]);++q){double a=read_real(inputs[j],(int64_t)q);
        if(!isfinite(a)||(j==0 && fabs(a)>1048576))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour inputs must be finite and coordinates bounded");}
    if(!workspace||!workspace->data||workspace->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"opaque colour workspace capacity insufficient");
    if((uintptr_t)workspace->data%alignment)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour workspace alignment");
    for(j=0;j<4;++j)if(inputs[j]){
        if(gffx_mesh_range_overlaps_view(workspace->data,bytes,inputs[j]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour input/workspace overlap");
        for(k=0;k<(gradient?2:1);++k)if(gffx_mesh_views_overlap(inputs[j],outputs[k]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour input/output overlap");}
    for(k=0;k<(gradient?2:1);++k)if(gffx_mesh_range_overlaps_view(workspace->data,bytes,outputs[k]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour output/workspace overlap");
    if(gradient&&gffx_mesh_views_overlap(out,gc))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque colour gradient outputs overlap");
    for(k=0;k<(gradient?2:1);++k)for(q=0;q<gffx_mesh_element_count(outputs[k]);++q)write_real(outputs[k],(int64_t)q,0);
    s=(Scratch*)workspace->data;faces=(const int32_t*)gffx_mesh_elements_const(f);
    for(y=0;y<h;++y)for(x=0;x<w;++x){double px=ox+(double)x,py=oy+(double)y;int64_t vertices[3*CANDIDATES];int nv=0;D rgba[4];Ctx ctx={0,0};n=0;
        for(i=0;i<f->shape[0];++i){double lx=read_real(v,3*(int64_t)faces[3*i]),hx=lx,ly=read_real(v,3*(int64_t)faces[3*i]+1),hy=ly;
            for(k=1;k<3;++k){double a=read_real(v,3*(int64_t)faces[3*i+k]),b=read_real(v,3*(int64_t)faces[3*i+k]+1);if(a<lx)lx=a;if(a>hx)hx=a;if(b<ly)ly=b;if(b>hy)hy=b;}
            if(hx<px||lx>px+1||hy<py||ly>py+1)continue;
            if(n==CANDIDATES)return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"opaque colour candidate capacity32 exceeded; partial outputs invalid");s->ids[n++]=i;}
        for(j=0;j<n;++j)s->triangles[j]=triangle(v,faces+3*s->ids[j],-1,px,py);
        pixel(s,n,colour,rgba,&ctx);if(ctx.failed)return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"opaque colour polygon/work/arithmetic overflow; partial outputs invalid");
        if(!gradient){for(k=0;k<4;++k)write_real(out,4*(y*w+x)+k,rgba[k].v);continue;}
        for(j=0;j<n;++j)for(k=0;k<3;++k){int c;for(c=0;c<3;++c){int64_t index=9*s->ids[j]+3*k+c;write_real(gc,index,read_real(gc,index)+s->weights[j][k].v*read_real(gradient,4*(y*w+x)+c));}}
        for(j=0;j<n;++j)for(k=0;k<3;++k){int a;int64_t id=faces[3*s->ids[j]+k];for(a=0;a<nv;++a)if(vertices[a]==id)break;if(a==nv)vertices[nv++]=id;}
        for(j=0;j<nv;++j)for(k=0;k<3;++k){int a;int64_t seed=3*vertices[j]+k;double value=0;ctx.failed=0;ctx.work=0;
            for(a=0;a<n;++a)s->triangles[a]=triangle(v,faces+3*s->ids[a],seed,px,py);
            pixel(s,n,colour,rgba,&ctx);if(ctx.failed)return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"opaque colour derivative capacity/arithmetic overflow; partial outputs invalid");
            for(a=0;a<4;++a)value+=rgba[a].d*read_real(gradient,4*(y*w+x)+a);
            write_real(out,seed,read_real(out,seed)+value);}
    }
    for(k=0;k<(gradient?2:1);++k)for(q=0;q<gffx_mesh_element_count(outputs[k]);++q)if(!isfinite(read_real(outputs[k],(int64_t)q)))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"opaque colour output accumulation overflow");
    return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_integrate(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *c,int64_t h,int64_t w,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {return run(v,f,c,h,w,ox,oy,NULL,ctx,out,NULL,scratch,d);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_integrate_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *c,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {return run(v,f,c,h,w,ox,oy,g,ctx,gv,gc,scratch,d);}

/* Quadrature nodes over the same visible polygons. Shader evaluation remains a
 * composition over these native outputs. A quadrature rule is not an a priori
 * error certificate for an arbitrary supplied shader. See the source contract. */
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_quadrature_workspace(gffx_dtype dtype,const gffx_execution_context *ctx,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
    gffx_status s=gffx_internal_prepare_diagnostic(d);const gffx_cuda_operations *ops;if(s)return s;
    s=gffx_validate_execution_context(ctx,d);if(s)return s;
    if(!bytes||!alignment)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature workspace results required");
    if(ctx->device_type==GFFX_DEVICE_CUDA){ops=gffx_cuda_loader_operations();
        if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_quadrature))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA opaque quadrature unavailable");
        return ops->workspace_query(GFFX_CUDA_OP_RENDER_OPAQUE_QUADRATURE,NULL,0,dtype,ctx,bytes,alignment,d);}
    return gffx_render_opaque_integrate_workspace(dtype,ctx,bytes,alignment,d);
}
static gffx_status quadrature_cuda_metadata(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int edge,const int32_t *cursor,int32_t *receipt,int rule,int derivatives,int32_t *prefix){
    const gffx_tensor_view *a[6]={v,f,cuts,g,out,other};int j,k;uint64_t bytes,alignment;gffx_status s;
    s=gffx_render_opaque_quadrature_workspace(v?v->dtype:GFFX_DTYPE_FLOAT64,ctx,&bytes,&alignment,d);if(s)return s;
    for(j=0;j<6;++j){if(j==3&&!g)continue;s=gffx_validate_tensor_view(a[j],d);if(s)return s;
        if(a[j]->device_type!=ctx->device_type||a[j]->device_index!=ctx->device_index||((a[j]->flags&GFFX_TENSOR_OUTPUT)!=0)!=(j>=4))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature device/flags mismatch");
        if(a[j]->dtype!=((j==1||(!g&&j==4))?GFFX_DTYPE_INT32:v->dtype))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature dtype mismatch");}
    if(v->rank!=2||v->shape[1]!=3||f->rank!=2||f->shape[1]!=3||cuts->rank!=3||cuts->shape[0]!=f->shape[0]||!((cuts->shape[1]==3&&cuts->shape[2]<=(rule==4?40:16))||(cuts->shape[1]==1&&cuts->shape[2]>=6&&cuts->shape[2]<=51&&(cuts->shape[2]-6)%3==0)))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature input shapes");
    if(g){if(g->rank!=3||g->shape[0]!=h*w||g->shape[1]!=800||g->shape[2]!=4||out->rank!=2||out->shape[0]!=v->shape[0]||out->shape[1]!=3||other->rank!=3||other->shape[0]!=f->shape[0]||other->shape[1]!=cuts->shape[1]||other->shape[2]!=cuts->shape[2])return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature backward shapes");}
    else if(out->rank!=2||out->shape[0]!=h*w||out->shape[1]!=800||other->rank!=3||other->shape[0]!=h*w||other->shape[1]!=800||other->shape[2]!=4)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature output shapes");
    if(!scratch)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"quadrature caller scratch required");s=gffx_validate_buffer(scratch,d);if(s)return s;
    if(scratch->device_type!=ctx->device_type||scratch->device_index!=ctx->device_index||scratch->alignment<alignment||(uintptr_t)scratch->data%alignment)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature scratch device/alignment");
    if(!scratch->data||scratch->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"quadrature scratch capacity");
    for(j=0;j<6;++j)if(a[j]){if(gffx_mesh_range_overlaps_view(scratch->data,scratch->capacity_bytes,a[j]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature scratch alias");
        for(k=j+1;k<6;++k)if(a[k]&&k>=4&&gffx_mesh_views_overlap(a[j],a[k]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature output alias");}
    if(prefix){
      if((uintptr_t)prefix%4)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix alignment");
      for(j=0;j<6;++j)if(a[j]&&gffx_mesh_range_overlaps_view(prefix,8,a[j]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix tensor alias");
      if(((uintptr_t)prefix<(uintptr_t)scratch->data+bytes&&(uintptr_t)scratch->data<(uintptr_t)prefix+8)||((uintptr_t)prefix<(uintptr_t)cursor+16&&(uintptr_t)cursor<(uintptr_t)prefix+8)||((uintptr_t)prefix<(uintptr_t)receipt+32&&(uintptr_t)receipt<(uintptr_t)prefix+8))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix metadata/scratch alias");
    }
    if(cursor){const gffx_cuda_operations *ops=gffx_cuda_loader_operations();
      if(edge>=4&&rule==4){if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_boundary_batch80))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA boundary stream unavailable");return ops->render_opaque_boundary_batch80(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,edge-4,0);}
      if(edge>=4){if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_boundary_batch64))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA boundary stream unavailable");return ops->render_opaque_boundary_batch64(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,edge-4,0);}
      if(prefix&&rule==4){if(!GFFX_CUDA_HAS_OPERATION(ops,opaque_prefix_backward80))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA persistent prefix unavailable");return ops->opaque_prefix_backward80(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,derivatives,prefix);}
      if(prefix&&rule==3){if(!GFFX_CUDA_HAS_OPERATION(ops,opaque_prefix_backward64))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA persistent prefix unavailable");return ops->opaque_prefix_backward64(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,derivatives,prefix);}
      if(prefix){if(!GFFX_CUDA_HAS_OPERATION(ops,opaque_prefix_backward))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA persistent prefix unavailable");return ops->opaque_prefix_backward(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,derivatives,prefix);}

      if(rule==4){if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_batch_rule80))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA explicit opaque rule unavailable");return ops->render_opaque_batch_rule80(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,rule,derivatives);}
      if(rule==3){if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_batch_rule64))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA explicit opaque rule unavailable");return ops->render_opaque_batch_rule64(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,rule,derivatives);}
      if(rule){if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_batch_rule))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA explicit opaque rule unavailable");return ops->render_opaque_batch_rule(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d,rule,derivatives);}
      if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_batch))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA opaque batch unavailable");
      return ops->render_opaque_batch(v,f,cuts,ox,oy,cursor,receipt,g,ctx,out,other,scratch,d);}
    if(edge>=0){const gffx_cuda_operations *ops=gffx_cuda_loader_operations();
      if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_boundary))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA opaque boundary unavailable");
      return ops->render_opaque_boundary(v,f,cuts,h,w,ox,oy,edge,ctx,out,other,scratch,d);}
    if(edge==-2){const gffx_cuda_operations *ops=gffx_cuda_loader_operations();
      if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_material))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA opaque material transport unavailable");
      return ops->render_opaque_material(v,f,cuts,h,w,ox,oy,g,ctx,out,other,scratch,d);}
    return gffx_cuda_loader_operations()->render_opaque_quadrature(v,f,cuts,h,w,ox,oy,g,ctx,out,other,scratch,d);
}
static gffx_status quadrature_run(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *gradient,const gffx_execution_context *context,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *workspace,gffx_diagnostic_buffer *d,int edge,const int32_t *cursor,int32_t *receipt,int rule,int derivatives,int32_t *prefix){
    gffx_status status;uint64_t bytes,alignment,q;int64_t i,x,y,pixels;int j,k,n,nc;Scratch *s;const int32_t *faces;
    const gffx_tensor_view *inputs[4]={v,f,cuts,gradient};gffx_tensor_view *outputs[2]={out,other};
    status=gffx_internal_prepare_diagnostic(d);if(status)return status;
    if(derivatives<0||derivatives>3||rule<0||rule>4||(rule && ((edge!=-3&&edge<4)||!cuts||cuts->rank!=3||cuts->shape[1]!=3)))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque batch rule requires DEFAULT, SH_BILINEAR25 or SH_BILINEAR25_STREAM52 without curves");
    if(edge==-3||edge>=4){QuadBatch checked;
      if(!cursor||!receipt||quad_batch_init_profile(&checked,cursor[0],cursor[1],cursor[2],cursor[3],rule==4?80:(rule==3?64:(rule==2?52:16))))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque batch cursor/receipt");
      if((uintptr_t)cursor<(uintptr_t)receipt+32&&(uintptr_t)receipt<(uintptr_t)cursor+16)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque batch metadata alias");
    }
    if(h<1||w<1||h>1048576||w>1048576||!isfinite(ox)||!isfinite(oy)||fabs(ox)>1048576||fabs(oy)>1048576)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque quadrature image/origin range");pixels=h*w;
    status=gffx_validate_execution_context(context,d);if(status)return status;
    if(prefix && ((rule!=2&&rule!=3&&rule!=4)||!gradient))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"prefix requires SH52 orSH64 first VJP");
    if(context->device_type==GFFX_DEVICE_CUDA)return quadrature_cuda_metadata(v,f,cuts,h,w,ox,oy,gradient,context,out,other,workspace,d,edge,cursor,receipt,rule,derivatives,prefix);
    status=gffx_mesh_check_common(v,f,0,context,workspace,d);if(status)return status;
    status=gffx_render_opaque_quadrature_workspace(v->dtype,context,&bytes,&alignment,d);if(status)return status;
    status=gffx_mesh_check_view(cuts,"quadrature cuts shape",3,f->shape[0],0,0,d);if(status)return status;
    if(!((cuts->shape[1]==3&&cuts->shape[2]>=0&&cuts->shape[2]<=(rule==4?40:16))||(cuts->shape[1]==1&&cuts->shape[2]>=6&&cuts->shape[2]<=51&&(cuts->shape[2]-6)%3==0)) || cuts->dtype!=v->dtype)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque quadrature cuts require[F,3,K] K<=16 or packed[F,1,3*K+6] K<=15");nc=(int)cuts->shape[2];
    if(gradient){
        status=view3(gradient,pixels,QUAD_SLOTS,4,0,v->dtype,d);if(status)return status;
        status=gffx_mesh_check_view(out,"quadrature vertex gradient",2,v->shape[0],3,1,d);if(status)return status;
        if(out->dtype!=v->dtype)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature vertex gradient dtype");
        status=view3(other,f->shape[0],cuts->shape[1],nc,1,v->dtype,d);if(status)return status;
    }else{
        status=gffx_mesh_check_view(out,"quadrature owners",2,pixels,QUAD_SLOTS,1,d);if(status)return status;
        if(out->dtype!=GFFX_DTYPE_INT32)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature owners require int32");
        status=view3(other,pixels,QUAD_SLOTS,4,1,v->dtype,d);if(status)return status;
    }
    for(j=0;j<4;++j)if(inputs[j] && j!=1)for(q=0;q<gffx_mesh_element_count(inputs[j]);++q){double value=read_real(inputs[j],(int64_t)q);
        if(!isfinite(value)||(j==0 && fabs(value)>1048576))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature inputs must be finite and coordinates bounded");}
    if(!workspace||!workspace->data||workspace->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"quadrature workspace capacity insufficient");
    if((uintptr_t)workspace->data%alignment)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature workspace alignment");
    for(j=0;j<4;++j)if(inputs[j]){
        if(gffx_mesh_range_overlaps_view(workspace->data,bytes,inputs[j]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature input/workspace overlap");
        for(k=0;k<2;++k)if(gffx_mesh_views_overlap(inputs[j],outputs[k]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature input/output overlap");}
    for(k=0;k<2;++k)if(gffx_mesh_range_overlaps_view(workspace->data,bytes,outputs[k]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature output/workspace overlap");
    if(gffx_mesh_views_overlap(out,other))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"quadrature outputs overlap");
    if(cursor){const gffx_tensor_view *all[6]={v,f,cuts,gradient,out,other};
      for(j=0;j<6;++j)if(all[j]&&(gffx_mesh_range_overlaps_view(cursor,16,all[j])||gffx_mesh_range_overlaps_view(receipt,32,all[j])))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque batch metadata/tensor alias");
      if(((uintptr_t)cursor<(uintptr_t)workspace->data+bytes&&(uintptr_t)workspace->data<(uintptr_t)cursor+16)||((uintptr_t)receipt<(uintptr_t)workspace->data+bytes&&(uintptr_t)workspace->data<(uintptr_t)receipt+32))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque batch metadata/scratch alias");
    }
    if(prefix){const gffx_tensor_view *all[6]={v,f,cuts,gradient,out,other};if((uintptr_t)prefix%4)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix alignment");
      for(j=0;j<6;++j)if(all[j]&&gffx_mesh_range_overlaps_view(prefix,8,all[j]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix tensor alias");
      if(((uintptr_t)prefix<(uintptr_t)workspace->data+bytes&&(uintptr_t)workspace->data<(uintptr_t)prefix+8)||((uintptr_t)prefix<(uintptr_t)cursor+16&&(uintptr_t)cursor<(uintptr_t)prefix+8)||((uintptr_t)prefix<(uintptr_t)receipt+32&&(uintptr_t)receipt<(uintptr_t)prefix+8))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix metadata/scratch alias");
    }
    if(edge>=4){if((rule==4?opaque_boundary_batch_execute80:opaque_boundary_batch_execute)(v,f,cuts,ox,oy,edge-4,cursor,receipt,out,other,(Scratch*)workspace->data))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"opaque boundary stream capacity/arithmetic overflow; partial outputs invalid");return GFFX_STATUS_OK;}
    if(cursor){if(opaque_batch_execute_prefix_shared(v,f,cuts,ox,oy,cursor,receipt,gradient,out,other,(Scratch*)workspace->data,rule,derivatives,prefix))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"opaque batch capacity/arithmetic overflow; partial outputs invalid");return GFFX_STATUS_OK;}
    if(edge>=0?opaque_boundary_execute(v,f,cuts,h,w,ox,oy,edge,out,other,(Scratch*)workspace->data):opaque_quadrature_execute(v,f,cuts,h,w,ox,oy,gradient,out,other,(Scratch*)workspace->data,edge==-2))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"quadrature capacity/arithmetic overflow; partial outputs invalid");
    return GFFX_STATUS_OK;
}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_quadrature(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return quadrature_run(v,f,cuts,h,w,ox,oy,NULL,ctx,ids,samples,scratch,d,-1,NULL,NULL,0,3,NULL);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_quadrature_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *gradient,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return quadrature_run(v,f,cuts,h,w,ox,oy,gradient,ctx,gv,gc,scratch,d,-1,NULL,NULL,0,3,NULL);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_boundary_quadrature(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,int32_t edge,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    gffx_status prepared=gffx_internal_prepare_diagnostic(d);if(prepared)return prepared;
    if(edge<0||edge>3)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"opaque edge must be0..3");
    return quadrature_run(v,f,cuts,h,w,ox,oy,NULL,ctx,ids,samples,scratch,d,edge,NULL,NULL,0,3,NULL);
}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_quadrature(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return quadrature_run(v,f,cuts,h,w,ox,oy,NULL,ctx,ids,samples,scratch,d,-2,NULL,NULL,0,3,NULL);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_quadrature_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *gradient,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return quadrature_run(v,f,cuts,h,w,ox,oy,gradient,ctx,gv,gc,scratch,d,-2,NULL,NULL,0,3,NULL);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return quadrature_run(v,f,cuts,1,1,ox,oy,NULL,ctx,ids,samples,scratch,d,-3,cursor,receipt,0,3,NULL);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return quadrature_run(v,f,cuts,1,1,ox,oy,g,ctx,gv,gc,scratch,d,-3,cursor,receipt,0,3,NULL);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_quadrature_workspace(gffx_dtype dtype,const gffx_execution_context *ctx,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
    gffx_status status=gffx_render_opaque_quadrature_workspace(dtype,ctx,bytes,alignment,d);
    if(status)return status;
    if(ctx->device_type==GFFX_DEVICE_CUDA){const gffx_cuda_operations *ops=gffx_cuda_loader_operations();
        if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_material))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA opaque material operation unavailable");}
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_workspace(gffx_dtype dtype,const gffx_execution_context *ctx,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d){
    gffx_status status=gffx_render_opaque_quadrature_workspace(dtype,ctx,bytes,alignment,d);
    if(status)return status;
    if(ctx->device_type==GFFX_DEVICE_CUDA){const gffx_cuda_operations *ops=gffx_cuda_loader_operations();
        if(!GFFX_CUDA_HAS_OPERATION(ops,render_opaque_batch))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA opaque material operation unavailable");}
    return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_rule(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t rule){return quadrature_run(v,f,cuts,1,1,ox,oy,NULL,ctx,ids,samples,scratch,d,-3,cursor,receipt,rule,3,NULL);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_batch_rule_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t rule,int32_t derivatives){return quadrature_run(v,f,cuts,1,1,ox,oy,g,ctx,gv,gc,scratch,d,-3,cursor,receipt,rule,derivatives,NULL);}

static gffx_status partition_cuda(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,gffx_tensor_view *rows,int32_t *result,int count,int begin,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_execution_context *ctx,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int capacity){
 const gffx_tensor_view *views[6]={v,f,cuts,rows,ids,samples};int prepare=v!=NULL,j,k;uint64_t bytes=1048576;gffx_status status;const gffx_cuda_operations *ops;
 if(!scratch)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"partition caller scratch required");status=gffx_validate_buffer(scratch,d);if(status)return status;
 if(!scratch->data||scratch->capacity_bytes<bytes||scratch->alignment<8||(uintptr_t)scratch->data%8||scratch->device_type!=ctx->device_type||scratch->device_index!=ctx->device_index)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition scratch device/capacity/alignment");
 for(j=0;j<6;++j)if(views[j]){const gffx_tensor_view *a=views[j];status=gffx_validate_tensor_view(a,d);if(status)return status;
  if(a->device_type!=ctx->device_type||a->device_index!=ctx->device_index||a->dtype!=((j==1||j==4)?GFFX_DTYPE_INT32:GFFX_DTYPE_FLOAT64)||((a->flags&GFFX_TENSOR_OUTPUT)!=0)!=(prepare?j==3:j>=4))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition dtype/device/flags");
  if(gffx_mesh_range_overlaps_view(scratch->data,bytes,a))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition scratch alias");
  if(result&&gffx_mesh_range_overlaps_view(result,8,a))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition result alias");
  for(k=j+1;k<6;++k)if(views[k]&&gffx_mesh_views_overlap(a,views[k]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition view alias");
 }
 if(!rows||rows->rank!=2||rows->shape[0]!=capacity||rows->shape[1]!=17)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition row shape");
 if(prepare){if(!f||!cuts||!result||(uintptr_t)result%4||!isfinite(ox)||!isfinite(oy)||fabs(ox)>1048576||fabs(oy)>1048576||v->rank!=2||v->shape[1]!=3||f->rank!=2||f->shape[1]!=3||cuts->rank!=3||cuts->shape[0]!=f->shape[0]||cuts->shape[1]!=3||cuts->shape[2]>(capacity==2560?40:16))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition preparation metadata");}
 else if(count<0||count>capacity||begin<0||begin>count||begin%(capacity==2560?28:32)||!ids||!samples||ids->rank!=2||ids->shape[0]!=1||ids->shape[1]!=800||samples->rank!=3||samples->shape[0]!=1||samples->shape[1]!=800||samples->shape[2]!=4)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition emission metadata");
 if(capacity==2560){ops=gffx_cuda_loader_operations();if(!GFFX_CUDA_HAS_OPERATION(ops,opaque_partition80))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA SH64 partition unavailable");return ops->opaque_partition80(v,f,cuts,ox,oy,rows,result,count,begin,ids,samples,ctx,scratch,d);}
 if(capacity==2048){ops=gffx_cuda_loader_operations();if(!GFFX_CUDA_HAS_OPERATION(ops,opaque_partition64))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA SH64 partition unavailable");return ops->opaque_partition64(v,f,cuts,ox,oy,rows,result,count,begin,ids,samples,ctx,scratch,d);}
 ops=gffx_cuda_loader_operations();if(!GFFX_CUDA_HAS_OPERATION(ops,opaque_partition))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA partition unavailable");
 return ops->opaque_partition(v,f,cuts,ox,oy,rows,result,count,begin,ids,samples,ctx,scratch,d);
}
static gffx_status gffx_render_opaque_partition_prepare_profile(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *rows,int32_t result[2],const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int capacity){
 gffx_status status;uint64_t bytes,alignment,q;int j,k,n=0,curves,count;double value;Scratch *s;Ctx work={0};QuadBatch batch;QuadPartition partition;
 const int32_t *faces;const gffx_tensor_view *inputs[3]={v,f,cuts};
 status=gffx_internal_prepare_diagnostic(d);if(status)return status;
 status=gffx_validate_execution_context(ctx,d);if(status)return status;
 if(ctx->device_type==GFFX_DEVICE_CUDA)return partition_cuda(v,f,cuts,ox,oy,rows,result,0,0,NULL,NULL,ctx,scratch,d,capacity);
 if(!result||!isfinite(ox)||!isfinite(oy)||fabs(ox)>1048576||fabs(oy)>1048576)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition origin/result");
 if((uintptr_t)result%sizeof(int32_t))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition metadata alignment");
 status=gffx_mesh_check_common(v,f,0,ctx,scratch,d);if(status)return status;
 if(v->dtype!=GFFX_DTYPE_FLOAT64)return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"partition float64 qualification only");
 status=gffx_mesh_check_view(cuts,"partition cuts",3,f->shape[0],3,0,d);if(status)return status;
 if(cuts->shape[1]!=3||cuts->shape[2]>(capacity==2560?40:16)||cuts->dtype!=v->dtype)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition cuts dtype/capacity");
 status=gffx_mesh_check_view(rows,"partition storage",2,capacity,17,1,d);if(status)return status;
 if(rows->dtype!=v->dtype)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition storage dtype");
 status=gffx_render_opaque_quadrature_workspace(v->dtype,ctx,&bytes,&alignment,d);if(status)return status;
 if(!scratch||!scratch->data||scratch->capacity_bytes<bytes||(uintptr_t)scratch->data%alignment)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"partition scratch");
 for(j=0;j<3;++j){
  if(inputs[j]->device_type!=ctx->device_type||inputs[j]->device_index!=ctx->device_index)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition input device");
  if(gffx_mesh_range_overlaps_view(result,8,inputs[j])||gffx_mesh_views_overlap(inputs[j],rows)||gffx_mesh_range_overlaps_view(scratch->data,bytes,inputs[j]))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition input alias");
 }
 if(gffx_mesh_range_overlaps_view(result,8,rows)||((uintptr_t)result<(uintptr_t)scratch->data+bytes&&(uintptr_t)scratch->data<(uintptr_t)result+8)||gffx_mesh_range_overlaps_view(scratch->data,bytes,rows))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition storage alias");
 result[0]=result[1]=0;
 for(j=0;j<3;j+=2)for(q=0;q<gffx_mesh_element_count(inputs[j]);++q){double z=read_real(inputs[j],q);if(!isfinite(z)||(j==0&&fabs(z)>1048576))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition finite bounded inputs");}
 if(opaque_partition_prepare_execute(v,f,cuts,ox,oy,rows,result,(Scratch*)scratch->data))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"partition capacity/arithmetic overflow; all rows invalid");return GFFX_STATUS_OK;
}
static gffx_status gffx_render_opaque_partition_emit_profile(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,gffx_diagnostic_buffer *d,int capacity){
 const double nodes[5]={0.0469100770306680036,0.2307653449471584545,0.5,0.7692346550528415455,0.9530899229693319964};
 const double weights[5]={0.1184634425280945438,0.2393143352496832340,0.2844444444444444444,0.2393143352496832340,0.1184634425280945438};
 gffx_status status=gffx_internal_prepare_diagnostic(d);int i,k,ns,nt,used=0,end=begin+32;uint64_t z;if(status)return status;
 status=gffx_validate_execution_context(ctx,d);if(status)return status;
 if(ctx->device_type!=GFFX_DEVICE_CPU)return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"partition emit CPU qualification only");
 if(count<0||count>capacity||begin<0||begin>count||begin%(capacity==2560?28:32))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition range");
 status=gffx_mesh_check_view(rows,"partition rows",2,capacity,17,0,d);if(status)return status;
 status=gffx_mesh_check_view(ids,"partition owners",2,1,800,1,d);if(status)return status;
 status=view3(samples,1,800,4,1,GFFX_DTYPE_FLOAT64,d);if(status)return status;
 if(rows->dtype!=GFFX_DTYPE_FLOAT64||ids->dtype!=GFFX_DTYPE_INT32||rows->device_type!=ctx->device_type||ids->device_type!=ctx->device_type||samples->device_type!=ctx->device_type)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition dtype/device");
 if(gffx_mesh_views_overlap(rows,ids)||gffx_mesh_views_overlap(rows,samples)||gffx_mesh_views_overlap(ids,samples))return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"partition alias");
 if(opaque_partition_emit_execute(rows,count,begin,ids,samples))return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"partition row/arithmetic failure; outputs invalid");return GFFX_STATUS_OK;
}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_prefix_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t prefix[2]){if(!prefix)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix required");return quadrature_run(v,f,cuts,1,1,ox,oy,g,ctx,gv,gc,scratch,d,-3,cursor,receipt,2,derivatives,prefix);}

static gffx_status gffx_render_opaque_partition_emit_workspace_profile(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int capacity){gffx_status s=gffx_internal_prepare_diagnostic(d);if(s)return s;s=gffx_validate_execution_context(ctx,d);if(s)return s;if(ctx->device_type==GFFX_DEVICE_CUDA)return partition_cuda(NULL,NULL,NULL,0,0,(gffx_tensor_view*)rows,NULL,count,begin,ids,samples,ctx,scratch,d,capacity);return gffx_render_opaque_partition_emit_profile(rows,count,begin,ctx,ids,samples,d,capacity);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_prefix_backward64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t prefix[2]){if(!prefix)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix required");return quadrature_run(v,f,cuts,1,1,ox,oy,g,ctx,gv,gc,scratch,d,-3,cursor,receipt,3,derivatives,prefix);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_prepare(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *rows,int32_t result[2],const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_prepare_profile(v,f,cuts,ox,oy,ctx,rows,result,scratch,d,1664);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_prepare64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *rows,int32_t result[2],const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_prepare_profile(v,f,cuts,ox,oy,ctx,rows,result,scratch,d,2048);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_emit_profile(rows,count,begin,ctx,ids,samples,d,1664);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit64(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_emit_profile(rows,count,begin,ctx,ids,samples,d,2048);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit_workspace(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_emit_workspace_profile(rows,count,begin,ctx,ids,samples,scratch,d,1664);}
GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit_workspace64(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_emit_workspace_profile(rows,count,begin,ctx,ids,samples,scratch,d,2048);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_boundary_batch64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,int32_t edge,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
 if(edge<0||edge>3)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"boundary stream edge range");
 return quadrature_run(v,f,cuts,1,1,ox,oy,NULL,ctx,ids,samples,scratch,d,edge+4,cursor,receipt,3,0,NULL);
}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_material_prefix_backward80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t cursor[4],int32_t receipt[8],const gffx_tensor_view *g,const gffx_execution_context *ctx,gffx_tensor_view *gv,gffx_tensor_view *gc,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t prefix[2]){if(!prefix)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"prefix required");return quadrature_run(v,f,cuts,1,1,ox,oy,g,ctx,gv,gc,scratch,d,-3,cursor,receipt,4,derivatives,prefix);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_prepare80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const gffx_execution_context *ctx,gffx_tensor_view *rows,int32_t result[2],const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_prepare_profile(v,f,cuts,ox,oy,ctx,rows,result,scratch,d,2560);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit80(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_emit_profile(rows,count,begin,ctx,ids,samples,d,2560);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_partition_emit_workspace80(const gffx_tensor_view *rows,int32_t count,int32_t begin,const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_render_opaque_partition_emit_workspace_profile(rows,count,begin,ctx,ids,samples,scratch,d,2560);}

GFFX_API gffx_status GFFX_CALL gffx_render_opaque_boundary_batch80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,int32_t edge,const int32_t cursor[4],int32_t receipt[8],const gffx_execution_context *ctx,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
 if(edge<0||edge>3)return gffx_internal_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"boundary stream edge range");
 return quadrature_run(v,f,cuts,1,1,ox,oy,NULL,ctx,ids,samples,scratch,d,edge+4,cursor,receipt,4,0,NULL);
}
