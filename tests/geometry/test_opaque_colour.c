#include <gffx/render.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"failed line %d\n",__LINE__);return 1;} } while(0)
static gffx_tensor_view tensor(void *data,int rank,int64_t *shape,int64_t *strides,gffx_dtype dtype,int output) {
    gffx_tensor_view t={0};t.struct_size=sizeof(t);t.abi_version=GFFX_ABI_VERSION;t.data=data;t.rank=rank;t.shape=shape;t.strides=strides;t.dtype=dtype;t.device_type=GFFX_DEVICE_CPU;t.flags=output?GFFX_TENSOR_OUTPUT:GFFX_TENSOR_READ_ONLY;return t;
}
int main(void) {
    char msg[512];gffx_diagnostic_buffer diag=GFFX_DIAGNOSTIC_INIT;gffx_execution_context ctx={0};gffx_buffer scratch={0};uint64_t bytes=0,alignment=0;void *memory;int i;
    double vertices[]={.125,.125,0,.625,.125,0,.125,.625,0},colour[]={1,0,0,0,1,0,0,0,1},rgba[4],cot[]={1,2,3,4},grad[9],cg[9];int32_t faces[]={0,1,2};
    int64_t vs[]={3,3},fs[]={1,3},cs[]={1,3,3},os[]={1,1,4},s2[]={3,1},s3[]={9,3,1},s4[]={4,4,1};
    gffx_tensor_view v=tensor(vertices,2,vs,s2,GFFX_DTYPE_FLOAT64,0),f=tensor(faces,2,fs,s2,GFFX_DTYPE_INT32,0),c=tensor(colour,3,cs,s3,GFFX_DTYPE_FLOAT64,0),o=tensor(rgba,3,os,s4,GFFX_DTYPE_FLOAT64,1),g=tensor(cot,3,os,s4,GFFX_DTYPE_FLOAT64,0),gv=tensor(grad,2,vs,s2,GFFX_DTYPE_FLOAT64,1),gc=tensor(cg,3,cs,s3,GFFX_DTYPE_FLOAT64,1),bad;
    diag.data=msg;diag.capacity_bytes=sizeof(msg);ctx.struct_size=sizeof(ctx);ctx.abi_version=GFFX_ABI_VERSION;ctx.device_type=GFFX_DEVICE_CPU;
    CHECK(gffx_render_opaque_integrate_workspace(GFFX_DTYPE_FLOAT64,&ctx,&bytes,&alignment,&diag)==0);CHECK(bytes>0 && bytes<1048576 && alignment==8);
    memory=calloc(1,(size_t)bytes+8);CHECK(memory!=NULL);scratch.struct_size=sizeof(scratch);scratch.abi_version=GFFX_ABI_VERSION;scratch.data=memory;scratch.capacity_bytes=bytes;scratch.alignment=8;scratch.device_type=GFFX_DEVICE_CPU;
    CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&o,&scratch,&diag)==0);CHECK(rgba[3]==.125);for(i=0;i<3;++i)CHECK(fabs(rgba[i]-1.0/24)<1e-14);
    CHECK(gffx_render_opaque_integrate_backward(&v,&f,&c,1,1,0,0,&g,&ctx,&gv,&gc,&scratch,&diag)==0);
    for(i=0;i<9;++i)CHECK(fabs(cg[i]-(i%3+1)/24.0)<1e-14);
    CHECK(fabs(grad[3]-1.5)<1e-14 && fabs(grad[7]-1.5)<1e-14);CHECK(grad[2]==0 && grad[5]==0 && grad[8]==0);
    rgba[0]=91;faces[2]=9;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);CHECK(rgba[0]==91);faces[2]=2;
    scratch.capacity_bytes=1;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INSUFFICIENT_WORKSPACE);CHECK(rgba[0]==91);scratch.capacity_bytes=bytes;
    scratch.data=(char*)memory+1;scratch.alignment=1;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);scratch.data=memory;scratch.alignment=8;
    bad=o;bad.data=colour;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);CHECK(colour[0]==1);
    bad=o;bad.data=memory;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);
    bad=gc;bad.data=grad;CHECK(gffx_render_opaque_integrate_backward(&v,&f,&c,1,1,0,0,&g,&ctx,&gv,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);
    bad=o;bad.flags=GFFX_TENSOR_READ_ONLY;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);
    colour[0]=NAN;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);CHECK(rgba[0]==91);colour[0]=1;
    cs[2]=2;s3[0]=6;s3[1]=2;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);cs[2]=3;s3[0]=9;s3[1]=3;
    fs[0]=0;cs[0]=0;CHECK(gffx_render_opaque_integrate(&v,&f,&c,1,1,0,0,&ctx,&o,&scratch,&diag)==0);for(i=0;i<4;++i)CHECK(rgba[i]==0);
    ctx.device_type=GFFX_DEVICE_CUDA;CHECK(gffx_render_opaque_integrate_workspace(GFFX_DTYPE_FLOAT64,&ctx,&bytes,&alignment,&diag)==GFFX_STATUS_UNSUPPORTED);
    ctx.device_type=GFFX_DEVICE_CPU;fs[0]=1;
    {
        static int32_t owner[800];static double nodes[3200],gn[3200];double cut[3]={-.25,.25,-.25},gcq[3],sum=0;
        int64_t ks[]={1,3,1},ksr[]={3,1,1},is[]={1,800},ist[]={800,1},qs[]={1,800,4},qst[]={3200,4,1};
        gffx_tensor_view cv=tensor(cut,3,ks,ksr,GFFX_DTYPE_FLOAT64,0),oi=tensor(owner,2,is,ist,GFFX_DTYPE_INT32,1),oq=tensor(nodes,3,qs,qst,GFFX_DTYPE_FLOAT64,1),qg=tensor(gn,3,qs,qst,GFFX_DTYPE_FLOAT64,0),qc=tensor(gcq,3,ks,ksr,GFFX_DTYPE_FLOAT64,1);
        CHECK(gffx_render_opaque_quadrature_workspace(GFFX_DTYPE_FLOAT64,&ctx,&bytes,&alignment,&diag)==0);
        CHECK(bytes<=scratch.capacity_bytes);
        CHECK(gffx_render_opaque_quadrature(&v,&f,&cv,1,1,0,0,&ctx,&oi,&oq,&scratch,&diag)==0);
        for(i=0;i<800;++i){sum+=nodes[4*i+3];gn[4*i+3]=1;if(owner[i]>=0)CHECK(owner[i]==0 && fabs(nodes[4*i]+nodes[4*i+1]+nodes[4*i+2]-1)<1e-14);else CHECK(nodes[4*i+3]==0);}
        CHECK(fabs(sum-.125)<1e-14);
        CHECK(gffx_render_opaque_quadrature_backward(&v,&f,&cv,1,1,0,0,&qg,&ctx,&gv,&qc,&scratch,&diag)==0);
        CHECK(fabs(grad[3]-.25)<1e-14 && fabs(grad[7]-.25)<1e-14);for(i=0;i<3;++i)CHECK(fabs(gcq[i])<1e-14);
        nodes[0]=91;scratch.capacity_bytes=1;CHECK(gffx_render_opaque_quadrature(&v,&f,&cv,1,1,0,0,&ctx,&oi,&oq,&scratch,&diag)==GFFX_STATUS_INSUFFICIENT_WORKSPACE);CHECK(nodes[0]==91);scratch.capacity_bytes=bytes;
        bad=oq;bad.flags=GFFX_TENSOR_READ_ONLY;CHECK(gffx_render_opaque_quadrature(&v,&f,&cv,1,1,0,0,&ctx,&oi,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);
        bad=oq;bad.data=memory;CHECK(gffx_render_opaque_quadrature(&v,&f,&cv,1,1,0,0,&ctx,&oi,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);
        bad=qc;bad.data=grad;CHECK(gffx_render_opaque_quadrature_backward(&v,&f,&cv,1,1,0,0,&qg,&ctx,&gv,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);
        cut[0]=NAN;CHECK(gffx_render_opaque_quadrature(&v,&f,&cv,1,1,0,0,&ctx,&oi,&oq,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);cut[0]=-.25;
        faces[2]=9;CHECK(gffx_render_opaque_quadrature(&v,&f,&cv,1,1,0,0,&ctx,&oi,&oq,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);faces[2]=2;
        ctx.device_type=GFFX_DEVICE_CUDA;{gffx_status status=gffx_render_opaque_quadrature_workspace(GFFX_DTYPE_FLOAT64,&ctx,&bytes,&alignment,&diag);CHECK(status==GFFX_STATUS_UNSUPPORTED || (status==GFFX_STATUS_OK && bytes==1048576 && alignment==8));}
    }
    free(memory);puts("opaque colour and quadrature native C boundary passed");return 0;
}
