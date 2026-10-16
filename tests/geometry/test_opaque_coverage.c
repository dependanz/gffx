#include <gffx/render.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"failed line %d\n",__LINE__);return 1;} } while(0)
static gffx_tensor_view tensor(void *data,int64_t *shape,int64_t *strides,gffx_dtype dtype,int output) {
    gffx_tensor_view t={0};t.struct_size=sizeof(t);t.abi_version=GFFX_ABI_VERSION;t.data=data;t.rank=2;t.shape=shape;t.strides=strides;t.dtype=dtype;t.device_type=GFFX_DEVICE_CPU;t.flags=output?GFFX_TENSOR_OUTPUT:GFFX_TENSOR_READ_ONLY;return t;
}
int main(void) {
    char msg[512];gffx_diagnostic_buffer diag=GFFX_DIAGNOSTIC_INIT;
    gffx_execution_context ctx={0};gffx_buffer scratch={0};uint64_t bytes=0,alignment=0;void *memory;
    double vertices[]={.125,.125,0,.625,.125,0,.125,.625,0},value=91,cot=2,grad[9];int32_t faces[]={0,1,2};
    int64_t vs[]={3,3},fs[]={1,3},os[]={1,1},strides[]={3,1},one[]={1,1};
    gffx_tensor_view v=tensor(vertices,vs,strides,GFFX_DTYPE_FLOAT64,0),f=tensor(faces,fs,strides,GFFX_DTYPE_INT32,0),o=tensor(&value,os,one,GFFX_DTYPE_FLOAT64,1),g=tensor(&cot,os,one,GFFX_DTYPE_FLOAT64,0),gv=tensor(grad,vs,strides,GFFX_DTYPE_FLOAT64,1),bad;
    diag.data=msg;diag.capacity_bytes=sizeof(msg);ctx.struct_size=sizeof(ctx);ctx.abi_version=GFFX_ABI_VERSION;ctx.device_type=GFFX_DEVICE_CPU;
    CHECK(gffx_render_opaque_coverage_workspace(GFFX_DTYPE_FLOAT64,&ctx,&bytes,&alignment,&diag)==0);CHECK(bytes>0 && bytes<1048576 && alignment==8);
    memory=calloc(1,(size_t)bytes+8);CHECK(memory!=NULL);scratch.struct_size=sizeof(scratch);scratch.abi_version=GFFX_ABI_VERSION;scratch.data=memory;scratch.capacity_bytes=bytes;scratch.alignment=8;scratch.device_type=GFFX_DEVICE_CPU;
    CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&o,&scratch,&diag)==0);CHECK(value==.125);
    CHECK(gffx_render_opaque_coverage_backward(&v,&f,1,1,0,0,&g,&ctx,&gv,&scratch,&diag)==0);CHECK(grad[3]==.5 && grad[7]==.5 && grad[2]==0 && grad[5]==0 && grad[8]==0);
    value=91;faces[2]=9;CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);CHECK(value==91);faces[2]=2;
    scratch.capacity_bytes=1;CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INSUFFICIENT_WORKSPACE);CHECK(value==91);scratch.capacity_bytes=bytes;
    scratch.data=(char*)memory+1;scratch.alignment=1;CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);scratch.data=memory;scratch.alignment=8;
    bad=o;bad.data=vertices;CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);CHECK(vertices[0]==.125);
    bad=o;bad.flags=GFFX_TENSOR_READ_ONLY;CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&bad,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);
    vertices[0]=NAN;CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&o,&scratch,&diag)==GFFX_STATUS_INVALID_ARGUMENT);CHECK(value==91);vertices[0]=.125;
    fs[0]=0;CHECK(gffx_render_opaque_coverage(&v,&f,1,1,0,0,&ctx,&o,&scratch,&diag)==0);CHECK(value==0);fs[0]=1;
    ctx.device_type=GFFX_DEVICE_CUDA;CHECK(gffx_render_opaque_coverage_workspace(GFFX_DTYPE_FLOAT64,&ctx,&bytes,&alignment,&diag)==GFFX_STATUS_UNSUPPORTED);
    free(memory);puts("opaque native C boundary passed");return 0;
}
