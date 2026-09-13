/* Public routing fails closed without a GPU or access to the fake device pointers. */
#include <gffx/animation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"animation provider line %d\n",__LINE__);return 1;} } while(0)
int main(int argc,char **argv) {
    gffx_execution_context c={0};gffx_tensor_view t={0},delta,w,y;
    int64_t ts[2]={1,3},ds[3]={1,1,3},ws[2]={1,1},ys[3]={1,1,3};
    int64_t stride2[2]={3,1},stride3[3]={3,3,1},weight_stride[2]={1,1};uint64_t bytes=0,alignment=0;
    CHECK(argc==2);
#if defined(_WIN32)
    CHECK(_putenv_s("GFFX_CUDA_PLUGIN_PATH",argv[1])==0);
#else
    CHECK(setenv("GFFX_CUDA_PLUGIN_PATH",argv[1],1)==0);
#endif
    c.struct_size=sizeof(c);c.abi_version=GFFX_ABI_VERSION;c.device_type=GFFX_DEVICE_CUDA;
    t.struct_size=sizeof(t);t.abi_version=GFFX_ABI_VERSION;t.device_type=GFFX_DEVICE_CUDA;
    t.dtype=GFFX_DTYPE_FLOAT32;t.flags=GFFX_TENSOR_READ_ONLY;t.data=(void *)(uintptr_t)0x1000;
    t.rank=2;t.shape=ts;t.strides=stride2;
    delta=t;delta.rank=3;delta.shape=ds;delta.strides=stride3;delta.data=(void *)(uintptr_t)0x2000;
    w=t;w.shape=ws;w.strides=weight_stride;w.data=(void *)(uintptr_t)0x3000;
    y=delta;y.shape=ys;y.flags=GFFX_TENSOR_OUTPUT;y.data=(void *)(uintptr_t)0x4000;
    t.struct_size=0;
    CHECK(gffx_mesh_blend_shapes(&t,&delta,&w,&c,&y,NULL,NULL)==GFFX_STATUS_ABI_MISMATCH);
    t.struct_size=sizeof(t);t.shape=NULL;
    CHECK(gffx_mesh_blend_shapes(&t,&delta,&w,&c,&y,NULL,NULL)==GFFX_STATUS_INVALID_ARGUMENT);
    t.shape=ts;w.device_index=1;
    CHECK(gffx_mesh_blend_shapes(&t,&delta,&w,&c,&y,NULL,NULL)==GFFX_STATUS_INVALID_ARGUMENT);
    w.device_index=0;
    CHECK(gffx_mesh_blend_shapes(&t,&delta,&w,&c,&y,NULL,NULL)==GFFX_STATUS_UNSUPPORTED);
    CHECK(gffx_mesh_blend_shapes_workspace(1,1,1,GFFX_DTYPE_FLOAT32,&c,&bytes,&alignment,NULL)==GFFX_STATUS_UNSUPPORTED);
    return 0;
}
