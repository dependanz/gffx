/* Real core routing against a synthetic old/null provider; fake pointers must not be read. */
#include <gffx/image.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"image provider line %d\n",__LINE__);return 1;}}while(0)
int main(int argc,char **argv){
    gffx_execution_context c={0};gffx_tensor_view x={0},g,y;int64_t shape=3,stride=1;uint64_t bytes=91,align=92;
    CHECK(argc==2);
#if defined(_WIN32)
    CHECK(_putenv_s("GFFX_CUDA_PLUGIN_PATH",argv[1])==0);
#else
    CHECK(setenv("GFFX_CUDA_PLUGIN_PATH",argv[1],1)==0);
#endif
    c.struct_size=sizeof(c);c.abi_version=GFFX_ABI_VERSION;c.device_type=GFFX_DEVICE_CUDA;
    x.struct_size=sizeof(x);x.abi_version=GFFX_ABI_VERSION;x.device_type=GFFX_DEVICE_CUDA;
    x.dtype=GFFX_DTYPE_FLOAT32;x.flags=GFFX_TENSOR_READ_ONLY;x.data=(void *)(uintptr_t)0x1000;
    x.rank=1;x.shape=&shape;x.strides=&stride;g=x;g.data=(void *)(uintptr_t)0x2000;
    y=x;y.flags=GFFX_TENSOR_OUTPUT;y.data=(void *)(uintptr_t)0x3000;
    CHECK(gffx_image_transfer_workspace(3,x.dtype,&c,&bytes,&align,NULL)==GFFX_STATUS_UNSUPPORTED);
    CHECK(bytes==91&&align==92);
    CHECK(gffx_image_transfer(&x,0,&c,&y,NULL,NULL)==GFFX_STATUS_UNSUPPORTED);
    CHECK(gffx_image_transfer_backward(&x,&g,1,&c,&y,NULL,NULL)==GFFX_STATUS_UNSUPPORTED);
    x.struct_size=8;CHECK(gffx_image_transfer(&x,0,&c,&y,NULL,NULL)==GFFX_STATUS_ABI_MISMATCH);
    return 0;
}
