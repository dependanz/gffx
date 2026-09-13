/* Framework-independent descriptor, ownership and arithmetic controls. */
#include <gffx/image.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"image check line %d\n",__LINE__);return 1;}}while(0)
static gffx_tensor_view view(void *data,int output,uint32_t rank,const int64_t *shape,const int64_t *stride){
    gffx_tensor_view v={0};v.struct_size=sizeof(v);v.abi_version=GFFX_ABI_VERSION;v.data=data;
    v.dtype=GFFX_DTYPE_FLOAT64;v.device_type=GFFX_DEVICE_CPU;v.rank=rank;v.shape=shape;v.strides=stride;
    v.flags=output?GFFX_TENSOR_OUTPUT:GFFX_TENSOR_READ_ONLY;return v;
}
int main(void){
    gffx_execution_context c={0};gffx_diagnostic_buffer d={0};uint64_t bytes=91,align=92;
    double x[4]={0.,-0.,1.,.2},g[4]={1.,1.,1.,1.},y[4],dy[4];int64_t shape[2]={2,2},stride[2]={2,1};
    gffx_tensor_view a=view(x,0,2,shape,stride),b=view(y,1,2,shape,stride),v=view(g,0,2,shape,stride),w=view(dy,1,2,shape,stride);
    c.struct_size=sizeof(c);c.abi_version=GFFX_ABI_VERSION;c.device_type=GFFX_DEVICE_CPU;
    d.struct_size=sizeof(d);d.abi_version=GFFX_ABI_VERSION;
    CHECK(gffx_image_transfer_workspace(4,GFFX_DTYPE_FLOAT64,&c,&bytes,&align,&d)==GFFX_STATUS_OK&&bytes==0&&align==1);
    CHECK(gffx_image_transfer(&a,0,&c,&b,NULL,&d)==GFFX_STATUS_OK);
    CHECK(y[0]==0.&&signbit(y[1])&&y[2]==1.&&fabs(y[3]-.033104766570885055)<=2e-12);
    CHECK(gffx_image_transfer_backward(&a,&v,0,&c,&w,NULL,&d)==GFFX_STATUS_OK);
    CHECK(fabs(dy[0]-1./12.92)<=2e-12&&fabs(dy[2]-2.4/1.055)<=2e-12);
    y[0]=91.;x[3]=NAN;
    CHECK(gffx_image_transfer(&a,0,&c,&b,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT&&y[0]==91.);x[3]=.2;
    CHECK(gffx_image_transfer(&a,3,&c,&b,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT&&y[0]==91.);
    b.data=x;CHECK(gffx_image_transfer(&a,0,&c,&b,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT);b.data=y;
    CHECK(gffx_image_transfer_backward(&a,NULL,0,&c,&w,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT);
    b.dtype=GFFX_DTYPE_FLOAT32;CHECK(gffx_image_transfer(&a,0,&c,&b,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT);b.dtype=a.dtype;
    a.struct_size=8;CHECK(gffx_image_transfer(&a,0,&c,&b,NULL,&d)==GFFX_STATUS_ABI_MISMATCH);a.struct_size=sizeof(a);
    x[2]=DBL_MAX;CHECK(gffx_image_transfer(&a,0,&c,&b,NULL,&d)==GFFX_STATUS_OVERFLOW);x[2]=1.;
    a=view(x,0,0,NULL,NULL);b=view(y,1,0,NULL,NULL);
    CHECK(gffx_image_transfer(&a,1,&c,&b,NULL,&d)==GFFX_STATUS_OK);
    {int64_t empty_shape[2]={0,3},empty_stride[2]={3,1};
     a=view(NULL,0,2,empty_shape,empty_stride);b=view(NULL,1,2,empty_shape,empty_stride);
     v=view(NULL,0,2,empty_shape,empty_stride);w=view(NULL,1,2,empty_shape,empty_stride);
     CHECK(gffx_image_transfer(&a,0,&c,&b,NULL,&d)==GFFX_STATUS_OK);
     CHECK(gffx_image_transfer_backward(&a,&v,0,&c,&w,NULL,&d)==GFFX_STATUS_OK);}
    return 0;
}
