#define GFFX_IMAGE_SCALAR static __device__
#include "../core/image_transfer_scalar.h"
#undef GFFX_IMAGE_SCALAR
#define GFFX_IMAGE_KERNEL(T,S) \
extern "C" __global__ void gffx_image_finite_##S(const T *x,long long count,int *status){ \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&!isfinite(x[i]))atomicExch(status,1); \
} \
extern "C" __global__ void gffx_image_transfer_##S(const T *x,const T *g,T *out,long long count, \
    unsigned transfer,int backward,int *status){ \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return; \
    T y,dy;int bad=gffx_im_transfer_##S(x[i],(int)transfer,&y,&dy);if(backward)y=g[i]*dy; \
    if(bad||!isfinite(y)){atomicExch(status,2);return;}out[i]=y; \
}
GFFX_IMAGE_KERNEL(float,f32)
GFFX_IMAGE_KERNEL(double,f64)
#undef GFFX_IMAGE_KERNEL
