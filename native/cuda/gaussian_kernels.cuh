#define GFFX_GW_SCALAR static __device__
#include "../core/gaussian_scalar.h"
#undef GFFX_GW_SCALAR
#define GFFX_GAUSSIAN_KERNEL(T,S) \
extern "C" __global__ void gffx_gaussian_domain_##S(const T *sigma,long long count,int *status){ \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&(!isfinite(sigma[i])||sigma[i]<(T).1||sigma[i]>(T)32))atomicExch(status,1); \
} \
extern "C" __global__ void gffx_gaussian_##S(const T *sigma,const T *gradient,T *output,long long batch,unsigned radius,int backward,int *status){ \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;int j,count=2*(int)radius+1;double w[65],d[65];T products[65]; \
    if(i>=batch)return;gffx_gw_evaluate((double)sigma[i],(int)radius,w,d); \
    if(!backward){for(j=0;j<count;++j)output[i*count+j]=(T)w[j];return;} \
    if(radius==0){output[i]=(T)0;return;} \
    for(j=0;j<count;++j){T derivative=(T)d[j];products[j]=derivative*gradient[i*count+j];if(!isfinite(products[j])){atomicExch(status,2);return;}} \
    while(count>1){int n=0;for(j=0;j+1<count;j+=2){T sum=products[j]+products[j+1];if(!isfinite(sum)){atomicExch(status,2);return;}products[n++]=sum;} \
        if(count%2)products[n++]=products[count-1];count=n;}output[i]=products[0]; \
}
GFFX_GAUSSIAN_KERNEL(float,f32)
GFFX_GAUSSIAN_KERNEL(double,f64)
#undef GFFX_GAUSSIAN_KERNEL
