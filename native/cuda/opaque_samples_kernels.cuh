#define GFFX_OC_LOCAL static __device__
#include "../core/opaque_samples_scalar.h"
extern "C" __global__ void gffx_opaque_samples_domain(GffxOpaqueSamples x,long long largest,int *status){
 long long j=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
 for(;j<largest;j+=stride)if(oc_domain(&x,j))atomicExch(status,1);
}
extern "C" __global__ void gffx_opaque_samples_evaluate(GffxOpaqueSamples x,void *scratch){
 int pixel=blockIdx.x,lane=threadIdx.x,start,j,k;
 double *work=(double*)((char*)scratch+8)+3200*pixel;
 for(start=0;start<800;start+=128){j=start+lane;
  if(j<800){if(oc_sample(&x,pixel*800+j))atomicExch((int*)scratch,1);
   for(k=0;k<4;++k)work[j*4+k]=x.weighted[(pixel*800+j)*4+k];}
  __syncthreads();
 }
 if(!lane&&oc_reduce(&x,pixel,work))atomicExch((int*)scratch,1);
}
#undef GFFX_OC_LOCAL
