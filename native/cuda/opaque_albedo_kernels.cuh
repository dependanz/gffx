#define GFFX_OC_LOCAL static __device__
#include "../core/opaque_albedo_scalar.h"
extern "C" __global__ void gffx_opaque_albedo_domain(GffxOpaqueSamples x,const double *cotangent,long long largest,int *status){
 long long j=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
 for(;j<largest;j+=stride)if(oc_domain(&x,j)||(j<x.pixels*3&&!isfinite(cotangent[j])))atomicExch(status,1);
}
extern "C" __global__ void gffx_opaque_albedo_evaluate(GffxOpaqueSamples x,const double *cotangent,double *gradient,void *scratch){
 int pixel=blockIdx.x,lane=threadIdx.x,start,j,k;
 double *work=(double*)((char*)scratch+8)+5600*pixel,*adjoint=work+3200;
 for(start=0;start<800;start+=128){j=start+lane;
  if(j<800){if(oa_sample(&x,cotangent,adjoint,pixel*800+j))atomicExch((int*)scratch,1);
   for(k=0;k<4;++k)work[j*4+k]=x.weighted[(pixel*800+j)*4+k];}
  __syncthreads();
 }
 if(!lane&&(oc_reduce(&x,pixel,work)||oa_reduce(&x,pixel,adjoint,gradient)))atomicExch((int*)scratch,1);
}
#undef GFFX_OC_LOCAL
