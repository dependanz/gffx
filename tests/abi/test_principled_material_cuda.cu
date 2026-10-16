/* Standalone fixed16-input material probe, not installed or resource admission. */
#include <cuda_runtime.h>
#define GFFX_PC_SCALAR static __device__
#include "principled_material_scalar.h"
__global__ void material_point_probe(int n,const double *cn,const double *logs,const double *curve,const double *partial,const double *mean,double width,const double *input,int side,double *out,int *status){
 if(blockIdx.x==0&&threadIdx.x==0)*status=gffx_pm_point(n,cn,logs,curve,partial,mean,width,input,side,out);
}
extern "C" __declspec(dllexport) int principled_test_material_point_cuda(int n,const double *cn,const double *logs,const double *curve,const double *partial,const double *mean,double width,const double *input,int side,double *out,int *status,void *stream){
 if(n<1||n>512||!cn||!logs||!curve||!partial||!mean||!input||!out||!status)return -1;
 material_point_probe<<<1,1,0,(cudaStream_t)stream>>>(n,cn,logs,curve,partial,mean,width,input,side,out,status);return (int)cudaGetLastError();
}
