/* Standalone CUDA arithmetic probe only: not built into the wheel/provider. */
#include <cuda_runtime.h>
#define GFFX_PC_SCALAR static __host__ __device__
#include "principled_curve_scalar.h"
__global__ void curve_probe(int count,const double *b,const double *u,const double *g,const double *x,double *out){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<count){double value=gffx_pc_curve(b+4*i,u[i]);
        out[3*i]=value;out[3*i+1]=gffx_pc_curve_vjp(b+4*i,u[i],g[i],value);out[3*i+2]=gffx_pc_log(x[i]);}
}
extern "C" __declspec(dllexport) int principled_test_cuda(int count,const double *b,const double *u,const double *g,const double *x,double *out,void *stream){
    if(count<0||count>32768||!b||!u||!g||!x||!out)return -1;
    if(count==0)return 0;
    curve_probe<<<(count+63)/64,64,0,(cudaStream_t)stream>>>(count,b,u,g,x,out);
    return (int)cudaGetLastError();
}

__global__ void slopes_probe(int n,const double *x,const double *y,const double *dy,double *out,double *dout){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n)gffx_pc_pair_slopes(n,i,x,y,dy,out+2*i,dout+2*i);
}
extern "C" __declspec(dllexport) int principled_test_slopes_cuda(int n,const double *x,const double *y,const double *dy,double *out,double *dout,void *stream){
    if(n<1||n>1024||!x||!y||!dy||!out||!dout)return -1;
    slopes_probe<<<(n+63)/64,64,0,(cudaStream_t)stream>>>(n,x,y,dy,out,dout);
    return (int)cudaGetLastError();
}

__global__ void build_probe(int n,const double *x,const double *b,double u,double *out,double *dout,double *scratch){
    if(blockIdx.x==0&&threadIdx.x==0)gffx_pc_curve_build(n,x,b,u,out,dout,scratch);
}
extern "C" __declspec(dllexport) int principled_test_build_cuda(int n,const double *x,const double *b,double u,double *out,double *dout,double *scratch,void *stream){
    if(n<1||n>512||!x||!b||!out||!dout||!scratch)return -1;
    build_probe<<<1,1,0,(cudaStream_t)stream>>>(n,x,b,u,out,dout,scratch);
    return (int)cudaGetLastError();
}

__global__ void query_probe(int n,const double *cn,const double *logs,const double *weights,const double *curve,const double *dcurve,double c,double *out,double *scratch){
    if(blockIdx.x==0&&threadIdx.x==0)gffx_pc_curve_query(n,cn,logs,weights,curve,dcurve,c,out,scratch);
}
extern "C" __declspec(dllexport) int principled_test_query_cuda(int n,const double *cn,const double *logs,const double *weights,const double *curve,const double *dcurve,double c,double *out,double *scratch,void *stream){
    if(n<1||n>512||!cn||!logs||!weights||!curve||!dcurve||!out||!scratch)return -1;
    query_probe<<<1,1,0,(cudaStream_t)stream>>>(n,cn,logs,weights,curve,dcurve,c,out,scratch);
    return (int)cudaGetLastError();
}
