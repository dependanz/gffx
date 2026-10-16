#define GFFX_PC_SCALAR static __device__ __forceinline__
#include "../core/principled_curve_scalar.h"
/* Each node has the original scalar operation order. Kernel boundaries make
 * values visible before slopes and slopes visible before angular controls. */
extern "C" __global__ void gffx_pc_domain(const double *x,const double *b,const double *u,long long batch,int n,int *status){
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(j<n&&(!isfinite(x[j])||x[j]>0.||(j&&(!(x[j]>x[j-1])||!isfinite(x[j]-x[j-1])))))atomicExch(status,1);
 if(j<batch*(2*n+1)*4&&(!isfinite(b[j])||b[j]<-745.||b[j]>0.))atomicExch(status,1);
 if(j<batch&&(!isfinite(u[j])||u[j]<0.||u[j]>1.))atomicExch(status,1);
}
extern "C" __global__ void gffx_pc_values(const double *x,const double *b,const double *u,double *y,double *dy,double *scratch,long long batch,int n,int *status){
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;int count=2*n+1,size=10*n-6;
 if(j>=batch*count)return;long long row=j/count;int a=(int)(j%count);double v=gffx_pc_curve(b+4*j,u[row]);double d=gffx_pc_curve_vjp(b+4*j,u[row],1.,v);
 y[row*size+a]=v;dy[row*size+a]=d;
 if(a==2*n){y[row*size+a+1]=v*0.;dy[row*size+a+1]=d*0.;}
 if(!isfinite(v)||!isfinite(d))atomicExch(status,1);
}
extern "C" __global__ void gffx_pc_slopes(const double *x,const double *b,const double *u,double *y,double *dy,double *scratch,long long batch,int n,int *status){
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(j>=batch*n)return;
 long long row=j/n;int node=(int)(j%n),size=10*n-6,stride=size>4*n?size:4*n;double *m=scratch+row*stride;
 gffx_pc_pair_slopes(n,node,x,y+row*size,dy+row*size,m+2*node,m+2*n+2*node);
}
extern "C" __global__ void gffx_pc_controls(const double *x,const double *b,const double *u,double *y,double *dy,double *scratch,long long batch,int n,int *status){
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(j>=batch*(n-1)*2)return;
 long long row=j/((n-1)*2);int i=(int)(j%((n-1)*2))/2,c=(int)(j%2),size=10*n-6,stride=size>4*n?size:4*n;
 double *out=y+row*size,*dout=dy+row*size,*m=scratch+row*stride,*dm=m+2*n;double h=x[i+1]-x[i];int a=2*i+c,z=2*n+2+8*i+c;
 out[z]=out[a];out[z+2]=out[a]+gffx_pc_div(h*m[a],3.);out[z+4]=out[a+2]-gffx_pc_div(h*m[a+2],3.);out[z+6]=out[a+2];
 dout[z]=dout[a];dout[z+2]=dout[a]+gffx_pc_div(h*dm[a],3.);dout[z+4]=dout[a+2]-gffx_pc_div(h*dm[a+2],3.);dout[z+6]=dout[a+2];
 for(int k=0;k<4;++k)if(!isfinite(out[z+2*k])||!isfinite(dout[z+2*k]))atomicExch(status,1);
}
extern "C" __global__ void gffx_pc_contract(const double *p,const double *g,double *out,double *scratch,long long batch,int size,int *status){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=batch)return;
 double *row=scratch+i*size;
 for(int j=0;j<size;++j){row[j]=p[i*size+j]*g[i*size+j];if(!isfinite(row[j]))atomicExch(status,1);}
 out[i]=gffx_pc_sum(row,size);if(!isfinite(out[i]))atomicExch(status,1);
}

extern "C" __global__ void gffx_pq_domain(const double *cn,const double *logs,const double *weights,const double *v,const double *p,const double *cosine,const int32_t *owners,int b,int n,int q,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x,size=10*n-6;
 if(j<n&&(!isfinite(cn[j])||cn[j]<=0.||cn[j]>1.||(j&&cn[j]<=cn[j-1])||(j==n-1&&cn[j]!=1.)||!isfinite(logs[j])||logs[j]>0.||(j&&(!(logs[j]>logs[j-1])||!isfinite(logs[j]-logs[j-1])))))atomicExch(status,1);
 if(j<4*(n-1)&&(!isfinite(weights[j])||weights[j]<0.))atomicExch(status,1);
 if(j<b*size&&(!isfinite(v[j])||!isfinite(p[j])))atomicExch(status,1);
 if(j<q&&(!isfinite(cosine[j])||cosine[j]<0.||cosine[j]>1.||owners[j]<0||owners[j]>=b))atomicExch(status,1);
}
extern "C" __global__ void gffx_pq_evaluate(const double *cn,const double *logs,const double *weights,const double *v,const double *p,const double *cosine,const int32_t *owners,double *out,double *mean,double *scratch,int b,int n,int q,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x,size=10*n-6,stride=4*n>q?4*n:q;
 if(j<q){gffx_pq_direction(n,cn,logs,v+owners[j]*size,p+owners[j]*size,cosine[j],out+6*j);for(int k=0;k<6;++k)if(!isfinite(out[6*j+k]))atomicExch(status,1);}
 if(j<b){gffx_pq_mean(n,cn,logs,weights,v+j*size,p+j*size,mean+4*j,scratch+j*stride);for(int k=0;k<4;++k)if(!isfinite(mean[4*j+k]))atomicExch(status,1);}
}
extern "C" __global__ void gffx_pq_backward_domain(const double *v,const double *m,const int32_t *owners,const double *g,const double *h,int b,int q,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j<6*q&&!isfinite(v[j]))atomicExch(status,1);if(j<4*b&&!isfinite(m[j]))atomicExch(status,1);
 if(j<2*q&&!isfinite(g[j]))atomicExch(status,1);if(j<2*b&&!isfinite(h[j]))atomicExch(status,1);
 if(j<q&&(owners[j]<0||owners[j]>=b))atomicExch(status,1);
}
extern "C" __global__ void gffx_pq_backward(const double *v,const double *m,const int32_t *owners,const double *g,const double *h,double *u,double *c,double *scratch,int b,int q,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j<q){c[j]=v[6*j+4]*g[2*j]+v[6*j+5]*g[2*j+1];if(!isfinite(c[j]))atomicExch(status,1);}
 if(j<b){u[j]=gffx_pq_contract(j,q,v,m,owners,g,h,scratch+j*(q>4?q:4));if(!isfinite(u[j]))atomicExch(status,1);}
}
