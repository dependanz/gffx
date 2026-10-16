#include "../core/principled_material_scalar.h"
extern "C" __global__ void gffx_pp_domain(const double *cn,const double *logs,const double *curve,const double *partial,const double *mean,const double *width,const double *normal,const double *view,const double *light,const double *albedo,const double *roughness,const double *f0,int p,int q,int n,int curve_rows,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x,size=10*n-6;
 if(j<n&&(!isfinite(cn[j])||cn[j]<=0.||cn[j]>1.||(j==n-1&&cn[j]!=1.)||!isfinite(logs[j])||logs[j]>0.||(j&&(cn[j]<=cn[j-1]||logs[j]<=logs[j-1]||!isfinite(logs[j]-logs[j-1])))))atomicExch(status,1);
 if(j<curve_rows*size&&(!isfinite(curve[j])||!isfinite(partial[j])||(!(j%2)&&(curve[j]<curve[j+1]||curve[j]>1.||curve[j+1]<0.))))atomicExch(status,1);
 if(j<curve_rows*4&&!isfinite(mean[j]))atomicExch(status,1);
 if(j<curve_rows&&(!isfinite(width[j])||width[j]<=0.||mean[4*j]<=0.||mean[4*j]>1.||mean[4*j+1]<=0.||mean[4*j+1]>mean[4*j]))atomicExch(status,1);
 if(j<p&&(!isfinite(roughness[j])||roughness[j]<=0.||roughness[j]>1.||(normal[3*j]==0.&&normal[3*j+1]==0.&&normal[3*j+2]==0.)||(view[3*j]==0.&&view[3*j+1]==0.&&view[3*j+2]==0.)))atomicExch(status,1);
 if(j<p*3&&(!isfinite(normal[j])||!isfinite(view[j])||!isfinite(albedo[j])||albedo[j]<0.||albedo[j]>1.||!isfinite(f0[j])||f0[j]<0.||f0[j]>1.))atomicExch(status,1);
 if(j<p*q*3&&!isfinite(light[j]))atomicExch(status,1);if(j<p*q&&light[3*j]==0.&&light[3*j+1]==0.&&light[3*j+2]==0.)atomicExch(status,1);
}
extern "C" __global__ void gffx_pp_evaluate(const double *cn,const double *logs,const double *curve,const double *partial,const double *mean,const double *width,const double *normal,const double *view,const double *light,const double *albedo,const double *roughness,const double *f0,double *value,double *jacobian,int p,int q,int n,int curve_rows,int options,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=p*q)return;int owner=j/q,size=10*n-6,curve_owner=curve_rows==1?0:owner;double input[16],out[51];gffx_pp_pack(owner,j,normal,view,light,albedo,roughness,f0,input);
 int fail=(options&2)?gffx_pm_point(n,cn,logs,curve+curve_owner*size,partial+curve_owner*size,mean+4*curve_owner,width[curve_owner],input,options&1,out):gffx_pv_point(n,cn,logs,curve+curve_owner*size,partial+curve_owner*size,mean+4*curve_owner,width[curve_owner],input,options&1,out);
 if(fail){atomicExch(status,1);return;}for(int k=0;k<3;++k)value[3*j+k]=out[k];if(options&2)for(int k=0;k<48;++k)jacobian[48*j+k]=out[3+k];
}
extern "C" __global__ void gffx_pp_backward(const double *jac,const double *gradient,double *point,double *light,double *scratch,int p,int q,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j<p){gffx_pp_contract(q,jac+j*q*48,gradient+j*q*3,point+j*16,scratch+j*(q?q:1));for(int k=0;k<16;++k)if(!isfinite(point[j*16+k]))atomicExch(status,1);}
 if(j<p*q)for(int k=0;k<3;++k){light[j*3+k]=gffx_pp_rgb(jac+j*48,gradient+j*3,k+6);if(!isfinite(light[j*3+k]))atomicExch(status,1);}
}

extern "C" __global__ void gffx_ordered_rgba(const double *carry,const double *values,double *output,int n,int *status){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=4)return;double total=carry[k];
 for(int j=0;j<n;++j)total=__dadd_rn(total,values[4*j+k]);
 output[k]=total;if(!isfinite(total))atomicExch(status,1);
}

#include "../core/unit_newton_scalar.h"
extern "C" __global__ void gffx_unit_newton_domain(const double *x,int n,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j<n&&(!isfinite(x[j])||x[j]<=0.||x[j]>4.))atomicExch(status,1);
}
extern "C" __global__ void gffx_unit_newton_evaluate(const double *x,double *y,int n){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)y[j]=gffx_unit_newton_scalar(x[j]);
}

extern "C" __global__ void gffx_unit_vectors_domain(const double *x,int n,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
 if(!isfinite(x[3*j])||!isfinite(x[3*j+1])||!isfinite(x[3*j+2])||(x[3*j]==0.&&x[3*j+1]==0.&&x[3*j+2]==0.))atomicExch(status,1);
}
extern "C" __global__ void gffx_unit_vectors_evaluate(const double *x,double *y,int n){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)gffx_unit_vectors_scalar(x+3*j,y+3*j);
}

#include "../core/delta_brdf_scalar.h"
extern "C" __global__ void gffx_delta_brdf_domain(const double *n,const double *b,const double *l,const double *i,const double *g,int p,int q,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<p&&gffx_dl_domain(n+3*j,(q?b+3*q*j:b),(q?l+3*q*j:l),(q?i+3*q*j:i),g+3*j,q))atomicExch(status,1);
}
extern "C" __global__ void gffx_delta_brdf_evaluate(const double *n,const double *b,const double *l,const double *i,const double *g,double *v,double *out,int p,int q,int *status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<p&&gffx_dl_point(n+3*j,(q?b+3*q*j:b),(q?l+3*q*j:l),(q?i+3*q*j:i),g+3*j,q,v+3*j,(q?out+3*q*j:out)))atomicExch(status,1);
}
