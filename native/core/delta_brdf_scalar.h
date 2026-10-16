#ifndef GFFX_DELTA_BRDF_SCALAR_H
#define GFFX_DELTA_BRDF_SCALAR_H
#include "unit_newton_scalar.h"
#if defined(__CUDACC__)
#define GFFX_DL_INLINE static __device__ __forceinline__
#else
#define GFFX_DL_INLINE static
#endif
GFFX_DL_INLINE double gffx_dl_add(double a,double b){
#if defined(__CUDA_ARCH__)
 return __dadd_rn(a,b);
#else
 volatile double x=a+b;return x;
#endif
}
GFFX_DL_INLINE double gffx_dl_mul(double a,double b){
#if defined(__CUDA_ARCH__)
 return __dmul_rn(a,b);
#else
 volatile double x=a*b;return x;
#endif
}
GFFX_DL_INLINE int gffx_dl_domain(const double *normal,const double *brdf,const double *direction,const double *incoming,const double *cotangent,int q){
 int k,j;for(k=0;k<3;++k)if(!isfinite(normal[k])||!isfinite(cotangent[k]))return 1;
 if(normal[0]==0.&&normal[1]==0.&&normal[2]==0.)return 1;
 for(j=0;j<q;++j){
  for(k=0;k<3;++k)if(!isfinite(brdf[3*j+k])||brdf[3*j+k]<0.||!isfinite(direction[3*j+k])||!isfinite(incoming[3*j+k])||incoming[3*j+k]<0.)return 1;
  if(direction[3*j]==0.&&direction[3*j+1]==0.&&direction[3*j+2]==0.)return 1;
 }return 0;
}
/* Exact eager delta integration and BRDF-only reverse factors. No FMA or
   changed reduction tree. The original full Principled Jacobian is separate. */
GFFX_DL_INLINE int gffx_dl_point(const double *normal,const double *brdf,const double *direction,const double *incoming,const double *cotangent,int q,double *value,double *gradient){
 double n[3],d[3],values[3][8];int j,k,count,i;gffx_unit_vectors_scalar(normal,n);
 for(j=0;j<q;++j){double cosine;int active;gffx_unit_vectors_scalar(direction+3*j,d);
  cosine=gffx_dl_add(gffx_dl_add(gffx_dl_mul(n[0],d[0]),gffx_dl_mul(n[1],d[1])),gffx_dl_mul(n[2],d[2]));active=cosine>0.;if(!active)cosine=0.;
  for(k=0;k<3;++k){double b=active?brdf[3*j+k]:0.,irr=active?incoming[3*j+k]:0.;
   values[k][j]=gffx_dl_mul(gffx_dl_mul(b,irr),cosine);
   /* For Q>1 the eager adjacent-pair reduction VJP adds disjoint
      slice gradients into zero-filled arrays. Preserve that signed-zero
      addition before reversing lighting factors; Q==1 only selects a slot. */
   gradient[3*j+k]=active?gffx_dl_mul(gffx_dl_mul(q>1?gffx_dl_add(cotangent[k],0.):cotangent[k],cosine),irr):0.;
   if(!isfinite(gradient[3*j+k]))return 1;
  }
 }
 for(k=0;k<3;++k){count=q;while(count>1){int pairs=count/2;for(i=0;i<pairs;++i)values[k][i]=gffx_dl_add(values[k][2*i],values[k][2*i+1]);if(count&1)values[k][pairs]=values[k][count-1];count=pairs+(count&1);}
  value[k]=q?values[k][0]:0.;if(!isfinite(value[k]))return 1;
 }return 0;
}
#undef GFFX_DL_INLINE
#endif
