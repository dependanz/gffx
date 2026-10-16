#ifndef GFFX_UNIT_NEWTON_SCALAR_H
#define GFFX_UNIT_NEWTON_SCALAR_H
/* Six prescribed IEEE binary64 steps, not a platform sqrt substitution. */
#if defined(__CUDACC__)
#define GFFX_UN_INLINE static __device__ __forceinline__
#else
#define GFFX_UN_INLINE static
#endif
GFFX_UN_INLINE double gffx_unit_newton_scalar(double x){
 int k;double y=1.5;
 for(k=0;k<6;++k){
#if defined(__CUDA_ARCH__)
  y=__dmul_rn(__dadd_rn(y,__ddiv_rn(x,y)),.5);
#else
  volatile double quotient=x/y;volatile double sum=y+quotient;
  volatile double half=sum*.5;y=half;
#endif
 }
 return y;
}
GFFX_UN_INLINE void gffx_unit_vectors_scalar(const double *x,double *y){
 double m=fabs(x[0]),a,b,c,root;int k;
 for(k=1;k<3;++k)if(fabs(x[k])>m)m=fabs(x[k]);
#if defined(__CUDA_ARCH__)
 a=__ddiv_rn(x[0],m);b=__ddiv_rn(x[1],m);c=__ddiv_rn(x[2],m);
 root=gffx_unit_newton_scalar(__dadd_rn(__dadd_rn(__dmul_rn(a,a),__dmul_rn(b,b)),__dmul_rn(c,c)));
 y[0]=__ddiv_rn(a,root);y[1]=__ddiv_rn(b,root);y[2]=__ddiv_rn(c,root);
#else
 {volatile double aa=x[0]/m,bb=x[1]/m,cc=x[2]/m;
  volatile double a2=aa*aa,b2=bb*bb,c2=cc*cc;volatile double ab=a2+b2,abc=ab+c2;
  a=aa;b=bb;c=cc;root=gffx_unit_newton_scalar(abc);}
 {volatile double aa=a/root,bb=b/root,cc=c/root;y[0]=aa;y[1]=bb;y[2]=cc;}
#endif
}
#undef GFFX_UN_INLINE
#endif
