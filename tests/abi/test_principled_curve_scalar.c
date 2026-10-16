/* Test-only exports; never linked or installed as part of the gffx core. */
#include "principled_curve_scalar.h"
#include <math.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
EXPORT int principled_test_log(double x,double *out){
    double result;
    if(!out||!isfinite(x)||x<=0.||x>1.)return 1;
    result=gffx_pc_log(x);if(!isfinite(result))return 2;
    *out=result;return 0;
}
EXPORT int principled_test_curve(const double *b,double u,double g,double *out,double *grad){
    double value,derivative;int i;
    if(!b||!out||!grad||out==grad||!isfinite(u)||u<0.||u>1.||!isfinite(g))return 1;
    for(i=0;i<4;++i)if(!isfinite(b[i])||b[i]<-745.||b[i]>0.)return 1;
    value=gffx_pc_curve(b,u);derivative=gffx_pc_curve_vjp(b,u,g,value);
    if(!isfinite(value)||!isfinite(derivative))return 2;
    *out=value;*grad=derivative;return 0;
}

EXPORT int principled_test_slopes(int n,const double *x,const double *y,const double *dy,double *out,double *dout){
    int i;
    if(n<1||n>1024||!x||!y||!dy||!out||!dout)return 1;
    for(i=0;i<n;++i)if(!isfinite(x[i])||(i&&x[i]<=x[i-1]))return 1;
    for(i=0;i<2*n;++i)if(!isfinite(y[i])||!isfinite(dy[i]))return 1;
    for(i=0;i<n;++i)gffx_pc_pair_slopes(n,i,x,y,dy,out+2*i,dout+2*i);
    return 0;
}

EXPORT int principled_test_build(int n,const double *x,const double *b,double u,double *out,double *dout,double *scratch){
    int i;
    if(n<1||n>512||!x||!b||!out||!dout||!scratch||!isfinite(u)||u<0.||u>1.)return 1;
    for(i=0;i<n;++i)if(!isfinite(x[i])||(i&&x[i]<=x[i-1]))return 1;
    for(i=0;i<(2*n+1)*4;++i)if(!isfinite(b[i])||b[i]<-745.||b[i]>0.)return 1;
    gffx_pc_curve_build(n,x,b,u,out,dout,scratch);return 0;
}

EXPORT int principled_test_query(int n,const double *cn,const double *logs,const double *weights,const double *curve,const double *dcurve,double c,double *out,double *scratch){
    if(n<1||n>512||!cn||!logs||!weights||!curve||!dcurve||!out||!scratch||!isfinite(c)||c<0.||c>1.)return 1;
    gffx_pc_curve_query(n,cn,logs,weights,curve,dcurve,c,out,scratch);return 0;
}
