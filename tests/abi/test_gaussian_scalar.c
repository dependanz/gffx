/* Standalone shared arithmetic qualification; not public ABI or device evidence. */
#include "gaussian_scalar.h"
#include <assert.h>
#include <float.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
EXPORT int gaussian_test_scale(double value,int exponent,double *result){return gffx_gw_scale(value,exponent,result);}
EXPORT int gaussian_test_weights(double sigma,int radius,int precision,double *weights,double *derivatives){
    int i;gffx_gw_evaluate(sigma,radius,weights,derivatives);
    if(precision==32)for(i=0;i<2*radius+1;++i){weights[i]=gffx_gw_float(weights[i]);derivatives[i]=gffx_gw_float(derivatives[i]);}
    return 0;
}
EXPORT int gaussian_test_vjp(double sigma,int radius,int precision,const double *gradients,double *result){
    double weights[65],derivatives[65];int i;gffx_gw_evaluate(sigma,radius,weights,derivatives);
    if(radius==0){*result=0.;return 0;}
    if(precision==32){float products[65];for(i=0;i<2*radius+1;++i){products[i]=gffx_gw_float(derivatives[i])*(float)gradients[i];if(!isfinite(products[i]))return 1;}
        {float sum=0.;int status=gffx_gw_sum32(products,2*radius+1,&sum);*result=sum;return status;}}
    for(i=0;i<2*radius+1;++i){derivatives[i]*=gradients[i];if(!isfinite(derivatives[i]))return 1;}
    return gffx_gw_sum64(derivatives,2*radius+1,result);
}
int main(void){
    double y,w[65],d[65],g[65]={0.};int i;
    assert(!gffx_gw_scale(1.,-1074,&y)&&gffx_gw_bits(y)==1);
    assert(!gffx_gw_scale(1.,-1075,&y)&&gffx_gw_bits(y)==0);
    assert(!gffx_gw_scale(-1.5,-1074,&y)&&gffx_gw_bits(y)==UINT64_C(0x8000000000000002));
    assert(gffx_gw_scale(DBL_MAX,1,&y));
    gffx_gw_evaluate(.83,32,w,d);for(i=0;i<65;++i){assert(w[i]>=0.&&isfinite(d[i]));assert(gffx_gw_bits(w[i])==gffx_gw_bits(w[64-i]));assert(gffx_gw_bits(d[i])==gffx_gw_bits(d[64-i]));}
    gffx_gw_evaluate(.1033,4,w,d);assert(w[0]==0.&&d[0]>0.);
    g[0]=-3.;assert(!gaussian_test_vjp(.83,0,64,g,&y)&&gffx_gw_bits(y)==0);
    return 0;
}
