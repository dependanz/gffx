#ifndef GFFX_GAUSSIAN_SCALAR_H
#define GFFX_GAUSSIAN_SCALAR_H
/* Bounded Gaussian arithmetic shared by C11 and CUDA. Strict RN, no FMA/FTZ.
 * Constants: independent rational certificate in tools/gaussian_weights_bounds.py.
 * This internal scalar is not a public generic exponential. Maximum 65 taps.
 * Independent oracle, compiled scalar and installed CPU/CUDA gates are separate. */
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifndef GFFX_GW_SCALAR
#define GFFX_GW_SCALAR static
#endif
GFFX_GW_SCALAR uint64_t gffx_gw_bits(double x){
#if defined(__CUDA_ARCH__)
    return (uint64_t)__double_as_longlong(x);
#else
    uint64_t b;memcpy(&b,&x,sizeof b);return b;
#endif
}
GFFX_GW_SCALAR double gffx_gw_from_bits(uint64_t b){
#if defined(__CUDA_ARCH__)
    return __longlong_as_double((long long)b);
#else
    double x;memcpy(&x,&b,sizeof x);return x;
#endif
}
GFFX_GW_SCALAR double gffx_gw_div(double a,double b){
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a,b);
#else
    return a/b;
#endif
}
GFFX_GW_SCALAR float gffx_gw_float(double x){
#if defined(__CUDA_ARCH__)
    return __double2float_rn(x);
#else
    return (float)x;
#endif
}
/* Exact integer scaling. int64 arithmetic bounds even INT_MIN/MAX exponents;
 * at most 52 normalization shifts, and every variable shift guarded below 64. */
GFFX_GW_SCALAR int gffx_gw_scale(double x,int exponent,double *out){
    uint64_t bits=gffx_gw_bits(x),sign=bits&UINT64_C(0x8000000000000000);
    uint64_t sig=bits&UINT64_C(0x000fffffffffffff),stored=(bits>>52)&2047;
    int64_t power;int leading=52;
    if(stored==2047)return 1;
    if(!stored&&!sig){*out=x;return 0;}
    if(stored){sig|=UINT64_C(0x0010000000000000);power=(int64_t)stored-1023-52+exponent;}
    else{uint64_t scan=sig;leading=0;while(scan>>1){scan>>=1;++leading;}power=(int64_t)-1074+exponent;}
    if(power+leading>1023)return 1;
    if(power+leading>=-1022){uint64_t normalized=sig<<(52-leading);
        *out=gffx_gw_from_bits(sign|((uint64_t)(power+leading+1023)<<52)|(normalized-UINT64_C(0x0010000000000000)));return 0;}
    {int64_t shift=-1074-power;uint64_t rounded;
     if(shift>leading+1){*out=gffx_gw_from_bits(sign);return 0;}
     if(shift>0){uint64_t half=UINT64_C(1)<<(shift-1),discard;
        rounded=sig>>shift;discard=sig-(rounded<<shift);
        if(discard>half||(discard==half&&(rounded&1)))++rounded;
     }else rounded=sig<<(-shift);
     *out=gffx_gw_from_bits(sign|rounded);return 0;}
}
GFFX_GW_SCALAR double gffx_gw_scaled(double value,int exponent){
    double result=0.;/* Domain bounds prove no overflow in Gaussian intermediates. */
    (void)gffx_gw_scale(value,exponent,&result);return result;
}
GFFX_GW_SCALAR void gffx_gw_exp(double x,double *mantissa,int *exponent){
    const double coefficients[19]={0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p-1,0x1.5555555555555p-3,0x1.5555555555555p-5,0x1.1111111111111p-7,0x1.6c16c16c16c17p-10,0x1.a01a01a01a01ap-13,0x1.a01a01a01a01ap-16,0x1.71de3a556c734p-19,0x1.27e4fb7789f5cp-22,0x1.ae64567f544e4p-26,0x1.1eed8eff8d898p-29,0x1.6124613a86d09p-33,0x1.93974a8c07c9dp-37,0x1.ae7f3e733b81fp-41,0x1.ae7f3e733b81fp-45,0x1.952c77030ad4ap-49,0x1.6827863b97d97p-53};
    double selected,remainder,product,p;int n,k;
    if(x < -800.){*mantissa=0.;*exponent=0;return;}
    selected=x*0x1.71547652b82fep+0;n=(int)selected;
    if(selected-n < -.5||(selected-n==-.5&&(n%2!=0)))--n;
    product=n*0x1.62e4200000000p-1;remainder=x-product;product=n*0x1.fdf473de6af28p-22;remainder=remainder-product;
    p=coefficients[18];for(k=17;k>=0;--k){product=p*remainder;p=coefficients[k]+product;}
    *mantissa=p;*exponent=n;
}
GFFX_GW_SCALAR int gffx_gw_sum64(double *values,int count,double *out){
    int i,n;while(count>1){n=0;for(i=0;i+1<count;i+=2){double sum=values[i]+values[i+1];if(!isfinite(sum))return 1;values[n++]=sum;}
        if(count%2)values[n++]=values[count-1];count=n;}*out=values[0];return 0;
}
GFFX_GW_SCALAR int gffx_gw_sum32(float *values,int count,float *out){
    int i,n;while(count>1){n=0;for(i=0;i+1<count;i+=2){float sum=values[i]+values[i+1];if(!isfinite(sum))return 1;values[n++]=sum;}
        if(count%2)values[n++]=values[count-1];count=n;}*out=values[0];return 0;
}
GFFX_GW_SCALAR void gffx_gw_evaluate(double sigma,int radius,double *weights,double *derivatives){
    double m[33],a[33],scratch[65],z,mu;int e[33],i,j,count=2*radius+1;
    if(radius==0){weights[0]=1.;derivatives[0]=0.;return;}
    m[0]=1.;a[0]=0.;e[0]=0;
    for(j=1;j<=radius;++j){double t=gffx_gw_div((double)j,sigma),b=t*t;
        a[j]=gffx_gw_div(b,sigma);gffx_gw_exp(-.5*b,&m[j],&e[j]);}
    for(i=0;i<count;++i){j=i<radius?radius-i:i-radius;scratch[i]=gffx_gw_scaled(m[j],e[j]);}
    (void)gffx_gw_sum64(scratch,count,&z);
    for(j=0;j<=radius;++j){double product=m[j]*a[j];double normalized=gffx_gw_div(product,z);
        scratch[radius-j]=scratch[radius+j]=gffx_gw_scaled(normalized,e[j]);}
    (void)gffx_gw_sum64(scratch,count,&mu);
    for(j=0;j<=radius;++j){double weight=gffx_gw_scaled(gffx_gw_div(m[j],z),e[j]);
        double coefficient=a[j]-mu,derivative=0.;
        if(coefficient!=0.){double product=m[j]*coefficient;double normalized=gffx_gw_div(product,z);derivative=gffx_gw_scaled(normalized,e[j]);}
        weights[radius-j]=weights[radius+j]=weight;
        derivatives[radius-j]=derivatives[radius+j]=derivative;}
}
#endif
