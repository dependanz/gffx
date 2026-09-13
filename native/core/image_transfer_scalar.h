#ifndef GFFX_IMAGE_TRANSFER_SCALAR_H
#define GFFX_IMAGE_TRANSFER_SCALAR_H
/* Rounded reference evaluation, NOT an exactly rounded generic rational power.
 * Only sRGB's positive-normal nonlinear domains enter the root routine. Fixed
 * 24 Newton steps from2 on m in[1,2), fixed integer multiplication order, exponent
 * reduction, no libm pow, no FMA. CPU/CUDA qualification and Decimal envelopes are
 * required separately. Compile host strict FP and CUDA fmad=false, ftz=false.
 * 2^(r/q) constants below were generated with Python Decimal precision90.
 * Float32 transfers evaluate in float64, then round value/first derivative once
 * to float32. Their branch decision still uses the float32-represented threshold.
 * This prevents float32 intermediate error from exhausting the top exponent bin;
 * it is not a clamp, an overflow-policy exemption or a float64 boundary proof.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifndef GFFX_IMAGE_SCALAR
#define GFFX_IMAGE_SCALAR static
#endif

GFFX_IMAGE_SCALAR float gffx_im_div_f32(float a,float b){
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a,b);
#else
    return a/b;
#endif
}
GFFX_IMAGE_SCALAR double gffx_im_div_f64(double a,double b){
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a,b);
#else
    return a/b;
#endif
}
GFFX_IMAGE_SCALAR uint32_t gffx_im_bits_f32(float x){
#if defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    uint32_t b;memcpy(&b,&x,sizeof(b));return b;
#endif
}
GFFX_IMAGE_SCALAR float gffx_im_from_f32(uint32_t b){
#if defined(__CUDA_ARCH__)
    return __uint_as_float(b);
#else
    float x;memcpy(&x,&b,sizeof(x));return x;
#endif
}
GFFX_IMAGE_SCALAR uint64_t gffx_im_bits_f64(double x){
#if defined(__CUDA_ARCH__)
    return (uint64_t)__double_as_longlong(x);
#else
    uint64_t b;memcpy(&b,&x,sizeof(b));return b;
#endif
}
GFFX_IMAGE_SCALAR double gffx_im_from_f64(uint64_t b){
#if defined(__CUDA_ARCH__)
    return __longlong_as_double((long long)b);
#else
    double x;memcpy(&x,&b,sizeof(x));return x;
#endif
}

#define GFFX_IMAGE_MATH(T,S,U,MASK,SHIFT,BIAS,EXPMAX,LDEXP) \
GFFX_IMAGE_SCALAR T gffx_im_factor_##S(int r,int q){ \
    if(q==5){switch(r){case 0:return (T)1; \
        case 1:return (T)1.14869835499703500679862694677792758944; \
        case 2:return (T)1.31950791077289425937400197122964013303; \
        case 3:return (T)1.51571656651039808234725980130644523868; \
        default:return (T)1.74110112659224827827254003495949219796;}} \
    switch(r){case 0:return (T)1; \
        case 1:return (T)1.05946309435929526456182529494634170078; \
        case 2:return (T)1.12246204830937298143353304967917951623; \
        case 3:return (T)1.18920711500272106671749997056047591529; \
        case 4:return (T)1.25992104989487316476721060727822835057; \
        case 5:return (T)1.33483985417003436483083188118445277491; \
        case 6:return (T)1.41421356237309504880168872420969807857; \
        case 7:return (T)1.49830707687668149879928073202979579630; \
        case 8:return (T)1.58740105196819947475170563927230826039; \
        case 9:return (T)1.68179283050742908606225095246642979008; \
        case 10:return (T)1.78179743628067860948045241118102501597; \
        default:return (T)1.88774862536338699328382631333506875202;} \
} \
GFFX_IMAGE_SCALAR T gffx_im_scale_##S(T x,int k){ \
    U bits=gffx_im_bits_##S(x);int exponent=(int)(bits>>SHIFT)+k; \
    if(exponent>=EXPMAX)return (T)INFINITY; \
    if(exponent<=0)return LDEXP(x,k); \
    return gffx_im_from_##S((bits&(U)(MASK))|((U)exponent<<SHIFT)); \
} \
GFFX_IMAGE_SCALAR T gffx_im_power_##S(T x,int p,int q){ \
    U bits=gffx_im_bits_##S(x);int e=(int)(bits>>SHIFT)-BIAS; \
    T m=gffx_im_from_##S((bits&(U)(MASK))|((U)BIAS<<SHIFT)),root=(T)2; \
    int iteration,j,k,r,absolute=p<0?-p:p;T result=(T)1; \
    for(iteration=0;iteration<24;++iteration){T denominator=(T)1,numerator; \
        for(j=0;j<q-1;++j)denominator=denominator*root; \
        numerator=(T)(q-1)*root; numerator=numerator+gffx_im_div_##S(m,denominator); \
        root=gffx_im_div_##S(numerator,(T)q); } \
    for(j=0;j<absolute;++j)result=result*root; \
    if(p<0)result=gffx_im_div_##S((T)1,result); \
    k=(p*e)/q;r=(p*e)%q;if(r<0){r+=q;--k;} \
    result=result*gffx_im_factor_##S(r,q); \
    return gffx_im_scale_##S(result,k); \
} \
GFFX_IMAGE_SCALAR int gffx_im_evaluate_##S(T x,int transfer,int linear,T *value,T *derivative){ \
    T y,d;if(!isfinite(x)||transfer<0||transfer>1){*value=(T)NAN;*derivative=(T)NAN;return 1;} \
    if(transfer==0){ \
        if(linear){y=gffx_im_div_##S(x,(T)12.92);d=gffx_im_div_##S((T)1,(T)12.92);} \
        else{T base=gffx_im_div_##S(x+(T).055,(T)1.055); \
            y=gffx_im_power_##S(base,12,5);d=gffx_im_div_##S((T)12,(T)5)*gffx_im_power_##S(base,7,5); \
            d=gffx_im_div_##S(d,(T)1.055);} \
    }else{ \
        if(linear){y=(T)12.92*x;d=(T)12.92;} \
        else{y=(T)1.055*gffx_im_power_##S(x,5,12);y=y-(T).055; \
            d=(T)1.055*gffx_im_div_##S((T)5,(T)12);d=d*gffx_im_power_##S(x,-7,12);} \
    } \
    *value=y;*derivative=d;return !isfinite(y)||!isfinite(d); \
}
GFFX_IMAGE_MATH(float,f32,uint32_t,UINT32_C(0x007fffff),23,127,255,ldexpf)
GFFX_IMAGE_MATH(double,f64,uint64_t,UINT64_C(0x000fffffffffffff),52,1023,2047,ldexp)
#undef GFFX_IMAGE_MATH
#include "image_transfer_boundary.h"

GFFX_IMAGE_SCALAR int gffx_im_transfer_f64(double x,int transfer,double *value,double *derivative){
    int status=gffx_im_evaluate_f64(x,transfer,x<=(transfer==0?.04045:.0031308),value,derivative);
    if(isfinite(x)&&transfer==0&&x>=0x1p+425){
        *value=gffx_im_boundary_decode_f64(x);return !isfinite(*value)||!isfinite(*derivative);
    }
    if(isfinite(x)&&transfer==1&&x<=-0x1p+1018){
        *value=gffx_im_boundary_negative_encode_f64(x);return !isfinite(*value)||!isfinite(*derivative);
    }
    return status;
}
GFFX_IMAGE_SCALAR int gffx_im_transfer_f32(float x,int transfer,float *value,float *derivative){
    double y,d;
    int status=gffx_im_evaluate_f64((double)x,transfer,
        x<=(transfer==0?.04045f:.0031308f),&y,&d);
    *value=(float)y;*derivative=(float)d;
    return status||!isfinite(*value)||!isfinite(*derivative);
}
#endif
