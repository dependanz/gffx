/* Standalone arithmetic fixture, without Torch or CUDA links. --stream accepts
 * dtype(32/64), transfer(0/1), input, incoming gradient; outputs status/value/VJP. */
#include "image_transfer_scalar.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define IMAGE_TEST_EXPORT __declspec(dllexport)
#else
#define IMAGE_TEST_EXPORT __attribute__((visibility("default")))
#endif
IMAGE_TEST_EXPORT int image_test_transfer(int bits,int mode,double x,double gradient,double *value,double *vjp){
    int status;
    if(bits==32){float y,d;status=gffx_im_transfer_f32((float)x,mode,&y,&d);*value=y;*vjp=(float)((float)gradient*d);}
    else{double d;status=gffx_im_transfer_f64(x,mode,value,&d);*vjp=gradient*d;}
    return status;
}
IMAGE_TEST_EXPORT double image_test_root(int bits,int q,double x){
    return bits==32?gffx_im_power_f32((float)x,1,q):gffx_im_power_f64(x,1,q);
}
IMAGE_TEST_EXPORT void image_test_dd(int operation,double ah,double al,double bh,double bl,double *hi,double *lo){
    gffx_im_dd a=gffx_im_dd_make(ah,al),b=gffx_im_dd_make(bh,bl),r;
    if(operation==0)r=gffx_im_dd_add(a,b);
    else if(operation==1)r=gffx_im_dd_mul(a,b);
    else if(operation==2)r=gffx_im_dd_div(a,b);
    else if(operation==3)r=gffx_im_dd_fifth_root(a);
    else r=gffx_im_dd_factor5((int)ah);
    *hi=r.hi;*lo=r.lo;
}

int main(int argc,char **argv) {
    double y,d;float yf,df;
    if(argc==2&&strcmp(argv[1],"--boundary")==0){
        /* Attribution only: the same represented f32 input in both precisions.
         * This is not a replacement for the independent Decimal gate. */
        float x=0x1.54478ap+53f,base=gffx_im_div_f32(x+.055f,1.055f);
        uint32_t bits=gffx_im_bits_f32(base);
        int exponent=(int)(bits>>23)-127;
        float mantissa=gffx_im_from_f32((bits&UINT32_C(0x007fffff))|UINT32_C(0x3f800000));
        int status32=gffx_im_transfer_f32(x,0,&yf,&df);
        int status64=gffx_im_transfer_f64((double)x,0,&y,&d);
        printf("input=%a base32=%a exponent32=%d mantissa32=%a root32=%a status32=%d value32=%a derivative32=%a status64=%d value64=%a derivative64=%a\n",
               (double)x,(double)base,exponent,(double)mantissa,
               (double)gffx_im_power_f32(mantissa,1,5),status32,(double)yf,(double)df,status64,y,d);
        return 0;
    }
    if(argc==2&&strcmp(argv[1],"--roots")==0){
        int bits,q;double x;
        while(scanf("%d %d %lf",&bits,&q,&x)==3){
            if((q!=5&&q!=12)||x<1.||x>=2.)return 2;
            y=bits==32?gffx_im_power_f32((float)x,1,q):gffx_im_power_f64(x,1,q);
            printf("%.17g\n",y);
        }
        return 0;
    }
    if(argc==2&&strcmp(argv[1],"--stream")==0) {
        int bits,mode;double x,g;
        while(scanf("%d %d %lf %lf",&bits,&mode,&x,&g)==4) {
            int status;
            if(bits==32){status=gffx_im_transfer_f32((float)x,mode,&yf,&df);y=yf;d=(float)((float)g*df);}
            else{status=gffx_im_transfer_f64(x,mode,&y,&d);d=g*d;}
            printf("%d %.17g %.17g\n",status,y,d);
        }
        return 0;
    }
    assert(!gffx_im_transfer_f64(1.,0,&y,&d)&&y==1.);
    assert(!gffx_im_transfer_f64(1.,1,&y,&d)&&fabs(y-1.)<=2e-12);
    assert(!gffx_im_transfer_f32(1.f,0,&yf,&df)&&yf==1.f);
    assert(!gffx_im_transfer_f32(1.f,1,&yf,&df)&&fabsf(yf-1.f)<=2e-5f);
    assert(!gffx_im_transfer_f64(-0.,0,&y,&d)&&y==0.&&signbit(y));
    assert(!gffx_im_transfer_f32(-0.f,1,&yf,&df)&&yf==0.f&&signbit(yf));
    assert(gffx_im_transfer_f64(1.,2,&y,&d));
    /* Decimal preflight1 found a false overflow below FLT_MAX. Preserve the
     * exact represented input; widening tolerances cannot repair this status. */
    assert(!gffx_im_transfer_f32(0x1.54478ap+53f,0,&yf,&df));
    assert(isfinite(yf)&&isfinite(df));
    /* Widening intermediates must not move float32's branch boundary. */
    assert(!gffx_im_transfer_f32(.04045f,0,&yf,&df));
    assert(yf==(float)((double).04045f/12.92)&&df==(float)(1./12.92));
    assert(!gffx_im_transfer_f32(.0031308f,1,&yf,&df));
    assert(yf==(float)((double).0031308f*12.92)&&df==(float)12.92);
    assert(!gffx_im_transfer_f32(-0.f,0,&yf,&df)&&signbit(yf));
    assert(gffx_im_transfer_f32(0x1.fffffep+127f,0,&yf,&df));
    assert(gffx_im_transfer_f32(-0x1.fffffep+127f,1,&yf,&df));
    /* Widened-f32 preflight then exposed the independent f64 boundary defect. */
    assert(!gffx_im_transfer_f64(0x1.acb9abb20cf2dp+426,0,&y,&d));
    assert(isfinite(y)&&isfinite(d));
    {
        gffx_im_dd r=gffx_im_dd_sum(1.,0x1p-53),q;
        assert(r.hi==1.&&r.lo==0x1p-53);
        r=gffx_im_dd_product(1.+0x1p-27,1.-0x1p-27);
        assert(r.hi==1.&&r.lo==-0x1p-54);
        r=gffx_im_dd_add(gffx_im_dd_make(1.,0x1p-54),gffx_im_dd_make(-1.,0.));
        assert(r.hi==0x1p-54&&r.lo==0.);
        q=gffx_im_dd_div(gffx_im_dd_make(1.,0.),gffx_im_dd_make(3.,0.));
        r=gffx_im_dd_add(gffx_im_dd_mul(q,gffx_im_dd_make(3.,0.)),gffx_im_dd_make(-1.,0.));
        assert(fabs(r.hi)+fabs(r.lo)<0x1p-99);
        assert(gffx_im_dd_round_scaled(gffx_im_dd_make(1.,-0x1p-54-0x1p-106),1024)==0x1.fffffffffffffp+1023);
        assert(isinf(gffx_im_dd_round_scaled(gffx_im_dd_make(1.,-0x1p-54),1024)));
        assert(isinf(gffx_im_dd_round_scaled(gffx_im_dd_make(1.,-0x1p-54+0x1p-106),1024)));
        /* This rounds finite but its conditional uncertainty overlaps T. */
        r=gffx_im_dd_make(0x1.fffffffffffffp+0,0x1p-53-0x1p-85);
        assert(isfinite(gffx_im_dd_round_scaled(r,1023)));
        assert(isnan(gffx_im_boundary_round_finite(r,1023)));
        assert(isnan(gffx_im_boundary_round_finite(gffx_im_dd_make(1.,0.),1024)));
        assert(gffx_im_boundary_round_finite(gffx_im_dd_make(1.5,0.),1023)==0x1.8p+1023);
    }
    assert(gffx_im_transfer_f64(0x1.acb9abb20cf2ep+426,0,&y,&d));
    assert(!gffx_im_transfer_f64(-0x1.3d0722149b57fp+1020,1,&y,&d)&&isfinite(y));
    assert(gffx_im_transfer_f64(-0x1.3d0722149b580p+1020,1,&y,&d));
    return 0;
}
