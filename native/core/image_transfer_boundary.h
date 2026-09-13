#ifndef GFFX_IMAGE_TRANSFER_BOUNDARY_H
#define GFFX_IMAGE_TRANSFER_BOUNDARY_H
/* Bounded compensated HDR boundary candidate. Included after scalar primitives.
 * No generic pow, FMA, long double, allocation, or convergence-dependent loop.
 * See IMAGE_TRANSFER_BOUNDARY_V0_1.md for proof obligations and qualification.
 * Constants are hi/lo splits generated with Decimal precision180. Exact input
 * cutoffs are certified by tools/image_transfer_cutoffs.py, not fitted samples.
 */
typedef struct gffx_im_dd { double hi,lo; } gffx_im_dd;
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_make(double hi,double lo){
    gffx_im_dd x;x.hi=hi;x.lo=lo;return x;
}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_sum(double a,double b){
    double s=a+b,v=s-a,e=(a-(s-v))+(b-v);
    return gffx_im_dd_make(s,e);
}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_add(gffx_im_dd a,gffx_im_dd b){
    gffx_im_dd s=gffx_im_dd_sum(a.hi,b.hi),t=gffx_im_dd_sum(a.lo,b.lo);
    double e=s.lo+t.hi;
    s=gffx_im_dd_sum(s.hi,e);e=s.lo+t.lo;
    return gffx_im_dd_sum(s.hi,e);
}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_neg(gffx_im_dd a){return gffx_im_dd_make(-a.hi,-a.lo);}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_product(double a,double b){
    double p=a*b,ca=134217729.*a,cb=134217729.*b;
    double ah=ca-(ca-a),al=a-ah,bh=cb-(cb-b),bl=b-bh;
    double e=ah*bh-p;e=e+ah*bl;e=e+al*bh;e=e+al*bl;
    return gffx_im_dd_make(p,e);
}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_mul(gffx_im_dd a,gffx_im_dd b){
    gffx_im_dd p=gffx_im_dd_product(a.hi,b.hi);
    double e=p.lo+a.hi*b.lo;e=e+a.lo*b.hi;e=e+a.lo*b.lo;
    return gffx_im_dd_sum(p.hi,e);
}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_div(gffx_im_dd a,gffx_im_dd b){
    double q1=gffx_im_div_f64(a.hi,b.hi),q2,q3;
    gffx_im_dd q=gffx_im_dd_make(q1,0.),r;
    r=gffx_im_dd_add(a,gffx_im_dd_neg(gffx_im_dd_mul(b,q)));
    q2=gffx_im_div_f64(r.hi,b.hi);q=gffx_im_dd_add(q,gffx_im_dd_make(q2,0.));
    r=gffx_im_dd_add(a,gffx_im_dd_neg(gffx_im_dd_mul(b,q)));
    q3=gffx_im_div_f64(r.hi,b.hi);return gffx_im_dd_add(q,gffx_im_dd_make(q3,0.));
}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_factor5(int r){
    switch(r){
    case 0:return gffx_im_dd_make(1.,0.);
    case 1:return gffx_im_dd_make(0x1.2611186bae675p+0,-0x1.a56b786f7e27cp-54);
    case 2:return gffx_im_dd_make(0x1.51cb453b9536cp+0,0x1.15a4ea11db2e6p-54);
    case 3:return gffx_im_dd_make(0x1.8406003b2ae5cp+0,0x1.ae64d1cc83ab8p-54);
    default:return gffx_im_dd_make(0x1.bdb8cdadbe120p+0,0x1.0a7b0e98593fdp-55);
    }
}
GFFX_IMAGE_SCALAR gffx_im_dd gffx_im_dd_fifth_root(gffx_im_dd m){
    gffx_im_dd root=gffx_im_dd_make(2.,0.),four=gffx_im_dd_make(4.,0.),five=gffx_im_dd_make(5.,0.);
    int i,j;
    for(i=0;i<24;++i){
        gffx_im_dd denominator=gffx_im_dd_make(1.,0.),numerator;
        for(j=0;j<4;++j)denominator=gffx_im_dd_mul(denominator,root);
        numerator=gffx_im_dd_add(gffx_im_dd_mul(four,root),gffx_im_dd_div(m,denominator));
        root=gffx_im_dd_div(numerator,five);
    }
    return root;
}
GFFX_IMAGE_SCALAR double gffx_im_dd_round_scaled(gffx_im_dd a,int exponent){
    int i;
    /* In this caller domain mantissa <32; four exact binary shifts suffice. */
    for(i=0;i<4;++i){if(a.hi>=2.){a.hi=a.hi*.5;a.lo=a.lo*.5;++exponent;}}
    if(exponent==1024&&a.hi==1.){
        /* Explicit RN-even top-bin midpoint; no blanket saturation. */
        return a.lo < -0x1p-54 ? 0x1.fffffffffffffp+1023 : INFINITY;
    }
    if(exponent>1023)return INFINITY;
    return ldexp(a.hi,exponent)+ldexp(a.lo,exponent);
}
GFFX_IMAGE_SCALAR double gffx_im_boundary_round_finite(gffx_im_dd a,int exponent){
    gffx_im_dd upper,midpoint,difference;double value;
    /* Conditional arithmetic consistency, not runtime accuracy certification.
     * Inflate the <2^-84 analytic budget to 2^-83 to cover construction error.
     * Actual admitted inputs are >2^-80 from the exact cutoff midpoint. */
    if(!isfinite(a.hi)||!isfinite(a.lo)||a.hi<=0.||exponent<1017||exponent>1024||
       fabs(a.lo)>0x1p-52*a.hi)return NAN;
    upper=gffx_im_dd_mul(a,gffx_im_dd_make(1.,0x1p-83));
    midpoint=gffx_im_dd_make(ldexp(1.,1024-exponent),-ldexp(1.,970-exponent));
    difference=gffx_im_dd_add(upper,gffx_im_dd_neg(midpoint));
    if(difference.hi>0.||(difference.hi==0.&&difference.lo>=0.))return NAN;
    value=gffx_im_dd_round_scaled(a,exponent);
    return isfinite(value)?value:NAN;
}
GFFX_IMAGE_SCALAR double gffx_im_boundary_decode_f64(double x){
    uint64_t bits;int exponent,k,r,j;gffx_im_dd m,base,root,result,offset,denominator;
    if(x>0x1.acb9abb20cf2dp+426)return INFINITY;
    bits=gffx_im_bits_f64(x);exponent=(int)(bits>>52)-1023;
    m=gffx_im_dd_make(gffx_im_from_f64((bits&UINT64_C(0x000fffffffffffff))|UINT64_C(0x3ff0000000000000)),0.);
    offset=gffx_im_dd_make(ldexp(0x1.c28f5c28f5c29p-5,-exponent),ldexp(-0x1.47ae147ae147bp-62,-exponent));
    denominator=gffx_im_dd_make(0x1.0e147ae147ae1p+0,0x1.1eb851eb851ecp-54);
    base=gffx_im_dd_div(gffx_im_dd_add(m,offset),denominator);
    if(base.hi<1.||(base.hi==1.&&base.lo<0.)){base.hi=base.hi*2.;base.lo=base.lo*2.;--exponent;}
    root=gffx_im_dd_fifth_root(base);result=gffx_im_dd_make(1.,0.);
    for(j=0;j<12;++j)result=gffx_im_dd_mul(result,root);
    k=(12*exponent)/5;r=(12*exponent)%5;
    result=gffx_im_dd_mul(result,gffx_im_dd_factor5(r));
    return gffx_im_boundary_round_finite(result,k);
}
GFFX_IMAGE_SCALAR double gffx_im_boundary_negative_encode_f64(double x){
    uint64_t bits;int exponent;gffx_im_dd m,factor;
    if(-x>0x1.3d0722149b57fp+1020)return -INFINITY;
    bits=gffx_im_bits_f64(-x);exponent=(int)(bits>>52)-1023;
    m=gffx_im_dd_make(gffx_im_from_f64((bits&UINT64_C(0x000fffffffffffff))|UINT64_C(0x3ff0000000000000)),0.);
    factor=gffx_im_dd_make(0x1.9d70a3d70a3d7p+3,0x1.47ae147ae147bp-54);
    return -gffx_im_boundary_round_finite(gffx_im_dd_mul(m,factor),exponent);
}
#endif
