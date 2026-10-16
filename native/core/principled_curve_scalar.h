#ifndef GFFX_PRINCIPLED_CURVE_SCALAR_H
#define GFFX_PRINCIPLED_CURVE_SCALAR_H
/* Private bounded-loss arithmetic, shared C11/CUDA. Requires validated finite
 * domain inputs, RN, no contraction and no FTZ. No allocation, mutable state or
 * exported ABI. See docs/PRINCIPLED_NATIVE_DESIGN.md and independent tests. */
#include <stdint.h>
#include <string.h>
#ifndef GFFX_PC_SCALAR
#define GFFX_PC_SCALAR static
#endif
GFFX_PC_SCALAR uint64_t gffx_pc_bits(double x){
#if defined(__CUDA_ARCH__)
    return (uint64_t)__double_as_longlong(x);
#else
    uint64_t bits;memcpy(&bits,&x,sizeof bits);return bits;
#endif
}
GFFX_PC_SCALAR double gffx_pc_from_bits(uint64_t bits){
#if defined(__CUDA_ARCH__)
    return __longlong_as_double((long long)bits);
#else
    double x;memcpy(&x,&bits,sizeof x);return x;
#endif
}
GFFX_PC_SCALAR double gffx_pc_div(double a,double b){
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a,b);
#else
    return a/b;
#endif
}
GFFX_PC_SCALAR double gffx_pc_log(double x){
    int small=x<0x1p-1022,exponent,k;uint64_t bits;
    double scaled=small?x*0x1p52:x,mantissa,z,square,p,local;
    bits=gffx_pc_bits(scaled);exponent=(int)((bits>>52)&2047)-1023-small*52;
    mantissa=gffx_pc_from_bits((bits&UINT64_C(0x000fffffffffffff))|UINT64_C(0x3ff0000000000000));
    if(mantissa>0x1.6a09e667f3bcdp+0){mantissa=mantissa*.5;++exponent;}
    z=gffx_pc_div(mantissa-1.,mantissa+1.);square=z*z;p=1./39.;
    for(k=37;k>0;k-=2)p=p*square+gffx_pc_div(1.,(double)k);
    local=(2.*z)*p;
    return ((double)exponent*0x1.62e4200000000p-1+local)+(double)exponent*0x1.fdf473de6af28p-22;
}
GFFX_PC_SCALAR double gffx_pc_exp(double x){
    const double coefficients[19]={0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p-1,0x1.5555555555555p-3,0x1.5555555555555p-5,0x1.1111111111111p-7,0x1.6c16c16c16c17p-10,0x1.a01a01a01a01ap-13,0x1.a01a01a01a01ap-16,0x1.71de3a556c734p-19,0x1.27e4fb7789f5cp-22,0x1.ae64567f544e4p-26,0x1.1eed8eff8d898p-29,0x1.6124613a86d09p-33,0x1.93974a8c07c9dp-37,0x1.ae7f3e733b81fp-41,0x1.ae7f3e733b81fp-45,0x1.952c77030ad4ap-49,0x1.6827863b97d97p-53};
    double selected=x*0x1.71547652b82fep+0,remainder,p;int n=(int)selected,high,k;
    /* x <= 0: truncation followed by the negative half/tie correction exactly
     * implements torch.round, including ties to even. */
    if(selected-(double)n<-.5||(selected-(double)n==-.5&&(n%2!=0)))--n;
    remainder=(x-(double)n*0x1.62e4200000000p-1)-(double)n*0x1.fdf473de6af28p-22;
    p=coefficients[18];for(k=17;k>=0;--k)p=p*remainder+coefficients[k];
    high=n < -1022 ? -1022:n;
    return (p*gffx_pc_from_bits((uint64_t)(n-high+1023)<<52))*gffx_pc_from_bits((uint64_t)(high+1023)<<52);
}
GFFX_PC_SCALAR double gffx_pc_curve(const double *b,double u){
    double v=1.-u,p=v*b[0]+u*b[1],q=v*b[1]+u*b[2],s=v*b[2]+u*b[3];
    return gffx_pc_exp(v*(v*p+u*q)+u*(v*q+u*s));
}
GFFX_PC_SCALAR double gffx_pc_curve_vjp(const double *b,double u,double g,double value){
    double v=1.-u,p=v*(b[1]-b[0])+u*(b[2]-b[1]);
    double q=v*(b[2]-b[1])+u*(b[3]-b[2]);
    /* Keep the two custom backward nodes' order, including subnormal rounding. */
    return (g*value)*(3.*(v*p+u*q));
}
/* Directional chain rule for this fixed angular-limiter formula only. */
typedef struct gffx_pc_dual { double v,d; } gffx_pc_dual;
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_d(double v,double d){gffx_pc_dual a;a.v=v;a.d=d;return a;}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_add(gffx_pc_dual a,gffx_pc_dual b){return gffx_pc_d(a.v+b.v,a.d+b.d);}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_sub(gffx_pc_dual a,gffx_pc_dual b){return gffx_pc_d(a.v-b.v,a.d-b.d);}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_mul(gffx_pc_dual a,gffx_pc_dual b){return gffx_pc_d(a.v*b.v,a.d*b.v+a.v*b.d);}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_quot(gffx_pc_dual a,gffx_pc_dual b){double v=gffx_pc_div(a.v,b.v);return gffx_pc_d(v,gffx_pc_div(a.d-v*b.d,b.v));}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_min(gffx_pc_dual a,gffx_pc_dual b){return a.v<b.v?a:(b.v<a.v?b:gffx_pc_d(a.v,(a.d+b.d)*.5));}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_max(gffx_pc_dual a,gffx_pc_dual b){return a.v>b.v?a:(b.v>a.v?b:gffx_pc_d(a.v,(a.d+b.d)*.5));}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_abs(gffx_pc_dual a){return gffx_pc_d(a.v<0.?-a.v:a.v,a.v<0.?-a.d:(a.v>0.?a.d:0.));}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_secant(int i,int c,const double *x,const double *y,const double *dy){
    return gffx_pc_quot(gffx_pc_sub(gffx_pc_d(y[2*i+2+c],dy[2*i+2+c]),gffx_pc_d(y[2*i+c],dy[2*i+c])),gffx_pc_d(x[i+1]-x[i],0.));
}
GFFX_PC_SCALAR gffx_pc_dual gffx_pc_slope(int n,int i,int c,const double *x,const double *y,const double *dy){
    int prev=i?i-1:0,next=i<n-1?i:n-2;double sign;
    gffx_pc_dual before=gffx_pc_secant(prev,c,x,y,dy),after=gffx_pc_secant(next,c,x,y,dy),raw,bound;
    if(n==2)return before;
    if(i==0){double h0=x[1]-x[0],h1=x[2]-x[1];
        raw=gffx_pc_quot(gffx_pc_sub(gffx_pc_mul(gffx_pc_d(2.*h0+h1,0.),before),gffx_pc_mul(gffx_pc_d(h0,0.),gffx_pc_secant(1,c,x,y,dy))),gffx_pc_d(h0+h1,0.));}
    else if(i==n-1){double h0=x[n-1]-x[n-2],h1=x[n-2]-x[n-3];
        raw=gffx_pc_quot(gffx_pc_sub(gffx_pc_mul(gffx_pc_d(2.*h0+h1,0.),after),gffx_pc_mul(gffx_pc_d(h0,0.),gffx_pc_secant(n-3,c,x,y,dy))),gffx_pc_d(h0+h1,0.));}
    else{double hp=x[i]-x[i-1],hn=x[i+1]-x[i];
        raw=gffx_pc_quot(gffx_pc_add(gffx_pc_mul(gffx_pc_d(hn,0.),before),gffx_pc_mul(gffx_pc_d(hp,0.),after)),gffx_pc_d(hp+hn,0.));}
    sign=(before.v>0.&&after.v>0.)?1.:((before.v<0.&&after.v<0.)?-1.:0.);
    bound=gffx_pc_mul(gffx_pc_d(1.5,0.),gffx_pc_min(gffx_pc_abs(before),gffx_pc_abs(after)));
    return gffx_pc_mul(gffx_pc_d(sign,0.),gffx_pc_min(gffx_pc_max(gffx_pc_mul(gffx_pc_d(sign,0.),raw),gffx_pc_d(0.,0.)),bound));
}
GFFX_PC_SCALAR void gffx_pc_pair_slopes(int n,int i,const double *x,const double *y,const double *dy,double *out,double *dout){
    gffx_pc_dual m0,m1,gap,delta,risk,scale;
    if(n==1){out[0]=y[0]*0.;out[1]=y[1]*0.;dout[0]=dy[0]*0.;dout[1]=dy[1]*0.;return;}
    m0=gffx_pc_slope(n,i,0,x,y,dy);m1=gffx_pc_slope(n,i,1,x,y,dy);
    gap=gffx_pc_sub(gffx_pc_d(y[2*i],dy[2*i]),gffx_pc_d(y[2*i+1],dy[2*i+1]));delta=gffx_pc_sub(m0,m1);
    risk=delta.v>0.?gffx_pc_mul(delta,gffx_pc_d(i?x[i]-x[i-1]:0.,0.)):gffx_pc_mul(gffx_pc_d(-delta.v,-delta.d),gffx_pc_d(i<n-1?x[i+1]-x[i]:0.,0.));
    scale=risk.v>0.?gffx_pc_min(gffx_pc_d(1.,0.),gffx_pc_quot(gffx_pc_mul(gffx_pc_d(3.,0.),gap),risk)):gffx_pc_d(1.,0.);
    if(scale.v<1.)scale=gffx_pc_mul(scale,gffx_pc_d(1.-8.*0x1p-52,0.));
    m0=gffx_pc_mul(m0,scale);m1=gffx_pc_mul(m1,scale);
    out[0]=m0.v;out[1]=m1.v;dout[0]=m0.d;dout[1]=m1.d;
}
GFFX_PC_SCALAR void gffx_pc_curve_build(int n,const double *x,const double *b,double u,double *out,double *dout,double *scratch){
    int j,i,c;double *m=scratch,*dm=scratch+2*n;
    for(j=0;j<2*n+1;++j){out[j]=gffx_pc_curve(b+4*j,u);dout[j]=gffx_pc_curve_vjp(b+4*j,u,1.,out[j]);}
    out[2*n+1]=out[2*n]*0.;dout[2*n+1]=dout[2*n]*0.;
    for(i=0;i<n;++i)gffx_pc_pair_slopes(n,i,x,out,dout,m+2*i,dm+2*i);
    for(i=0;i<n-1;++i){double h=x[i+1]-x[i];
        for(c=0;c<2;++c){int a=2*i+c,z=2*n+2+8*i+c;
            out[z]=out[a];out[z+2]=out[a]+gffx_pc_div(h*m[a],3.);
            out[z+4]=out[a+2]-gffx_pc_div(h*m[a+2],3.);out[z+6]=out[a+2];
            dout[z]=dout[a];dout[z+2]=dout[a]+gffx_pc_div(h*dm[a],3.);
            dout[z+4]=dout[a+2]-gffx_pc_div(h*dm[a+2],3.);dout[z+6]=dout[a+2];
        }
    }
}
GFFX_PC_SCALAR double gffx_pc_bezier(const double *b,double u){
    double v=1.-u,p=v*b[0]+u*b[1],q=v*b[1]+u*b[2],a=v*b[2]+u*b[3];
    return v*(v*p+u*q)+u*(v*q+u*a);
}
GFFX_PC_SCALAR double gffx_pc_bezier_derivative(const double *b,double u){
    double v=1.-u,p=v*(b[1]-b[0])+u*(b[2]-b[1]),q=v*(b[2]-b[1])+u*(b[3]-b[2]);
    return 3.*(v*p+u*q);
}
GFFX_PC_SCALAR double gffx_pc_sum(double *x,int n){
    int i,k;if(!n)return 0.;
    while(n>1){k=0;for(i=0;i+1<n;i+=2)x[k++]=x[i]+x[i+1];if(n%2)x[k++]=x[n-1];n=k;}
    return x[0];
}
GFFX_PC_SCALAR void gffx_pc_curve_query(int n,const double *cn,const double *logs,const double *weights,const double *curve,const double *dcurve,double c,double *out,double *scratch){
    int channel,i,k,segment=0;double u=0.,width=1.;
    if(n>1&&c>=cn[0]){while(segment<n-2&&c>=cn[segment+1])++segment;
        width=logs[segment+1]-logs[segment];u=gffx_pc_div(gffx_pc_log(c)-logs[segment],width);}
    for(channel=0;channel<2;++channel){double y0=curve[channel],h=curve[2*n+channel],dy0=dcurve[channel],dh=dcurve[2*n+channel];
        double average=gffx_pc_div((cn[0]*cn[0])*(h+2.*y0),3.);
        double daverage=gffx_pc_div((cn[0]*cn[0])*(dh+2.*dy0),3.);
        if(n==1||c<cn[0]){out[channel]=h+gffx_pc_div(c,cn[0])*(y0-h);
            out[4+channel]=dh+gffx_pc_div(c,cn[0])*(dy0-dh);out[8+channel]=gffx_pc_div(y0-h,cn[0]);}
        else{double b[4],db[4];for(k=0;k<4;++k){int a=2*n+2+8*segment+2*k+channel;b[k]=curve[a];db[k]=dcurve[a];}
            out[channel]=gffx_pc_bezier(b,u);out[4+channel]=gffx_pc_bezier(db,u);
            out[8+channel]=gffx_pc_div(gffx_pc_div(gffx_pc_bezier_derivative(b,u),width),c);}
        for(i=0;i<n-1;++i){double coefficient=(2.*(cn[i+1]*cn[i+1]))*(logs[i+1]-logs[i]);
            for(k=0;k<4;++k){int a=2*n+2+8*i+2*k+channel,j=4*i+k;
                scratch[j]=(coefficient*curve[a])*weights[j];scratch[4*n+j]=(coefficient*dcurve[a])*weights[j];}}
        out[2+channel]=average+gffx_pc_sum(scratch,4*(n-1));
        out[6+channel]=daverage+gffx_pc_sum(scratch+4*n,4*(n-1));
    }
}

/* Query only the four selected angular controls; mean is evaluated once/owner. */
GFFX_PC_SCALAR void gffx_pq_direction(int n,const double *cn,const double *logs,const double *v,const double *d,double c,double *out){
 int ch,k,segment=0;double width=1.,u=0.;
 if(n>1&&c>=cn[0]){while(segment<n-2&&c>=cn[segment+1])++segment;width=logs[segment+1]-logs[segment];u=gffx_pc_div(gffx_pc_log(c)-logs[segment],width);}
 for(ch=0;ch<2;++ch){
  if(n==1||c<cn[0]){double h=v[2*n+ch],dh=d[2*n+ch];out[ch]=h+gffx_pc_div(c,cn[0])*(v[ch]-h);out[2+ch]=dh+gffx_pc_div(c,cn[0])*(d[ch]-dh);out[4+ch]=gffx_pc_div(v[ch]-h,cn[0]);}
  else{double b[4],db[4];for(k=0;k<4;++k){int a=2*n+2+8*segment+2*k+ch;b[k]=v[a];db[k]=d[a];}out[ch]=gffx_pc_bezier(b,u);out[2+ch]=gffx_pc_bezier(db,u);out[4+ch]=gffx_pc_div(gffx_pc_div(gffx_pc_bezier_derivative(b,u),width),c);}
 }
}
GFFX_PC_SCALAR void gffx_pq_mean(int n,const double *cn,const double *logs,const double *weights,const double *v,const double *d,double *out,double *scratch){
 int ch,deriv,i,k;for(deriv=0;deriv<2;++deriv){const double *curve=deriv?d:v;
  for(ch=0;ch<2;++ch){double average=gffx_pc_div((cn[0]*cn[0])*(curve[2*n+ch]+2.*curve[ch]),3.);
   for(i=0;i<n-1;++i){double coefficient=(2.*(cn[i+1]*cn[i+1]))*(logs[i+1]-logs[i]);for(k=0;k<4;++k){int a=2*n+2+8*i+2*k+ch,j=4*i+k;scratch[j]=(coefficient*curve[a])*weights[j];}}
   out[2*deriv+ch]=average+gffx_pc_sum(scratch,4*(n-1));
  }
 }
}
GFFX_PC_SCALAR double gffx_pq_contract(int row,int count,const double *query,const double *mean,const int32_t *owners,const double *g,const double *h,double *scratch){
 int j,k=0;for(j=0;j<count;++j)if(owners[j]==row)scratch[k++]=query[6*j+2]*g[2*j]+query[6*j+3]*g[2*j+1];
 return gffx_pc_sum(scratch,k)+(mean[4*row+2]*h[2*row]+mean[4*row+3]*h[2*row+1]);
}

#endif
