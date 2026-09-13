#ifndef GFFX_ANIMATION_SCALAR_H
#define GFFX_ANIMATION_SCALAR_H
/* Shared reference arithmetic, compiled as C11 host or CUDA device routines.
 * Keep operation/loop order identical; status formatting belongs to the caller. */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifndef GFFX_ANIMATION_SCALAR
#define GFFX_ANIMATION_SCALAR static
#endif

#define AN_RODRIGUES(T, suffix, SQRT, SIN, COS) \
GFFX_ANIMATION_SCALAR int gffx_an_rodrigues_##suffix(const T *r, const T *g, T *y, int64_t count) { \
    int64_t n; int i,j,l; \
    for(n=0;n<count;++n) { \
        const T *p=r+n*3; T q=p[0]*p[0]+p[1]*p[1]+p[2]*p[2],a,b,da,db; \
        T k[9]={(T)0,-p[2],p[1],p[2],(T)0,-p[0],-p[1],p[0],(T)0}; \
        if(!isfinite(q)) return 1; \
        if(q <= (T)0.00390625) { \
            a=(T)1+q*((T)(-1.0/6)+q*((T)(1.0/120)+q*((T)(-1.0/5040)+q*(T)(1.0/362880)))); \
            b=(T)0.5+q*((T)(-1.0/24)+q*((T)(1.0/720)+q*((T)(-1.0/40320)+q*(T)(1.0/3628800)))); \
            da=(T)(-1.0/6)+q*((T)(2.0/120)+q*((T)(-3.0/5040)+q*(T)(4.0/362880))); \
            db=(T)(-1.0/24)+q*((T)(2.0/720)+q*((T)(-3.0/40320)+q*(T)(4.0/3628800))); \
        } else { \
            T theta=SQRT(q),half_sine=SIN(theta*(T)0.5); \
            a=SIN(theta)/theta; b=(T)2*half_sine*half_sine/q; \
            da=(COS(theta)-a)/((T)2*q); db=(a-(T)2*b)/((T)2*q); \
        } \
        if(g==NULL) { \
            for(i=0;i<3;++i) for(j=0;j<3;++j) { T eye=(T)(i==j),s=p[i]*p[j]-q*eye; y[n*9+i*3+j]=eye+a*k[i*3+j]+b*s; } \
        } else { \
            for(l=0;l<3;++l) { T sum=0; \
                for(i=0;i<3;++i) for(j=0;j<3;++j) { \
                    T eye=(T)(i==j),s=p[i]*p[j]-q*eye,dk=0; \
                    if((i==1&&j==2&&l==0)||(i==2&&j==0&&l==1)||(i==0&&j==1&&l==2)) dk=(T)-1; \
                    if((i==2&&j==1&&l==0)||(i==0&&j==2&&l==1)||(i==1&&j==0&&l==2)) dk=(T)1; \
                    { T ds=(T)(i==l)*p[j]+p[i]*(T)(j==l)-(T)2*p[l]*eye; \
                      T dr=(T)2*p[l]*da*k[i*3+j]+a*dk+(T)2*p[l]*db*s+b*ds; sum=sum+g[n*9+i*3+j]*dr; } \
                } y[n*3+l]=sum; \
            } \
        } \
    } return 0; \
}
AN_RODRIGUES(float,f32,sqrtf,sinf,cosf)
AN_RODRIGUES(double,f64,sqrt,sin,cos)
#undef AN_RODRIGUES

#define AN_FK(T,suffix) \
GFFX_ANIMATION_SCALAR int gffx_an_fk_##suffix(const T *local,const int32_t *parents,const int32_t *order,const T *g, \
    T *out,T *world,T *adj,int64_t B,int64_t J) { \
    int64_t b,step,j,p,child;int r,c,k; \
    for(b=0;b<B;++b)for(step=0;step<J;++step){j=order[step];p=parents[j]; \
        for(r=0;r<4;++r)for(c=0;c<4;++c){T sum; \
            if(p<0)sum=local[(b*J+j)*16+r*4+c];else{sum=0;for(k=0;k<4;++k)sum=sum+world[(b*J+p)*16+r*4+k]*local[(b*J+j)*16+k*4+c];} \
            if(!isfinite(sum))return 1;world[(b*J+j)*16+r*4+c]=sum;}} \
    if(g==NULL){for(b=0;b<B*J*16;++b)out[b]=world[b];return 0;} \
    for(b=0;b<B;++b)for(step=J;step-->0;){j=order[step];p=parents[j]; \
        for(r=0;r<4;++r)for(c=0;c<4;++c){T sum=g[(b*J+j)*16+r*4+c]; \
            for(child=0;child<J;++child)if(parents[child]==j){T part=0;for(k=0;k<4;++k)part=part+adj[(b*J+child)*16+r*4+k]*local[(b*J+child)*16+c*4+k];sum=sum+part;} \
            adj[(b*J+j)*16+r*4+c]=sum;} \
        for(r=0;r<4;++r)for(c=0;c<4;++c){T sum; \
            if(p<0)sum=adj[(b*J+j)*16+r*4+c];else{sum=0;for(k=0;k<4;++k)sum=sum+world[(b*J+p)*16+k*4+r]*adj[(b*J+j)*16+k*4+c];} \
            out[(b*J+j)*16+r*4+c]=sum;} \
    }return 0; \
}
AN_FK(float,f32)
AN_FK(double,f64)
#undef AN_FK
#endif
