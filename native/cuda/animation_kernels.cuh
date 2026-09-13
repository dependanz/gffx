/* Fixed-order reference reductions. Each output element has one owning thread. */
#define GFFX_ANIMATION_SCALAR static __device__
#include "../core/animation_scalar.h"
#undef GFFX_ANIMATION_SCALAR
#define GFFX_ANIMATION_CUDA(T,S) \
extern "C" __global__ void gffx_animation_finite_##S(const T *x,long long count,int *status) { \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; \
    if(i<count&&!isfinite(x[i]))atomicExch(status,1); \
} \
extern "C" __global__ void gffx_animation_blend_##S(const T *t,const T *d,const T *w,const T *g,T *out, \
    long long B,long long V,long long K,int mode,long long count) { \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,b,v,k,c;T sum=0; \
    if(i>=count)return; \
    if(mode==0){b=i/(V*3);v=(i/3)%V;c=i%3;sum=t[v*3+c]; \
        for(k=0;k<K;++k)sum=sum+w[b*K+k]*d[(k*V+v)*3+c];} \
    else if(mode==1){v=i/3;c=i%3;for(b=0;b<B;++b)sum=sum+g[(b*V+v)*3+c];} \
    else if(mode==2){k=i/(V*3);v=(i/3)%V;c=i%3;for(b=0;b<B;++b)sum=sum+w[b*K+k]*g[(b*V+v)*3+c];} \
    else {b=i/K;k=i%K;for(v=0;v<V;++v)for(c=0;c<3;++c)sum=sum+d[(k*V+v)*3+c]*g[(b*V+v)*3+c];} \
    out[i]=sum; \
}
GFFX_ANIMATION_CUDA(float,f32)
GFFX_ANIMATION_CUDA(double,f64)
#undef GFFX_ANIMATION_CUDA

#define GFFX_ANIMATION_SKIN(T,S) \
extern "C" __global__ void gffx_animation_skin_##S(const T *v,const T *t,const T *w,const T *g,T *out, \
    long long B,long long V,long long J,int mode,long long count) { \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,b,n,j;int c,a;T sum=0; \
    if(i>=count)return; \
    if(mode==0){b=i/(V*3);n=(i/3)%V;c=(int)(i%3); \
        for(j=0;j<J;++j){const T *m=t+(b*J+j)*16;T p=m[c*4]*v[(b*V+n)*3]; \
            p=p+m[c*4+1]*v[(b*V+n)*3+1];p=p+m[c*4+2]*v[(b*V+n)*3+2];p=p+m[c*4+3];sum=sum+w[n*J+j]*p;}} \
    else if(mode==1){b=i/(V*3);n=(i/3)%V;a=(int)(i%3); \
        for(j=0;j<J;++j)for(c=0;c<3;++c)sum=sum+w[n*J+j]*t[(b*J+j)*16+c*4+a]*g[(b*V+n)*3+c];} \
    else if(mode==2){b=i/(J*16);j=(i/16)%J;c=(int)((i/4)%4);a=(int)(i%4); \
        if(c<3)for(n=0;n<V;++n)sum=sum+w[n*J+j]*(a<3?v[(b*V+n)*3+a]:(T)1)*g[(b*V+n)*3+c];} \
    else {n=i/J;j=i%J;for(b=0;b<B;++b)for(c=0;c<3;++c){const T *m=t+(b*J+j)*16;T p=m[c*4]*v[(b*V+n)*3]; \
        p=p+m[c*4+1]*v[(b*V+n)*3+1];p=p+m[c*4+2]*v[(b*V+n)*3+2];p=p+m[c*4+3];sum=sum+g[(b*V+n)*3+c]*p;}} \
    out[i]=sum; \
}
GFFX_ANIMATION_SKIN(float,f32)
GFFX_ANIMATION_SKIN(double,f64)
#undef GFFX_ANIMATION_SKIN

extern "C" __global__ void gffx_animation_validate_index(const int *x,long long count,long long limit,int *status) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count&&(x[i]<0||x[i]>=limit))atomicExch(status,1);
}
#define GFFX_ANIMATION_BARY(T,S) \
extern "C" __global__ void gffx_animation_bary_##S(const T *v,const int *f,const int *idx,const T *w,const T *g,T *out, \
    long long B,long long V,long long N,int mode,long long count) { \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,b,n,p;int c,k;T sum=0; \
    if(i>=count)return; \
    if(mode==0){b=i/(N*3);n=(i/3)%N;c=(int)(i%3); \
        for(k=0;k<3;++k)sum=sum+w[n*3+k]*v[(b*V+f[(long long)idx[n]*3+k])*3+c];} \
    else if(mode==1){b=i/(V*3);p=(i/3)%V;c=(int)(i%3); \
        for(n=0;n<N;++n)for(k=0;k<3;++k)if(f[(long long)idx[n]*3+k]==p)sum=sum+w[n*3+k]*g[(b*N+n)*3+c];} \
    else {n=i/3;k=(int)(i%3);for(b=0;b<B;++b)for(c=0;c<3;++c)sum=sum+v[(b*V+f[(long long)idx[n]*3+k])*3+c]*g[(b*N+n)*3+c];} \
    out[i]=sum; \
}
GFFX_ANIMATION_BARY(float,f32)
GFFX_ANIMATION_BARY(double,f64)
#undef GFFX_ANIMATION_BARY

extern "C" __global__ void gffx_animation_order(const int *parents,long long J,int *order,int *done,int *status) {
    if(blockIdx.x||threadIdx.x)return;
    for(long long j=0;j<J;++j){done[j]=0;if(parents[j]<-1||parents[j]>=J||parents[j]==j){*status=1;return;}}
    for(long long step=0;step<J;++step){int found=0;
        for(long long j=0;j<J;++j)if(!done[j]&&(parents[j]==-1||done[parents[j]])){
            order[step]=(int)j;done[j]=1;found=1;break;}
        if(!found){*status=1;return;}}
}
#define GFFX_ANIMATION_POSE(T,S) \
extern "C" __global__ void gffx_animation_rodrigues_##S(const T *r,const T *g,T *out,long long count,int *status) { \
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return; \
    if(gffx_an_rodrigues_##S(r+i*3,g?g+i*9:NULL,out+i*(g?3:9),1))atomicExch(status,2); \
} \
extern "C" __global__ void gffx_animation_fk_##S(const T *local,const int *parents,const int *order,const T *g,T *out, \
    T *world,T *adj,long long B,long long J,int *status) { \
    long long b=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;long long offset=b*J*16; \
    if(gffx_an_fk_##S(local+offset,parents,order,g?g+offset:NULL,out+offset,world+offset,adj+offset,1,J))atomicExch(status,2); \
}
GFFX_ANIMATION_POSE(float,f32)
GFFX_ANIMATION_POSE(double,f64)
#undef GFFX_ANIMATION_POSE
