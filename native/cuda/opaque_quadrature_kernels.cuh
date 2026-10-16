#include <stdint.h>
namespace gffx_opaque_device {
struct gffx_tensor_view { void *data; int dtype; int64_t shape[3]; unsigned rank; };
enum {GFFX_DTYPE_FLOAT64=1};
static __device__ void *gffx_mesh_elements(const gffx_tensor_view *v){return v->data;}
static __device__ const void *gffx_mesh_elements_const(const gffx_tensor_view *v){return v->data;}
static __device__ uint64_t gffx_mesh_element_count(const gffx_tensor_view *v){uint64_t n=1;for(unsigned i=0;i<v->rank;++i)n*=v->shape[i];return n;}
#define GFFX_OPAQUE_LOCAL static __device__
/* Avoid duplicating interval predicates into every forward/backward kernel. */
#define GFFX_OPAQUE_SEPARATION_LOCAL static __device__ __noinline__
#define GFFX_OPAQUE_CLIP_LOCAL static __device__ __noinline__
#define GFFX_OPAQUE_PRECLEARED 1
#include "../core/opaque_quadrature_scalar.h"
#include "../core/opaque_boundary_scalar.h"
#include "../core/opaque_batch_scalar.h"
#include "../core/opaque_partition_scalar.h"
#include "../core/opaque_pair_dispatch.h"
static_assert(sizeof(Scratch)+64<=1048576,"opaque scratch slab too small");
static_assert(sizeof(pair_Scratch)<=sizeof(Scratch),"paired scratch exceeds scalar allocation");
}
extern "C" __global__ void gffx_opaque_quadrature_kernel(void *vertices,int *faces,void *cuts,void *gradient,void *out,void *other,long long nv,long long nf,int nc,int rows,long long h,long long w,double ox,double oy,int real64,int backward,int domain,int edge,void *scratch){
    if(blockIdx.x||threadIdx.x)return;
    using namespace gffx_opaque_device;
    int *status=(int*)scratch;
    gffx_tensor_view v={vertices,real64,{nv,3,0},2},f={faces,0,{nf,3,0},2},c={cuts,real64,{nf,rows,nc},3},g={gradient,real64,{h*w,800,4},3};
    gffx_tensor_view a={out,real64,{backward?nv:h*w,backward?3:800,0},2},b={other,real64,{backward?nf:h*w,backward?rows:800,backward?nc:4},3};
    if(domain){
        for(long long i=0;i<nf*3;++i)if(faces[i]<0||faces[i]>=nv){*status=1;return;}
        const gffx_tensor_view *inputs[3]={&v,&c,&g};
        for(int j=0;j<(backward?3:2);++j)for(uint64_t i=0;i<gffx_mesh_element_count(inputs[j]);++i){double x=read_real(inputs[j],i);if(!isfinite(x)||(j==0&&fabs(x)>1048576)){*status=1;return;}}
        return;
    }
    *status=edge>=0?opaque_boundary_execute(&v,&f,&c,h,w,ox,oy,edge,&a,&b,(Scratch*)((char*)scratch+8)):opaque_quadrature_execute(&v,&f,&c,h,w,ox,oy,backward?&g:0,&a,&b,(Scratch*)((char*)scratch+8),edge==-2);
}
extern "C" __global__ void gffx_opaque_batch_kernel(void *vertices,int *faces,void *cuts,void *gradient,void *out,void *other,long long nv,long long nf,int nc,int rows,double ox,double oy,int begin,int batch,int slots,int blocks,int real64,int backward,int domain,void *scratch,int rule,int derivatives){
 if(blockIdx.x||threadIdx.x)return;using namespace gffx_opaque_device;int *status=(int*)scratch;int32_t cursor[4]={begin,batch,slots,blocks};
 gffx_tensor_view v={vertices,real64,{nv,3,0},2},f={faces,0,{nf,3,0},2},c={cuts,real64,{nf,rows,nc},3},g={gradient,real64,{1,800,4},3};
 gffx_tensor_view a={out,real64,{backward?nv:1,backward?3:800,0},2},b={other,real64,{backward?nf:1,backward?rows:800,backward?nc:4},3};
 if(domain){for(long long i=0;i<nf*3;++i)if(faces[i]<0||faces[i]>=nv){*status=1;return;}
  const gffx_tensor_view *inputs[3]={&v,&c,&g};for(int j=0;j<(backward?3:2);++j)for(uint64_t i=0;i<gffx_mesh_element_count(inputs[j]);++i){double x=read_real(inputs[j],i);if(!isfinite(x)||(j==0&&fabs(x)>1048576)){*status=1;return;}}return;}
 *status=opaque_batch_execute_shared(&v,&f,&c,ox,oy,cursor,status+1,backward?&g:0,&a,&b,(Scratch*)((char*)scratch+64),rule,derivatives);
}

/* Only independent forward pixels share a launch. Scratch, work guards, sample
 * order and outputs are private to each lane; backward accumulation stays serial. */
extern "C" __global__ void gffx_opaque_quadrature_pixel_lanes(void *vertices,int *faces,void *cuts,void *out,void *other,long long nv,long long nf,int nc,int rows,long long h,long long w,double ox,double oy,int real64,int material,void *scratch){
 using namespace gffx_opaque_device;
 if(threadIdx.x)return;
 int lane=blockIdx.x;int *status=(int*)scratch;
 gffx_tensor_view v={vertices,real64,{nv,3,0},2},f={faces,0,{nf,3,0},2},c={cuts,real64,{nf,rows,nc},3};
 Scratch *work=(Scratch*)((char*)scratch+(long long)lane*1048576+8);
 for(long long pixel=lane;pixel<h*w;pixel+=gridDim.x){
  gffx_tensor_view a={(int*)out+pixel*800,0,{1,800,0},2};
  gffx_tensor_view b={(char*)other+pixel*3200*(real64?8:4),real64,{1,800,4},3};
  int result=opaque_quadrature_execute(&v,&f,&c,1,1,ox+(double)(pixel%w),oy+(double)(pixel/w),0,&a,&b,work,material);
  if(result)atomicExch(status,result);
 }
}

#undef GFFX_OPAQUE_CLIP_LOCAL
#undef GFFX_OPAQUE_SEPARATION_LOCAL
#undef GFFX_OPAQUE_LOCAL
#undef GFFX_OPAQUE_PRECLEARED

/* Separate lightweight kernels: no polygon stack per validation lane. */
extern "C" __global__ void gffx_opaque_domain_parallel(void *vertices,int *faces,void *cuts,void *gradient,long long nv,long long nf,long long cut_count,long long gradient_count,int real64,void *scratch){
 unsigned long long lane=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;int *status=(int*)scratch;
 for(unsigned long long i=lane;i<(unsigned long long)nf*3;i+=stride)if(faces[i]<0||faces[i]>=nv)atomicExch(status,1);
 void *inputs[3]={vertices,cuts,gradient};unsigned long long counts[3]={(unsigned long long)nv*3,(unsigned long long)cut_count,(unsigned long long)gradient_count};
 for(int k=0;k<3;++k)for(unsigned long long i=lane;i<counts[k];i+=stride){double x=real64?((double*)inputs[k])[i]:((float*)inputs[k])[i];if(!isfinite(x)||(k==0&&fabs(x)>1048576))atomicExch(status,1);}
}
extern "C" __global__ void gffx_opaque_clear_parallel(void *out,void *other,long long a_count,long long b_count,int real64,int backward){
 unsigned long long lane=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
 for(unsigned long long i=lane;i<(unsigned long long)a_count;i+=stride){if(!backward)((int*)out)[i]=-1;else if(real64)((double*)out)[i]=0.;else ((float*)out)[i]=0.f;}
 for(unsigned long long i=lane;i<(unsigned long long)b_count;i+=stride){if(real64)((double*)other)[i]=0.;else ((float*)other)[i]=0.f;}
}

extern "C" __global__ void gffx_opaque_partition_kernel(void *vertices,int *faces,void *cuts,void *rows,void *ids,void *samples,long long nv,long long nf,int nc,double ox,double oy,int count,int begin,int capacity,void *scratch){
 if(blockIdx.x||threadIdx.x)return;using namespace gffx_opaque_device;int *status=(int*)scratch;
 gffx_tensor_view r={rows,1,{capacity,17,0},2};
 /* Preparation has no owner output; empty vertex tensors may have a null pointer. */
 if(!ids){gffx_tensor_view v={vertices,1,{nv,3,0},2},f={faces,0,{nf,3,0},2},c={cuts,1,{nf,3,nc},3};*status=opaque_partition_prepare_execute(&v,&f,&c,ox,oy,&r,status+1,(Scratch*)((char*)scratch+64));}
 else {gffx_tensor_view a={ids,0,{1,800,0},2},b={samples,1,{1,800,4},3};*status=opaque_partition_emit_execute(&r,count,begin,&a,&b);}
}

extern "C" __global__ void gffx_opaque_prefix_kernel(void *vertices,int *faces,void *cuts,void *gradient,void *out,void *other,long long nv,long long nf,int nc,int rows,double ox,double oy,int begin,int batch,int slots,int blocks,int real64,int backward,int domain,void *scratch,int rule,int derivatives,int face,int seen){
 if(blockIdx.x||threadIdx.x)return;using namespace gffx_opaque_device;int *status=(int*)scratch;int32_t cursor[4]={begin,batch,slots,blocks};
 gffx_tensor_view v={vertices,real64,{nv,3,0},2},f={faces,0,{nf,3,0},2},c={cuts,real64,{nf,rows,nc},3},g={gradient,real64,{1,800,4},3};
 gffx_tensor_view a={out,real64,{backward?nv:1,backward?3:800,0},2},b={other,real64,{backward?nf:1,backward?rows:800,backward?nc:4},3};
 if(domain){for(long long i=0;i<nf*3;++i)if(faces[i]<0||faces[i]>=nv){*status=1;return;}
  const gffx_tensor_view *inputs[3]={&v,&c,&g};for(int j=0;j<(backward?3:2);++j)for(uint64_t i=0;i<gffx_mesh_element_count(inputs[j]);++i){double x=read_real(inputs[j],i);if(!isfinite(x)||(j==0&&fabs(x)>1048576)){*status=1;return;}}return;}
 int32_t prefix[2]={face,seen};
 *status=opaque_batch_execute_prefix_shared(&v,&f,&c,ox,oy,cursor,status+1,backward?&g:0,&a,&b,(Scratch*)((char*)scratch+64),rule,derivatives,prefix);
 if(!*status){status[9]=prefix[0];status[10]=prefix[1];}
}

extern "C" __global__ void gffx_opaque_boundary_batch64_kernel(void *vertices,int *faces,void *cuts,void *out,void *other,long long nv,long long nf,int nc,double ox,double oy,int begin,int batch,int slots,int blocks,int real64,int edge,void *scratch){
 if(blockIdx.x||threadIdx.x)return;using namespace gffx_opaque_device;int *status=(int*)scratch;int32_t cursor[4]={begin,batch,slots,blocks};
 gffx_tensor_view v={vertices,real64,{nv,3,0},2},f={faces,0,{nf,3,0},2},c={cuts,real64,{nf,3,nc},3};
 gffx_tensor_view a={out,0,{1,800,0},2},b={other,real64,{1,800,4},3};
 *status=opaque_boundary_batch_execute(&v,&f,&c,ox,oy,edge,cursor,status+1,&a,&b,(Scratch*)((char*)scratch+64));
}

extern "C" __global__ void gffx_opaque_boundary_batch80_kernel(void *vertices,int *faces,void *cuts,void *out,void *other,long long nv,long long nf,int nc,double ox,double oy,int begin,int batch,int slots,int blocks,int real64,int edge,void *scratch){
 if(blockIdx.x||threadIdx.x)return;using namespace gffx_opaque_device;int *status=(int*)scratch;int32_t cursor[4]={begin,batch,slots,blocks};
 gffx_tensor_view v={vertices,real64,{nv,3,0},2},f={faces,0,{nf,3,0},2},c={cuts,real64,{nf,3,nc},3};
 gffx_tensor_view a={out,0,{1,800,0},2},b={other,real64,{1,800,4},3};
 *status=opaque_boundary_batch_execute80(&v,&f,&c,ox,oy,edge,cursor,status+1,&a,&b,(Scratch*)((char*)scratch+64));
}
