/* Private forward raster acceleration. Framework-free C11 and optional Driver API. */
#include <stdint.h>
#include "shade_scalar.h"
#include "shade_indirect.h"
#include "shade_subsurface.h"
#ifdef _WIN32
#define RS_EXPORT __declspec(dllexport)
#define RS_THREAD __declspec(thread)
#else
#define RS_EXPORT __attribute__((visibility("default")))
#define RS_THREAD _Thread_local
#endif
#ifdef GFFX_RASTER_CUDA
#include <cuda.h>
#include "shade_ptx.h"
static RS_THREAD CUcontext rs_context;
static RS_THREAD CUmodule rs_module;
static RS_THREAD CUfunction rs_kernel;
static RS_THREAD CUfunction ri_kernel;
static RS_THREAD CUfunction rss_diffusion_kernel;
static RS_THREAD CUfunction rss_transmission_kernel;
static RS_THREAD CUfunction rs_binned;
static RS_THREAD CUfunction rs_shadow_bins;
static RS_THREAD CUfunction rs_shadow_depth;
static RS_THREAD CUfunction rm_bins,rm_depth,rm_primary;

static int rs_load(void){
 CUcontext context;CUresult code=cuCtxGetCurrent(&context);if(code||!context)return code?code:-2;
 if(context!=rs_context){code=cuModuleLoadData(&rs_module,gffx_raster_ptx);if(code)return code;code=cuModuleGetFunction(&rs_kernel,rs_module,"gffx_raster_shade");if(code)return code;code=cuModuleGetFunction(&rs_binned,rs_module,"gffx_raster_binned");if(code)return code;code=cuModuleGetFunction(&rs_shadow_bins,rs_module,"gffx_shadow_bins");if(code)return code;code=cuModuleGetFunction(&rs_shadow_depth,rs_module,"gffx_shadow_depth");if(code)return code;code=cuModuleGetFunction(&ri_kernel,rs_module,"gffx_raster_indirect");if(code)return code;code=cuModuleGetFunction(&rss_diffusion_kernel,rs_module,"gffx_raster_diffusion");if(code)return code;code=cuModuleGetFunction(&rss_transmission_kernel,rs_module,"gffx_raster_transmission");if(code)return code;code=cuModuleGetFunction(&rm_bins,rs_module,"gffx_micro_bins");if(code)return code;code=cuModuleGetFunction(&rm_depth,rs_module,"gffx_micro_depth");if(code)return code;code=cuModuleGetFunction(&rm_primary,rs_module,"gffx_micro_primary");if(code)return code;rs_context=context;}
 return 0;
}
#endif
RS_EXPORT int gffx_raster_shade_v2_f32(int device,int count,int light_count,int size,int kind,const float *p,const float *n,const float *g,const float *v,const float *f,const float *l,const float *m,const float *d,const float *table,float *out,void *stream){
 if(count<0||light_count<0||size<1||kind<0||kind>2||!p||!n||!g||!v||!f||!out)return -1;
 if(!device){for(int i=0;i<count;i++)rs_shade(i,light_count,size,kind,p,n,g,v,f,l,m,d,table,out);return 0;}
#ifdef GFFX_RASTER_CUDA
 int code=rs_load();if(code)return code;
 void *args[]={&count,&light_count,&size,&kind,&p,&n,&g,&v,&f,&l,&m,&d,&table,&out};
 return cuLaunchKernel(rs_kernel,(count+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0);
#else
 (void)stream;return -3;
#endif
}

#ifdef GFFX_RASTER_CUDA
RS_EXPORT int gffx_raster_binned_f32(int height,int width,int cull,const float *ndc,const int *faces,const int *offsets,const int *candidates,int *indices,float *bary,float *depth,void *stream){
 if(height<1||width<1)return -1;int code=rs_load();if(code)return code;
 void *args[]={&height,&width,&cull,&ndc,&faces,&offsets,&candidates,&indices,&bary,&depth};
 return cuLaunchKernel(rs_binned,((width+15)/16)*((height+15)/16),1,1,256,1,1,0,(CUstream)stream,args,0);
}
RS_EXPORT int gffx_raster_shadow_bins_f32(int stage,int count,int height,int width,const float *ndc,const int *faces,int *counts,const int *offsets,int *candidates,float *depth,void *stream){
 if(count<0||height<1||width<1||stage<0||stage>2)return -1;int code=rs_load();if(code)return code;
 int tiles=((width+15)/16)*((height+15)/16);code=cuMemsetD32Async((CUdeviceptr)counts,0,tiles,(CUstream)stream);if(code)return code;
 void *args[]={&count,&height,&width,&stage,&ndc,&faces,&counts,&offsets,&candidates};
 if(count){code=cuLaunchKernel(rs_shadow_bins,(count+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0);if(code)return code;}
 if(stage==1){void *draw[]={&height,&width,&ndc,&faces,&offsets,&candidates,&depth};return cuLaunchKernel(rs_shadow_depth,tiles,1,1,256,1,1,0,(CUstream)stream,draw,0);}return 0;
}

#endif

RS_EXPORT int gffx_raster_indirect_f32(int device,int count,int cubes,int mirrors,int classic,const float *p,const float *n,const float *v,const float *f,const int *ci,const float *images,const float *meta,const float *sh,const float *gi,const float *mi,const float *mp,const int *ids,const float *dfg,float *out,void *stream){
 if(count<0||cubes<1||mirrors<0||!p||!n||!v||!f||!out)return -1;
 if(!device){for(int i=0;i<count;i++)ri_shade(i,cubes,mirrors,classic,p,n,v,f,ci,images,meta,sh,gi,mi,mp,ids,dfg,out);return 0;}
#ifdef GFFX_RASTER_CUDA
 int code=rs_load();if(code)return code;void *args[]={&count,&cubes,&mirrors,&classic,&p,&n,&v,&f,&ci,&images,&meta,&sh,&gi,&mi,&mp,&ids,&dfg,&out};
 return cuLaunchKernel(ri_kernel,(count+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0);
#else
 (void)stream;return -3;
#endif
}

RS_EXPORT int gffx_raster_diffusion_f32(int device,int h,int w,float fx,float fy,const float *image,const float *data,const int *ids,float *out,void *stream){
 if(h<1||w<1||fx==0||fy<=0||!image||!data||!ids||!out)return -1;
 if(!device){for(int i=0;i<h*w;i++)rss_diffuse(i,h,w,fx,fy,image,data,ids,out);return 0;}
#ifdef GFFX_RASTER_CUDA
 int code=rs_load();if(code)return code;void *args[]={&h,&w,&fx,&fy,&image,&data,&ids,&out};return cuLaunchKernel(rss_diffusion_kernel,(h*w+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0);
#else
 (void)stream;return -3;
#endif
}
RS_EXPORT int gffx_raster_transmission_f32(int device,int count,int lights,int size,const float *p,const float *n,const float *f,const float *l,const float *m,const float *d,float *out,void *stream){
 if(count<0||lights<0||size<1||!p||!n||!f||!out)return -1;
 if(!device){for(int i=0;i<count;i++)rss_transmit(i,lights,size,p,n,f,l,m,d,out);return 0;}
#ifdef GFFX_RASTER_CUDA
 int code=rs_load();if(code)return code;void *args[]={&count,&lights,&size,&p,&n,&f,&l,&m,&d,&out};return cuLaunchKernel(rss_transmission_kernel,(count+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0);
#else
 (void)stream;return -3;
#endif
}

#ifdef GFFX_RASTER_CUDA
RS_EXPORT int gffx_raster_micro_f32(int stage,int count,int h,int w,int cull,const float *ndc,const int *faces,int *counts,const int *offsets,int *candidates,int *indices,float *bary,float *depth,void *stream){
 if(stage<0||stage>3||count<0||h<1||w<1)return -1;int code=rs_load();if(code)return code;
 if(stage<=1){int buckets=h*w+((w+15)/16)*((h+15)/16);code=cuMemsetD32Async((CUdeviceptr)counts,0,buckets,(CUstream)stream);if(code)return code;void *args[]={&count,&h,&w,&stage,&ndc,&faces,&counts,&offsets,&candidates};return count?cuLaunchKernel(rm_bins,(count+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0):0;}
 if(stage==2){void *args[]={&h,&w,&cull,&ndc,&faces,&offsets,&candidates,&indices,&bary,&depth};return cuLaunchKernel(rm_primary,(h*w+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0);}
 void *args[]={&h,&w,&ndc,&faces,&offsets,&candidates,&depth};return cuLaunchKernel(rm_depth,(h*w+127)/128,1,1,128,1,1,0,(CUstream)stream,args,0);
}
#endif
