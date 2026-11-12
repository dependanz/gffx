#define RS_INLINE static __device__ __forceinline__
#include "shade_scalar.h"
#include "shade_indirect.h"
#include "shade_subsurface.h"

/* Shadow traversal needs no face-order sort: float depth minimum is invariant.
   Count and fill conservative bins directly on device, retaining native coverage. */
static __device__ bool rs_bounds(int face,int height,int width,const float *ndc,const int *faces,int *lo,int *hi){
 double lower[2]={1e300,1e300},upper[2]={-1e300,-1e300};
 for(int i=0;i<3;i++){const float *v=ndc+faces[face*3+i]*3;double x=((double)v[0]+1)*width*.5,y=(1-(double)v[1])*height*.5;lower[0]=fmin(lower[0],x);upper[0]=fmax(upper[0],x);lower[1]=fmin(lower[1],y);upper[1]=fmax(upper[1],y);}
 if(upper[0]<.5||upper[1]<.5||lower[0]>width-.5||lower[1]>height-.5)return false;
 lo[0]=max(0,min((width-1)/16,(int)floor(lower[0]/16)));hi[0]=max(0,min((width-1)/16,(int)floor(upper[0]/16)));
 lo[1]=max(0,min((height-1)/16,(int)floor(lower[1]/16)));hi[1]=max(0,min((height-1)/16,(int)floor(upper[1]/16)));return true;
}
extern "C" __global__ void gffx_shadow_bins(int count,int height,int width,int fill,const float *ndc,const int *faces,int *counts,const int *offsets,int *candidates){
 int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=count)return;int lo[2],hi[2];if(!rs_bounds(face,height,width,ndc,faces,lo,hi))return;
 for(int y=lo[1];y<=hi[1];y++)for(int x=lo[0];x<=hi[0];x++){
  int tile=y*((width+15)/16)+x,slot=atomicAdd(counts+tile,1);if(fill)candidates[offsets[tile]+slot]=face;
 }
}
extern "C" __global__ void gffx_shadow_depth(int height,int width,const float *ndc,const int *faces,const int *offsets,const int *candidates,float *depth){
 int tiles=(width+15)/16,x=(blockIdx.x%tiles)*16+threadIdx.x%16,y=(blockIdx.x/tiles)*16+threadIdx.x/16;if(x>=width||y>=height)return;
 float best=__int_as_float(0x7f800000);double px=x+.5,py=y+.5;
 for(int j=offsets[blockIdx.x];j<offsets[blockIdx.x+1];j++){
  int face=candidates[j];const float *a=ndc+faces[face*3]*3,*b=ndc+faces[face*3+1]*3,*c=ndc+faces[face*3+2]*3;
  double ax=((double)a[0]+1)*width*.5,ay=(1-(double)a[1])*height*.5,bx=((double)b[0]+1)*width*.5,by=(1-(double)b[1])*height*.5,cx=((double)c[0]+1)*width*.5,cy=(1-(double)c[1])*height*.5;
  double e0=(bx-px)*(cy-py)-(by-py)*(cx-px),e1=(cx-px)*(ay-py)-(cy-py)*(ax-px),e2=(ax-px)*(by-py)-(ay-py)*(bx-px),area=e0+e1+e2;
  if(!(fabs(area)>0.00000095367431640625))continue;
  /* Reject uncovered samples before costly double divisions. Edge signs are
     exactly the same closed-triangle rule for either winding; covered samples
     retain the original divisions and stored-float depth expression. */
  if(area>0 ? (e0<0||e1<0||e2<0) : (e0>0||e1>0||e2>0))continue;
  double a0=e0/area,a1=e1/area,a2=e2/area;
  best=fminf(best,(float)(a0*a[2]+a1*b[2]+a2*c[2]));
 }depth[y*width+x]=best;
}
extern "C" __global__ void gffx_raster_shade(int count,int light_count,int size,int kind,const float *p,const float *n,const float *g,const float *v,const float *f,const float *l,const float *m,const float *d,const float *table,float *out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)rs_shade(i,light_count,size,kind,p,n,g,v,f,l,m,d,table,out);
}

/* Sorted conservative tile lists change candidate traversal, not raster rules.
   Double edge arithmetic and float32 stored-depth comparisons match the
   existing batch-one hard raster path, including ascending-face ties. */
extern "C" __global__ void gffx_raster_binned(int height,int width,int cull,const float *ndc,const int *faces,const int *offsets,const int *candidates,int *indices,float *bary,float *depth){
 int tiles=(width+15)/16,x=(blockIdx.x%tiles)*16+threadIdx.x%16,y=(blockIdx.x/tiles)*16+threadIdx.x/16;
 if(x>=width||y>=height)return;int pixel=y*width+x,best=-1;float best_depth=__int_as_float(0x7f800000),b0=0,b1=0,b2=0;double px=x+.5,py=y+.5;
 for(int j=offsets[blockIdx.x];j<offsets[blockIdx.x+1];j++){
  int face=candidates[j];const float *a=ndc+faces[face*3]*3,*b=ndc+faces[face*3+1]*3,*c=ndc+faces[face*3+2]*3;
  double ax=((double)a[0]+1)*width*.5,ay=(1-(double)a[1])*height*.5;
  double bx=((double)b[0]+1)*width*.5,by=(1-(double)b[1])*height*.5;
  double cx=((double)c[0]+1)*width*.5,cy=(1-(double)c[1])*height*.5;
  double e0=(bx-px)*(cy-py)-(by-py)*(cx-px),e1=(cx-px)*(ay-py)-(cy-py)*(ax-px),e2=(ax-px)*(by-py)-(ay-py)*(bx-px),area=e0+e1+e2;
  if(!(fabs(area)>0.00000095367431640625)||(cull==2&&-area<=0)||(cull==3&&-area>0))continue;
  double w0=e0/area,w1=e1/area,w2=e2/area;if(w0<0||w1<0||w2<0)continue;
  double z=w0*a[2]+w1*b[2]+w2*c[2];
  if(best<0||z<(double)best_depth){best=face;best_depth=(float)z;b0=(float)w0;b1=(float)w1;b2=(float)w2;}
 }
 indices[pixel]=best;depth[pixel]=best_depth;bary[pixel*3]=b0;bary[pixel*3+1]=b1;bary[pixel*3+2]=b2;
}

extern "C" __global__ void gffx_raster_indirect(int count,int cubes,int mirrors,int classic,const float *p,const float *n,const float *v,const float *f,const int *ci,const float *images,const float *meta,const float *sh,const float *gi,const float *mi,const float *mp,const int *ids,const float *dfg,float *out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)ri_shade(i,cubes,mirrors,classic,p,n,v,f,ci,images,meta,sh,gi,mi,mp,ids,dfg,out);
}

extern "C" __global__ void gffx_raster_diffusion(int h,int w,float fx,float fy,const float *image,const float *data,const int *ids,float *out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<h*w)rss_diffuse(i,h,w,fx,fy,image,data,ids,out);}
extern "C" __global__ void gffx_raster_transmission(int count,int lights,int size,const float *p,const float *n,const float *f,const float *l,const float *m,const float *d,float *out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)rss_transmit(i,lights,size,p,n,f,l,m,d,out);}

/* Hybrid pixel/coarse bins. Fine faces are classified with the identical
   closed-edge coverage predicate, avoiding repeated rejection of subpixel hair. */
static __device__ bool rm_cover(const double *x,const double *y,double px,double py){
 double e0=(x[1]-px)*(y[2]-py)-(y[1]-py)*(x[2]-px),e1=(x[2]-px)*(y[0]-py)-(y[2]-py)*(x[0]-px),e2=(x[0]-px)*(y[1]-py)-(y[0]-py)*(x[1]-px),area=e0+e1+e2;
 return fabs(area)>0.00000095367431640625&&((area>0&&e0>=0&&e1>=0&&e2>=0)||(area<0&&e0<=0&&e1<=0&&e2<=0));
}
extern "C" __global__ void gffx_micro_bins(int count,int h,int w,int fill,const float *ndc,const int *faces,int *counts,const int *offsets,int *candidates){
 int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=count)return;
 double x[3],y[3],low[2]={1e300,1e300},high[2]={-1e300,-1e300};
 for(int j=0;j<3;j++){const float *v=ndc+faces[face*3+j]*3;x[j]=((double)v[0]+1)*w*.5;y[j]=(1-(double)v[1])*h*.5;low[0]=fmin(low[0],x[j]);high[0]=fmax(high[0],x[j]);low[1]=fmin(low[1],y[j]);high[1]=fmax(high[1],y[j]);}
 if(high[0]<.5||high[1]<.5||low[0]>w-.5||low[1]>h-.5)return;
 int lx=max(0,min(w-1,(int)floor(low[0]))),hx=max(0,min(w-1,(int)floor(high[0]))),ly=max(0,min(h-1,(int)floor(low[1]))),hy=max(0,min(h-1,(int)floor(high[1])));
 bool coarse=(double)(hx-lx+1)*(hy-ly+1)>64;
 if(coarse){lx/=16;hx/=16;ly/=16;hy/=16;}
 for(int py=ly;py<=hy;py++)for(int px=lx;px<=hx;px++){
  if(!coarse&&!rm_cover(x,y,px+.5,py+.5))continue;
  int bucket=coarse?h*w+py*((w+15)/16)+px:py*w+px,slot=atomicAdd(counts+bucket,1);
  if(fill)candidates[offsets[bucket]+slot]=face;
 }
}
extern "C" __global__ void gffx_micro_depth(int h,int w,const float *ndc,const int *faces,const int *offsets,const int *candidates,float *depth){
 int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=h*w)return;int x=pixel%w,y=pixel/w;
 float best=__int_as_float(0x7f800000);double px=x+.5,py=y+.5;
 int buckets[2]={pixel,h*w+(y/16)*((w+15)/16)+x/16};
 for(int k=0;k<2;k++)for(int j=offsets[buckets[k]];j<offsets[buckets[k]+1];j++){
  int face=candidates[j];const float *a=ndc+faces[face*3]*3,*b=ndc+faces[face*3+1]*3,*c=ndc+faces[face*3+2]*3;
  double ax=((double)a[0]+1)*w*.5,ay=(1-(double)a[1])*h*.5,bx=((double)b[0]+1)*w*.5,by=(1-(double)b[1])*h*.5,cx=((double)c[0]+1)*w*.5,cy=(1-(double)c[1])*h*.5;
  double e0=(bx-px)*(cy-py)-(by-py)*(cx-px),e1=(cx-px)*(ay-py)-(cy-py)*(ax-px),e2=(ax-px)*(by-py)-(ay-py)*(bx-px),area=e0+e1+e2;
  if(!(fabs(area)>0.00000095367431640625))continue;
  if((area>0&&(e0<0||e1<0||e2<0))||(area<0&&(e0>0||e1>0||e2>0)))continue;
  double w0=e0/area,w1=e1/area,w2=e2/area;
  float z=(float)(w0*a[2]+w1*b[2]+w2*c[2]);best=fminf(best,z);
 }
 depth[pixel]=best;
}
extern "C" __global__ void gffx_micro_primary(int h,int w,int cull,const float *ndc,const int *faces,const int *offsets,const int *candidates,int *indices,float *bary,float *depth){
 int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=h*w)return;int x=pixel%w,y=pixel/w;
 int coarse=h*w+(y/16)*((w+15)/16)+x/16,ia=offsets[pixel],ea=offsets[pixel+1],ib=offsets[coarse],eb=offsets[coarse+1],best=-1;
 float best_depth=__int_as_float(0x7f800000),b0=0,b1=0,b2=0;double px=x+.5,py=y+.5;
 while(ia<ea||ib<eb){
  int face;if(ib==eb||(ia<ea&&candidates[ia]<candidates[ib]))face=candidates[ia++];else face=candidates[ib++];
  const float *a=ndc+faces[face*3]*3,*b=ndc+faces[face*3+1]*3,*c=ndc+faces[face*3+2]*3;
  double ax=((double)a[0]+1)*w*.5,ay=(1-(double)a[1])*h*.5,bx=((double)b[0]+1)*w*.5,by=(1-(double)b[1])*h*.5,cx=((double)c[0]+1)*w*.5,cy=(1-(double)c[1])*h*.5;
  double e0=(bx-px)*(cy-py)-(by-py)*(cx-px),e1=(cx-px)*(ay-py)-(cy-py)*(ax-px),e2=(ax-px)*(by-py)-(ay-py)*(bx-px),area=e0+e1+e2;
  if(!(fabs(area)>0.00000095367431640625)||(cull==2&&-area<=0)||(cull==3&&-area>0))continue;
  double w0=e0/area,w1=e1/area,w2=e2/area;if(w0<0||w1<0||w2<0)continue;
  double z=w0*a[2]+w1*b[2]+w2*c[2];
  if(best<0||z<(double)best_depth){best=face;best_depth=(float)z;b0=(float)w0;b1=(float)w1;b2=(float)w2;}
 }
 indices[pixel]=best;depth[pixel]=best_depth;bary[pixel*3]=b0;bary[pixel*3+1]=b1;bary[pixel*3+2]=b2;
}
