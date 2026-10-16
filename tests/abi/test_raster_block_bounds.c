#include <stdio.h>
#include "raster_block_bounds.h"
static unsigned int seed=19414u;
static double sample(void){seed=1664525u*seed+1013904223u;return (double)(seed>>8)/16777216.;}
static double segment(double px,double py,double ux,double uy,double vx,double vy){
 double ex=vx-ux,ey=vy-uy,length=ex*ex+ey*ey,t=0.,cx,cy,dx,dy;
 if(length>0.){t=((px-ux)*ex+(py-uy)*ey)/length;if(t<0.)t=0.;if(t>1.)t=1.;}
 cx=ux+t*ex;cy=uy+t*ey;dx=px-cx;dy=py-cy;return dx*dx+dy*dy;
}
static int original_accept(double ax,double ay,double bx,double by,double cx,double cy,double px,double py,double blur){
 double e0=(bx-px)*(cy-py)-(by-py)*(cx-px),e1=(cx-px)*(ay-py)-(cy-py)*(ax-px),e2=(ax-px)*(by-py)-(ay-py)*(bx-px);
 double area=e0+e1+e2,d,c;
 if(!(fabs(area)>0.))return 0;
 if(e0/area>=0. && e1/area>=0. && e2/area>=0.)return 1;
 d=segment(px,py,ax,ay,bx,by);c=segment(px,py,bx,by,cx,cy);if(c<d)d=c;
 c=segment(px,py,cx,cy,ax,ay);if(c<d)d=c;return !(d>blur);
}
int main(void){
 gffx_raster_box box=gffx_raster_make_box(0.,0.,2.,0.,0.,2.,64.,64.);
 if(!gffx_raster_block_excluded(box,0.,0.,2.,0.,0.,2.,32.5,32.5,63.5,63.5,0.04))return 1;
 if(gffx_raster_block_excluded(box,0.,0.,2.,0.,0.,2.,0.5,0.5,1.5,1.5,0.04))return 2;
 box=gffx_raster_make_box(1e100,0.,2.,0.,0.,2.,64.,64.);
 if(gffx_raster_far(box,32.5,32.5,63.5,63.5,0.04))return 3;
 {
  int trial,x,y,rejected=0,checked=0;
  for(trial=0;trial<10000;++trial){
   double scale=trial%7==0?1048000.:(trial%7==1?1e-150:128.);
   double ax=(sample()-.5)*scale,ay=(sample()-.5)*scale,bx=(sample()-.5)*scale,by=(sample()-.5)*scale,cx=(sample()-.5)*scale,cy=(sample()-.5)*scale;
   double l=(double)(trial%4*32)+.5,t=(double)(trial%3*32)+.5,blur=trial%3==0?0.:(trial%3==1?.36:4096.);
   box=gffx_raster_make_box(ax,ay,bx,by,cx,cy,128.,128.);
   if(gffx_raster_block_excluded(box,ax,ay,bx,by,cx,cy,l,t,l+31.,t+31.,blur)){
    ++rejected;
    for(y=0;y<32;++y)for(x=0;x<32;++x){
     ++checked;if(original_accept(ax,ay,bx,by,cx,cy,l+x,t+y,blur))return 4;
    }
   }
  }
  if(rejected<100)return 5;
  printf("{\"trials\":10000,\"rejected_blocks\":%d,\"original_pixel_checks\":%d}\n",rejected,checked);
 }
 return 0;
}
