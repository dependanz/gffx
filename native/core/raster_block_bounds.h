/* Conservative rejection only; see docs/RASTER_BLOCK_ACCELERATION.md. */
#ifndef GFFX_RASTER_BLOCK_BOUNDS_H
#define GFFX_RASTER_BLOCK_BOUNDS_H
#include <math.h>
#ifndef GFFX_RASTER_INLINE
#define GFFX_RASTER_INLINE static inline
#endif
typedef struct gffx_raster_box { double xmin,xmax,ymin,ymax,error; int valid; } gffx_raster_box;
GFFX_RASTER_INLINE double gffx_rb_min(double a,double b){return a<b?a:b;}
GFFX_RASTER_INLINE double gffx_rb_max(double a,double b){return a>b?a:b;}
GFFX_RASTER_INLINE gffx_raster_box gffx_raster_make_box(double ax,double ay,double bx,double by,double cx,double cy,double width,double height){
 gffx_raster_box b;double m=1.;
 b.valid=fabs(ax)<=1048576. && fabs(ay)<=1048576. && fabs(bx)<=1048576. && fabs(by)<=1048576. && fabs(cx)<=1048576. && fabs(cy)<=1048576. && width<=1048576. && height<=1048576.;
 b.xmin=gffx_rb_min(ax,gffx_rb_min(bx,cx));b.xmax=gffx_rb_max(ax,gffx_rb_max(bx,cx));
 b.ymin=gffx_rb_min(ay,gffx_rb_min(by,cy));b.ymax=gffx_rb_max(ay,gffx_rb_max(by,cy));
 m=gffx_rb_max(m,gffx_rb_max(width,height));m=gffx_rb_max(m,gffx_rb_max(fabs(b.xmin),fabs(b.xmax)));m=gffx_rb_max(m,gffx_rb_max(fabs(b.ymin),fabs(b.ymax)));
 b.error=(m*m)*0x1p-44;return b;
}
GFFX_RASTER_INLINE int gffx_raster_far(gffx_raster_box b,double left,double top,double right,double bottom,double blur_squared){
 double gap=0.,value;
 if(!b.valid)return 0;
 gap=gffx_rb_max(gap,left-b.xmax);gap=gffx_rb_max(gap,b.xmin-right);
 gap=gffx_rb_max(gap,top-b.ymax);gap=gffx_rb_max(gap,b.ymin-bottom);
 if(!(gap>0x1p-20))return 0;
 value=gap-0x1p-20;return value*value>blur_squared;
}
GFFX_RASTER_INLINE double gffx_raster_edge(double ux,double uy,double vx,double vy,double px,double py){
 return (ux-px)*(vy-py)-(uy-py)*(vx-px);
}
GFFX_RASTER_INLINE int gffx_raster_edge_sign(double ux,double uy,double vx,double vy,double l,double t,double r,double b,double error){
 double a=gffx_raster_edge(ux,uy,vx,vy,l,t),c=gffx_raster_edge(ux,uy,vx,vy,r,t);
 double d=gffx_raster_edge(ux,uy,vx,vy,l,b),e=gffx_raster_edge(ux,uy,vx,vy,r,b);
 double low=gffx_rb_min(gffx_rb_min(a,c),gffx_rb_min(d,e)),high=gffx_rb_max(gffx_rb_max(a,c),gffx_rb_max(d,e));
 return low>2.*error?1:(high< -2.*error?-1:0);
}
GFFX_RASTER_INLINE int gffx_raster_block_excluded(gffx_raster_box box,double ax,double ay,double bx,double by,double cx,double cy,double l,double t,double r,double b,double blur_squared){
 int a,c,d;
 if(!gffx_raster_far(box,l,t,r,b,blur_squared))return 0;
 a=gffx_raster_edge_sign(bx,by,cx,cy,l,t,r,b,box.error);
 c=gffx_raster_edge_sign(cx,cy,ax,ay,l,t,r,b,box.error);
 d=gffx_raster_edge_sign(ax,ay,bx,by,l,t,r,b,box.error);
 return (a>0 || c>0 || d>0) && (a<0 || c<0 || d<0);
}
#endif
