/* Development diagnostic only: exact original-profile sample composition.
   Caller checks dense CPU shapes and the existing installed validators first.
   No public ABI, admission certificate, heap allocation or production bypass. */
#include <math.h>
#include <stdint.h>
#include "principled_material_scalar.h"
#include "delta_brdf_scalar.h"
static double add(double a,double b){volatile double x=a+b;return x;}
static double mul(double a,double b){volatile double x=a*b;return x;}
static double divd(double a,double b){volatile double x=a/b;return x;}
__declspec(dllexport) int compose(int pixels,int nv,int nf,int nodes,
 const int32_t *owners,const double *q,const double *world,const int32_t *faces,
 const double *clip,const double *normals,const double *camera,const double *cn,
 const double *logs,const double *curve,const double *partial,const double *mean,
 const double *width,const double *albedo,const double *roughness,const double *f0,
 const double *direction,const double *incoming,double *weighted,double *rgba){
 int pixel,start,j,k,c,i,count,pairs;
 if(pixels<1||pixels>16||nv<0||nf<0||nodes<1||nodes>512)return 1;
 for(pixel=0;pixel<pixels;++pixel){
  double reduction[800][4];
  /* Bounded128-sample cohorts preserve original800-slot order. */
  for(start=0;start<800;start+=128)for(j=start;j<800&&j<start+128;++j){
   int slot=pixel*800+j,face=owners[slot];double values[3]={0.,0.,0.};
   if(face>=0){
    double u[3],den,beta[3],position[3],rawnormal[3],normal[3],view[3],input[16],brdf[51],lit[3],gradient[3],cot[3]={0.,0.,0.};
    if(face>=nf)return 2;
    for(c=0;c<3;++c){int v=faces[face*3+c];if(v<0||v>=nv)return 2;u[c]=divd(q[slot*4+c],clip[v*4+3]);}
    den=add(add(u[0],u[1]),u[2]);if(!(den>0.))return 3;
    for(c=0;c<3;++c)beta[c]=divd(u[c],den);
    for(k=0;k<3;++k){position[k]=0.;rawnormal[k]=0.;
     for(c=0;c<3;++c){int v=faces[face*3+c];position[k]=add(position[k],mul(beta[c],world[v*3+k]));rawnormal[k]=add(rawnormal[k],mul(beta[c],normals[face*9+c*3+k]));}
     view[k]=add(camera[k],-position[k]);
    }
    gffx_unit_vectors_scalar(rawnormal,normal);
    for(k=0;k<3;++k){input[k]=normal[k];input[k+3]=view[k];input[k+6]=direction[k];input[k+9]=albedo[k];input[k+13]=f0[k];}input[12]=roughness[0];
    if(gffx_pv_point(nodes,cn,logs,curve,partial,mean,width[0],input,0,brdf))return 4;
    if(gffx_dl_domain(normal,brdf,direction,incoming,cot,1)||gffx_dl_point(normal,brdf,direction,incoming,cot,1,lit,gradient))return 5;
    for(k=0;k<3;++k){double initial=mul(position[k],0.);for(c=0;c<3;++c)initial=add(initial,0.);values[k]=mul(add(initial,lit[k]),1.);}
   }
   for(k=0;k<3;++k)weighted[slot*4+k]=reduction[j][k]=mul(values[k],q[slot*4+3]);
   weighted[slot*4+3]=reduction[j][3]=q[slot*4+3];
  }
  count=800;while(count>1){pairs=count/2;for(i=0;i<pairs;++i)for(k=0;k<4;++k)reduction[i][k]=add(reduction[2*i][k],reduction[2*i+1][k]);if(count&1)for(k=0;k<4;++k)reduction[pairs][k]=reduction[count-1][k];count=pairs+(count&1);}
  for(k=0;k<4;++k){rgba[pixel*4+k]=reduction[0][k];if(!isfinite(rgba[pixel*4+k]))return 6;}
 }return 0;
}
