#ifndef GFFX_OPAQUE_SAMPLES_SCALAR_H
#define GFFX_OPAQUE_SAMPLES_SCALAR_H
#ifndef GFFX_OC_LOCAL
#define GFFX_OC_LOCAL static
#endif
#include "opaque_samples_args.h"
#include <math.h>
#include <stdint.h>
#include "principled_material_scalar.h"
#include "delta_brdf_scalar.h"
GFFX_OC_LOCAL double oc_add(double a,double b){
#if defined(__CUDA_ARCH__)
return __dadd_rn(a,b);
#else
volatile double x=a+b;return x;
#endif
}
GFFX_OC_LOCAL double oc_mul(double a,double b){
#if defined(__CUDA_ARCH__)
return __dmul_rn(a,b);
#else
volatile double x=a*b;return x;
#endif
}
GFFX_OC_LOCAL double oc_div(double a,double b){
#if defined(__CUDA_ARCH__)
return __ddiv_rn(a,b);
#else
volatile double x=a/b;return x;
#endif
}
GFFX_OC_LOCAL int oc_sample(const GffxOpaqueSamples *x,int slot){
 int nv=x->nv,nf=x->nf,nodes=x->nodes,k,c;
 const int32_t *owners=(const int32_t*)x->data[0],*faces=(const int32_t*)x->data[3];
 const double *q=(const double*)x->data[1],*world=(const double*)x->data[2],*clip=(const double*)x->data[4],*normals=(const double*)x->data[5],*camera=(const double*)x->data[6],*cn=(const double*)x->data[7],*logs=(const double*)x->data[8],*curve=(const double*)x->data[9],*partial=(const double*)x->data[10],*mean=(const double*)x->data[11],*width=(const double*)x->data[12],*albedo=(const double*)x->data[13],*roughness=(const double*)x->data[14],*f0=(const double*)x->data[15],*direction=(const double*)x->data[16],*incoming=(const double*)x->data[17];
 double *weighted=x->weighted;int face=owners[slot];double values[3]={0.,0.,0.};
   if(face>=0){
    double u[3],den,beta[3],position[3],rawnormal[3],normal[3],view[3],input[16],brdf[51],lit[3],gradient[3],cot[3]={0.,0.,0.};
    if(face>=nf)return 2;
    for(c=0;c<3;++c){int v=faces[face*3+c];if(v<0||v>=nv)return 2;u[c]=oc_div(q[slot*4+c],clip[v*4+3]);}
    den=oc_add(oc_add(u[0],u[1]),u[2]);if(!(den>0.))return 3;
    for(c=0;c<3;++c)beta[c]=oc_div(u[c],den);
    for(k=0;k<3;++k){position[k]=0.;rawnormal[k]=0.;
     for(c=0;c<3;++c){int v=faces[face*3+c];position[k]=oc_add(position[k],oc_mul(beta[c],world[v*3+k]));rawnormal[k]=oc_add(rawnormal[k],oc_mul(beta[c],normals[face*9+c*3+k]));}
     view[k]=oc_add(camera[k],-position[k]);
    }
    gffx_unit_vectors_scalar(rawnormal,normal);
    for(k=0;k<3;++k){input[k]=normal[k];input[k+3]=view[k];input[k+6]=direction[k];input[k+9]=albedo[k];input[k+13]=f0[k];}input[12]=roughness[0];
    if(gffx_pv_point(nodes,cn,logs,curve,partial,mean,width[0],input,0,brdf))return 4;
    if(gffx_dl_domain(normal,brdf,direction,incoming,cot,1)||gffx_dl_point(normal,brdf,direction,incoming,cot,1,lit,gradient))return 5;
    for(k=0;k<3;++k){double initial=oc_mul(position[k],0.);for(c=0;c<3;++c)initial=oc_add(initial,0.);values[k]=oc_mul(oc_add(initial,lit[k]),1.);}
   }
   for(k=0;k<3;++k)weighted[slot*4+k]=oc_mul(values[k],q[slot*4+3]);
   weighted[slot*4+3]=q[slot*4+3];
 return 0;
}
GFFX_OC_LOCAL int oc_reduce(const GffxOpaqueSamples *x,int pixel,double *scratch){
 int count=800,pairs,i,k;double (*reduction)[4]=(double (*)[4])scratch;
 while(count>1){pairs=count/2;for(i=0;i<pairs;++i)for(k=0;k<4;++k)reduction[i][k]=oc_add(reduction[2*i][k],reduction[2*i+1][k]);if(count&1)for(k=0;k<4;++k)reduction[pairs][k]=reduction[count-1][k];count=pairs+(count&1);}
 for(k=0;k<4;++k){x->rgba[pixel*4+k]=reduction[0][k];if(!isfinite(x->rgba[pixel*4+k]))return 6;}return 0;
}
GFFX_OC_LOCAL int oc_value_domain(const GffxOpaqueSamples *x,int64_t j){
 int k;const int32_t *owner=(const int32_t*)x->data[0],*faces=(const int32_t*)x->data[3];
 const double *cn=(const double*)x->data[7],*logs=(const double*)x->data[8],*curve=(const double*)x->data[9],*mean=(const double*)x->data[11],*width=(const double*)x->data[12],*a=(const double*)x->data[13],*r=(const double*)x->data[14],*f=(const double*)x->data[15],*l=(const double*)x->data[16],*irr=(const double*)x->data[17];
 if(j<x->count[0]&&(owner[j]<-1||owner[j]>=x->nf))return 1;
 if(j<x->count[3]&&(faces[j]<0||faces[j]>=x->nv))return 1;
 if(j<x->nodes&&(cn[j]<=0.||cn[j]>1.||logs[j]>0.||(j==x->nodes-1&&cn[j]!=1.)||(j&&(cn[j]<=cn[j-1]||logs[j]<=logs[j-1]||!isfinite(logs[j]-logs[j-1])))))return 1;
 if(j<x->count[9]&&!(j%2)&&(curve[j]<curve[j+1]||curve[j]>1.||curve[j+1]<0.))return 1;
 if(j<3&&(a[j]<0.||a[j]>1.||f[j]<0.||f[j]>1.||irr[j]<0.))return 1;
 if(!j&&(width[0]<=0.||mean[0]<=0.||mean[0]>1.||mean[1]<=0.||mean[1]>mean[0]||r[0]<=0.||r[0]>1.||(l[0]==0.&&l[1]==0.&&l[2]==0.)))return 1;
 return 0;
}
GFFX_OC_LOCAL int oc_domain(const GffxOpaqueSamples *x,int64_t j){int k;
 for(k=1;k<18;++k)if(k!=3&&j<x->count[k]&&!isfinite(((const double*)x->data[k])[j]))return 1;
 return oc_value_domain(x,j);
}
GFFX_OC_LOCAL int oc_execute(const GffxOpaqueSamples *x,int pixel,double *scratch){
 int start,j,k,s;for(start=0;start<800;start+=128)for(j=start;j<800&&j<start+128;++j){
 s=oc_sample(x,pixel*800+j);if(s)return s;for(k=0;k<4;++k)scratch[j*4+k]=x->weighted[(pixel*800+j)*4+k];}
 return oc_reduce(x,pixel,scratch);
}
#endif
