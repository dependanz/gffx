#ifndef GFFX_OPAQUE_ALBEDO_SCALAR_H
#define GFFX_OPAQUE_ALBEDO_SCALAR_H
#include "opaque_samples_scalar.h"
/* Full16-column first Jacobian, fixed-light reverse factors and original
   four-point broadcast reduction. No albedo-only analytic specialization. */
GFFX_OC_LOCAL int oa_sample(const GffxOpaqueSamples *x,const double *cotangent,double *adjoint,int slot){
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
    for(k=0;k<3;++k)cot[k]=oc_mul(oc_mul(oc_add(cotangent[(slot/800)*3+k],0.),q[slot*4+3]),1.);
    if(gffx_pm_point(nodes,cn,logs,curve,partial,mean,width[0],input,0,brdf))return 4;
    if(gffx_dl_domain(normal,brdf,direction,incoming,cot,1)||gffx_dl_point(normal,brdf,direction,incoming,cot,1,lit,gradient))return 5;
    for(k=0;k<16;++k){double derivative=gffx_pp_rgb(brdf+3,gradient,k);if(!isfinite(derivative))return 7;if(k>=9&&k<12)adjoint[(slot%800)*3+k-9]=derivative;}
    for(k=0;k<3;++k){double initial=oc_mul(position[k],0.);for(c=0;c<3;++c)initial=oc_add(initial,0.);values[k]=oc_mul(oc_add(initial,lit[k]),1.);}
   }
   for(k=0;k<3;++k)weighted[slot*4+k]=oc_mul(values[k],q[slot*4+3]);
   weighted[slot*4+3]=q[slot*4+3];
 return 0;
}

GFFX_OC_LOCAL int oa_reduce(const GffxOpaqueSamples *x,int pixel,double *adjoint,double *gradient){
 const int32_t *owners=(const int32_t*)x->data[0];int j,k,count=0,start,size;double carry[3]={0.,0.,0.};
 for(j=0;j<800;++j)if(owners[pixel*800+j]>=0){for(k=0;k<3;++k)adjoint[count*3+k]=adjoint[j*3+k];++count;}
 for(start=((count+3)/4)*4-4;start>=0;start-=4){size=count-start;if(size>4)size=4;
  for(k=0;k<3;++k){double q=adjoint[start*3+k];if(size>1)q=oc_add(q,adjoint[(start+1)*3+k]);
   if(size==3)q=oc_add(q,adjoint[(start+2)*3+k]);
   if(size==4)q=oc_add(q,oc_add(adjoint[(start+2)*3+k],adjoint[(start+3)*3+k]));
   carry[k]=oc_add(carry[k],q);}
 }
 for(k=0;k<3;++k){gradient[pixel*3+k]=carry[k];if(!isfinite(carry[k]))return 8;}return 0;
}
GFFX_OC_LOCAL int oa_execute(const GffxOpaqueSamples *x,const double *cotangent,double *gradient,int pixel,double *scratch){
 double *adjoint=scratch+3200;int j,k,s;
 for(j=0;j<800;++j){s=oa_sample(x,cotangent,adjoint,pixel*800+j);if(s)return s;for(k=0;k<4;++k)scratch[j*4+k]=x->weighted[(pixel*800+j)*4+k];}
 s=oc_reduce(x,pixel,scratch);return s?s:oa_reduce(x,pixel,adjoint,gradient);
}
#endif
