#pragma once
#include <math.h>
#ifndef RS_INLINE
#define RS_INLINE static inline
#endif
RS_INLINE float rs_clamp(float x,float a,float b){return fminf(b,fmaxf(a,x));}
RS_INLINE float rs_dot(const float *a,const float *b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
RS_INLINE void rs_unit(float *a){float k=1.f/fmaxf(sqrtf(rs_dot(a,a)),1.17549435e-38f);for(int i=0;i<3;i++)a[i]*=k;}
RS_INLINE float rs_pcf(int id,const float *p,const float *n,const float *matrices,const float *depths,int size,float curvature){
 const float *m=matrices+id*64;float c[4];
 for(int j=0;j<4;j++)c[j]=m[j*4]*p[0]+m[j*4+1]*p[1]+m[j*4+2]*p[2]+m[j*4+3];
 if(c[3]<=0)return 1;float nx=c[0]/c[3],ny=c[1]/c[3],nz=c[2]/c[3];
 if(fabsf(nx)>1||fabsf(ny)>1||nz < -1||nz>1)return 1;
 float x=(nx+1)*size*.5f-.5f,y=(1-ny)*size*.5f-.5f,fx=x-floorf(x),fy=y-floorf(y),sum=0;
 int ix=(int)floorf(x),iy=(int)floorf(y);float cx=0,cy=0,cz=nz,wx[2]={1-fx,fx},wy[2]={1-fy,fy};
 if(m[51]!=0){float nv[3];for(int j=0;j<3;j++)nv[j]=n[0]*m[32+j*4]+n[1]*m[33+j*4]+n[2]*m[34+j*4];float D=n[0]*(m[52]-p[0])+n[1]*(m[53]-p[1])+n[2]*(m[54]-p[2]);if(fabsf(D)>1e-12f){cx=m[50]*nv[0]/(m[48]*D);cy=m[50]*nv[1]/(m[49]*D);cz=-m[50]*nv[2]/D;}else cz=0;}
 else {const float *inv=m+16;float plane[4],world[4]={n[0],n[1],n[2],-rs_dot(n,p)};for(int j=0;j<4;j++){plane[j]=0;for(int k=0;k<4;k++)plane[j]+=world[k]*inv[k*4+j];}if(fabsf(plane[2])>1e-12f){cx=-plane[0]/plane[2];cy=-plane[1]/plane[2];cz=-plane[3]/plane[2];}}
 float curve_factor=0,cap_base=0,slope_cap=.25f*(fabsf(cx)+fabsf(cy))*2/size;
 if(m[51]!=0&&curvature<=0&&(m[55]!=0||m[56]!=0||m[57]!=0||m[58]!=0)){
  float w_safe=fmaxf(c[3],1e-12f),ref0=-m[50]/w_safe,eta_mid=1.f/w_safe,eta_sum=0,eta_w=0;
  for(int iu=-2;iu<=2;iu++)for(int iv=-2;iv<=2;iv++){
   float su=iu*.625f,sv=iv*.625f;
   int sx=(int)floorf(x+eta_mid*(m[59]+su*m[55]+sv*m[57])+.5f);
   int sy=(int)floorf(y+eta_mid*(m[60]+su*m[56]+sv*m[58])+.5f);
   sx=sx<0?0:(sx>=size?size-1:sx);sy=sy<0?0:(sy>=size?size-1:sy);
   float zs=depths[(id*size+sy)*size+sx];
   float refs=cx*(2*(sx+.5f)/size-1)+cy*(1-2*(sy+.5f)/size)+cz,ec=(ref0-zs)/m[50];
   /* Relative reciprocal-depth separation, independent of near-plane scale. */
   float receiver_limit=fminf(ref0,refs),blocker_margin=2e-3f*fabsf(receiver_limit);
   if(isfinite(zs)&&zs<receiver_limit-blocker_margin&&ec>0){eta_sum+=ec;eta_w+=1.f;}
  }
  if(eta_w>0){
   float eta=eta_sum/eta_w,vsum=0;
   for(int ku=0;ku<6;ku++)for(int kv=0;kv<6;kv++){
    float xu=(ku+.5f)/6.f-.5f,xv=(kv+.5f)/6.f-.5f;
    float tx=x+eta*(m[59]+xu*m[55]+xv*m[57]),ty=y+eta*(m[60]+xu*m[56]+xv*m[58]);
    int tix=(int)floorf(tx),tiy=(int)floorf(ty);
    float tfx=tx-floorf(tx),tfy=ty-floorf(ty),twx[2]={1-tfx,tfx},twy[2]={1-tfy,tfy},tsum=0;
    for(int oy=0;oy<2;oy++)for(int ox=0;ox<2;ox++){
     int px=tix+ox,py=tiy+oy;px=px<0?0:(px>=size?size-1:px);py=py<0?0:(py>=size?size-1:py);
     float reference=cx*(2*(px+.5f)/size-1)+cy*(1-2*(py+.5f)/size)+cz;
     float bias=16*1.1920928955078125e-7f*fabsf(reference);
     tsum+=(reference<=depths[(id*size+py)*size+px]+bias?1:0)*twx[ox]*twy[oy];
    }
    vsum+=tsum*(1.f/36.f);
   }
   return rs_clamp(vsum,0,1);
  }
 }
 if(curvature>0){
  float nv[3];for(int j=0;j<3;j++)nv[j]=n[0]*m[32+j*4]+n[1]*m[33+j*4]+n[2]*m[34+j*4];
  float pixel=2*(m[51]!=0?c[3]:1)/fminf(m[48],m[49])/size;
  float scale=fminf(16,1+(nv[0]*nv[0]+nv[1]*nv[1])/fmaxf(nv[2]*nv[2],.01f));
  float error=curvature*pixel*pixel*scale;
  if(m[51]!=0){float D=n[0]*(m[52]-p[0])+n[1]*(m[53]-p[1])+n[2]*(m[54]-p[2]);curve_factor=error/fmaxf(fabsf(D),1e-12f);cap_base=.02f*pixel/c[3];}
  else {float zscale=sqrtf(m[8]*m[8]+m[9]*m[9]+m[10]*m[10]);curve_factor=error*zscale/fmaxf(fabsf(nv[2]),1e-3f);cap_base=.02f*pixel*zscale;}
 }
 /* Bilinear PCF with capped quadratic geometric receiver uncertainty. */
 for(int oy=0;oy<2;oy++)for(int ox=0;ox<2;ox++){
  int px=ix+ox,py=iy+oy;px=px<0?0:(px>=size?size-1:px);py=py<0?0:(py>=size?size-1:py);
  float reference=cx*(2*(px+.5f)/size-1)+cy*(1-2*(py+.5f)/size)+cz;
  float bias=16*1.1920928955078125e-7f*(m[51]!=0?fabsf(reference):1+fabsf(reference));
  if(curvature>0)bias+=fminf(m[51]!=0?fabsf(reference)*curve_factor:curve_factor,slope_cap+(m[51]!=0?fabsf(reference)*cap_base:cap_base));
  sum+=(reference<=depths[(id*size+py)*size+px]+bias?1:0)*wx[ox]*wy[oy];
 }return rs_clamp(sum,0,1);
}
#include "shade_area.h"
RS_INLINE void rs_shade(int point,int light_count,int size,int material_kind,const float *points,const float *normals,const float *geometric,const float *views,const float *fields,const float *lights,const float *matrices,const float *depths,const float *table,float *output){
 const float *p=points+point*3,*gn=geometric+point*3,*f=fields+point*12;float n[3],v[3],out[3]={0,0,0};
 for(int j=0;j<3;j++){n[j]=normals[point*3+j];v[j]=views[point*3+j];}rs_unit(n);rs_unit(v);
 float facing=rs_dot(gn,v);if(material_kind==2&&facing<0){for(int j=0;j<3;j++)n[j]=-n[j];facing=-facing;}
 if(facing>0)for(int li=0;li<light_count;li++){
  const float *light=lights+li*40;float l[3],incoming[3],distance=1;
  if(material_kind==0&&light[19]!=0){if(light[19]==1)rs_area(p,n,gn,v,f,light,table,matrices,depths,size,out);continue;}
  if(light[0]==2){for(int j=0;j<3;j++){l[j]=light[1+j];incoming[j]=light[4+j];}}
  else {for(int j=0;j<3;j++)l[j]=light[1+j]-p[j];distance=fmaxf(rs_dot(l,l),1e-12f);rs_unit(l);for(int j=0;j<3;j++)incoming[j]=light[4+j]/distance;}
  float nl=rs_clamp(rs_dot(n,l),0,1);if(nl<=0)continue;
  if(material_kind && rs_dot(gn,l)*(material_kind==2&&rs_dot(gn,v)<0?-1:1)<=0)continue;
  float weight=1;if(light[10]!=0)weight*=fmaxf(0,-(l[0]*light[7]+l[1]*light[8]+l[2]*light[9]));
  if(light[0]==1){float co=-(l[0]*light[11]+l[1]*light[12]+l[2]*light[13]);float t=light[14]==light[15]?(co>=light[14]?1:0):rs_clamp((co-light[14])/(light[15]-light[14]),0,1);weight*=t*t*(3-2*t);}
  int maps=(int)light[17],map=(int)light[16];if(maps==6){float d[3]={p[0]-light[1],p[1]-light[2],p[2]-light[3]};int axis=0;if(fabsf(d[1])>fabsf(d[axis]))axis=1;if(fabsf(d[2])>fabsf(d[axis]))axis=2;map+=axis*2+(d[axis]<0);}
  if(maps)weight*=rs_pcf(map,p,gn,matrices,depths,size,f[material_kind?3:8]);if(weight<=0)continue;
  float halfv[3]={v[0]+l[0],v[1]+l[1],v[2]+l[2]};rs_unit(halfv);float nh=rs_clamp(rs_dot(n,halfv),0,1);
  if(material_kind){float highlight=powf(nh,f[11])*(f[11]+8)/(8*3.14159265358979323846f);for(int j=0;j<3;j++)out[j]+=(f[j]/3.14159265358979323846f+f[8+j]*highlight)*incoming[j]*weight*nl;continue;}
  float nv=rs_clamp(rs_dot(n,v),0,1),vh=rs_clamp(rs_dot(v,halfv),0,1),a2=f[3]*f[3]*f[3]*f[3],den=nh*nh*(a2-1)+1;
  float d=a2/(3.14159265358979323846f*fmaxf(den*den,1e-20f));float vis=.5f/fmaxf(nl*sqrtf(nv*nv*(1-a2)+a2)+nv*sqrtf(nl*nl*(1-a2)+a2),1e-12f);
  float fres=powf(1-vh,5),dielectric=(f[5]-1)/(f[5]+1);dielectric*=dielectric;
  float coat=0,atten=1;if(f[6]!=0){float ca2=f[7]*f[7]*f[7]*f[7],cd=nh*nh*(ca2-1)+1;coat=ca2/(3.14159265358979323846f*fmaxf(cd*cd,1e-20f))*.5f/fmaxf(nl*sqrtf(nv*nv*(1-ca2)+ca2)+nv*sqrtf(nl*nl*(1-ca2)+ca2),1e-12f)*(.04f+.96f*fres)*f[6];atten=(1-f[6]*(.04f+.96f*powf(1-nv,5)))*(1-f[6]*(.04f+.96f*powf(1-nl,5)));}
  for(int j=0;j<3;j++){float f0=dielectric*(1-f[4])+f[j]*f[4],F=f0+(1-f0)*fres;float brdf=(d*vis*F+(1-f[4])*(1-F)*f[j]/3.14159265358979323846f)*atten+coat;out[j]+=brdf*incoming[j]*weight*nl;}
 }
 for(int j=0;j<3;j++)output[point*3+j]=out[j];
}
