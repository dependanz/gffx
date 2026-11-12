#pragma once
/* Independently implemented normalized radial diffusion and single light-space
   transmission. No Blender code. radius is world-space RGB diffusion length. */
RS_INLINE void rss_diffuse(int i,int h,int w,float fx,float fy,const float *image,const float *data,const int *ids,float *out){
 const float *d=data+i*6;int x=i%w,y=i/w;
 for(int c=0;c<3;c++){
  if(ids[i]<0||d[0]<=0||d[1+c]<=0){out[i*3+c]=image[i*3+c];continue;}
  float sum=0,weight=0;
  for(int k=0;k<32;k++){
   /* Stratified inverse-CDF mixture: .25 exp(-r/d) + .75 exp(-r/(3d)). */
   float u=((k/4)+.5f)/8.f,r=-logf(1-u)*(k%4==0?1.f:3.f)*d[1+c];
   float angle=k*2.39996322972865332f;
   /* Private negative-fx marker selects orthographic pixel scale; depth
      remains world-space for bilateral rejection. Positive path is unchanged. */
   float px,py;if(fx>0){px=x+r*fx/d[0]*cosf(angle);py=y+r*fy/d[0]*sinf(angle);}
   else{px=x-r*fx*cosf(angle);py=y+r*fy*sinf(angle);}
   int ix=(int)floorf(px),iy=(int)floorf(py);float ax=px-ix,ay=py-iy;
   for(int oy=0;oy<2;oy++)for(int ox=0;ox<2;ox++){
    int qx=ix+ox,qy=iy+oy;if(qx<0||qx>=w||qy<0||qy>=h)continue;
    int q=qy*w+qx;if(ids[q]!=ids[i]||data[q*6]<=0)continue;
    float a=(ox?ax:1-ax)*(oy?ay:1-ay);
    a*=expf(-fabsf(data[q*6]-d[0])/fmaxf(d[1+c]*2,1e-8f))*data[q*6+5];
    sum+=a*image[q*3+c];weight+=a;
   }
  }
  out[i*3+c]=weight>1e-12f?sum/weight:image[i*3+c];
 }
}
RS_INLINE float rss_path(int id,const float *p,const float *matrices,const float *depths,int size){
 const float *m=matrices+id*64;float c[4];
 for(int j=0;j<4;j++)c[j]=m[j*4]*p[0]+m[j*4+1]*p[1]+m[j*4+2]*p[2]+m[j*4+3];
 if(c[3]<=0)return 1e20f;float nx=c[0]/c[3],ny=c[1]/c[3];
 if(fabsf(nx)>1||fabsf(ny)>1)return 1e20f;
 int x=(int)((nx+1)*size*.5f),y=(int)((1-ny)*size*.5f);x=x>=size?size-1:x;y=y>=size?size-1:y;
 float z=depths[(id*size+y)*size+x];if(!isfinite(z))return 1e20f;
 if(m[51]!=0){
  float receiver=c[3],entry=-m[50]/z;
  float rx=nx/m[48],ry=ny/m[49];return fmaxf(0,receiver-entry)*sqrtf(1+rx*rx+ry*ry);
 }
 const float *inv=m+16;float a[4],b[4];
 for(int j=0;j<4;j++){a[j]=inv[j*4]*nx+inv[j*4+1]*ny+inv[j*4+2]*z+inv[j*4+3];b[j]=inv[j*4]*nx+inv[j*4+1]*ny+inv[j*4+2]*(c[2]/c[3])+inv[j*4+3];}
 float r=0;for(int j=0;j<3;j++){float t=a[j]/a[3]-b[j]/b[3];r+=t*t;}return sqrtf(r);
}
RS_INLINE void rss_transmit(int i,int count,int size,const float *points,const float *normals,const float *fields,const float *lights,const float *matrices,const float *depths,float *out){
 const float *p=points+i*3,*n=normals+i*3,*f=fields+i*12;float value[3]={0,0,0};
 for(int li=0;li<count;li++){
  const float *l=lights+li*40;int maps=(int)l[17],map=(int)l[16];if(!maps)continue;
  float d[3],incoming[3],dist=1;
  if(l[0]==2){for(int c=0;c<3;c++){d[c]=l[1+c];incoming[c]=l[4+c];}}
  else {for(int c=0;c<3;c++)d[c]=l[1+c]-p[c];dist=fmaxf(rs_dot(d,d),1e-12f);rs_unit(d);for(int c=0;c<3;c++)incoming[c]=l[4+c]/dist;}
  float back=fmaxf(0,-rs_dot(n,d));if(back<=0)continue;float weight=back;
  if(l[10]!=0)weight*=fmaxf(0,-(d[0]*l[7]+d[1]*l[8]+d[2]*l[9]));
  if(l[0]==1){float co=-(d[0]*l[11]+d[1]*l[12]+d[2]*l[13]);float t=l[14]==l[15]?(co>=l[14]?1:0):rs_clamp((co-l[14])/(l[15]-l[14]),0,1);weight*=t*t*(3-2*t);}
  if(maps==6){int axis=0;if(fabsf(d[1])>fabsf(d[axis]))axis=1;if(fabsf(d[2])>fabsf(d[axis]))axis=2;map+=axis*2+(d[axis]>0);}
  float path=rss_path(map,p,matrices,depths,size);
  for(int c=0;c<3;c++)value[c]+=incoming[c]*weight*f[c]*expf(-path/fmaxf(f[8+c],1e-8f))/3.14159265358979323846f*f[11];
 }
 for(int c=0;c<3;c++)out[i*3+c]=value[c];
}
