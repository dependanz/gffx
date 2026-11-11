#pragma once
/* Owned image-lighting data, shared C11/CUDA scalar evaluation. */
RS_INLINE void ri_bilinear(const float *image,int width,int height,float x,float y,int channels,float *out){
 int ix=(int)floorf(x),iy=(int)floorf(y);float fx=x-ix,fy=y-iy;
 for(int c=0;c<channels;c++)out[c]=0;
 for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){
  int px=ix+dx,py=iy+dy;px=px<0?0:(px>=width?width-1:px);py=py<0?0:(py>=height?height-1:py);
  float weight=(dx?fx:1-fx)*(dy?fy:1-fy);
  for(int c=0;c<channels;c++)out[c]+=weight*image[(py*width+px)*channels+c];
 }
}
RS_INLINE void ri_cube(int cube,float rough,const float *direction,const int *info,const float *images,float *out){
 const int *row=info+cube*32;for(int c=0;c<3;c++)out[c]=0;if(!row[0])return;
 int axis=0;if(fabsf(direction[1])>fabsf(direction[axis]))axis=1;if(fabsf(direction[2])>fabsf(direction[axis]))axis=2;
 int face=axis*2+(direction[axis]<0);float z=fmaxf(fabsf(direction[axis]),1e-20f),x,y;
 if(axis==0){x=(face==0?-direction[1]:direction[1])/z;y=direction[2]/z;}
 else if(axis==1){x=(face==2?direction[0]:-direction[0])/z;y=direction[2]/z;}
 else{x=(face==4?-direction[0]:direction[0])/z;y=direction[1]/z;}
 float lod=rs_clamp(rough,0,1)*(row[0]-1),fraction=lod-floorf(lod);int lo=(int)floorf(lod);
 for(int l=0;l<2;l++){
  int level=lo+l;if(level>=row[0])continue;int side=row[2+level*2],size=side-2;float value[3];
  ri_bilinear(images+row[1+level*2]*3,side,side*6,(x+1)*size*.5f+.5f,face*side+(1-y)*size*.5f+.5f,3,value);
  for(int c=0;c<3;c++)out[c]+=value[c]*(l?fraction:1-fraction);
 }
}
RS_INLINE void ri_sh(const float *n,float *b){
 float x=n[0],y=n[1],z=n[2];b[0]=.28209479177387814f;b[1]=.4886025119029199f*y;b[2]=.4886025119029199f*z;b[3]=.4886025119029199f*x;b[4]=1.0925484305920792f*x*y;b[5]=1.0925484305920792f*y*z;b[6]=.31539156525252005f*(3*z*z-1);b[7]=1.0925484305920792f*x*z;b[8]=.5462742152960396f*(x*x-y*y);
}
RS_INLINE void ri_parallax(const float *p,const float *d,const float *meta,float *out){
 int inside=meta[9]!=0;float t=1e30f;
 for(int c=0;c<3;c++){
  if(p[c]<meta[3+c]||p[c]>meta[6+c])inside=0;
  float safe=fabsf(d[c])>1e-10f?d[c]:1e-10f;
  t=fminf(t,fmaxf((meta[3+c]-p[c])/safe,(meta[6+c]-p[c])/safe));
 }
 for(int c=0;c<3;c++)out[c]=inside?p[c]+d[c]*fmaxf(t,0)-meta[c]:d[c];rs_unit(out);
}
RS_INLINE void ri_shade(int i,int cubes,int mirror_count,int classic,const float *points,const float *normals,const float *views,const float *fields,const int *cube_info,const float *cube_images,const float *probe_meta,const float *env_sh,const float *gi,const float *mirror_info,const float *mirror_images,const int *mirror_ids,const float *dfg,float *output){
 const float *p=points+i*3,*n=normals+i*3,*v=views+i*3,*f=fields+i*12;float b[9],incoming[3]={0,0,0};ri_sh(n,b);
 for(int j=0;j<9;j++)for(int c=0;c<3;c++)incoming[c]+=b[j]*env_sh[j*3+c];
 int within=gi[0]>0;for(int c=0;c<3;c++)if(p[c]<gi[3+c]||p[c]>gi[6+c])within=0;
 if(within){
  int lo[3],size[3]={(int)gi[0],(int)gi[1],(int)gi[2]};float frac[3];for(int c=0;c<3;c++){float q=rs_clamp((p[c]+n[c]*gi[9]-gi[3+c])/(gi[6+c]-gi[3+c])*(size[c]-1),0,(float)(size[c]-1));lo[c]=(int)floorf(q);frac[c]=q-lo[c];}
  float plain[3]={0,0,0},total=0;for(int c=0;c<3;c++)incoming[c]=0;
  for(int dx=0;dx<2;dx++)for(int dy=0;dy<2;dy++)for(int dz=0;dz<2;dz++){
   int x=lo[0]+dx,y=lo[1]+dy,z=lo[2]+dz;if(x>=size[0])x=size[0]-1;if(y>=size[1])y=size[1]-1;if(z>=size[2])z=size[2]-1;
   float w=(dx?frac[0]:1-frac[0])*(dy?frac[1]:1-frac[1])*(dz?frac[2]:1-frac[2]);const float *sh=gi+16+((x*size[1]+y)*size[2]+z)*27;
   if(gi[10]>0){
    float value[3]={0,0,0};for(int j=0;j<9;j++)for(int c=0;c<3;c++)value[c]+=b[j]*sh[j*3+c];
    for(int c=0;c<3;c++)plain[c]+=w*value[c];
    int index[3]={x,y,z};float ray[3];for(int c=0;c<3;c++)ray[c]=gi[3+c]+index[c]*(gi[6+c]-gi[3+c])/(size[c]-1)-p[c];
    float distance=sqrtf(rs_dot(ray,ray));float facing=distance<1e-8f?1:fmaxf(rs_dot(n,ray)/fmaxf(distance,1e-10f),0);w*=facing*facing;
    for(int c=0;c<3;c++)incoming[c]+=w*value[c];total+=w;
   }else{for(int j=0;j<9;j++)for(int c=0;c<3;c++)incoming[c]+=w*b[j]*sh[j*3+c];}
  }
  if(gi[10]>0)for(int c=0;c<3;c++)incoming[c]=total>1e-10f?incoming[c]/total:plain[c];
 }

 float nv=rs_clamp(rs_dot(n,v),0,1),fc=powf(1-nv,5),f0[3],out[3],dielectric=(f[5]-1)/(f[5]+1);dielectric*=dielectric;
 for(int c=0;c<3;c++){f0[c]=dielectric*(1-f[4])+f[c]*f[4];float F=f0[c]+(1-f0[c])*fc;out[c]=f[c]*(classic?1:(1-f[4])*(1-F))*fmaxf(incoming[c],0)/3.14159265358979323846f;}
 if(!classic){
  float direction[3];for(int c=0;c<3;c++)direction[c]=2*nv*n[c]-v[c];rs_unit(direction);
  int choice=0;float best=1e30f;for(int cube=1;cube<cubes;cube++){const float *meta=probe_meta+cube*12;float d=0;for(int c=0;c<3;c++){float a=p[c]-meta[c];d+=a*a;}if(d<best){best=d;choice=cube;}}
  float ray[3];if(choice)ri_parallax(p,direction,probe_meta+choice*12,ray);else for(int c=0;c<3;c++)ray[c]=direction[c];
  float radiance[3],ab[2];ri_cube(choice,f[3],ray,cube_info,cube_images,radiance);
  int mirror=mirror_ids[i];if(mirror>=0&&mirror<mirror_count){const float *m=mirror_info+mirror*20;float clip[4];for(int j=0;j<4;j++)clip[j]=m[j*4]*p[0]+m[j*4+1]*p[1]+m[j*4+2]*p[2]+m[j*4+3];float x=clip[0]/clip[3],y=clip[1]/clip[3];if(clip[3]>0&&fabsf(x)<=1&&fabsf(y)<=1)ri_bilinear(mirror_images+(int)m[18]*3,(int)m[16],(int)m[17],(x+1)*m[16]*.5f-.5f,(1-y)*m[17]*.5f-.5f,3,radiance);}
  ri_bilinear(dfg,64,64,nv*64-.5f,f[3]*64-.5f,2,ab);for(int c=0;c<3;c++)out[c]+=radiance[c]*(f0[c]*ab[0]+ab[1]);
  if(f[6]>0){ri_cube(choice,f[7],ray,cube_info,cube_images,radiance);ri_bilinear(dfg,64,64,nv*64-.5f,f[7]*64-.5f,2,ab);float cf=.04f+.96f*fc;for(int c=0;c<3;c++)out[c]=out[c]*(1-f[6]*cf)+f[6]*radiance[c]*(.04f*ab[0]+ab[1]);}
 }
 for(int c=0;c<3;c++)output[i*3+c]=out[c];
}
