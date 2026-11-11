/* Polygon area-light integration. GGX fits: Heitz et al., SIGGRAPH 2016.
   The lookup data and redistribution notice accompany the Python package.
   This implementation clips polygons geometrically; no clipless horizon LUT. */
RS_INLINE void rs_cross(const float *a,const float *b,float *c){
 c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];
}
RS_INLINE int rs_clip(float poly[12][3],int count,const float *normal){
 float result[12][3];int n=0;
 for(int i=0;i<count;i++){
  const float *a=poly[i],*b=poly[(i+1)%count];float da=rs_dot(a,normal),db=rs_dot(b,normal);
  if(da>=0){for(int j=0;j<3;j++)result[n][j]=a[j];n++;}
  if((da>0&&db<0)||(da<0&&db>0)){float t=da/(da-db);for(int j=0;j<3;j++)result[n][j]=a[j]+t*(b[j]-a[j]);n++;}
 }
 for(int i=0;i<n;i++)for(int j=0;j<3;j++)poly[i][j]=result[i][j];return n;
}
RS_INLINE void rs_ltc_lookup(float rough,float nv,const float *table,float *fit){
 float x=rs_clamp(rough,0,1)*63,y=sqrtf(1-rs_clamp(nv,0,1))*63;
 int ix=(int)floorf(x),iy=(int)floorf(y),jx=ix<63?ix+1:63,jy=iy<63?iy+1:63;x-=ix;y-=iy;
 for(int k=0;k<8;k++)fit[k]=(1-y)*((1-x)*table[(iy*64+ix)*8+k]+x*table[(iy*64+jx)*8+k])+y*((1-x)*table[(jy*64+ix)*8+k]+x*table[(jy*64+jx)*8+k]);
}
RS_INLINE float rs_polygon(const float *p,const float *n,const float *gn,const float *v,const float *light,const float *fit){
 float poly[12][3],t1[3],t2[3],nv=rs_dot(n,v);
 for(int j=0;j<3;j++)t1[j]=v[j]-nv*n[j];
 if(rs_dot(t1,t1)<1e-10f){float axis[3]={0,0,0};axis[fabsf(n[2])<.9f?2:1]=1;rs_cross(axis,n,t1);}rs_unit(t1);rs_cross(n,t1,t2);
 const int sx[4]={-1,1,1,-1},sy[4]={-1,-1,1,1};
 for(int i=0;i<4;i++)for(int j=0;j<3;j++)poly[i][j]=light[20+j]-p[j]+.5f*(sx[i]*light[23+j]+sy[i]*light[26+j]);
 int count=rs_clip(poly,4,gn);
 for(int i=0;i<count;i++){
  float x=rs_dot(poly[i],t1),y=rs_dot(poly[i],t2),z=rs_dot(poly[i],n);
  poly[i][0]=fit?fit[0]*x+fit[2]*z:x;poly[i][1]=y;poly[i][2]=fit?fit[1]*x+fit[3]*z:z;
 }
 float horizon[3]={0,0,1};count=rs_clip(poly,count,horizon);if(count<3)return 0;
 for(int i=0;i<count;i++)rs_unit(poly[i]);float sum=0;
 for(int i=0;i<count;i++){
  const float *a=poly[i],*b=poly[(i+1)%count];float cross[3];rs_cross(a,b,cross);float length=sqrtf(rs_dot(cross,cross));
  if(length>1e-10f)sum+=cross[2]*atan2f(length,rs_clamp(rs_dot(a,b),-1,1))/length;
 }
 return rs_clamp(fabsf(sum)/(2*3.14159265358979323846f),0,1);
}
RS_INLINE void rs_area(const float *p,const float *n,const float *gn,const float *v,const float *f,const float *light,const float *table,const float *matrices,const float *depths,int size,float *out){
 float to_receiver[3];for(int j=0;j<3;j++)to_receiver[j]=p[j]-light[20+j];if(rs_dot(to_receiver,light+7)<=0)return;
 float visibility=1;int maps=(int)light[33],axis=(int)light[34];
 if(maps){float numerator=0,denominator=0;
  for(int i=0;i<axis*axis;i++){
   /* Use the actual packed source. Its map may use multi-jittered strata. */
   const float *source=light+i*40;float l[3];for(int j=0;j<3;j++)l[j]=source[1+j]-p[j];
   float d2=rs_dot(l,l);rs_unit(l);float w=fmaxf(0,rs_dot(n,l))*fmaxf(0,-rs_dot(l,light+7))/fmaxf(d2,1e-12f);
   denominator+=w;if(w>0)numerator+=w*rs_pcf((int)light[32]+i,p,gn,matrices,depths,size,f[8]);
  }visibility=denominator>0?numerator/denominator:0;
 }
 if(visibility<=0)return;
 float nv=rs_clamp(rs_dot(n,v),0,1),fit[8],coat_fit[8],base_integral,coat_integral=0;
 rs_ltc_lookup(f[3],nv,table,fit);base_integral=rs_polygon(p,n,gn,v,light,fit);
 float diffuse_integral=rs_polygon(p,n,gn,v,light,0);
 float dielectric=(f[5]-1)/(f[5]+1);dielectric*=dielectric;float atten=1,coat_energy=0;
 if(f[6]>0){rs_ltc_lookup(f[7],nv,table,coat_fit);coat_integral=rs_polygon(p,n,gn,v,light,coat_fit);coat_energy=f[6]*(.04f*coat_fit[4]+.96f*coat_fit[5]);
  float l[3];for(int j=0;j<3;j++)l[j]=light[20+j]-p[j];rs_unit(l);float nl=rs_clamp(rs_dot(n,l),0,1);
  atten=(1-f[6]*(.04f+.96f*powf(1-nv,5)))*(1-f[6]*(.04f+.96f*powf(1-nl,5)));
 }
 for(int j=0;j<3;j++){
  float f0=dielectric*(1-f[4])+f[j]*f[4];float energy=f0*fit[4]+(1-f0)*fit[5];
  out[j]+=light[29+j]*visibility*((energy*base_integral+(1-f[4])*(1-dielectric)*f[j]*diffuse_integral)*atten+coat_energy*coat_integral);
 }
}
