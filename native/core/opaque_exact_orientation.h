/* Prototype-only exact sign fallback. Error-free binary64 expansions under
 * round-to-nearest, no contraction/FTZ; refuses unsafe exponent ranges.
 * A returned 2 is unknown, never a separating certificate. No epsilon cutoff. */
#ifndef GFFX_OPAQUE_EXACT_ORIENTATION_H
#define GFFX_OPAQUE_EXACT_ORIENTATION_H
GFFX_OPAQUE_LOCAL void orient_sum(double a,double b,double *hi,double *lo){
 double x=a+b,bv=x-a,av=x-bv,br=b-bv,ar=a-av;*hi=x;*lo=ar+br;
}
GFFX_OPAQUE_LOCAL int orient_safe(double x){return isfinite(x)&&(x==0.||(fabs(x)>=0x1p-450&&fabs(x)<=0x1p450));}
GFFX_OPAQUE_LOCAL void orient_product(double a,double b,double *hi,double *lo){
 double ca=134217729.*a,cb=134217729.*b,ah=ca-(ca-a),bh=cb-(cb-b),al=a-ah,bl=b-bh;
 double x=a*b,e1=x-ah*bh,e2=e1-al*bh,e3=e2-ah*bl;*hi=x;*lo=al*bl-e3;
}
GFFX_OPAQUE_LOCAL int orient_grow(double *e,int n,double scalar){
 double q=scalar;int i,m=0;
 for(i=0;i<n;++i){double h,l;orient_sum(q,e[i],&h,&l);if(l!=0.)e[m++]=l;q=h;}
 if(q!=0.||!m)e[m++]=q;return m;
}
GFFX_OPAQUE_LOCAL int orient_exact(double ax,double ay,double bx,double by,double cx,double cy){
 double u[2],v[2],w[2],z[2],e[32]={0.};int i,j,n=1;
 if(!orient_safe(ax)||!orient_safe(ay)||!orient_safe(bx)||!orient_safe(by)||!orient_safe(cx)||!orient_safe(cy))return 2;
 orient_sum(bx,-ax,&u[1],&u[0]);orient_sum(cy,-ay,&v[1],&v[0]);
 orient_sum(by,-ay,&w[1],&w[0]);orient_sum(cx,-ax,&z[1],&z[0]);
 for(i=0;i<2;++i)if(!orient_safe(u[i])||!orient_safe(v[i])||!orient_safe(w[i])||!orient_safe(z[i]))return 2;
 for(i=0;i<2;++i)for(j=0;j<2;++j){double h,l;orient_product(u[i],v[j],&h,&l);n=orient_grow(e,n,l);n=orient_grow(e,n,h);orient_product(w[i],z[j],&h,&l);n=orient_grow(e,n,-l);n=orient_grow(e,n,-h);}
 return e[n-1]>0.?1:(e[n-1]<0.?-1:0);
}
GFFX_OPAQUE_LOCAL int exact_separated_edge(const gffx_tensor_view *v,const int32_t *a,const int32_t *b){
 int orientation=orient_exact(read_real(v,3*(int64_t)a[0]),read_real(v,3*(int64_t)a[0]+1),read_real(v,3*(int64_t)a[1]),read_real(v,3*(int64_t)a[1]+1),read_real(v,3*(int64_t)a[2]),read_real(v,3*(int64_t)a[2]+1));
 int k,j;if(orientation==0||orientation==2)return 0;
 for(k=0;k<3;++k){int32_t x=a[k],y=a[(k+1)%3];int outside=1,strict=0;
  for(j=0;j<3;++j){int sign=orient_exact(read_real(v,3*(int64_t)x),read_real(v,3*(int64_t)x+1),read_real(v,3*(int64_t)y),read_real(v,3*(int64_t)y+1),read_real(v,3*(int64_t)b[j]),read_real(v,3*(int64_t)b[j]+1));
   if(sign==2||sign==orientation){outside=0;break;}if(sign==-orientation)strict=1;
  }
  if(outside&&strict)return 1;
 }
 return 0;
}
#endif
