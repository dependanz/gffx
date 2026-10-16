/* Development-only C11 moment prototype. No renderer admission or ABI change.
 * Exact cell polynomials times a bounded normal-reciprocal series.
 * All zero skips require BOTH value and tangent to be exactly zero.
 */
#include <math.h>
#include <stddef.h>
#include <string.h>
#define W 21
#define EXPORT __declspec(dllexport)
typedef struct { double v,d; } Dual;
typedef struct { int n; Dual a[W][W]; } Poly;
static Dual plus(Dual a,Dual b){return (Dual){a.v+b.v,a.d+b.d};}
static Dual times(Dual a,Dual b){return (Dual){a.v*b.v,a.d*b.v+a.v*b.d};}
static Dual scale(Dual a,double s){return (Dual){a.v*s,a.d*s};}
static int zero(Dual x){return x.v==0 && x.d==0;}
static void init(Poly *p,int n){memset(p,0,sizeof(*p));p->n=n;}
static void affine(Poly *p,const double *a,const double *d,int stride,int component){
 init(p,1);p->a[0][0]=(Dual){a[component],d[component]};
 p->a[1][0]=(Dual){a[stride+component]-a[component],d[stride+component]-d[component]};
 p->a[0][1]=(Dual){a[2*stride+component]-a[component],d[2*stride+component]-d[component]};
}
static void mul(Poly *out,const Poly *p,const Poly *q){
 init(out,p->n+q->n);
 for(int i=0;i<=p->n;i++)for(int j=0;j<=p->n-i;j++){
  Dual x=p->a[i][j];if(zero(x))continue;
  for(int k=0;k<=q->n;k++)for(int l=0;l<=q->n-k;l++){
   Dual y=q->a[k][l];if(zero(y))continue;
   out->a[i+k][j+l]=plus(out->a[i+k][j+l],times(x,y));
  }
 }
}
static void add_scaled(Poly *out,const Poly *p,Dual s){
 if(out->n<p->n)out->n=p->n;
 for(int i=0;i<=p->n;i++)for(int j=0;j<=p->n-i;j++)out->a[i][j]=plus(out->a[i][j],times(p->a[i][j],s));
}
static Dual integral(const Poly *p,const double m[W][W]){
 Dual s={0,0};for(int i=0;i<=p->n;i++)for(int j=0;j<=p->n-i;j++)s=plus(s,scale(p->a[i][j],m[i][j]));return s;
}
static Dual product_integral(const Poly *p,const Poly *q,const double m[W][W]){
 Dual s={0,0};
 for(int i=0;i<=p->n;i++)for(int j=0;j<=p->n-i;j++){
  Dual x=p->a[i][j];if(zero(x))continue;
  for(int k=0;k<=q->n;k++)for(int l=0;l<=q->n-k;l++){
   Dual y=q->a[k][l];if(zero(y))continue;
   s=plus(s,scale(times(x,y),m[i+k][j+l]));
  }
 }
 return s;
}
/* Packed fields: normal9 UV6 tex12 SH27 det1 =55; then55 tangents.
 * q0/degree are validated externally by the directed-interval obligation.
 */
EXPORT int sh_moments(const double *a,int degree,double q0,double *out){
 if(degree<0||degree>8||!isfinite(q0)||q0<=0)return 1;
 for(int i=0;i<110;i++)if(!isfinite(a[i]))return 2;
 if(a[54]<=0)return 3;
 const double *d=a+55;
 Poly n[3],s,t,st,q,power,tmp,next,albedo,lin,quad,al,aq;
 double m[W][W]={{0}},fac[W+2];fac[0]=1;
 for(int i=1;i<W+2;i++)fac[i]=fac[i-1]*i;
 for(int i=0;i<W;i++)for(int j=0;j<W-i;j++)m[i][j]=fac[i]*fac[j]/fac[i+j+2];
 for(int k=0;k<3;k++)affine(&n[k],a,d,3,k);
 affine(&s,a+9,d+9,2,0);affine(&t,a+9,d+9,2,1);mul(&st,&s,&t);
 init(&q,2);
 for(int k=0;k<3;k++){mul(&tmp,&n[k],&n[k]);add_scaled(&q,&tmp,(Dual){1/q0,0});}
 q.a[0][0].v-=1;
 for(int c=0;c<3;c++){
  Dual tex[4],sh[9];for(int k=0;k<4;k++)tex[k]=(Dual){a[15+3*k+c],d[15+3*k+c]};
  for(int k=0;k<9;k++)sh[k]=(Dual){a[27+3*k+c],d[27+3*k+c]};
  init(&albedo,2);albedo.a[0][0]=tex[0];
  add_scaled(&albedo,&s,plus(tex[1],scale(tex[0],-1)));
  add_scaled(&albedo,&t,plus(tex[2],scale(tex[0],-1)));
  add_scaled(&albedo,&st,plus(plus(tex[3],tex[0]),scale(plus(tex[1],tex[2]),-1)));
  init(&lin,1);for(int k=0;k<3;k++)add_scaled(&lin,&n[k],sh[k+1]);
  init(&quad,2);
  const int ix[6]={0,1,2,0,0,1},iy[6]={1,2,2,2,0,1},ic[6]={4,5,6,7,8,8};
  const double sc[6]={1,1,3,1,1,-1};
  for(int k=0;k<6;k++){mul(&tmp,&n[ix[k]],&n[iy[k]]);add_scaled(&quad,&tmp,scale(sh[ic[k]],sc[k]));}
  mul(&al,&albedo,&lin);mul(&aq,&albedo,&quad);
  Dual result=times(integral(&albedo,m),plus(sh[0],scale(sh[6],-1)));
  init(&power,0);power.a[0][0].v=1;double half=1,inv=1;
  for(int k=0;k<=degree;k++){
   result=plus(result,scale(product_integral(&al,&power,m),half/sqrt(q0)));
   result=plus(result,scale(product_integral(&aq,&power,m),inv/q0));
   if(k<degree){mul(&next,&power,&q);power=next;half*=(-2.*k-1)/(2.*k+2);inv=-inv;}
  }
  result=times(result,(Dual){a[54],d[54]});out[c]=result.v;out[c+3]=result.d;
  if(!isfinite(result.v)||!isfinite(result.d))return 4;
 }
 return 0;
}
EXPORT int sh_gauss_rule(const double *a,int order,double *out){
 if(order<3||order>5)return 1;
 for(int k=0;k<110;k++)if(!isfinite(a[k]))return 2;
 if(a[54]<=0)return 3;
 const double *d=a+55;
 double nodes[5]={.0469100770306680036,.2307653449471584545,.5,.7692346550528415455,.9530899229693319964};
 double weights[5]={.1184634425280945438,.2393143352496832340,.2844444444444444444,.2393143352496832340,.1184634425280945438};
 if(order==3){double x=sqrt(3./5.);nodes[0]=(1-x)/2;nodes[1]=.5;nodes[2]=(1+x)/2;weights[0]=weights[2]=5./18.;weights[1]=4./9.;}
 if(order==4){double x=sqrt((3+2*sqrt(6./5.))/7),y=sqrt((3-2*sqrt(6./5.))/7);nodes[0]=(1-x)/2;nodes[1]=(1-y)/2;nodes[2]=(1+y)/2;nodes[3]=(1+x)/2;weights[0]=weights[3]=(18-sqrt(30.))/72;weights[1]=weights[2]=(18+sqrt(30.))/72;}
 for(int k=0;k<6;k++)out[k]=0;
 for(int i=0;i<order;i++)for(int j=0;j<order;j++){
  double u=nodes[i],v=nodes[j],beta[3]={(1-u)*(1-v),u,(1-u)*v};Dual n[3]={{0}},uv[2]={{0}},tex[4][3],sh[9][3];
  for(int k=0;k<3;k++)for(int r=0;r<3;r++)n[k]=plus(n[k],scale((Dual){a[3*r+k],d[3*r+k]},beta[r]));
  for(int k=0;k<2;k++)for(int r=0;r<3;r++)uv[k]=plus(uv[k],scale((Dual){a[9+2*r+k],d[9+2*r+k]},beta[r]));
  Dual q={0};for(int k=0;k<3;k++)q=plus(q,times(n[k],n[k]));double len=sqrt(q.v);Dual inv={1/len,-.5*q.d/(q.v*len)};
  if(!isfinite(inv.v)||!isfinite(inv.d))return 4;
 for(int k=0;k<3;k++)n[k]=times(n[k],inv);
  for(int k=0;k<4;k++)for(int c=0;c<3;c++)tex[k][c]=(Dual){a[15+3*k+c],d[15+3*k+c]};
  for(int k=0;k<9;k++)for(int c=0;c<3;c++)sh[k][c]=(Dual){a[27+3*k+c],d[27+3*k+c]};
  Dual b[9]={{1,0},n[0],n[1],n[2],times(n[0],n[1]),times(n[1],n[2]),plus(scale(times(n[2],n[2]),3),(Dual){-1,0}),times(n[0],n[2]),plus(times(n[0],n[0]),scale(times(n[1],n[1]),-1))};
  for(int c=0;c<3;c++){
   Dual colour=tex[0][c];colour=plus(colour,times(uv[0],plus(tex[1][c],scale(tex[0][c],-1))));colour=plus(colour,times(uv[1],plus(tex[2][c],scale(tex[0][c],-1))));
   colour=plus(colour,times(times(uv[0],uv[1]),plus(plus(tex[3][c],tex[0][c]),scale(plus(tex[1][c],tex[2][c]),-1))));
   Dual light={0};for(int k=0;k<9;k++)light=plus(light,times(b[k],sh[k][c]));
   Dual result=scale(times(times(colour,light),(Dual){a[54],d[54]}),weights[i]*weights[j]*(1-u));out[c]+=result.v;out[c+3]+=result.d;
  }
 }
 for(int k=0;k<6;k++)if(!isfinite(out[k]))return 5;
 return 0;
}
/* Repeated calls for bounded same-compiler timing without Python call overhead. */
EXPORT int sh_repeat(const double *a,int degree,double q0,int mode,int count,double *out){
 if(count<1||count>10000)return 1;
 for(int i=0;i<count;i++){int status=mode==1?sh_moments(a,degree,q0,out):sh_gauss_rule(a,mode==0?5:mode,out);if(status)return status;}
 return 0;
}
