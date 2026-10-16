/* Qualification-only outward convolution. No renderer linkage or FMA. */
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <fenv.h>
#include <xmmintrin.h>
#pragma fenv_access(on)
typedef struct { int i,j; double lo,hi; } term;
static int valid(const term *a,int n,int order) {
 uint32_t seen=0;
 for(int t=0;t<n;t++) {
  if(a[t].i<0||a[t].i>order||a[t].j<0||a[t].j>1||!isfinite(a[t].lo)||!isfinite(a[t].hi)||a[t].lo>a[t].hi)return 0;
  uint32_t bit=1u<<(2*a[t].i+a[t].j);if(seen&bit)return 0;seen|=bit;
 }
 return 1;
}
__declspec(dllexport) int gffx_proof_convolve(int order,const term *a,int na,const term *b,int nb,double *out,uint32_t *mask) {
 if(!a||!b||!out||!mask||order<0||order>10||na<0||na>22||nb<0||nb>22)return 2;
 if(!valid(a,na,order)||!valid(b,nb,order))return 2;
 if(FLT_RADIX!=2||DBL_MANT_DIG!=53||fegetround()!=FE_TONEAREST||(_mm_getcsr()&0x8040))return 3;
 *mask=0;for(int k=0;k<44;k++)out[k]=0;
 for(int x=0;x<na;x++)for(int y=0;y<nb;y++) {
  int i=a[x].i+b[y].i,j=a[x].j+b[y].j;if(i<3||i>order||j>1)continue;
  if((a[x].lo==0&&a[x].hi==0)||(b[y].lo==0&&b[y].hi==0))continue;
  volatile double p0=a[x].lo*b[y].lo,p1=a[x].lo*b[y].hi,p2=a[x].hi*b[y].lo,p3=a[x].hi*b[y].hi;
  double lo=nextafter(fmin(fmin(p0,p1),fmin(p2,p3)),-INFINITY),hi=nextafter(fmax(fmax(p0,p1),fmax(p2,p3)),INFINITY);
  int key=2*i+j;uint32_t bit=1u<<key;
  if(*mask&bit){volatile double l=out[2*key]+lo,h=out[2*key+1]+hi;lo=nextafter(l,-INFINITY);hi=nextafter(h,INFINITY);}
  if(!isfinite(lo)||!isfinite(hi))return 1;
  out[2*key]=lo;out[2*key+1]=hi;*mask|=bit;
 }
 return 0;
}
