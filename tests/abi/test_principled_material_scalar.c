/* Standalone prospective complete-pair probe; not linked into installed core. */
#include "principled_material_scalar.h"
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
EXPORT int principled_test_material_point(int n,const double *cn,const double *logs,const double *curve,const double *partial,const double *mean,double width,const double *input,int two_sided,double *out){
 int i;if(n<1||n>512||!cn||!logs||!curve||!partial||!mean||!input||!out||!isfinite(width)||width<=0.||(two_sided!=0&&two_sided!=1))return 1;
 for(i=0;i<16;++i)if(!isfinite(input[i]))return 1;
 if(input[12]<=0.||input[12]>1.)return 1;for(i=9;i<12;++i)if(input[i]<0.||input[i]>1.)return 1;for(i=13;i<16;++i)if(input[i]<0.||input[i]>1.)return 1;
 return gffx_pm_point(n,cn,logs,curve,partial,mean,width,input,two_sided,out);
}
