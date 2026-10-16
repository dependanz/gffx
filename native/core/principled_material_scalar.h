#ifndef GFFX_PRINCIPLED_MATERIAL_SCALAR_H
#define GFFX_PRINCIPLED_MATERIAL_SCALAR_H
/* Exactly two fixed specializations: primal and16-input first Jacobian. */
#include "principled_curve_scalar.h"
#include <math.h>
#define PM_DIM 16
#define PM_NAME(name) gffx_pm_ ## name
#include "principled_material_impl.inc"
#undef PM_NAME
#undef PM_DIM
#define PM_DIM 0
#define PM_NAME(name) gffx_pv_ ## name
#include "principled_material_impl.inc"
#undef PM_NAME
#undef PM_DIM

GFFX_PC_SCALAR double gffx_pp_rgb(const double *jac,const double *g,int k){return (jac[k]*g[0]+jac[16+k]*g[1])+jac[32+k]*g[2];}
GFFX_PC_SCALAR void gffx_pp_contract(int q,const double *jac,const double *g,double *out,double *scratch){
 int k,j;for(k=0;k<16;++k){if(k>=6&&k<9){out[k]=0.;continue;}for(j=0;j<q;++j)scratch[j]=gffx_pp_rgb(jac+48*j,g+3*j,k);out[k]=gffx_pc_sum(scratch,q);}
}
GFFX_PC_SCALAR void gffx_pp_pack(int p,int row,const double *normal,const double *view,const double *light,const double *albedo,const double *roughness,const double *f0,double *out){
 int k;for(k=0;k<3;++k){out[k]=normal[3*p+k];out[3+k]=view[3*p+k];out[6+k]=light[3*row+k];out[9+k]=albedo[3*p+k];out[13+k]=f0[3*p+k];}out[12]=roughness[p];
}
#endif
