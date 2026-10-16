/* Flat SH52 paired x/y tangents, with fully charged scalar fallback.
 * Both scratch layouts fit the same existing allocation. A failed trial never
 * publishes its prefix. Clear partial tangents before the scalar retry, also
 * on CUDA where the outer launch normally supplies zeroed outputs. */
#include "opaque_pair_scalar.h"
GFFX_OPAQUE_CLIP_LOCAL int opaque_batch_execute_prefix_shared(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *c,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,gffx_tensor_view *a,gffx_tensor_view *b,Scratch *s,int rule,int derivatives,int32_t *prefix){
 int spent=0;
 if(g&&rule>=2&&c->shape[1]==3&&(derivatives&1)){
  int result=pair_opaque_batch_execute_prefix_charged(v,f,c,ox,oy,cursor,receipt,g,a,b,(pair_Scratch*)s,rule,derivatives,prefix,0);
  if(!result)return 0;
  spent=receipt[7];if(spent<0||spent>2000000)return 2;
  gffx_tensor_view *outputs[2]={a,b};
  for(int k=0;k<2;++k)for(uint64_t q=0;q<gffx_mesh_element_count(outputs[k]);++q)write_real(outputs[k],(int64_t)q,0.);
 }
 return opaque_batch_execute_prefix_charged(v,f,c,ox,oy,cursor,receipt,g,a,b,s,rule,derivatives,prefix,spent);
}
GFFX_OPAQUE_LOCAL int opaque_batch_execute_shared(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *c,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,gffx_tensor_view *a,gffx_tensor_view *b,Scratch *s,int rule,int derivatives){return opaque_batch_execute_prefix_shared(v,f,c,ox,oy,cursor,receipt,g,a,b,s,rule,derivatives,NULL);}
