/* Shared bounded single-pixel continuation. No allocation or hidden state.
 * Requires opaque_quadrature_scalar.h; all replay shares the clipping counter. */
#include "opaque_exact_orientation.h"
/* Compare numerical fields, never C/CUDA struct padding or ABI return copies. */
GFFX_OPAQUE_LOCAL int opaque_dual_same(D a,D b){return a.v==b.v&&a.d==b.d&&signbit(a.v)==signbit(b.v)&&signbit(a.d)==signbit(b.d);}
GFFX_OPAQUE_LOCAL int opaque_line_same(Line a,Line b){return opaque_dual_same(a.a,b.a)&&opaque_dual_same(a.b,b.b)&&opaque_dual_same(a.c,b.c);}
GFFX_OPAQUE_LOCAL int opaque_triangle_same(ColourTri a,ColourTri b){int k;if(a.tri.valid!=b.tri.valid||!opaque_line_same(a.depth,b.depth)||!opaque_dual_same(a.determinant,b.determinant))return 0;
 for(k=0;k<3;++k)if(!opaque_line_same(a.tri.edge[k],b.tri.edge[k])||!opaque_line_same(a.bary[k],b.bary[k])||!opaque_dual_same(a.vertex[k].x,b.vertex[k].x)||!opaque_dual_same(a.vertex[k].y,b.vertex[k].y))return 0;return 1;}
/* Bits 0..2: unchanged edges; 3: unchanged depth; 4: all owner geometry.
 * Full dual equality still controls restoration between adjacent seeds. */
GFFX_OPAQUE_LOCAL int opaque_triangle_edge_mask(ColourTri a,ColourTri b){int k,mask=0,geometry=opaque_dual_same(a.determinant,b.determinant);
 if(a.tri.valid!=b.tri.valid)return 0;
 for(k=0;k<3;++k){
  if(opaque_line_same(a.tri.edge[k],b.tri.edge[k]))mask|=1<<k;
  else geometry=0;
  if(!opaque_line_same(a.bary[k],b.bary[k])||!opaque_dual_same(a.vertex[k].x,b.vertex[k].x)||!opaque_dual_same(a.vertex[k].y,b.vertex[k].y))geometry=0;
 }
 if(opaque_line_same(a.depth,b.depth))mask|=8;
 if(geometry)mask|=16;
 return mask;
}
GFFX_OPAQUE_CLIP_LOCAL int opaque_batch_walk(Scratch *s,int n,const gffx_tensor_view *v,const int32_t *faces,const gffx_tensor_view *cuts,double ox,double oy,int64_t seed,int64_t cut_seed,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,gffx_tensor_view *a,gffx_tensor_view *b,double *value,Ctx *ctx,int rule){
 QuadBatch batch;int j,curves,count;
 if(quad_batch_init_profile(&batch,cursor[0],cursor[1],cursor[2],cursor[3],rule==4?80:(rule==3?64:(rule==2?52:16))))return 2;
 /* A seeded walk only changes triangles incident to its vertex. Restore
  * the preceding seed's changed triangles and construct the current ones;
  * all other stored dual triangles are still exactly the baseline. */
 if(rule>=2 && g && s->reuse.mode==2){
  for(j=0;j<n;++j){
   int affected=0;for(int corner=0;corner<3;++corner)if(seed>=0 && seed/3==faces[3*s->ids[j]+corner])affected=1;
   if(affected || !s->reuse.unchanged[j])s->triangles[j]=(rule==4||n>CANDIDATES)?triangle_shared_edges(v,faces+3*s->ids[j],seed,ox,oy):triangle(v,faces+3*s->ids[j],seed,ox,oy);
   if(!affected){s->reuse.unchanged[j]=1;s->reuse.geometry_unchanged[j]=31;}
   else {ColourTri baseline=(rule==4||n>CANDIDATES)?triangle_shared_edges(v,faces+3*s->ids[j],-1,ox,oy):triangle(v,faces+3*s->ids[j],-1,ox,oy);s->reuse.unchanged[j]=opaque_triangle_same(s->triangles[j],baseline);s->reuse.geometry_unchanged[j]=opaque_triangle_edge_mask(s->triangles[j],baseline);}
  }
 }else {
  for(j=0;j<n;++j){s->triangles[j]=(rule==4||(rule!=0&&n>CANDIDATES))?triangle_shared_edges(v,faces+3*s->ids[j],seed,ox,oy):triangle(v,faces+3*s->ids[j],seed,ox,oy);if(rule>=2&&g){s->reuse.unchanged[j]=1;s->reuse.geometry_unchanged[j]=31;}}
 }

 quadrature_pixel_ex(s,n,cuts,cut_seed,0,12,rule!=0?5:(quad_linear_count(cuts)==0?-44:8),&curves,&count,a,b,g,value,ctx,&batch,NULL,rule>=2&&g?&s->reuse:NULL);
 if(ctx->failed||quad_batch_finish(&batch)||count!=batch.slots)return 2;
 receipt[0]=batch.seen;receipt[1]=!batch.full;receipt[2]=batch.slots;receipt[3]=batch.blocks;receipt[4]=batch.total_slots;receipt[5]=batch.total_blocks;receipt[6]=batch.emitted;receipt[7]=ctx->work;return 0;
}
GFFX_OPAQUE_CLIP_LOCAL int opaque_batch_execute_prefix_charged(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,gffx_tensor_view *out,gffx_tensor_view *other,Scratch *s,int rule,int derivatives,int32_t *prefix,int initial_work){
 const int32_t *faces=(const int32_t*)gffx_mesh_elements_const(f);int32_t first[8],next[8];int n=0,i,j,k,nv=0;int64_t vertices[3*SH_CANDIDATES];double value;Ctx ctx={0,initial_work};uint64_t q;gffx_tensor_view *outputs[2]={out,other};
#ifndef GFFX_OPAQUE_PRECLEARED
 for(k=0;k<2;++k)for(q=0;q<gffx_mesh_element_count(outputs[k]);++q){if(!g&&k==0)((int32_t*)gffx_mesh_elements(out))[q]=-1;else write_real(outputs[k],(int64_t)q,0.);}
#endif
 for(i=0;i<8;++i)receipt[i]=0;receipt[7]=initial_work;if(initial_work<0||initial_work>2000000){receipt[7]=ctx.work;return 2;}
 for(int64_t face=0;face<f->shape[0];++face){double lx=read_real(v,3*(int64_t)faces[3*face]),hx=lx,ly=read_real(v,3*(int64_t)faces[3*face]+1),hy=ly;
  for(k=1;k<3;++k){double x=read_real(v,3*(int64_t)faces[3*face+k]),y=read_real(v,3*(int64_t)faces[3*face+k]+1);if(x<lx)lx=x;if(x>hx)hx=x;if(y<ly)ly=y;if(y>hy)hy=y;}
  if(hx<ox||lx>ox+1||hy<oy||ly>oy+1)continue;if(n==(rule!=0?SH_CANDIDATES:CANDIDATES)){receipt[7]=ctx.work;return 2;}s->ids[n++]=face;
 }
 ctx.canonical_supports=rule==4||(rule!=0&&n>CANDIDATES);
 separation_clear(s,n);
 for(j=0;j<n;++j)for(k=j+1;k<n;++k)if(separated_edge(v,faces+3*s->ids[j],faces+3*s->ids[k])||separated_edge(v,faces+3*s->ids[k],faces+3*s->ids[j])||exact_separated_edge(v,faces+3*s->ids[j],faces+3*s->ids[k])||exact_separated_edge(v,faces+3*s->ids[k],faces+3*s->ids[j])){separation_set(s,j,k);separation_set(s,k,j);}
 if(rule>=2 && g){memset(&s->reuse,0,sizeof(s->reuse));s->reuse.mode=1;if(prefix){if(prefix[0]<0||prefix[0]>n||prefix[1]<0||prefix[1]>cursor[0]||(!cursor[1]&&(prefix[0]||prefix[1]))){receipt[7]=ctx.work;return 2;}s->reuse.prefix_face=s->reuse.next_face=prefix[0];s->reuse.prefix_seen=s->reuse.next_seen=prefix[1];}}
 if(opaque_batch_walk(s,n,v,faces,cuts,ox,oy,-1,-1,cursor,first,g,out,other,&value,&ctx,rule)){receipt[7]=ctx.work;return 2;}
 if(rule>=2 && g)s->reuse.mode=2;
 if(g){
  for(j=0;j<n;++j)for(k=0;k<3;++k){int a;int64_t id=faces[3*s->ids[j]+k];for(a=0;a<nv;++a)if(vertices[a]==id)break;if(a==nv)vertices[nv++]=id;}
  if(derivatives&1)for(j=0;j<nv;++j)for(k=0;k<3;++k){int64_t seed=3*vertices[j]+k;
   if(opaque_batch_walk(s,n,v,faces,cuts,ox,oy,seed,-1,cursor,next,g,NULL,NULL,&value,&ctx,rule)){receipt[7]=ctx.work;return 2;}
   for(i=0;i<7;++i)if(next[i]!=first[i]){receipt[7]=ctx.work;return 2;}write_real(out,seed,value);
  }
  if(derivatives&2)for(j=0;j<n;++j)for(k=0;k<cuts->shape[1]*cuts->shape[2];++k){int64_t seed=cuts->shape[1]*cuts->shape[2]*s->ids[j]+k;
   if(opaque_batch_walk(s,n,v,faces,cuts,ox,oy,-1,seed,cursor,next,g,NULL,NULL,&value,&ctx,rule)){receipt[7]=ctx.work;return 2;}
   for(i=0;i<7;++i)if(next[i]!=first[i]){receipt[7]=ctx.work;return 2;}write_real(other,seed,value);
  }
  for(k=0;k<2;++k)for(q=0;q<gffx_mesh_element_count(outputs[k]);++q)if(!isfinite(read_real(outputs[k],(int64_t)q))){receipt[7]=ctx.work;return 2;}
 }
 for(i=0;i<7;++i)receipt[i]=first[i];receipt[7]=ctx.work;if(prefix){prefix[0]=s->reuse.next_face;prefix[1]=s->reuse.next_seen;}return 0;
}

GFFX_OPAQUE_LOCAL int opaque_batch_execute_prefix(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *c,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,gffx_tensor_view *a,gffx_tensor_view *b,Scratch *s,int rule,int derivatives,int32_t *prefix){return opaque_batch_execute_prefix_charged(v,f,c,ox,oy,cursor,receipt,g,a,b,s,rule,derivatives,prefix,0);}

GFFX_OPAQUE_LOCAL int opaque_batch_execute(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *c,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,gffx_tensor_view *a,gffx_tensor_view *b,Scratch *s,int rule,int derivatives){return opaque_batch_execute_prefix(v,f,c,ox,oy,cursor,receipt,g,a,b,s,rule,derivatives,NULL);}
