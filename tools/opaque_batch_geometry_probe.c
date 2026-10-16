#include <gffx/tensor.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static void *gffx_mesh_elements(const gffx_tensor_view *v){return (char*)v->data+v->byte_offset;}
static const void *gffx_mesh_elements_const(const gffx_tensor_view *v){return (const char*)v->data+v->byte_offset;}
static uint64_t gffx_mesh_element_count(const gffx_tensor_view *v){uint64_t n=1;for(uint32_t i=0;i<v->rank;i++)n*=v->shape[i];return n;}
#include "../native/core/opaque_quadrature_scalar.h"
#include "../native/core/opaque_exact_orientation.h"
__declspec(dllexport) int trace_run(double *vertices,int nv,int32_t *faces,int nf,double *cuts,int columns,double ox,double oy,int32_t *ids,double *samples,int *trace){
 int64_t vs[2]={nv,3},fs[2]={nf,3},cs[3]={nf,1,columns},is[2]={1,800},qs[3]={1,800,4};
 gffx_tensor_view v={0},f={0},c={0},a={0},b={0};
 v.data=vertices;v.shape=vs;v.rank=2;v.dtype=GFFX_DTYPE_FLOAT64;f.data=faces;f.shape=fs;f.rank=2;f.dtype=GFFX_DTYPE_INT32;c.data=cuts;c.shape=cs;c.rank=3;c.dtype=GFFX_DTYPE_FLOAT64;a.data=ids;a.shape=is;a.rank=2;a.dtype=GFFX_DTYPE_INT32;b.data=samples;b.shape=qs;b.rank=3;b.dtype=GFFX_DTYPE_FLOAT64;
 Scratch *scratch=calloc(1,sizeof(Scratch));if(!scratch)return -1;
 int result=opaque_quadrature_execute(&v,&f,&c,1,1,ox,oy,NULL,&a,&b,scratch,1);
 trace[0]=trace[1]=trace[2]=0;free(scratch);return result;
}

/* Research adapter, no public ABI or installed-wheel qualification. */
static int batch_execute(double *vertices,int nv,int32_t *faces,int nf,double *cuts,int columns,double ox,double oy,int *cursor,int32_t *ids,double *samples,int *receipt,int64_t vertex_seed,int64_t cut_seed,const double *upstream,double *answer){
 int64_t vs[2]={nv,3},fs[2]={nf,3},cs[3]={nf,1,columns},is[2]={1,800},qs[3]={1,800,4};
 gffx_tensor_view v={0},f={0},c={0},a={0},b={0},g={0};
 int n=0,i,j,k,curves=0,count=0,result=2;double value=0.;Ctx ctx={0,0};QuadBatch batch;Scratch *s=NULL;
 if(answer)*answer=0.;
 for(i=0;i<8;++i)receipt[i]=0;
 for(i=0;i<800;++i)ids[i]=-1;memset(samples,0,3200*sizeof(double));
 if(nv<0||nf<0||columns<6||(columns-6)%3||(columns-6)/3>16||!isfinite(ox)||!isfinite(oy))return 2;
 if(vertex_seed < -1 || vertex_seed >= (int64_t)nv*3 || cut_seed < -1 || cut_seed >= (int64_t)nf*columns || (vertex_seed>=0&&cut_seed>=0))return 2;
 if(upstream)for(i=0;i<3200;++i)if(!isfinite(upstream[i]))return 2;
 if(quad_batch_init(&batch,cursor[0],cursor[1],cursor[2],cursor[3]))return 2;
 for(i=0;i<nv*3;++i)if(!isfinite(vertices[i]))return 2;
 for(i=0;i<nf*3;++i)if(faces[i]<0||faces[i]>=nv)return 2;
 for(i=0;i<nf*columns;++i)if(!isfinite(cuts[i]))return 2;
 v.data=vertices;v.shape=vs;v.rank=2;v.dtype=GFFX_DTYPE_FLOAT64;
 f.data=faces;f.shape=fs;f.rank=2;f.dtype=GFFX_DTYPE_INT32;
 c.data=cuts;c.shape=cs;c.rank=3;c.dtype=GFFX_DTYPE_FLOAT64;
 a.data=ids;a.shape=is;a.rank=2;a.dtype=GFFX_DTYPE_INT32;
 b.data=samples;b.shape=qs;b.rank=3;b.dtype=GFFX_DTYPE_FLOAT64;g=b;g.data=(void*)upstream;
 s=calloc(1,sizeof(Scratch));if(!s)return 2;
 for(i=0;i<nf;++i){double lx=vertices[3*faces[3*i]],hx=lx,ly=vertices[3*faces[3*i]+1],hy=ly;
   for(k=1;k<3;++k){double x=vertices[3*faces[3*i+k]],y=vertices[3*faces[3*i+k]+1];if(x<lx)lx=x;if(x>hx)hx=x;if(y<ly)ly=y;if(y>hy)hy=y;}
   if(hx<ox||lx>ox+1||hy<oy||ly>oy+1)continue;
   if(n==CANDIDATES)goto end;s->ids[n++]=i;
 }
 for(j=0;j<n;++j)s->triangles[j]=triangle(&v,faces+3*s->ids[j],vertex_seed,ox,oy);
 for(j=0;j<n;++j)for(k=j+1;k<n;++k)
   if(separated_edge(&v,faces+3*s->ids[j],faces+3*s->ids[k])||separated_edge(&v,faces+3*s->ids[k],faces+3*s->ids[j])||exact_separated_edge(&v,faces+3*s->ids[j],faces+3*s->ids[k])||exact_separated_edge(&v,faces+3*s->ids[k],faces+3*s->ids[j])){s->separate[j]|=(uint32_t)1<<k;s->separate[k]|=(uint32_t)1<<j;}
 quadrature_pixel(s,n,&c,cut_seed,0,12,columns==6?-44:8,&curves,&count,&a,&b,upstream?&g:NULL,&value,&ctx,&batch);
 if(ctx.failed||quad_batch_finish(&batch)||count!=batch.slots)goto end;
 receipt[0]=batch.seen;receipt[1]=!batch.full;receipt[2]=batch.slots;receipt[3]=batch.blocks;
 receipt[4]=batch.total_slots;receipt[5]=batch.total_blocks;receipt[6]=batch.emitted;receipt[7]=ctx.work;if(answer)*answer=value;result=0;
end:
 if(result){for(i=0;i<800;++i)ids[i]=-1;memset(samples,0,3200*sizeof(double));}
 free(s);return result;
}

__declspec(dllexport) int batch_geometry(double *vertices,int nv,int32_t *faces,int nf,double *cuts,int columns,double ox,double oy,int *cursor,int32_t *ids,double *samples,int *receipt){
 return batch_execute(vertices,nv,faces,nf,cuts,columns,ox,oy,cursor,ids,samples,receipt,-1,-1,NULL,NULL);
}
__declspec(dllexport) int batch_derivative(double *vertices,int nv,int32_t *faces,int nf,double *cuts,int columns,int *cursor,int64_t vertex_seed,int64_t cut_seed,const double *upstream,double *answer,int *receipt){
 int32_t ids[800];double samples[3200];
 return batch_execute(vertices,nv,faces,nf,cuts,columns,0.,0.,cursor,ids,samples,receipt,vertex_seed,cut_seed,upstream,answer);
}

/* Diagnostic: replay the same seed with sample writes to audit layout invariance. */
__declspec(dllexport) int batch_seed_values(double *vertices,int nv,int32_t *faces,int nf,double *cuts,int columns,int *cursor,int64_t vertex_seed,int32_t *ids,double *samples,int *receipt){
 return batch_execute(vertices,nv,faces,nf,cuts,columns,0.,0.,cursor,ids,samples,receipt,vertex_seed,-1,NULL,NULL);
}

__declspec(dllexport) int orientation_sign(const double *x){return orient_exact(x[0],x[1],x[2],x[3],x[4],x[5]);}
