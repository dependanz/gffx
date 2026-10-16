/* Shared C11 scalar-dual CPU/CUDA arithmetic; caller owns all scratch. */
#ifndef GFFX_OPAQUE_LOCAL
#define GFFX_OPAQUE_LOCAL static
#endif
#include "opaque_polygon.h"
#include "opaque_batch_cursor.h"
#define CANDIDATES 32
/* Expanded storage is admitted only by the explicit SH25 batch rule. */
#define SH_CANDIDATES 512
#define SEPARATION_WORDS (SH_CANDIDATES/32)
typedef struct { Tri tri; Line depth,bary[3]; P vertex[3]; D determinant; } ColourTri;
/* Every cursor field is a nonnegative bounded count (maximum41600).
 * Store all fields losslessly; reject a nonrepresentable snapshot rather
 * than truncate it. Explicit fields avoid C/CUDA padding dependence. */
typedef struct { uint16_t begin,batch,prefix_slots,prefix_blocks,seen,emitted,slots,blocks,total_slots,total_blocks,full,failed,max_batches; } QuadMark;
GFFX_OPAQUE_LOCAL int quad_mark_save(QuadMark *m,const QuadBatch *b){
#define QUAD_MARK_SAVE(field) if(b->field<0 || b->field>65535)return 0;m->field=(uint16_t)b->field;
 QUAD_MARK_SAVE(begin) QUAD_MARK_SAVE(batch) QUAD_MARK_SAVE(prefix_slots) QUAD_MARK_SAVE(prefix_blocks)
 QUAD_MARK_SAVE(seen) QUAD_MARK_SAVE(emitted) QUAD_MARK_SAVE(slots) QUAD_MARK_SAVE(blocks)
 QUAD_MARK_SAVE(total_slots) QUAD_MARK_SAVE(total_blocks) QUAD_MARK_SAVE(full) QUAD_MARK_SAVE(failed) QUAD_MARK_SAVE(max_batches)
#undef QUAD_MARK_SAVE
 return 1;
}
GFFX_OPAQUE_LOCAL void quad_mark_load(QuadBatch *b,const QuadMark *m){
#define QUAD_MARK_LOAD(field) b->field=m->field;
 QUAD_MARK_LOAD(begin) QUAD_MARK_LOAD(batch) QUAD_MARK_LOAD(prefix_slots) QUAD_MARK_LOAD(prefix_blocks)
 QUAD_MARK_LOAD(seen) QUAD_MARK_LOAD(emitted) QUAD_MARK_LOAD(slots) QUAD_MARK_LOAD(blocks)
 QUAD_MARK_LOAD(total_slots) QUAD_MARK_LOAD(total_blocks) QUAD_MARK_LOAD(full) QUAD_MARK_LOAD(failed) QUAD_MARK_LOAD(max_batches)
#undef QUAD_MARK_LOAD
}
GFFX_OPAQUE_LOCAL int quad_mark_same(const QuadMark *m,const QuadBatch *b){
 QuadBatch original={0};quad_mark_load(&original,m);
#define QUAD_MARK_SAME(field) if(original.field!=b->field)return 0;
 QUAD_MARK_SAME(begin) QUAD_MARK_SAME(batch) QUAD_MARK_SAME(prefix_slots) QUAD_MARK_SAME(prefix_blocks)
 QUAD_MARK_SAME(seen) QUAD_MARK_SAME(emitted) QUAD_MARK_SAME(slots) QUAD_MARK_SAME(blocks)
 QUAD_MARK_SAME(total_slots) QUAD_MARK_SAME(total_blocks) QUAD_MARK_SAME(full) QUAD_MARK_SAME(failed) QUAD_MARK_SAME(max_batches)
#undef QUAD_MARK_SAME
 return 1;
}
#define QUAD_GUARDS 512
#define QUAD_GUARD_POINTS 2976
typedef struct { uint16_t owner,other,packed,offset; int cost; } QuadGuard;
typedef struct { int16_t face; uint8_t edges,depth; } QuadDependency;
typedef struct { int prefix_face,prefix_seen,next_face,next_seen,mode,valid[SH_CANDIDATES],unchanged[SH_CANDIDATES],geometry_unchanged[SH_CANDIDATES],used[SH_CANDIDATES],triangles[SH_CANDIDATES]; QuadMark before[SH_CANDIDATES],after[SH_CANDIDATES]; int16_t dependency_count[SH_CANDIDATES]; uint16_t dependency_begin[SH_CANDIDATES]; int dependency_used; uint8_t empty[SH_CANDIDATES],owner_depth[SH_CANDIDATES]; QuadDependency dependencies[SH_CANDIDATES*32]; QuadGuard guards[QUAD_GUARDS]; double guard_xy[2*QUAD_GUARD_POINTS]; int guard_points; } QuadReuse;
typedef struct { QuadReuse reuse; Poly work[2*MAX_PARTS]; ColourTri triangles[SH_CANDIDATES]; int64_t ids[SH_CANDIDATES]; D weights[CANDIDATES][3]; uint32_t separate[SH_CANDIDATES][SEPARATION_WORDS]; Line support[MAX_SUPPORT_LINES]; } Scratch;
/* Dependencies append owner-by-owner during the primal walk. Share the same
 * 16384 entries instead of refusing an owner at32 while other rows are empty.
 * Pool exhaustion or noncontiguous insertion disables that owner's reuse;
 * entries are never dropped and numerical walks/accumulation are unchanged. */
GFFX_OPAQUE_LOCAL void quad_dependency_add(QuadReuse *r,int owner,int face,int edges,int depth){
 int count=r->dependency_count[owner];
 if(count<0 || (!edges && !depth))return;
 if(r->dependency_used==SH_CANDIDATES*32 || (count && r->dependency_begin[owner]+count!=r->dependency_used)){r->dependency_count[owner]=-1;return;}
 if(!count)r->dependency_begin[owner]=(uint16_t)r->dependency_used;
 {QuadDependency dep={(int16_t)face,(uint8_t)edges,(uint8_t)depth};r->dependencies[r->dependency_used++]=dep;r->dependency_count[owner]++;}
}
GFFX_OPAQUE_LOCAL void separation_clear(Scratch *s,int n){int i,j;for(i=0;i<n;++i)for(j=0;j<(n+31)/32;++j)s->separate[i][j]=0;}
GFFX_OPAQUE_LOCAL void separation_set(Scratch *s,int i,int j){s->separate[i][j/32]|=(uint32_t)1<<(j%32);}
GFFX_OPAQUE_LOCAL int separation_has(const Scratch *s,int i,int j){return (s->separate[i][j/32]&((uint32_t)1<<(j%32)))!=0;}
GFFX_OPAQUE_LOCAL double read_real(const gffx_tensor_view *v,int64_t i) {
    return v->dtype==GFFX_DTYPE_FLOAT64?((const double*)gffx_mesh_elements_const(v))[i]:((const float*)gffx_mesh_elements_const(v))[i];
}
GFFX_OPAQUE_LOCAL void write_real(gffx_tensor_view *v,int64_t i,double a) {
    if(v->dtype==GFFX_DTYPE_FLOAT64)((double*)gffx_mesh_elements(v))[i]=a;
    else ((float*)gffx_mesh_elements(v))[i]=(float)a;
}
GFFX_OPAQUE_LOCAL Line scaled(Line a,D s) { Line r={mul(a.a,s),mul(a.b,s),mul(a.c,s)};return r; }
GFFX_OPAQUE_LOCAL Line plus(Line a,Line b) { Line r={add(a.a,b.a),add(a.b,b.b),add(a.c,b.c)};return r; }
GFFX_OPAQUE_LOCAL ColourTri triangle(const gffx_tensor_view *v,const int32_t *f,int64_t seed,double ox,double oy) {
    D xyz[3][3],ux,uy,vx,vy,det;ColourTri t;int i,k;memset(&t,0,sizeof(t));
    for(i=0;i<3;++i)for(k=0;k<3;++k){xyz[i][k].v=read_real(v,3*(int64_t)f[i]+k)-(k==0?ox:(k==1?oy:0));xyz[i][k].d=seed==3*(int64_t)f[i]+k?1:0;}
    ux=sub(xyz[1][0],xyz[0][0]);uy=sub(xyz[1][1],xyz[0][1]);vx=sub(xyz[2][0],xyz[0][0]);vy=sub(xyz[2][1],xyz[0][1]);
    det=sub(mul(ux,vy),mul(uy,vx));if(det.v==0)return t;
    t.bary[1].a=divd(vy,det);t.bary[1].b=divd(neg(vx),det);
    t.bary[2].a=divd(neg(uy),det);t.bary[2].b=divd(ux,det);
    for(i=1;i<3;++i)t.bary[i].c=neg(add(mul(t.bary[i].a,xyz[0][0]),mul(t.bary[i].b,xyz[0][1])));
    t.bary[0].a=neg(add(t.bary[1].a,t.bary[2].a));t.bary[0].b=neg(add(t.bary[1].b,t.bary[2].b));t.bary[0].c=sub(val(1),add(t.bary[1].c,t.bary[2].c));
    t.depth.a=val(0);t.depth.b=val(0);t.depth.c=xyz[0][2];
    for(i=1;i<3;++i)t.depth=plus(t.depth,scaled(t.bary[i],sub(xyz[i][2],xyz[0][2])));
    for(i=0;i<3;++i){int j=(i+1)%3;D dx=sub(xyz[j][0],xyz[i][0]),dy=sub(xyz[j][1],xyz[i][1]);
        t.tri.edge[i].a=neg(dy);t.tri.edge[i].b=dx;t.tri.edge[i].c=sub(mul(dy,xyz[i][0]),mul(dx,xyz[i][1]));
        if(det.v<0)t.tri.edge[i]=negative(t.tri.edge[i]);}
    for(i=0;i<3;++i){t.vertex[i].x=xyz[i][0];t.vertex[i].y=xyz[i][1];}t.determinant=det;
    t.tri.valid=1;return t;
}
/* A mesh edge uses one endpoint order regardless of the incident face.
 * This is confined to newly admitted SH pixels; no epsilon or area cutoff. */
GFFX_OPAQUE_LOCAL ColourTri triangle_shared_edges(const gffx_tensor_view *v,const int32_t *f,int64_t seed,double ox,double oy){
    ColourTri t=triangle(v,f,seed,ox,oy);int k;
    for(k=0;k<3;++k){int j=(k+1)%3,lo=f[k]<f[j]?k:j,hi=lo==k?j:k;
        D dx=sub(t.vertex[hi].x,t.vertex[lo].x),dy=sub(t.vertex[hi].y,t.vertex[lo].y);Line edge;
        edge.a=neg(dy);edge.b=dx;edge.c=sub(mul(dy,t.vertex[lo].x),mul(dx,t.vertex[lo].y));
        if(lo!=k)edge=negative(edge);if(t.determinant.v<0)edge=negative(edge);t.tri.edge[k]=edge;
    }
    return t;
}
GFFX_OPAQUE_LOCAL int strict_outside(const Poly *p,Line l,Ctx *ctx);
GFFX_OPAQUE_LOCAL int strict_polygon_separation(const Poly *p,const ColourTri *other,Ctx *ctx);
GFFX_OPAQUE_LOCAL int joint_depth_separation(const Poly *p,const ColourTri *other,Line depth,Ctx *ctx);
GFFX_OPAQUE_LOCAL int strict_depth_inactive(const Poly *p,Line l,Ctx *ctx);
GFFX_OPAQUE_LOCAL int strict_edge_inactive(const Poly *p,Line l,Ctx *ctx);
/* These separation predicates use primal coordinates only. Cache at most512
 * expensive baseline results within this call for polygons with at most8
 * vertices in a2976-point arena. Larger or changed rounded polygons take the original proof path.
 * Every compared vertex is charged. Misses preserve the original proof path. */
GFFX_OPAQUE_LOCAL int quad_guard_lookup(QuadReuse *reuse,int owner,int other,const Poly *p,Ctx *ctx){
 int j,k;if(!reuse || reuse->mode!=2 || p->n>8)return -1;
 for(j=0;j<QUAD_GUARDS;++j){QuadGuard *q=reuse->guards+j;int same=1;
  if(!q->cost||q->owner!=owner||q->other!=other||(q->packed&255)!=p->n)continue;
  for(k=0;k<p->n;++k){double x=reuse->guard_xy[2*(q->offset+k)],y=reuse->guard_xy[2*(q->offset+k)+1];
   if(++ctx->work>2000000){ctx->failed=1;return -1;}
   if(x!=p->p[k].x.v||y!=p->p[k].y.v||signbit(x)!=signbit(p->p[k].x.v)||signbit(y)!=signbit(p->p[k].y.v)){same=0;break;}
  }
  if(same)return q->packed>>8;
 }
 return -1;
}
GFFX_OPAQUE_LOCAL void quad_guard_remove(QuadReuse *r,int slot){
 QuadGuard *q=r->guards+slot;int n=q->packed&255,start=q->offset,j;
 if(!q->cost)return;
 for(j=2*(start+n);j<2*r->guard_points;++j)r->guard_xy[j-2*n]=r->guard_xy[j];
 r->guard_points-=n;q->cost=0;
 for(j=0;j<QUAD_GUARDS;++j)if(r->guards[j].cost && r->guards[j].offset>start)r->guards[j].offset-=n;
}
GFFX_OPAQUE_LOCAL void quad_guard_store(QuadReuse *reuse,int owner,int other,const Poly *p,int result,int cost){
 int j,k,slot=-1;if(!reuse||reuse->mode!=1||p->n<1||p->n>8||cost<=p->n||owner<0||owner>=SH_CANDIDATES||other<0||other>=SH_CANDIDATES||(result!=0&&result!=1))return;
 for(;;){int lowest=-1;slot=-1;
  for(j=0;j<QUAD_GUARDS;++j){if(!reuse->guards[j].cost){if(slot<0)slot=j;}else if(lowest<0||reuse->guards[j].cost<reuse->guards[lowest].cost)lowest=j;}
  if(slot>=0 && reuse->guard_points+p->n<=QUAD_GUARD_POINTS)break;
  if(lowest<0 || cost<=reuse->guards[lowest].cost)return;
  quad_guard_remove(reuse,lowest);
 }
 {QuadGuard *q=reuse->guards+slot;q->owner=(uint16_t)owner;q->other=(uint16_t)other;q->packed=(uint16_t)(p->n|(result<<8));q->offset=(uint16_t)reuse->guard_points;q->cost=cost;
  for(k=0;k<p->n;++k){reuse->guard_xy[2*(q->offset+k)]=p->p[k].x.v;reuse->guard_xy[2*(q->offset+k)+1]=p->p[k].y.v;}reuse->guard_points+=p->n;}
}
GFFX_OPAQUE_LOCAL int quad_separation_guard(const Poly *p,const ColourTri *other,Line depth,int tie,Ctx *ctx){
 int k;
 if(!tie && strict_outside(p,depth,ctx))return 1;
 for(k=0;k<3 && !ctx->failed;++k)if(strict_outside(p,other->tri.edge[k],ctx))return 1;
 if(ctx->canonical_supports && strict_polygon_separation(p,other,ctx))return 1;
 if(ctx->canonical_supports && !tie && joint_depth_separation(p,other,depth,ctx))return 1;
 return 0;
}
GFFX_OPAQUE_LOCAL void subtract_visible(const Poly *p,const ColourTri *other,Line depth,int tie,Poly *out,int *count,Ctx *ctx,int *dependency,int *depth_dependency,int *edge_dependencies,QuadReuse *reuse,int owner,int occluder) {
    Poly remain=*p;int k;
    if(ctx->support){
        int skip=quad_guard_lookup(reuse,owner,occluder,p,ctx);
        if(ctx->failed)return;
        if(skip<0){int before=ctx->work;skip=quad_separation_guard(p,other,depth,tie,ctx);
            if(ctx->failed)return;
            quad_guard_store(reuse,owner,occluder,p,skip,ctx->work-before);
        }
        if(skip){append(out,count,*p,ctx);return;}
    }
    if(dependency)*dependency=1;
    if(depth_dependency && tie)*depth_dependency=1;
    for(k=0;k<(tie?3:4);++k){Line l=k<3?other->tri.edge[k]:depth;
        int known_side=0;
        if(k<3 && edge_dependencies){known_side=strict_edge_inactive(&remain,l,ctx);if(!known_side)*edge_dependencies|=1<<k;}
        if(k==3 && depth_dependency){known_side=strict_depth_inactive(&remain,l,ctx);if(!known_side)*depth_dependency=1;}
        if(known_side){
            /* The baseline dependency proof already evaluated every vertex.
             * Reuse its strict side for both emissions; keep support order. */
            append(out,count,clip_complement(&remain,negative(l),known_side,ctx),ctx);
            remain=clip_complement(&remain,l,-known_side,ctx);
        }else {int side=0;append(out,count,clip_classified(&remain,negative(l),ctx,reuse?&side:NULL),ctx);remain=clip_complement(&remain,l,side,ctx);}
        if(!remain.n || area(&remain).v==0 || ctx->failed)return;}
}
#define QUAD_SLOTS 800
/* Outward binary64 enclosures certify signs for the represented polynomial.
   They are topology guards, not an integration-error bound. */
typedef struct {double lo,hi;} QR;
GFFX_OPAQUE_LOCAL QR qr(double x){QR r={x,x};return r;}
GFFX_OPAQUE_LOCAL QR qr_add(QR a,QR b){QR r={nextafter(a.lo+b.lo,-HUGE_VAL),nextafter(a.hi+b.hi,HUGE_VAL)};return r;}
GFFX_OPAQUE_LOCAL QR qr_neg(QR a){QR r={-a.hi,-a.lo};return r;}
GFFX_OPAQUE_LOCAL QR qr_sub(QR a,QR b){return qr_add(a,qr_neg(b));}
GFFX_OPAQUE_LOCAL QR qr_mul(QR a,QR b){double z[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi},lo=z[0],hi=z[0];int i;for(i=1;i<4;++i){if(z[i]<lo)lo=z[i];if(z[i]>hi)hi=z[i];} {QR r={nextafter(lo,-HUGE_VAL),nextafter(hi,HUGE_VAL)};return r;}}
GFFX_OPAQUE_LOCAL QR qr_at(Line l,P p){return qr_add(qr_add(qr_mul(qr(l.a.v),qr(p.x.v)),qr_mul(qr(l.b.v),qr(p.y.v))),qr(l.c.v));}
GFFX_OPAQUE_LOCAL int qr_sign(QR r){return r.lo>0?1:(r.hi<0?-1:0);}
/* Certify only the actual depth-clip input after the three edge clips.
 * Strict interval positivity plus the original provenance-aware predicate
 * excludes contact, canonical zero constraints and rounded sign uncertainty. */
GFFX_OPAQUE_LOCAL int strict_depth_inactive(const Poly *p,Line l,Ctx *ctx){
 int i;if(!p->n)return 0;
 for(i=0;i<p->n;++i){
  if(++ctx->work>2000000){ctx->failed=1;return 0;}
  if(!(support_at(p,i,l,ctx).v>0.) || !(qr_at(l,p->p[i]).lo>0.))return 0;
 }
 return 1;
}
/* An inactive edge is strictly one-sided on the actual incoming polygon.
 * Check both interval signs and the provenance-aware predicate. Contact and
 * uncertainty retain the original dependency and clipping path. */
GFFX_OPAQUE_LOCAL int strict_edge_inactive(const Poly *p,Line l,Ctx *ctx){
 int i,side=0;if(!p->n)return 0;
 for(i=0;i<p->n;++i){int sign;D a;
  if(++ctx->work>2000000){ctx->failed=1;return 0;}
  sign=qr_sign(qr_at(l,p->p[i]));a=support_at(p,i,l,ctx);
  if(!sign || !(sign>0?a.v>0.:a.v<0.) || (side && side!=sign))return 0;
  side=sign;
 }
 return side;
}
GFFX_OPAQUE_LOCAL int strict_outside(const Poly *p,Line l,Ctx *ctx){
    int i;if(!p->n)return 0;
    for(i=0;i<p->n;++i){
        if(++ctx->work>2000000){ctx->failed=1;return 0;}
        if(!(qr_at(l,p->p[i]).hi<0.))return 0;
    }
    return 1;
}
/* The surviving polygon can supply a separating axis absent from the
 * original face. Require a strict certified gap: contact/unknown keeps clipping.
 * Every predicate is charged to the existing aggregate clipping-work guard. */
GFFX_OPAQUE_LOCAL int polygon_side(P a,P b,P c,Ctx *ctx){
    QR ax=qr(a.x.v),ay=qr(a.y.v),bx=qr(b.x.v),by=qr(b.y.v),cx=qr(c.x.v),cy=qr(c.y.v);
    if(++ctx->work>2000000){ctx->failed=1;return 0;}
    return qr_sign(qr_sub(qr_mul(qr_sub(bx,ax),qr_sub(cy,ay)),qr_mul(qr_sub(by,ay),qr_sub(cx,ax))));
}
GFFX_OPAQUE_LOCAL int strict_polygon_separation(const Poly *p,const ColourTri *other,Ctx *ctx){
    int e,j;if(p->n<3)return 0;
    for(e=0;e<p->n&&!ctx->failed;++e){int next=(e+1)%p->n,side=0,ok=1;
        for(j=0;j<3;++j){int sign=polygon_side(p->p[e],p->p[next],other->vertex[j],ctx);
            if(!sign||(side&&sign!=side)){ok=0;break;}side=sign;}
        if(!ok)continue;
        for(j=0;j<p->n;++j)if(j!=e&&j!=next){
            if(polygon_side(p->p[e],p->p[next],p->p[j],ctx)!=-side){ok=0;break;}}
        if(ok&&!ctx->failed)return 1;
    }
    return 0;
}
/* A joint half-plane certificate: occlusion needs depth >= 0 and each
 * occluder edge >= 0. If depth + lambda*edge is strictly negative at every
 * polygon vertex for a finite lambda >= 0, it is negative throughout this
 * convex polygon. Thus the joint occlusion region is empty. Floating ratios
 * only propose lambda; outward verification alone authorizes skipping cuts.
 * No texture cut is involved. Unknown/contact signs retain the old path. */
GFFX_OPAQUE_LOCAL int joint_depth_separation(const Poly *p,const ColourTri *other,Line depth,Ctx *ctx){
    int k,i;if(p->n<3||ctx->failed)return 0;
    for(k=0;k<3&&!ctx->failed;++k){Line edge=other->tri.edge[k];double lo=0.,hi=HUGE_VAL,lambda;int ok=1;
        for(i=0;i<p->n;++i){double d,e;
            if(++ctx->work>2000000){ctx->failed=1;return 0;}
            d=at(depth,p->p[i]).v;e=at(edge,p->p[i]).v;
            if(!isfinite(d)||!isfinite(e)){ok=0;break;}
            if(e<0.){double q=-d/e;if(q>lo)lo=q;}
            else if(e>0.){double q=-d/e;if(q<hi)hi=q;}
            else if(d>=0.){ok=0;break;}
        }
        if(!ok||!isfinite(lo)||!(lo<hi))continue;
        lambda=isfinite(hi)?lo+(hi-lo)*.5:lo+(lo>1.?lo:1.);
        if(!isfinite(lambda)||lambda<0.)continue;
        for(i=0;i<p->n;++i){QR proof;
            if(++ctx->work>2000000){ctx->failed=1;return 0;}
            proof=qr_add(qr_at(depth,p->p[i]),qr_mul(qr(lambda),qr_at(edge,p->p[i])));
            if(!(proof.hi<0.)){ok=0;break;}
        }
        if(ok)return 1;
    }
    return 0;
}
/* A certified separating edge proves empty INTERIOR overlap. Shared vertex
   identities supply exact endpoint zeros; all other signs are outward-rounded
   from original screen coordinates, never rounded line coefficients. Unknown
   signs retain clipping. No area threshold or contribution is discarded. */
#ifndef GFFX_OPAQUE_SEPARATION_LOCAL
#define GFFX_OPAQUE_SEPARATION_LOCAL GFFX_OPAQUE_LOCAL
#endif
GFFX_OPAQUE_SEPARATION_LOCAL QR orient_interval(const gffx_tensor_view *v,int32_t a,int32_t b,int32_t c) {
    QR ax=qr(read_real(v,3*(int64_t)a)),ay=qr(read_real(v,3*(int64_t)a+1));
    QR bx=qr(read_real(v,3*(int64_t)b)),by=qr(read_real(v,3*(int64_t)b+1));
    QR cx=qr(read_real(v,3*(int64_t)c)),cy=qr(read_real(v,3*(int64_t)c+1));
    return qr_sub(qr_mul(qr_sub(bx,ax),qr_sub(cy,ay)),qr_mul(qr_sub(by,ay),qr_sub(cx,ax)));
}
GFFX_OPAQUE_SEPARATION_LOCAL int separated_edge(const gffx_tensor_view *v,const int32_t *a,const int32_t *b) {
    int orientation=qr_sign(orient_interval(v,a[0],a[1],a[2])),e,k;
    if(!orientation)return 0;
    for(e=0;e<3;++e){int outside=1,strict=0;int32_t x=a[e],y=a[(e+1)%3];
        for(k=0;k<3;++k){int sign;if(b[k]==x||b[k]==y)continue;
            sign=qr_sign(orient_interval(v,x,y,b[k]));
            if(sign!=-orientation){outside=0;break;}strict=1;
        }
        if(outside&&strict)return 1;
    }
    return 0;
}
GFFX_OPAQUE_LOCAL D dsqrt(D a){double r=sqrt(a.v);D b={r,a.d/(2*r)};return b;}
/* Packed [F,1,3*K+6]: K linear corner columns, then q00,q11,q22,
   doubled q01,q02,q12. A zero six-vector explicitly means no curved cut. */
GFFX_OPAQUE_LOCAL int quad_linear_count(const gffx_tensor_view *c){return c->shape[1]==3?(int)c->shape[2]:((int)c->shape[2]-6)/3;}
GFFX_OPAQUE_LOCAL int curved_samples(ColourTri *tri,P *p,D *coef,int64_t face,int64_t pixel_index,int *used,int *pieces,int radial_order,int *curves,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_tensor_view *gradient,double *contract,Ctx *ctx,QuadBatch *batch){
    const double nodes[5]={0.0469100770306680036,0.2307653449471584545,0.5,0.7692346550528415455,0.9530899229693319964};
    const double weights[5]={0.1184634425280945438,0.2393143352496832340,0.2844444444444444444,0.2393143352496832340,0.1184634425280945438};
    /* One Gauss12 angular rule replaces four Gauss5 strips. Both curved
       regions use120 or288 nodes,charged as five logical blocks; all caps unchanged. */
    const double angular_nodes[12]={0.0092196828766403782,0.047941371814762601,0.11504866290284765,0.20634102285669126,0.31608425050090994,0.43738329574426554,0.5626167042557344,0.68391574949909006,0.79365897714330869,0.88495133709715235,0.95205862818523745,0.99078031712335957};
    const double angular_weights[12]={0.023587668193256011,0.053469662997659442,0.080039164271673055,0.10158371336153282,0.11674626826917732,0.12457352290670134,0.12457352290670134,0.11674626826917732,0.10158371336153282,0.080039164271673055,0.053469662997659442,0.023587668193256011};
    const int ia[6]={0,1,2,0,0,1},ib[6]={0,1,2,1,2,2};
    D bary[3][3],control[3][3],a,b,c,ab,ac,bc,aa[3],bb[2],det;QR rb[3][3],rc[3][3];
    int i,j,k,t,sign[3],allpos=1,allneg=1,isolated=-1,sg,order[3],side,ns,nt;P o,u,v;
    for(i=0;i<3;++i)for(j=0;j<3;++j){bary[i][j]=at(tri->bary[j],p[i]);rb[i][j]=qr_at(tri->bary[j],p[i]);}
    for(i=0;i<3;++i)for(j=i;j<3;++j){D d=val(0);QR r=qr(0);
        for(k=0;k<6;++k){int x=ia[k],y=ib[k];D z=mul(bary[i][x],bary[j][y]);QR rz=qr_mul(rb[i][x],rb[j][y]);
            if(k>=3){z=mul(val(.5),add(z,mul(bary[i][y],bary[j][x])));rz=qr_mul(qr(.5),qr_add(rz,qr_mul(rb[i][y],rb[j][x])));}
            d=add(d,mul(coef[k],z));r=qr_add(r,qr_mul(qr(coef[k].v),rz));}
        control[i][j]=control[j][i]=d;rc[i][j]=rc[j][i]=r;
        if(qr_sign(r)!=1)allpos=0;if(qr_sign(r)!=-1)allneg=0;
    }
    if(allpos||allneg)return 0;
    for(i=0;i<3;++i)sign[i]=qr_sign(rc[i][i]);
    for(i=0;i<3;++i)if(sign[i]&&sign[(i+1)%3]==-sign[i]&&sign[(i+2)%3]==-sign[i])isolated=i;
    if(isolated<0){ctx->failed=1;return 1;}
    for(i=0;i<3;++i)order[i]=(isolated+i)%3;sg=-sign[isolated];
    /* Normalize the isolated vertex negative. Strict Bernstein certificates
       establish a positive opposite edge and positive radial derivative. */
    {QR ra=rc[order[0]][order[0]],rbb=rc[order[1]][order[1]],rcc=rc[order[2]][order[2]],rab=rc[order[0]][order[1]],rac=rc[order[0]][order[2]],rbc=rc[order[1]][order[2]],guards[7];
      if(sg<0){ra=qr_neg(ra);rbb=qr_neg(rbb);rcc=qr_neg(rcc);rab=qr_neg(rab);rac=qr_neg(rac);rbc=qr_neg(rbc);}
      guards[0]=rbb;guards[1]=rbc;guards[2]=rcc;guards[3]=qr_sub(rab,ra);guards[4]=qr_sub(rac,ra);guards[5]=qr_sub(rbb,rab);guards[6]=qr_sub(rcc,rac);
      for(i=0;i<7;++i)if(qr_sign(guards[i])!=1){ctx->failed=1;return 1;}
      if(qr_sign(qr_sub(qr_mul(qr(2),rbc),qr_add(rab,rac)))!=1){ctx->failed=1;return 1;}
    }
    o=p[order[0]];u=p[order[1]];v=p[order[2]];
    a=mul(val(sg),control[order[0]][order[0]]);b=mul(val(sg),control[order[1]][order[1]]);c=mul(val(sg),control[order[2]][order[2]]);
    ab=mul(val(sg),control[order[0]][order[1]]);ac=mul(val(sg),control[order[0]][order[2]]);bc=mul(val(sg),control[order[1]][order[2]]);
    aa[0]=add(sub(b,mul(val(2),ab)),a);aa[1]=mul(val(2),add(sub(sub(bc,b),ac),ab));aa[2]=add(sub(b,mul(val(2),bc)),c);
    bb[0]=mul(val(2),sub(ab,a));bb[1]=mul(val(2),sub(ac,ab));
    det=sub(mul(sub(u.x,o.x),sub(v.y,o.y)),mul(sub(u.y,o.y),sub(v.x,o.x)));if(det.v<0)det=neg(det);
    if(batch){int offered=quad_batch_offer(batch,24*radial_order,5);
        if(offered<0)ctx->failed=1;if(offered!=1)return 1;
    }
    if((*pieces+=5)>32){ctx->failed=1;return 1;}
    ++*curves;
    if(!radial_order){if(*used>QUAD_SLOTS-120){ctx->failed=1;return 1;}*used+=120;return 1;}
    if(*used>QUAD_SLOTS-24*radial_order){ctx->failed=1;return 1;}
    for(side=0;side<2;++side)for(ns=0;ns<12;++ns)for(nt=0;nt<radial_order;++nt){
        /* Concentrate the same five radial nodes at the curved horizon.
           u(t) is monotone on both sides; include du/dt in the measure.
           Constants carry no geometry seed, so shared dual arithmetic still
           differentiates root, span, position and complete transformed weight. */
        double tnode=radial_order==12?angular_nodes[nt]:nodes[nt],radial_weight=radial_order==12?angular_weights[nt]:weights[nt];
        double radial_node=side?tnode*tnode:1.-(1.-tnode)*(1.-tnode);
        double radial_jacobian=side?2.*tnode:2.*(1.-tnode);
        D ss=val(angular_nodes[ns]),uu=val(radial_node),av=add(aa[0],mul(ss,add(aa[1],mul(ss,aa[2])))),bv=add(bb[0],mul(ss,bb[1]));
        D disc=sub(mul(bv,bv),mul(val(4),mul(av,a))),root,span,rad,q[4];P point;
        int64_t index=pixel_index*QUAD_SLOTS+(*used)++;
        if(!(disc.v>0)){ctx->failed=1;return 1;}
        root=divd(mul(val(-2),a),add(bv,dsqrt(disc)));
        if(!(root.v>0&&root.v<1)){ctx->failed=1;return 1;}
        span=side?sub(val(1),root):root;rad=mul(uu,span);if(side)rad=add(root,rad);
        point.x=add(o.x,mul(rad,add(sub(u.x,o.x),mul(ss,sub(v.x,u.x)))));
        point.y=add(o.y,mul(rad,add(sub(u.y,o.y),mul(ss,sub(v.y,u.y)))));
        for(t=0;t<3;++t)q[t]=at(tri->bary[t],point);
        q[3]=mul(mul(mul(det,rad),span),val((angular_weights[ns]*radial_weight)*radial_jacobian));
        for(t=0;t<4;++t){if(!isfinite(q[t].v)||!isfinite(q[t].d)){ctx->failed=1;return 1;}
          if(gradient)*contract+=q[t].d*read_real(gradient,4*index+t);else write_real(samples,4*index+t,q[t].v);}
        if(!gradient)((int32_t*)gffx_mesh_elements(ids))[index]=(int32_t)face;
    }
    return 1;
}
typedef struct { gffx_tensor_view *rows; int count,capacity; } QuadPartition;
GFFX_OPAQUE_CLIP_LOCAL void quadrature_pixel_ex(Scratch *s,int n,const gffx_tensor_view *cuts,int64_t cut_seed,int64_t pixel_index,int radial_order,int flat_order,int *curves,int *sample_count,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_tensor_view *gradient,double *contract,Ctx *ctx,QuadBatch *batch,QuadPartition *partition,QuadReuse *reuse) {
    const double nodes[5]={0.0469100770306680036,0.2307653449471584545,0.5,0.7692346550528415455,0.9530899229693319964};
    const double weights[5]={0.1184634425280945438,0.2393143352496832340,0.2844444444444444444,0.2393143352496832340,0.1184634425280945438};
    /* Gauss-Jacobi5 integrates the Duffy factor (1-u) in its positive weights.
       SH keeps25 samples per triangle; other integration profiles are unchanged. */
    const double sh_nodes[5]={0.03980985705146874,0.19801341787360818,0.43797481024738616,0.6954642733536361,0.9014649142011736};
    const double sh_weights[5]={0.09678159022665168,0.16717463809436955,0.1463869870846698,0.07390887007261666,0.015747914521692274};
    /* Positive interior Xiao-Gimbutas degree11 rule,28 samples.
       Explicit SH80 only; other profiles retain their qualified rule. */
    const double sh_triangle28[28][3]={
        {0.030846895635588036,0.93830620872882398,0.0061246484753539809},
        {0.93830620872882398,0.030846895635588036,0.0061246484753539809},
        {0.030846895635588036,0.030846895635588036,0.0061246484753539809},
        {0.4987801651784608,0.0024396696430784462,0.0062327459369406896},
        {0.0024396696430784462,0.4987801651784608,0.0062327459369406896},
        {0.4987801651784608,0.4987801651784608,0.0062327459369406896},
        {0.15930361983769351,0.82632971759275087,0.0072788116689046231},
        {0.82632971759275087,0.15930361983769351,0.0072788116689046231},
        {0.014366662569555592,0.82632971759275087,0.0072788116689046231},
        {0.82632971759275087,0.014366662569555592,0.0072788116689046231},
        {0.014366662569555592,0.15930361983769351,0.0072788116689046231},
        {0.15930361983769351,0.014366662569555592,0.0072788116689046231},
        {0.33333333333333331,0.33333333333333331,0.040722567354675644},
        {0.11320782728669394,0.77358434542661214,0.020064621190654157},
        {0.77358434542661214,0.11320782728669394,0.020064621190654157},
        {0.11320782728669394,0.11320782728669394,0.020064621190654157},
        {0.43665501639317611,0.12668996721364778,0.031547436079949337},
        {0.12668996721364778,0.43665501639317611,0.031547436079949337},
        {0.43665501639317611,0.43665501639317611,0.031547436079949337},
        {0.21448345861926932,0.57103308276146136,0.033922553871847567},
        {0.57103308276146136,0.21448345861926932,0.033922553871847567},
        {0.21448345861926932,0.21448345861926932,0.033922553871847567},
        {0.31063121631346313,0.64170471671438611,0.020321424327943236},
        {0.64170471671438611,0.31063121631346313,0.020321424327943236},
        {0.047664066972150747,0.64170471671438611,0.020321424327943236},
        {0.64170471671438611,0.047664066972150747,0.020321424327943236},
        {0.047664066972150747,0.31063121631346313,0.020321424327943236},
        {0.31063121631346313,0.047664066972150747,0.020321424327943236}
    };
    /* Inner Gauss6/7/8 constants are shared literally by CPU and CUDA. */
    const double flat_nodes6[6]={0.03376524289842403,0.16939530676686776,0.38069040695840156,0.6193095930415985,0.8306046932331322,0.9662347571015759};
    const double flat_weights6[6]={0.08566224618958514,0.18038078652406936,0.23395696728634552,0.23395696728634552,0.18038078652406936,0.08566224618958514};
    const double flat_nodes7[7]={0.0254460438286207,0.12923440720030277,0.2970774243113014,0.5,0.7029225756886985,0.8707655927996972,0.9745539561713793};
    const double flat_weights7[7]={0.06474248308443487,0.13985269574463843,0.19091502525255935,0.20897959183673465,0.19091502525255935,0.13985269574463843,0.06474248308443487};
    const double flat_nodes8[8]={0.019855071751231912,0.10166676129318664,0.2372337950418355,0.4082826787521751,0.5917173212478248,0.7627662049581645,0.8983332387068134,0.9801449282487681};
    const double flat_weights8[8]={0.05061426814518853,0.11119051722668721,0.15685332293894344,0.18134189168918083,0.18134189168918083,0.15685332293894344,0.11119051722668721,0.05061426814518853};
    const double *flat_nodes=flat_order==8?flat_nodes8:(flat_order==7?flat_nodes7:(flat_order==6?flat_nodes6:nodes));
    const double *flat_weights=flat_order==8?flat_weights8:(flat_order==7?flat_weights7:(flat_order==6?flat_weights6:weights));
    Poly p;int i,j,k,used=0,triangles=0,nc=quad_linear_count(cuts);p.n=4;p.p[0]=point(0,0);p.p[1]=point(1,0);p.p[2]=point(1,1);p.p[3]=point(0,1);
    ctx->support=batch?s->support:NULL;
    if(batch){
        Line square[4]={ { {1,0},{0,0},{0,0} }, { {0,0},{1,0},{0,0} }, { {1,0},{0,0},{-1,0} }, { {0,0},{1,0},{-1,0} } };
        for(i=0;i<4;++i){p.edge[i]=i;s->support[i]=square[i];}
    }
    *contract=0.;*curves=0;*sample_count=0;
    if(reuse){batch->seen=reuse->prefix_seen;batch->total_slots=(batch->max_batches==80?28:25)*reuse->prefix_seen;batch->total_blocks=reuse->prefix_seen;}
    for(i=reuse?reuse->prefix_face:0;i<n && !ctx->failed && !(batch&&batch->full);++i)if(s->triangles[i].tri.valid){
        /* Reuse only unchanged owner data and the occluder supports actually
         * needed by the baseline. Strictly inactive edges may move. A bounded
         * sparse dependency table overflowing falls back to full replay. */
        if(reuse && reuse->mode==2 && reuse->valid[i]){
            int safe=(reuse->unchanged[i] || ((reuse->geometry_unchanged[i]&16) && !reuse->owner_depth[i])) && reuse->dependency_count[i]>=0 && (reuse->empty[i] || !(cut_seed>=0 && cut_seed/(cuts->shape[1]*cuts->shape[2])==s->ids[i]));
            for(j=0;j<reuse->dependency_count[i] && safe;++j){
                QuadDependency dep=reuse->dependencies[reuse->dependency_begin[i]+j];int q=dep.face;
                if((dep.depth && !(reuse->geometry_unchanged[q]&8)) || ((dep.edges & reuse->geometry_unchanged[q])!=dep.edges))safe=0;
            }
            if(safe){
                if(!quad_mark_same(reuse->before+i,batch)){ctx->failed=1;return;}
                quad_mark_load(batch,reuse->after+i);used=reuse->used[i];triangles=reuse->triangles[i];continue;
            }
        }
        if(reuse && reuse->mode==1 && !quad_mark_save(reuse->before+i,batch)){ctx->failed=1;return;}
        Poly subject=p,*a=s->work,*b=s->work+MAX_PARTS;int count=1;
        if(batch)ctx->support_count=4;
        for(k=0;k<3;++k)subject=clip(&subject,s->triangles[i].tri.edge[k],ctx);
        if(!subject.n || area(&subject).v==0)goto quad_face_done;a[0]=subject;
        for(j=0;j<n && count && !ctx->failed;++j)if(i!=j && s->triangles[j].tri.valid && !separation_has(s,i,j)){
            Line d=difference(s->triangles[i].depth,s->triangles[j].depth);int m=0,tie=d.a.v==0 && d.b.v==0 && d.c.v==0;
            if(reuse && reuse->mode==1 && tie)reuse->owner_depth[i]=1;
            if(tie && j>i)continue;
            int dependency=0,depth_dependency=0,edge_dependencies=0;
            for(k=0;k<count && !ctx->failed;++k)subtract_visible(a+k,s->triangles+j,d,tie,b,&m,ctx,&dependency,reuse && reuse->mode==1?&depth_dependency:NULL,reuse && reuse->mode==1?&edge_dependencies:NULL,reuse,i,j);
            if(reuse && reuse->mode==1 && depth_dependency)reuse->owner_depth[i]=1;
            if(reuse && reuse->mode==1 && dependency)quad_dependency_add(reuse,i,j,edge_dependencies,depth_dependency);
            {Poly *tmp=a;a=b;b=tmp;}count=m;
        }
        if(reuse && reuse->mode==1 && count==0 && !ctx->failed)reuse->empty[i]=1;
        for(j=0;j<nc && !ctx->failed;++j){Line line={ {0,0},{0,0},{0,0} };int m=0;
            for(k=0;k<3;++k){int64_t index=s->ids[i]*cuts->shape[1]*cuts->shape[2]+k*nc+j;D q=val(read_real(cuts,index));q.d=index==cut_seed?1.:0.;line=plus(line,scaled(s->triangles[i].bary[k],q));}
            if(line.a.v==0. && line.b.v==0. && line.c.v==0.)continue;
            for(k=0;k<count && !ctx->failed;++k){int side=0;append(b,&m,clip_classified(a+k,negative(line),ctx,reuse?&side:NULL),ctx);append(b,&m,clip_complement(a+k,line,side,ctx),ctx);}
            {Poly *tmp=a;a=b;b=tmp;}count=m;
        }
        for(k=0;k<count && !ctx->failed && !(batch&&batch->full);++k){int fan_index;
            if(batch){int vertex;
                for(vertex=0;vertex<a[k].n;++vertex){
                    int ia=a[k].edge[vertex],ib=a[k].edge[(vertex+1)%a[k].n];
                    if(ia<0||ib<0){ctx->failed=1;return;}
                    Line la=ctx->support[ia],lb=ctx->support[ib];P *q=a[k].p+vertex;
                    double d=la.a.v*lb.b.v-la.b.v*lb.a.v;
                    if(d!=0.)support_tangent(q,la,lb,ctx);
                    else {
                        double ra=-((la.a.d*q->x.v+la.b.d*q->y.v)+la.c.d),rb=-((lb.a.d*q->x.v+lb.b.d*q->y.v)+lb.c.d);
                        if(ra*lb.a.v!=rb*la.a.v||ra*lb.b.v!=rb*la.b.v){ctx->failed=1;return;}
                        if(fabs(la.a.v)>=fabs(la.b.v)){
                            if(la.a.v==0.){ctx->failed=1;return;}q->x.d=(ra-la.b.v*q->y.d)/la.a.v;
                        }else q->y.d=(ra-la.a.v*q->x.d)/la.b.v;
                    }
                    if(ctx->failed)return;
                }
            }
            for(fan_index=1;fan_index<a[k].n-1 && !ctx->failed && !(batch&&batch->full);++fan_index){
                P o=a[k].p[0],u=a[k].p[fan_index],v=a[k].p[fan_index+1];D ux=sub(u.x,o.x),uy=sub(u.y,o.y),vx=sub(v.x,o.x),vy=sub(v.y,o.y),det=sub(mul(ux,vy),mul(uy,vx));int ns,nt,c;
                if(det.v==0.)continue;if(det.v<0.)det=neg(det);
                if(cuts->shape[1]==1){D coeff[6];P points[3]={o,u,v};int active=0;
                    for(c=0;c<6;++c){int64_t ci=s->ids[i]*cuts->shape[2]+3*nc+c;coeff[c]=val(read_real(cuts,ci));coeff[c].d=ci==cut_seed?1.:0.;if(coeff[c].v!=0)active=1;}
                    if(active&&curved_samples(s->triangles+i,points,coeff,s->ids[i],pixel_index,&used,&triangles,radial_order,curves,ids,samples,gradient,contract,ctx,batch))continue;
                }
                if(partition){
                    int q;P corners[3]={o,u,v};int64_t base=(int64_t)partition->count*17;
                    if(flat_order!=5||partition->count==partition->capacity){ctx->failed=1;return;}
                    write_real(partition->rows,base,s->ids[i]);
                    for(q=0;q<3;++q){write_real(partition->rows,base+1+2*q,corners[q].x.v);write_real(partition->rows,base+2+2*q,corners[q].y.v);}
                    for(q=0;q<3;++q){Line l=s->triangles[i].bary[q];write_real(partition->rows,base+7+3*q,l.a.v);write_real(partition->rows,base+8+3*q,l.b.v);write_real(partition->rows,base+9+3*q,l.c.v);}
                    write_real(partition->rows,base+16,det.v);partition->count++;continue;
                }
                if(batch){int offered=quad_batch_offer(batch,flat_order==-44?44:((flat_order==5&&batch&&batch->max_batches==80)?28:5*flat_order),1);
                    if(offered<0){ctx->failed=1;break;}if(offered!=1)continue;
                }
                if(++triangles>32 || used>QUAD_SLOTS-(flat_order==-44?44:((flat_order==5&&batch&&batch->max_batches==80)?28:5*flat_order))){ctx->failed=1;break;}
                if(!radial_order){used+=(batch&&batch->max_batches==80&&flat_order==5)?28:25;continue;}
                if(flat_order==-44){
                    /* Material-following Reynolds identity. These44slots are
                       integral linearization data, not raw node derivatives. */
                    const double en[3]={0.1127016653792583,0.5,0.8872983346207417};
                    const double ew[3]={0.2777777777777778,0.4444444444444444,0.2777777777777778};
                    P corners[3]={o,u,v};ColourTri *source=s->triangles+i;
                    int support_edges[3]={-1,-1,-1};
                    if(batch){support_edges[0]=fan_index==1?a[k].edge[fan_index]:-1;support_edges[1]=a[k].edge[fan_index+1];support_edges[2]=fan_index+1==a[k].n-1?a[k].edge[0]:-1;}
                    double divergence=source->determinant.d/source->determinant.v;
                    double orientation=(ux.v*vy.v-uy.v*vx.v)<0.?-1.:1.;
                    for(ns=0;ns<5;++ns)for(nt=0;nt<7;++nt){
                        D ss=val(nodes[ns]),tt=val((1.-nodes[ns])*flat_nodes7[nt]),q[4];P point;
                        int64_t index=pixel_index*QUAD_SLOTS+used++;
                        point.x=add(o.x,add(mul(ss,ux),mul(tt,vx)));point.y=add(o.y,add(mul(ss,uy),mul(tt,vy)));
                        for(c=0;c<3;++c){q[c]=at(source->bary[c],point);q[c].d=0.;}
                        q[3]=mul(det,val((1.-nodes[ns])*weights[ns]*flat_weights7[nt]));q[3].d=q[3].v*divergence;
                        for(c=0;c<4;++c){if(!isfinite(q[c].v)||!isfinite(q[c].d)){ctx->failed=1;return;}if(gradient)*contract+=q[c].d*read_real(gradient,4*index+c);else write_real(samples,4*index+c,q[c].v);}
                        if(!gradient)((int32_t*)gffx_mesh_elements(ids))[index]=(int32_t)s->ids[i];
                    }
                    for(ns=0;ns<3;++ns)for(nt=0;nt<3;++nt){
                        P a=corners[ns],b=corners[(ns+1)%3],point;D ex=sub(b.x,a.x),ey=sub(b.y,a.y),q[4];double wx=0.,wy=0.;
                        int64_t index=pixel_index*QUAD_SLOTS+used++;
                        point.x=add(a.x,mul(val(en[nt]),ex));point.y=add(a.y,mul(val(en[nt]),ey));
                        for(c=0;c<3;++c){q[c]=at(source->bary[c],point);q[c].d=0.;wx+=q[c].v*source->vertex[c].x.d;wy+=q[c].v*source->vertex[c].y.d;}
                        q[3]=val(0.);q[3].d=orientation*ew[nt]*((point.x.d-wx)*ey.v-(point.y.d-wy)*ex.v);
                        if(batch){
                            int edge_id=support_edges[ns];
                            if(edge_id>=0){
                                Line line=ctx->support[edge_id];double rhs=-((line.a.d*point.x.v+line.b.d*point.y.v)+line.c.d),motion;
                                if(fabs(line.a.v)>=fabs(line.b.v)){
                                    if(line.a.v==0.){ctx->failed=1;return;}motion=(rhs/line.a.v)*ey.v;
                                }else motion=-(rhs/line.b.v)*ex.v;
                                q[3].d=orientation*ew[nt]*(motion-(wx*ey.v-wy*ex.v));
                            }
                        }
                        for(c=0;c<4;++c){if(!isfinite(q[c].v)||!isfinite(q[c].d)){ctx->failed=1;return;}if(gradient)*contract+=q[c].d*read_real(gradient,4*index+c);else write_real(samples,4*index+c,q[c].v);}
                        if(!gradient)((int32_t*)gffx_mesh_elements(ids))[index]=(int32_t)s->ids[i];
                    }
                    continue;
                }
                if(batch&&batch->max_batches==80&&flat_order==5){
                    for(ns=0;ns<28;++ns){
                        D ss=val(sh_triangle28[ns][0]),tt=val(sh_triangle28[ns][1]),q[4];P point;
                        int64_t index=pixel_index*QUAD_SLOTS+used++;
                        point.x=add(o.x,add(mul(ss,ux),mul(tt,vx)));point.y=add(o.y,add(mul(ss,uy),mul(tt,vy)));
                        for(c=0;c<3;++c)q[c]=at(s->triangles[i].bary[c],point);
                        q[3]=mul(det,val(sh_triangle28[ns][2]));
                        for(c=0;c<4;++c){if(!isfinite(q[c].v)||!isfinite(q[c].d)){ctx->failed=1;return;}
                            if(gradient)*contract+=q[c].d*read_real(gradient,4*index+c);else write_real(samples,4*index+c,q[c].v);}
                        if(!gradient)((int32_t*)gffx_mesh_elements(ids))[index]=(int32_t)s->ids[i];
                    }
                    continue;
                }
                for(ns=0;ns<5;++ns)for(nt=0;nt<flat_order;++nt){
                    double un=flat_order==5?sh_nodes[ns]:nodes[ns];
                    D ss=val(un),tt=val((1.-un)*flat_nodes[nt]),q[4];P point;
                    int64_t index=pixel_index*QUAD_SLOTS+used++;
                    point.x=add(o.x,add(mul(ss,ux),mul(tt,vx)));point.y=add(o.y,add(mul(ss,uy),mul(tt,vy)));
                    for(c=0;c<3;++c)q[c]=at(s->triangles[i].bary[c],point);
                    q[3]=mul(det,val((flat_order==5?sh_weights[ns]:(1.-nodes[ns])*weights[ns])*flat_weights[nt]));
                    for(c=0;c<4;++c){if(!isfinite(q[c].v)||!isfinite(q[c].d)){ctx->failed=1;return;}
                        if(gradient)*contract+=q[c].d*read_real(gradient,4*index+c);else write_real(samples,4*index+c,q[c].v);}
                    if(!gradient)((int32_t*)gffx_mesh_elements(ids))[index]=(int32_t)s->ids[i];
                }
            }
        }
quad_face_done:
        if(reuse && reuse->mode==1 && !ctx->failed){if(!quad_mark_save(reuse->after+i,batch)){ctx->failed=1;return;}reuse->used[i]=used;reuse->triangles[i]=triangles;reuse->valid[i]=1;if(!batch->full){reuse->next_face=i+1;reuse->next_seen=batch->seen;}}
        if(reuse && reuse->mode==2 && reuse->valid[i]){if(!quad_mark_same(reuse->after+i,batch)){ctx->failed=1;return;}}
    }
    *sample_count=used;
    if(!isfinite(*contract))ctx->failed=1;
}
GFFX_OPAQUE_LOCAL void quadrature_pixel(Scratch *s,int n,const gffx_tensor_view *cuts,int64_t cut_seed,int64_t pixel_index,int radial_order,int flat_order,int *curves,int *sample_count,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_tensor_view *gradient,double *contract,Ctx *ctx,QuadBatch *batch){
 quadrature_pixel_ex(s,n,cuts,cut_seed,pixel_index,radial_order,flat_order,curves,sample_count,ids,samples,gradient,contract,ctx,batch,NULL,NULL);
}
GFFX_OPAQUE_LOCAL int opaque_quadrature_execute(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *gradient,gffx_tensor_view *out,gffx_tensor_view *other,Scratch *s,int material_transport) {
    uint64_t q;int64_t i,x,y;int j,k,n,nc=quad_linear_count(cuts);const int32_t *faces;
    gffx_tensor_view *outputs[2]={out,other};
#ifndef GFFX_OPAQUE_PRECLEARED
    for(k=0;k<2;++k)for(q=0;q<gffx_mesh_element_count(outputs[k]);++q){
        if(!gradient && k==0)((int32_t*)gffx_mesh_elements(out))[q]=-1;else write_real(outputs[k],(int64_t)q,0.);}
#endif
    faces=(const int32_t*)gffx_mesh_elements_const(f);
    for(y=0;y<h;++y)for(x=0;x<w;++x){double px=ox+(double)x,py=oy+(double)y,value;int64_t vertices[3*CANDIDATES];int nv=0,curves=0,sample_count=0,radial_order=5,flat_order=5;Ctx ctx={0,0};n=0;
        for(i=0;i<f->shape[0];++i){double lx=read_real(v,3*(int64_t)faces[3*i]),hx=lx,ly=read_real(v,3*(int64_t)faces[3*i]+1),hy=ly;
            for(k=1;k<3;++k){double a=read_real(v,3*(int64_t)faces[3*i+k]),b=read_real(v,3*(int64_t)faces[3*i+k]+1);if(a<lx)lx=a;if(a>hx)hx=a;if(b<ly)ly=b;if(b>hy)hy=b;}
            if(hx<px||lx>px+1||hy<py||ly>py+1)continue;
            if(n==CANDIDATES)return 2;s->ids[n++]=i;}
        for(j=0;j<n;++j)s->triangles[j]=triangle(v,faces+3*s->ids[j],-1,px,py);
        separation_clear(s,n);
        for(j=0;j<n;++j)for(k=j+1;k<n;++k)
            if(separated_edge(v,faces+3*s->ids[j],faces+3*s->ids[k]) || separated_edge(v,faces+3*s->ids[k],faces+3*s->ids[j])){
                separation_set(s,j,k);separation_set(s,k,j);
            }
        /* Count topology with no sample writes. A single choice per pixel is
           reused for every derivative seed, and never exceeds the same800slots.
           Count and emission share ctx.work, so the existing work guard stays. */
        quadrature_pixel(s,n,cuts,-1,y*w+x,0,5,&curves,&sample_count,NULL,NULL,NULL,&value,&ctx,NULL);
        if(ctx.failed)return 2;
        if(curves && sample_count+curves*(288-120)<=QUAD_SLOTS)radial_order=12;
        /* Retain curved priority. The remaining slots choose one flat inner
           order for every fan; no contribution is removed or extra storage used.
           Freeze this unseeded discrete choice for every first-VJP seed. */
        {int flat_count=(sample_count-120*curves)/25;
          if(flat_count){flat_order=(QUAD_SLOTS-24*radial_order*curves)/(5*flat_count);
            if(flat_order>8)flat_order=8;if(flat_order<5)return 2;
            if(material_transport && quad_linear_count(cuts)==0 && 44*flat_count+24*radial_order*curves<=QUAD_SLOTS)flat_order=-44;}
        }
        quadrature_pixel(s,n,cuts,-1,y*w+x,radial_order,flat_order,&curves,&sample_count,out,other,gradient,&value,&ctx,NULL);
        if(ctx.failed)return 2;
        if(!gradient)continue;
        for(j=0;j<n;++j)for(k=0;k<3;++k){int a;int64_t id=faces[3*s->ids[j]+k];for(a=0;a<nv;++a)if(vertices[a]==id)break;if(a==nv)vertices[nv++]=id;}
        for(j=0;j<nv;++j)for(k=0;k<3;++k){int a;int64_t seed=3*vertices[j]+k;ctx.failed=0;ctx.work=0;
            for(a=0;a<n;++a)s->triangles[a]=triangle(v,faces+3*s->ids[a],seed,px,py);
            quadrature_pixel(s,n,cuts,-1,y*w+x,radial_order,flat_order,&curves,&sample_count,NULL,NULL,gradient,&value,&ctx,NULL);
            if(ctx.failed)return 2;
            write_real(out,seed,read_real(out,seed)+value);}
        for(j=0;j<n;++j)s->triangles[j]=triangle(v,faces+3*s->ids[j],-1,px,py);
        for(j=0;j<n;++j)for(k=0;k<cuts->shape[1]*cuts->shape[2];++k){int64_t seed=cuts->shape[1]*cuts->shape[2]*s->ids[j]+k;ctx.failed=0;ctx.work=0;
            quadrature_pixel(s,n,cuts,seed,y*w+x,radial_order,flat_order,&curves,&sample_count,NULL,NULL,gradient,&value,&ctx,NULL);
            if(ctx.failed)return 2;
            write_real(other,seed,read_real(other,seed)+value);}
    }
    if(gradient)for(k=0;k<2;++k)for(q=0;q<gffx_mesh_element_count(outputs[k]);++q)if(!isfinite(read_real(outputs[k],(int64_t)q)))return 2;
    return 0;
}
