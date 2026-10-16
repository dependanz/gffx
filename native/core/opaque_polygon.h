/* Original fixed-capacity scalar-dual polygon arithmetic. Internal only. */

#ifndef GFFX_OPAQUE_POLYGON_H

#define GFFX_OPAQUE_POLYGON_H

#ifndef GFFX_OPAQUE_LOCAL

#define GFFX_OPAQUE_LOCAL static

#endif

#define MAX_P 64

#define MAX_PARTS 128

typedef struct { double v,d; } D;

typedef struct { D x,y; } P;

typedef struct { D a,b,c; } Line;

typedef struct { int n; P p[MAX_P]; int edge[MAX_P]; } Poly;

typedef struct { int valid; Line edge[3]; } Tri;

/* Optional batch-only line provenance lives in caller scratch. Legacy clipping

 * does not use it and keeps its original arithmetic. */

#define MAX_SUPPORT_LINES 512

typedef struct { int failed,work; Line *support; int support_count; int canonical_supports; } Ctx;

GFFX_OPAQUE_LOCAL D val(double v) { D r={v,0};return r; }

GFFX_OPAQUE_LOCAL D add(D a,D b) { D r={a.v+b.v,a.d+b.d};return r; }

GFFX_OPAQUE_LOCAL D neg(D a) { D r={-a.v,-a.d};return r; }

GFFX_OPAQUE_LOCAL D sub(D a,D b) { return add(a,neg(b)); }

GFFX_OPAQUE_LOCAL D mul(D a,D b) { D r={a.v*b.v,a.d*b.v+a.v*b.d};return r; }

GFFX_OPAQUE_LOCAL D divd(D a,D b) { D r={a.v/b.v,(a.d*b.v-a.v*b.d)/(b.v*b.v)};return r; }

GFFX_OPAQUE_LOCAL D at(Line l,P p) { return add(add(mul(l.a,p.x),mul(l.b,p.y)),l.c); }

GFFX_OPAQUE_LOCAL Line negative(Line l) { Line r={neg(l.a),neg(l.b),neg(l.c)};return r; }

GFFX_OPAQUE_LOCAL Line difference(Line a,Line b) { Line r={sub(a.a,b.a),sub(a.b,b.b),sub(a.c,b.c)};return r; }

GFFX_OPAQUE_LOCAL P point(double x,double y) { P p={ {x,0},{y,0} };return p; }

GFFX_OPAQUE_LOCAL void put(Poly *p,P q,Ctx *ctx) {

    if(p->n && p->p[p->n-1].x.v==q.x.v && p->p[p->n-1].y.v==q.y.v && (ctx->support || (p->p[p->n-1].x.d==q.x.d && p->p[p->n-1].y.d==q.y.d)))return;

    if(p->n==MAX_P) {ctx->failed=1;return;}

    p->edge[p->n]=-1;p->p[p->n++]=q;

}

GFFX_OPAQUE_LOCAL int support_line(Ctx *ctx,Line l) {

    int i;for(i=0;i<ctx->support_count;++i){Line q=ctx->support[i];

        if(q.a.v==l.a.v&&q.a.d==l.a.d&&q.b.v==l.b.v&&q.b.d==l.b.d&&q.c.v==l.c.v&&q.c.d==l.c.d)return i;

    }

    if(i==MAX_SUPPORT_LINES){ctx->failed=1;return -1;}ctx->support[i]=l;ctx->support_count++;return i;

}

GFFX_OPAQUE_LOCAL void put_edge(Poly *p,P q,int edge,Ctx *ctx){

    int n=p->n;put(p,q,ctx);if(p->n>n)p->edge[p->n-1]=edge;

}

/* Differentiate the two original support constraints at the rounded hit.

 * Chained interpolation through an almost-zero clipped edge can corrupt its

 * tangent even when its forward coordinates are accurate. No epsilon, snapping

 * or contribution removal is used; only the derivative calculation changes. */

GFFX_OPAQUE_LOCAL void support_tangent(P *hit,Line a,Line b,Ctx *ctx){

    double det=a.a.v*b.b.v-a.b.v*b.a.v;

    double ra=-((a.a.d*hit->x.v+a.b.d*hit->y.v)+a.c.d);

    double rb=-((b.a.d*hit->x.v+b.b.d*hit->y.v)+b.c.d);

    if(det==0.||!isfinite(det)){ctx->failed=1;return;}

    hit->x.d=(ra*b.b.v-a.b.v*rb)/det;hit->y.d=(a.a.v*rb-ra*b.a.v)/det;

    if(!isfinite(hit->x.d)||!isfinite(hit->y.d))ctx->failed=1;

}

#ifndef GFFX_OPAQUE_CLIP_LOCAL

#define GFFX_OPAQUE_CLIP_LOCAL GFFX_OPAQUE_LOCAL

#endif

/* A vertex on an identical moving support has identically zero signed
 * constraint value. Use provenance, not an epsilon or coordinate snapping. */
GFFX_OPAQUE_LOCAL int support_same(Line a,Line b){
    return a.a.v==b.a.v&&a.a.d==b.a.d&&a.b.v==b.b.v&&a.b.d==b.b.d&&a.c.v==b.c.v&&a.c.d==b.c.d;
}
GFFX_OPAQUE_LOCAL D support_at(const Poly *p,int i,Line l,Ctx *ctx){
    if(ctx->canonical_supports&&ctx->support){int k;
        for(k=0;k<2;++k){int id=p->edge[(i+k)%p->n];
            if(id>=0&&id<ctx->support_count){Line a=ctx->support[id];if(support_same(a,l)||support_same(negative(a),l))return val(0);}
        }
    }
    return at(l,p->p[i]);
}
GFFX_OPAQUE_CLIP_LOCAL Poly clip_classified(const Poly *p,Line l,Ctx *ctx,int *strict_side) {

    Poly r;int i,line_id=-1,allpos=1,allneg=1;P before;D a;r.n=0;
    if(strict_side)*strict_side=0;

    if(!p->n || ctx->failed)return r;

    if(ctx->support){line_id=support_line(ctx,l);if(ctx->failed)return r;}

    before=p->p[p->n-1];a=support_at(p,p->n-1,l,ctx);

    for(i=0;i<p->n;++i) {

        P now=p->p[i];D b=support_at(p,i,l,ctx);
        if(strict_side){if(!isfinite(b.v)||!(b.v>0.))allpos=0;if(!isfinite(b.v)||!(b.v<0.))allneg=0;}

        if(++ctx->work>2000000){ctx->failed=1;r.n=0;return r;}

        if((a.v>=0)!=(b.v>=0)) {

            /* Interpolate from the nearer endpoint. Forming 1-t at the farther

               endpoint can erase a strictly positive subpixel contribution. */

            P hit;

            if(fabs(b.v)<fabs(a.v)) {

                D t=divd(b,sub(b,a));hit.x=add(now.x,mul(t,sub(before.x,now.x)));hit.y=add(now.y,mul(t,sub(before.y,now.y)));

            } else {

                D t=divd(a,sub(a,b));hit.x=add(before.x,mul(t,sub(now.x,before.x)));hit.y=add(before.y,mul(t,sub(now.y,before.y)));

            }

            if(ctx->support){

                int incoming=p->edge[i];if(incoming<0||incoming>=ctx->support_count){ctx->failed=1;return r;}

                support_tangent(&hit,ctx->support[incoming],l,ctx);if(ctx->failed)return r;

                put_edge(&r,hit,a.v>=0?incoming:line_id,ctx);

            }else put(&r,hit,ctx);

        }

        if(b.v>=0){if(ctx->support)put_edge(&r,now,p->edge[i],ctx);else put(&r,now,ctx);}

        before=now;a=b;

    }

    if(r.n>1 && r.p[0].x.v==r.p[r.n-1].x.v && r.p[0].y.v==r.p[r.n-1].y.v && (ctx->support || (r.p[0].x.d==r.p[r.n-1].x.d && r.p[0].y.d==r.p[r.n-1].y.d))){if(ctx->support)r.edge[0]=r.edge[r.n-1];--r.n;}

    if(r.n<3)r.n=0;

    if(strict_side && !ctx->failed)*strict_side=allpos?1:(allneg?-1:0);
    return r;
}
GFFX_OPAQUE_CLIP_LOCAL Poly clip(const Poly *p,Line l,Ctx *ctx){return clip_classified(p,l,ctx,NULL);}
/* The preceding clip used -l. For finite strictly one-sided values, the
 * opposite predicate has the opposite sign: symmetric rounded products/adds
 * and the same provenance-zero test. Zero/contact/nonfinite always replays.
 * Register l in the original order, including capacity failure, and perform
 * the original all-inside emission/normalization without recalculating signs.
 * Vertex predicates eliminated here are genuinely not evaluated or charged. */
GFFX_OPAQUE_CLIP_LOCAL Poly clip_complement(const Poly *p,Line l,int negative_side,Ctx *ctx){
 Poly r;int i;r.n=0;
 if(!negative_side)return clip(p,l,ctx);
 if(!p->n || ctx->failed)return r;
 if(ctx->support){support_line(ctx,l);if(ctx->failed)return r;}
 if(negative_side>0)return r;
 for(i=0;i<p->n;++i){if(ctx->support)put_edge(&r,p->p[i],p->edge[i],ctx);else put(&r,p->p[i],ctx);}
 if(r.n>1 && r.p[0].x.v==r.p[r.n-1].x.v && r.p[0].y.v==r.p[r.n-1].y.v && (ctx->support || (r.p[0].x.d==r.p[r.n-1].x.d && r.p[0].y.d==r.p[r.n-1].y.d))){if(ctx->support)r.edge[0]=r.edge[r.n-1];--r.n;}
 if(r.n<3)r.n=0;
 return r;
}

GFFX_OPAQUE_LOCAL D area(const Poly *p) {
    D total=val(0);int i;
    if(p->n<3)return total;
    /* Translate each fan to its first vertex before forming products. The
     * absolute-coordinate shoelace sum can round a positive thin strip to
     * zero and drop its finite boundary derivative. Convex fan terms have
     * the same orientation; no epsilon or small-area cutoff is used. */
    P origin=p->p[0];
    for(i=1;i<p->n-1;++i) {
        D ax=sub(p->p[i].x,origin.x),ay=sub(p->p[i].y,origin.y);
        D bx=sub(p->p[i+1].x,origin.x),by=sub(p->p[i+1].y,origin.y);
        total=add(total,sub(mul(ax,by),mul(ay,bx)));
    }
    if(total.v<0)total=neg(total);
    return mul(total,val(.5));
}

GFFX_OPAQUE_LOCAL void append(Poly *p,int *n,Poly part,Ctx *ctx) {

    if(part.n && area(&part).v>0) {

        if(*n==MAX_PARTS){ctx->failed=1;return;}

        p[(*n)++]=part;

    }

}

GFFX_OPAQUE_LOCAL void subtract_triangle(const Poly *p,const Tri *other,Poly *out,int *count,Ctx *ctx) {

    Poly remain=*p;int k;

    for(k=0;k<3;++k) {

        Line l=other->edge[k];append(out,count,clip(&remain,negative(l),ctx),ctx);

        remain=clip(&remain,l,ctx);

        if(!remain.n || area(&remain).v==0 || ctx->failed)return;

    }

}

#endif
