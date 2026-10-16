/* Original bounded C11 research models, not a production GFFX operation.
 * Complete analytic coverage; centre-sample-discovered boundary coverage;
 * and an EdgeGrad-inspired pixel-pair estimator for constant face colours.
 * This is not either upstream implementation or its benchmark reproduction. */
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#define MAX_V 64
#define MAX_F 32
#define MAX_P 64
#define MAX_PARTS 128
typedef struct { double v,d; } D;
typedef struct { D x,y; } P;
typedef struct { D a,b,c; } Line;
typedef struct { int n; P p[MAX_P]; } Poly;
typedef struct { int valid; Line edge[3],depth; } Tri;
typedef struct { int failed,work; } Ctx;
static D val(double v) { D r={v,0};return r; }
static D add(D a,D b) { D r={a.v+b.v,a.d+b.d};return r; }
static D neg(D a) { D r={-a.v,-a.d};return r; }
static D sub(D a,D b) { return add(a,neg(b)); }
static D mul(D a,D b) { D r={a.v*b.v,a.d*b.v+a.v*b.d};return r; }
static D divd(D a,D b) { D r={a.v/b.v,(a.d*b.v-a.v*b.d)/(b.v*b.v)};return r; }
static D at(Line l,P p) { return add(add(mul(l.a,p.x),mul(l.b,p.y)),l.c); }
static Line negative(Line l) { Line r={neg(l.a),neg(l.b),neg(l.c)};return r; }
static Line difference(Line a,Line b) { Line r={sub(a.a,b.a),sub(a.b,b.b),sub(a.c,b.c)};return r; }
static P point(double x,double y) { P p={ {x,0},{y,0} };return p; }
static void put(Poly *p,P q,Ctx *ctx) {
    if(p->n && p->p[p->n-1].x.v==q.x.v && p->p[p->n-1].y.v==q.y.v && p->p[p->n-1].x.d==q.x.d && p->p[p->n-1].y.d==q.y.d)return;
    if(p->n==MAX_P) {ctx->failed=1;return;}
    p->p[p->n++]=q;
}
static Poly clip(const Poly *p,Line l,Ctx *ctx) {
    Poly r;int i;P before;D a;r.n=0;
    if(!p->n || ctx->failed)return r;
    before=p->p[p->n-1];a=at(l,before);
    for(i=0;i<p->n;++i) {
        P now=p->p[i];D b=at(l,now);
        if(++ctx->work>2000000){ctx->failed=1;r.n=0;return r;}
        if((a.v>=0)!=(b.v>=0)) {
            D t=divd(a,sub(a,b));P hit={add(before.x,mul(t,sub(now.x,before.x))),add(before.y,mul(t,sub(now.y,before.y)))};
            put(&r,hit,ctx);
        }
        if(b.v>=0)put(&r,now,ctx);
        before=now;a=b;
    }
    if(r.n>1 && r.p[0].x.v==r.p[r.n-1].x.v && r.p[0].y.v==r.p[r.n-1].y.v && r.p[0].x.d==r.p[r.n-1].x.d && r.p[0].y.d==r.p[r.n-1].y.d)--r.n;
    if(r.n<3)r.n=0;
    return r;
}
static D area(const Poly *p) {
    D total=val(0);int i;
    for(i=0;i<p->n;++i) {
        P a=p->p[i],b=p->p[(i+1)%p->n];
        total=add(total,sub(mul(a.x,b.y),mul(a.y,b.x)));
    }
    if(total.v<0)total=neg(total);
    return mul(total,val(.5));
}
static Tri triangle(const double *v,const double *velocity,const int *f) {
    D xyz[3][3],ux,uy,vx,vy,det,dz1,dz2;P xy[3];Tri t;int i,k;
    memset(&t,0,sizeof(t));
    for(i=0;i<3;++i)for(k=0;k<3;++k){xyz[i][k].v=v[3*f[i]+k];xyz[i][k].d=velocity[3*f[i]+k];}
    ux=sub(xyz[1][0],xyz[0][0]);uy=sub(xyz[1][1],xyz[0][1]);
    vx=sub(xyz[2][0],xyz[0][0]);vy=sub(xyz[2][1],xyz[0][1]);
    det=sub(mul(ux,vy),mul(uy,vx));if(det.v==0)return t;
    dz1=sub(xyz[1][2],xyz[0][2]);dz2=sub(xyz[2][2],xyz[0][2]);
    t.depth.a=divd(sub(mul(dz1,vy),mul(dz2,uy)),det);
    t.depth.b=divd(sub(mul(ux,dz2),mul(vx,dz1)),det);
    t.depth.c=sub(sub(xyz[0][2],mul(t.depth.a,xyz[0][0])),mul(t.depth.b,xyz[0][1]));
    for(i=0;i<3;++i){k=det.v<0?2-i:i;xy[i].x=xyz[k][0];xy[i].y=xyz[k][1];}
    for(i=0;i<3;++i) {
        P a=xy[i],b=xy[(i+1)%3];D dx=sub(b.x,a.x),dy=sub(b.y,a.y);
        t.edge[i].a=neg(dy);t.edge[i].b=dx;t.edge[i].c=sub(mul(dy,a.x),mul(dx,a.y));
    }
    t.valid=1;return t;
}
static int inside(const Tri *t,P p) {
    return t->valid && at(t->edge[0],p).v>=0 && at(t->edge[1],p).v>=0 && at(t->edge[2],p).v>=0;
}
static int hard(const Tri *t,int nf,P p) {
    int i,best=-1;double z=0;
    for(i=0;i<nf;++i)if(inside(t+i,p)) {
        double q=at(t[i].depth,p).v;if(best<0 || q<z){best=i;z=q;}
    }
    return best;
}
static void append(Poly *p,int *n,Poly part,Ctx *ctx) {
    if(part.n && area(&part).v>0) {
        if(*n==MAX_PARTS){ctx->failed=1;return;}
        p[(*n)++]=part;
    }
}
static void subtract(const Poly *p,const Tri *other,Line depth,int tie,Poly *out,int *count,Ctx *ctx) {
    Poly remain=*p;int k;
    for(k=0;k<(tie?3:4);++k) {
        Line l=k<3?other->edge[k]:depth;
        append(out,count,clip(&remain,negative(l),ctx),ctx);
        remain=clip(&remain,l,ctx);
        if(!remain.n || area(&remain).v==0 || ctx->failed)return;
    }
}
static int pixel(const Tri *t,int nf,const double *rgb,double x,double y,const int *selected,
                 Poly *work,double *image,double *gradient,Ctx *ctx) {
    Poly p;int i,j,k,c;D total[4]={ {0,0},{0,0},{0,0},{0,0} };
    p.n=4;p.p[0]=point(x,y);p.p[1]=point(x+1,y);p.p[2]=point(x+1,y+1);p.p[3]=point(x,y+1);
    for(i=0;i<nf && !ctx->failed;++i)if(t[i].valid && selected[i]) {
        Poly subject=p,*a=work,*b=work+MAX_PARTS;int count=1;
        for(k=0;k<3;++k)subject=clip(&subject,t[i].edge[k],ctx);
        if(!subject.n || area(&subject).v==0)continue;
        a[0]=subject;
        for(j=0;j<nf && count && !ctx->failed;++j)if(j!=i && t[j].valid && selected[j]) {
            Line d=difference(t[i].depth,t[j].depth);int n=0;
            int tie=d.a.v==0 && d.b.v==0 && d.c.v==0;
            if(tie && j>i)continue;
            for(k=0;k<count && !ctx->failed;++k)subtract(a+k,t+j,d,tie,b,&n,ctx);
            {Poly *temp=a;a=b;b=temp;}count=n;
        }
        for(k=0;k<count;++k) {
            D ar=area(a+k);total[3]=add(total[3],ar);
            for(c=0;c<3;++c)total[c]=add(total[c],mul(ar,val(rgb[3*i+c])));
        }
    }
    for(c=0;c<4;++c) {
        image[c]=total[c].v;gradient[c]=total[c].d;
        if(!isfinite(image[c]) || !isfinite(gradient[c]))ctx->failed=1;
    }
    return ctx->failed;
}
static double edge_velocity(const Tri *t,int ia,int ib,P pa,P pb,int axis) {
    int ab=ia>=0 && inside(t+ia,pb),ba=ib>=0 && inside(t+ib,pa),k;
    Line l;D root;double fixed=axis?pa.x.v:pa.y.v;
    if(ia>=0 && ib>=0 && !ab && !ba)return 0; /* adjacent: no visibility jump */
    if(ab && ba)l=difference(t[ia].depth,t[ib].depth);
    else {
        int front=ia>=0 && !ab?ia:ib;
        if(front<0)return 0;
        for(k=0;k<3;++k) {
            double a=at(t[front].edge[k],pa).v,b=at(t[front].edge[k],pb).v;
            if((a>=0)!=(b>=0))break;
        }
        if(k==3)return 0;
        l=t[front].edge[k];
    }
    if((axis?l.b.v:l.a.v)==0)return 0;
    root=divd(neg(add(mul(axis?l.a:l.b,val(fixed)),l.c)),axis?l.b:l.a);
    return root.d;
}
static void micro_pair(const Tri *t,int nf,const double *rgb,int h,int w,double ox,double oy,
                       int x,int y,int axis,double *gradient) {
    int bx=x+(axis==0),by=y+(axis==1),c,ia,ib;double speed;
    P a=point(ox+x+.5,oy+y+.5),b=point(ox+bx+.5,oy+by+.5);
    ia=hard(t,nf,a);ib=hard(t,nf,b);if(ia==ib)return;
    speed=edge_velocity(t,ia,ib,a,b,axis);
    for(c=0;c<4;++c) {
        double va=ia<0?0:(c==3?1:rgb[3*ia+c]),vb=ib<0?0:(c==3?1:rgb[3*ib+c]);
        double g=.5*(va-vb)*speed;
        if(x>=0 && x<w && y>=0 && y<h)gradient[4*(y*w+x)+c]+=g;
        if(bx>=0 && bx<w && by>=0 && by<h)gradient[4*(by*w+bx)+c]+=g;
    }
}
EXPORT int opaque_model(int model,int nv,const double *vertices,const double *velocity,
                        int nf,const int *faces,const double *rgb,int h,int w,double ox,double oy,
                        double *image,double *gradient) {
    Tri t[MAX_F];Ctx ctx={0,0};Poly *work=NULL;int i,j,x,y,c;
    if(model<0 || model>2 || nv<0 || nv>MAX_V || nf<0 || nf>MAX_F || h<1 || h>8 || w<1 || w>8)return 1;
    for(i=0;i<nv*3;++i)if(!isfinite(vertices[i]) || !isfinite(velocity[i]) || fabs(vertices[i])>16 || fabs(velocity[i])>16)return 1;
    for(i=0;i<nf*3;++i)if(faces[i]<0 || faces[i]>=nv || !isfinite(rgb[i]) || fabs(rgb[i])>16)return 1;
    if(!isfinite(ox) || !isfinite(oy) || fabs(ox)>16 || fabs(oy)>16)return 1;
    memset(image,0,(size_t)h*w*4*sizeof(double));memset(gradient,0,(size_t)h*w*4*sizeof(double));
    for(i=0;i<nf;++i)t[i]=triangle(vertices,velocity,faces+3*i);
    if(model!=2){work=(Poly*)malloc(2*MAX_PARTS*sizeof(Poly));if(!work)return 2;}
    for(y=0;y<h && !ctx.failed;++y)for(x=0;x<w && !ctx.failed;++x) {
        int selected[MAX_F]={0};
        if(model==2) {
            i=hard(t,nf,point(ox+x+.5,oy+y+.5));
            if(i>=0)for(c=0;c<4;++c)image[4*(y*w+x)+c]=c==3?1:rgb[3*i+c];
        } else {
            if(model==0)for(i=0;i<nf;++i)selected[i]=1;
            else for(i=-1;i<=1;++i)for(j=-1;j<=1;++j) {
                int id=hard(t,nf,point(ox+x+j+.5,oy+y+i+.5));if(id>=0)selected[id]=1;
            }
            pixel(t,nf,rgb,ox+x,oy+y,selected,work,image+4*(y*w+x),gradient+4*(y*w+x),&ctx);
        }
    }
    free(work);
    if(model==2) {
        for(y=0;y<h;++y)for(x=-1;x<w;++x)micro_pair(t,nf,rgb,h,w,ox,oy,x,y,0,gradient);
        for(y=-1;y<h;++y)for(x=0;x<w;++x)micro_pair(t,nf,rgb,h,w,ox,oy,x,y,1,gradient);
        for(i=0;i<4*h*w;++i)if(!isfinite(gradient[i]))ctx.failed=1;
    }
    return ctx.failed?3:0;
}
