/* Shared native edge visibility. Requires opaque_quadrature_scalar.h first. */
typedef struct {double lo,hi;} EdgeInterval;
GFFX_OPAQUE_LOCAL int edge_half(EdgeInterval *s,double a,double b,Ctx *ctx){
    double root;if(++ctx->work>2000000){ctx->failed=1;return 0;}
    if(b==0)return a>=0;
    root=-a/b;if(!isfinite(root)){ctx->failed=1;return 0;}
    if(b>0){if(root>s->lo)s->lo=root;}else if(root<s->hi)s->hi=root;
    return s->hi>s->lo;
}
GFFX_OPAQUE_LOCAL int edge_append(EdgeInterval *out,int *n,double lo,double hi,Ctx *ctx){
    if(!(hi>lo))return 1;
    if(*n==MAX_PARTS){ctx->failed=1;return 0;}
    out[*n].lo=lo;out[*n].hi=hi;++*n;return 1;
}
GFFX_OPAQUE_LOCAL int edge_roots(const double *c,const double *b0,const double *bd,double roots[2],Ctx *ctx){
    const int ia[6]={0,1,2,0,0,1},ib[6]={0,1,2,1,2,2};
    double a=0,b=0,d=0,disc,q;int k,n=0,active=0;QR rd;
    for(k=0;k<6;++k){int i=ia[k],j=ib[k];if(c[k]!=0)active=1;
      a+=c[k]*(bd[i]*bd[j]);b+=c[k]*(b0[i]*bd[j]+bd[i]*b0[j]);d+=c[k]*(b0[i]*b0[j]);}
    if(!active)return 0;
    if(!isfinite(a)||!isfinite(b)||!isfinite(d)){ctx->failed=1;return 0;}
    {int s0=qr_sign(qr(d)),s1=qr_sign(qr_add(qr(d),qr_mul(qr(.5),qr(b)))),s2=qr_sign(qr_add(qr_add(qr(a),qr(b)),qr(d)));if(s0&&s1==s0&&s2==s0)return 0;}
    if(a==0){if(b==0){if(d==0)ctx->failed=1;return 0;}roots[0]=-d/b;return 1;}
    rd=qr_sub(qr_mul(qr(b),qr(b)),qr_mul(qr(4),qr_mul(qr(a),qr(d))));
    if(rd.hi<0)return 0;if(rd.lo<=0){ctx->failed=1;return 0;}
    disc=b*b-4*a*d;q=-.5*(b+copysign(sqrt(disc),b));
    roots[n++]=q/a;roots[n++]=d/q;
    if(roots[0]>roots[1]){double z=roots[0];roots[0]=roots[1];roots[1]=z;}
    if(!isfinite(roots[0])||!isfinite(roots[1]))ctx->failed=1;
    return n;
}
GFFX_OPAQUE_LOCAL int opaque_boundary_execute_common(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,int edge,gffx_tensor_view *ids,gffx_tensor_view *samples,Scratch *s,const int32_t *cursor,int32_t *receipt,int max_batches){
    const double nodes[12]={0.0092196828766403782,0.047941371814762601,0.11504866290284765,0.20634102285669126,0.31608425050090994,0.43738329574426554,0.5626167042557344,0.68391574949909006,0.79365897714330869,0.88495133709715235,0.95205862818523745,0.99078031712335957};
    const double weights[12]={0.023587668193256011,0.053469662997659442,0.080039164271673055,0.10158371336153282,0.11674626826917732,0.12457352290670134,0.12457352290670134,0.11674626826917732,0.10158371336153282,0.080039164271673055,0.053469662997659442,0.023587668193256011};
    EdgeInterval storage[2][MAX_PARTS];const int32_t *faces=(const int32_t*)gffx_mesh_elements_const(f);
    uint64_t index;int64_t x,y,fi;int nc=quad_linear_count(cuts);QuadBatch stream;
    if(cursor&&(h!=1||w!=1||!receipt||quad_batch_init_profile(&stream,cursor[0],cursor[1],cursor[2],cursor[3],max_batches)))return 1;
#ifndef GFFX_OPAQUE_PRECLEARED
    for(index=0;index<gffx_mesh_element_count(ids);++index)((int32_t*)gffx_mesh_elements(ids))[index]=-1;
    for(index=0;index<gffx_mesh_element_count(samples);++index)write_real(samples,(int64_t)index,0);
#endif
    for(y=0;y<h;++y)for(x=0;x<w;++x){
      int n=0,i,j,k,used=0,segments=0;double px=ox+x,py=oy+y;Ctx ctx={0,0};P start=edge<2?point(edge,0):point(0,edge-2);
      for(fi=0;fi<f->shape[0];++fi){double lx=read_real(v,3*(int64_t)faces[3*fi]),hx=lx,ly=read_real(v,3*(int64_t)faces[3*fi]+1),hy=ly;
        for(k=1;k<3;++k){double a=read_real(v,3*(int64_t)faces[3*fi+k]),b=read_real(v,3*(int64_t)faces[3*fi+k]+1);if(a<lx)lx=a;if(a>hx)hx=a;if(b<ly)ly=b;if(b>hy)hy=b;}
        if(hx<px||lx>px+1||hy<py||ly>py+1)continue;if(n==(cursor?SH_CANDIDATES:CANDIDATES))return 2;s->ids[n++]=fi;
      }
      for(i=0;i<n;++i)s->triangles[i]=triangle(v,faces+3*s->ids[i],-1,px,py);
      for(i=0;i<n;++i)if(s->triangles[i].tri.valid){
        EdgeInterval initial={0,1},*a=storage[0],*b=storage[1];int count=1,valid=1,coincident=0;double b0[3],bd[3];
        for(k=0;k<3;++k){Line l=s->triangles[i].bary[k];b0[k]=at(l,start).v;bd[k]=edge<2?l.b.v:l.a.v;
          if(b0[k]==0&&bd[k]==0)coincident=1;if(!edge_half(&initial,b0[k],bd[k],&ctx))valid=0;}
        if(ctx.failed)return 2;if(!valid)continue;if(coincident)return 2;a[0]=initial;
        for(j=0;j<n&&count;++j)if(i!=j&&s->triangles[j].tri.valid){
          EdgeInterval hidden={0,1};Line depth=difference(s->triangles[i].depth,s->triangles[j].depth);int ok=1,m=0,t;
          int tie=depth.a.v==0&&depth.b.v==0&&depth.c.v==0;
          if(tie&&j>i)continue;
          for(k=0;k<3;++k){Line l=s->triangles[j].bary[k];if(!edge_half(&hidden,at(l,start).v,edge<2?l.b.v:l.a.v,&ctx))ok=0;}
          if(!tie&&!edge_half(&hidden,at(depth,start).v,edge<2?depth.b.v:depth.a.v,&ctx))ok=0;
          if(ctx.failed)return 2;if(!ok)continue;
          for(t=0;t<count;++t){double lo=hidden.lo>a[t].lo?hidden.lo:a[t].lo,hi=hidden.hi<a[t].hi?hidden.hi:a[t].hi;
            if(hi<=lo)edge_append(b,&m,a[t].lo,a[t].hi,&ctx);
            else{edge_append(b,&m,a[t].lo,lo,&ctx);edge_append(b,&m,hi,a[t].hi,&ctx);}}
          {EdgeInterval *tmp=a;a=b;b=tmp;}count=m;if(ctx.failed)return 2;
        }
        for(j=0;j<nc&&count;++j){double u=0,d=0,root;int t,m=0,active=0;
          for(k=0;k<3;++k){double c=read_real(cuts,s->ids[i]*cuts->shape[1]*cuts->shape[2]+k*nc+j);if(c!=0)active=1;u+=c*b0[k];d+=c*bd[k];}
          if(d==0){if(active&&u==0)return 2;continue;}root=-u/d;if(!isfinite(root))return 2;
          for(t=0;t<count;++t){if(root>a[t].lo&&root<a[t].hi){edge_append(b,&m,a[t].lo,root,&ctx);edge_append(b,&m,root,a[t].hi,&ctx);}else edge_append(b,&m,a[t].lo,a[t].hi,&ctx);}
          {EdgeInterval *tmp=a;a=b;b=tmp;}count=m;if(ctx.failed)return 2;
        }
        if(cuts->shape[1]==1&&count){double c[6],roots[2];int nr,t,m;
          for(k=0;k<6;++k)c[k]=read_real(cuts,s->ids[i]*cuts->shape[2]+3*nc+k);
          nr=edge_roots(c,b0,bd,roots,&ctx);if(ctx.failed)return 2;
          for(j=0;j<nr;++j){m=0;for(t=0;t<count;++t){double root=roots[j];if(root>a[t].lo&&root<a[t].hi){edge_append(b,&m,a[t].lo,root,&ctx);edge_append(b,&m,root,a[t].hi,&ctx);}else edge_append(b,&m,a[t].lo,a[t].hi,&ctx);}
            {EdgeInterval *tmp=a;a=b;b=tmp;}count=m;if(ctx.failed)return 2;}
        }
        for(j=0;j<count;++j){int t;double span=a[j].hi-a[j].lo;if(cursor){int offer=quad_batch_offer(&stream,12,1);if(offer<0)return 2;if(offer==2)goto boundary_finish;if(!offer)continue;}else if(++segments>32||used>QUAD_SLOTS-12)return 2;
          for(t=0;t<12;++t){double at_t=a[j].lo+span*nodes[t];int64_t pos=(y*w+x)*QUAD_SLOTS+used++;
            for(k=0;k<3;++k){double bval=b0[k]+bd[k]*at_t;if(!isfinite(bval))return 2;write_real(samples,4*pos+k,bval);}
            write_real(samples,4*pos+3,span*weights[t]);((int32_t*)gffx_mesh_elements(ids))[pos]=(int32_t)s->ids[i];}
        }
      }
boundary_finish:
      if(cursor){if(quad_batch_finish(&stream))return 2;
        receipt[0]=stream.seen;receipt[1]=!stream.full;receipt[2]=stream.slots;receipt[3]=stream.blocks;
        receipt[4]=stream.total_slots;receipt[5]=stream.total_blocks;receipt[6]=stream.emitted;receipt[7]=ctx.work;}
    }
    return 0;
}
GFFX_OPAQUE_LOCAL int opaque_boundary_execute(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,int edge,gffx_tensor_view *ids,gffx_tensor_view *samples,Scratch *s){return opaque_boundary_execute_common(v,f,cuts,h,w,ox,oy,edge,ids,samples,s,NULL,NULL,64);}
GFFX_OPAQUE_LOCAL int opaque_boundary_batch_execute(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,int edge,const int32_t *cursor,int32_t *receipt,gffx_tensor_view *ids,gffx_tensor_view *samples,Scratch *s){return opaque_boundary_execute_common(v,f,cuts,1,1,ox,oy,edge,ids,samples,s,cursor,receipt,64);}


GFFX_OPAQUE_LOCAL int opaque_boundary_batch_execute80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,int edge,const int32_t *cursor,int32_t *receipt,gffx_tensor_view *ids,gffx_tensor_view *samples,Scratch *s){return opaque_boundary_execute_common(v,f,cuts,1,1,ox,oy,edge,ids,samples,s,cursor,receipt,80);}
