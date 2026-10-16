/* Shared CPU/CUDA persistent forward partition arithmetic. Caller owns all storage. */
GFFX_OPAQUE_LOCAL int opaque_partition_prepare_execute(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,gffx_tensor_view *rows,int32_t result[2],Scratch *s){
 int j,k,n=0,curves,count;double value;Ctx work={0};QuadBatch batch;QuadPartition partition;const int32_t *faces;
 faces=(const int32_t*)gffx_mesh_elements_const(f);
 for(int64_t i=0;i<f->shape[0];++i){double lx=read_real(v,3*(int64_t)faces[3*i]),hx=lx,ly=read_real(v,3*(int64_t)faces[3*i]+1),hy=ly;
  for(k=1;k<3;++k){double x=read_real(v,3*(int64_t)faces[3*i+k]),y=read_real(v,3*(int64_t)faces[3*i+k]+1);if(x<lx)lx=x;if(x>hx)hx=x;if(y<ly)ly=y;if(y>hy)hy=y;}
  if(hx<ox||lx>ox+1||hy<oy||ly>oy+1)continue;
  if(n==512||i>INT_MAX)return 2;s->ids[n++]=i;
 }
 work.canonical_supports=rows->shape[0]==2560||n>CANDIDATES;separation_clear(s,n);
 for(j=0;j<n;++j)for(k=j+1;k<n;++k)if(separated_edge(v,faces+3*s->ids[j],faces+3*s->ids[k])||separated_edge(v,faces+3*s->ids[k],faces+3*s->ids[j])||exact_separated_edge(v,faces+3*s->ids[j],faces+3*s->ids[k])||exact_separated_edge(v,faces+3*s->ids[k],faces+3*s->ids[j])){separation_set(s,j,k);separation_set(s,k,j);}
 for(j=0;j<n;++j)s->triangles[j]=work.canonical_supports?triangle_shared_edges(v,faces+3*s->ids[j],-1,ox,oy):triangle(v,faces+3*s->ids[j],-1,ox,oy);
 quad_batch_init_profile(&batch,0,0,0,0,(int)rows->shape[0]/32);partition.rows=rows;partition.count=0;partition.capacity=(int)rows->shape[0];
 quadrature_pixel_ex(s,n,cuts,-1,0,12,5,&curves,&count,NULL,NULL,NULL,&value,&work,&batch,&partition,NULL);
 result[1]=work.work;
 if(work.failed)return 2;
 result[0]=partition.count;return 0;
}
GFFX_OPAQUE_LOCAL int opaque_partition_emit_execute(const gffx_tensor_view *rows,int count,int begin,gffx_tensor_view *ids,gffx_tensor_view *samples){
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
 int i,k,ns,nt,used=0,end=begin+(rows->shape[0]==2560?28:32);uint64_t z;
 for(z=0;z<800;++z)((int32_t*)gffx_mesh_elements(ids))[z]=-1;
 for(z=0;z<3200;++z)write_real(samples,z,0.);
 if(end>count)end=count;
 for(i=begin;i<end;++i){int64_t base=(int64_t)i*17;P o=point(read_real(rows,base+1),read_real(rows,base+2)),u=point(read_real(rows,base+3),read_real(rows,base+4)),v=point(read_real(rows,base+5),read_real(rows,base+6));
  D ux=sub(u.x,o.x),uy=sub(u.y,o.y),vx=sub(v.x,o.x),vy=sub(v.y,o.y),det=val(read_real(rows,base+16));double face=read_real(rows,base);
  if(!isfinite(face)||face<0||face>INT_MAX||floor(face)!=face||!isfinite(det.v)||det.v<=0)return 1;
  if(rows->shape[0]==2560){
   for(ns=0;ns<28;++ns){D ss=val(sh_triangle28[ns][0]),tt=val(sh_triangle28[ns][1]),q[4];P point;
    point.x=add(o.x,add(mul(ss,ux),mul(tt,vx)));point.y=add(o.y,add(mul(ss,uy),mul(tt,vy)));
    for(k=0;k<3;++k){Line l={val(read_real(rows,base+7+3*k)),val(read_real(rows,base+8+3*k)),val(read_real(rows,base+9+3*k))};q[k]=at(l,point);}
    q[3]=mul(det,val(sh_triangle28[ns][2]));
    for(k=0;k<4;++k){if(!isfinite(q[k].v))return 2;write_real(samples,4*used+k,q[k].v);}
    ((int32_t*)gffx_mesh_elements(ids))[used++]=(int32_t)face;
   }
   continue;
  }
  for(ns=0;ns<5;++ns)for(nt=0;nt<5;++nt){D ss=val(sh_nodes[ns]),tt=val((1.-sh_nodes[ns])*nodes[nt]),q[4];P point;
   point.x=add(o.x,add(mul(ss,ux),mul(tt,vx)));point.y=add(o.y,add(mul(ss,uy),mul(tt,vy)));
   for(k=0;k<3;++k){Line l={val(read_real(rows,base+7+3*k)),val(read_real(rows,base+8+3*k)),val(read_real(rows,base+9+3*k))};q[k]=at(l,point);}
   q[3]=mul(det,val(sh_weights[ns]*weights[nt]));
   for(k=0;k<4;++k){if(!isfinite(q[k].v))return 2;write_real(samples,4*used+k,q[k].v);}
   ((int32_t*)gffx_mesh_elements(ids))[used++]=(int32_t)face;
  }
 }
 return 0;
}
