/* Fixed primal/16-input specializations. Included only by the owned header. */
typedef struct PM_NAME(dual) { double v,d[PM_DIM?PM_DIM:1]; } PM_NAME(dual);
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(constant)(double x){PM_NAME(dual) a;int i;a.v=x;for(i=0;i<PM_DIM;++i)a.d[i]=0.;return a;}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(input)(double x,int index){PM_NAME(dual) a=PM_NAME(constant)(x);if(PM_DIM)a.d[index]=1.;return a;}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(add)(PM_NAME(dual) a,PM_NAME(dual) b){int i;a.v+=b.v;for(i=0;i<PM_DIM;++i)a.d[i]+=b.d[i];return a;}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(sub)(PM_NAME(dual) a,PM_NAME(dual) b){int i;a.v-=b.v;for(i=0;i<PM_DIM;++i)a.d[i]-=b.d[i];return a;}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(mul)(PM_NAME(dual) a,PM_NAME(dual) b){PM_NAME(dual) out;int i;out.v=a.v*b.v;for(i=0;i<PM_DIM;++i)out.d[i]=a.d[i]*b.v+a.v*b.d[i];return out;}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(div)(PM_NAME(dual) a,PM_NAME(dual) b){PM_NAME(dual) out;int i;out.v=gffx_pc_div(a.v,b.v);for(i=0;i<PM_DIM;++i)out.d[i]=gffx_pc_div(a.d[i]-out.v*b.d[i],b.v);return out;}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(square)(PM_NAME(dual) a){return PM_NAME(mul)(a,a);}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(sqrt)(PM_NAME(dual) a){PM_NAME(dual) out;int i;
#if defined(__CUDA_ARCH__)
 out.v=__dsqrt_rn(a.v);
#else
 out.v=sqrt(a.v);
#endif
 for(i=0;i<PM_DIM;++i)out.d[i]=gffx_pc_div(a.d[i],2.*out.v);return out;
}
GFFX_PC_SCALAR int PM_NAME(unit)(const PM_NAME(dual) *a,PM_NAME(dual) *out){
 double maximum=0.,y=1.5;PM_NAME(dual) scale,length,root,s[3];int i,j,ties=0;
 for(j=0;j<3;++j){double v=a[j].v<0.?-a[j].v:a[j].v;if(v>maximum)maximum=v;}
 if(maximum==0.||!isfinite(maximum))return 1;scale=PM_NAME(constant)(maximum);
 for(j=0;j<3;++j)if((a[j].v<0.?-a[j].v:a[j].v)==maximum)++ties;
 for(j=0;j<3;++j)if((a[j].v<0.?-a[j].v:a[j].v)==maximum)for(i=0;i<PM_DIM;++i)scale.d[i]+=gffx_pc_div(a[j].v<0.?-a[j].d[i]:a[j].d[i],(double)ties);
 for(j=0;j<3;++j)s[j]=PM_NAME(div)(a[j],scale);
 length=PM_NAME(add)(PM_NAME(add)(PM_NAME(square)(s[0]),PM_NAME(square)(s[1])),PM_NAME(square)(s[2]));
 for(j=0;j<6;++j)y=(y+gffx_pc_div(length.v,y))*.5;
 root=PM_NAME(constant)(y);for(i=0;i<PM_DIM;++i)root.d[i]=gffx_pc_div(length.d[i],2.*y);
 for(j=0;j<3;++j)out[j]=PM_NAME(div)(s[j],root);return 0;
}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(dot)(const PM_NAME(dual) *a,const PM_NAME(dual) *b){
 PM_NAME(dual) out=PM_NAME(add)(PM_NAME(add)(PM_NAME(mul)(a[0],b[0]),PM_NAME(mul)(a[1],b[1])),PM_NAME(mul)(a[2],b[2]));
 if(out.v>1.&&out.v<=1.+8.*0x1p-52)out.v=1.;if(out.v< -1.&&out.v>=-1.-8.*0x1p-52)out.v=-1.;return out;
}
GFFX_PC_SCALAR PM_NAME(dual) PM_NAME(loss)(double value,double partial,double cosine_partial,PM_NAME(dual) cosine,double width,double roughness){
 PM_NAME(dual) out=PM_NAME(constant)(value);int i;for(i=0;i<PM_DIM;++i)out.d[i]=cosine_partial*cosine.d[i];if(PM_DIM)out.d[12]+=gffx_pc_div(gffx_pc_div(partial,width),roughness);return out;
}
GFFX_PC_SCALAR int PM_NAME(point)(int n,const double *cn,const double *logs,const double *curve,const double *partial,const double *mean,double width,const double *input,int two_sided,double *out){
 PM_NAME(dual) a[16],normal[3],view[3],light[3],half[3],sum[3],nv,nl,nh,vh,one=PM_NAME(constant)(1.),pi=PM_NAME(constant)(0x1.921fb54442d18p+1),r4,denominator,distribution,visibility,distance,square,fifth;
 double vq[6],lq[6];int i,j;for(i=0;i<16;++i)a[i]=PM_NAME(input)(input[i],i);
 if(PM_NAME(unit)(a,normal)||PM_NAME(unit)(a+3,view)||PM_NAME(unit)(a+6,light))return 1;
 nv=PM_NAME(dot)(normal,view);if(nv.v<0.&&two_sided)for(j=0;j<3;++j)normal[j]=PM_NAME(mul)(normal[j],PM_NAME(constant)(-1.));
 nv=PM_NAME(dot)(normal,view);nl=PM_NAME(dot)(normal,light);
 if(nv.v<=0.||nl.v<=0.){for(i=0;i<3+3*PM_DIM;++i)out[i]=0.;return 0;}
 for(j=0;j<3;++j)sum[j]=PM_NAME(add)(view[j],light[j]);if(PM_NAME(unit)(sum,half))return 1;
 nh=PM_NAME(dot)(normal,half);vh=PM_NAME(dot)(view,half);if(nv.v>1.||nl.v>1.||nh.v<0.||nh.v>1.||vh.v<0.||vh.v>1.)return 1;
 gffx_pq_direction(n,cn,logs,curve,partial,nv.v,vq);gffx_pq_direction(n,cn,logs,curve,partial,nl.v,lq);
 r4=PM_NAME(square)(PM_NAME(square)(a[12]));
 denominator=PM_NAME(square)(PM_NAME(add)(PM_NAME(mul)(PM_NAME(sub)(one,nh),PM_NAME(add)(one,nh)),PM_NAME(mul)(r4,PM_NAME(square)(nh))));
 distribution=PM_NAME(div)(PM_NAME(div)(r4,denominator),pi);
 denominator=PM_NAME(add)(PM_NAME(mul)(nl,PM_NAME(sqrt)(PM_NAME(add)(PM_NAME(mul)(PM_NAME(square)(nv),PM_NAME(sub)(one,r4)),r4))),PM_NAME(mul)(nv,PM_NAME(sqrt)(PM_NAME(add)(PM_NAME(mul)(PM_NAME(square)(nl),PM_NAME(sub)(one,r4)),r4))));
 visibility=PM_NAME(div)(PM_NAME(constant)(.5),denominator);
 distance=PM_NAME(sub)(one,vh);square=PM_NAME(square)(distance);fifth=PM_NAME(mul)(PM_NAME(square)(square),distance);
 for(j=0;j<3;++j){PM_NAME(dual) f=a[13+j],v[2],l[2],m[2],vl,ll,ml,fresnel,specular,matte,result;
  for(i=0;i<2;++i){v[i]=PM_NAME(loss)(vq[i],vq[2+i],vq[4+i],nv,width,input[12]);l[i]=PM_NAME(loss)(lq[i],lq[2+i],lq[4+i],nl,width,input[12]);m[i]=PM_NAME(loss)(mean[i],mean[2+i],0.,PM_NAME(constant)(0.),width,input[12]);}
  vl=PM_NAME(add)(PM_NAME(mul)(PM_NAME(sub)(one,f),v[0]),PM_NAME(mul)(f,v[1]));ll=PM_NAME(add)(PM_NAME(mul)(PM_NAME(sub)(one,f),l[0]),PM_NAME(mul)(f,l[1]));ml=PM_NAME(add)(PM_NAME(mul)(PM_NAME(sub)(one,f),m[0]),PM_NAME(mul)(f,m[1]));
  fresnel=PM_NAME(add)(f,PM_NAME(mul)(PM_NAME(sub)(one,f),fifth));specular=PM_NAME(mul)(PM_NAME(mul)(distribution,visibility),fresnel);
  matte=PM_NAME(div)(PM_NAME(mul)(PM_NAME(mul)(a[9+j],vl),ll),PM_NAME(mul)(pi,ml));result=PM_NAME(add)(specular,matte);out[j]=result.v;for(i=0;i<PM_DIM;++i)out[3+16*j+i]=result.d[i];
 }
 for(i=0;i<3+3*PM_DIM;++i)if(!isfinite(out[i]))return 1;return 0;
}
