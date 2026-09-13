/* Direct C ABI controls: no framework intermediary can hide layout/alias failures. */
#include <gffx/animation.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if(!(x)){fprintf(stderr,"animation check failed at line %d\n",__LINE__);return 1;} } while(0)

static gffx_tensor_view view(void *data,gffx_dtype dtype,uint32_t rank,const int64_t *shape,
                             int64_t *strides,int output) {
    gffx_tensor_view v={0};uint32_t i;int64_t stride=1;
    for(i=rank;i-->0;){strides[i]=stride;stride*=shape[i]>0?shape[i]:1;}
    v.struct_size=sizeof(v);v.abi_version=GFFX_ABI_VERSION;v.data=data;v.dtype=dtype;v.rank=rank;
    v.shape=shape;v.strides=strides;v.flags=output?GFFX_TENSOR_OUTPUT:GFFX_TENSOR_READ_ONLY;
    v.device_type=GFFX_DEVICE_CPU;return v;
}

int main(void) {
    gffx_execution_context c={0};gffx_diagnostic_buffer d={0};uint64_t bytes,alignment;
    double t[3]={1,2,3},delta[3]={2,0,-1},w[2]={1,-2},out[6],g[6]={1,1,1,1,1,1};
    double gt[3],gd[3],gw[2];int64_t ts[2]={1,3},ds[3]={1,1,3},ws[2]={2,1},ys[3]={2,1,3};
    int64_t strides[8][4];gffx_tensor_view tv,dv,wv,yv,gv,gtv,gdv,gwv;
    c.struct_size=sizeof(c);c.abi_version=GFFX_ABI_VERSION;c.device_type=GFFX_DEVICE_CPU;
    d.struct_size=sizeof(d);d.abi_version=GFFX_ABI_VERSION;
    tv=view(t,GFFX_DTYPE_FLOAT64,2,ts,strides[0],0);dv=view(delta,GFFX_DTYPE_FLOAT64,3,ds,strides[1],0);
    wv=view(w,GFFX_DTYPE_FLOAT64,2,ws,strides[2],0);yv=view(out,GFFX_DTYPE_FLOAT64,3,ys,strides[3],1);
    gv=view(g,GFFX_DTYPE_FLOAT64,3,ys,strides[4],0);gtv=view(gt,GFFX_DTYPE_FLOAT64,2,ts,strides[5],1);
    gdv=view(gd,GFFX_DTYPE_FLOAT64,3,ds,strides[6],1);gwv=view(gw,GFFX_DTYPE_FLOAT64,2,ws,strides[7],1);
    CHECK(gffx_mesh_blend_shapes_workspace(2,1,1,GFFX_DTYPE_FLOAT64,&c,&bytes,&alignment,&d)==GFFX_STATUS_OK);
    CHECK(bytes==0&&alignment==1);
    CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&c,&yv,NULL,&d)==GFFX_STATUS_OK);
    CHECK(out[0]==3&&out[1]==2&&out[2]==2&&out[3]==-3&&out[5]==5);
    CHECK(gffx_mesh_blend_shapes_backward(&tv,&dv,&wv,&gv,&c,&gtv,&gdv,&gwv,NULL,&d)==GFFX_STATUS_OK);
    CHECK(gt[0]==2&&gt[2]==2&&gd[0]==-1&&gd[2]==-1&&gw[0]==1&&gw[1]==1);
    /* Null gradient subset is legal; no caller output is required for unused basis. */
    CHECK(gffx_mesh_blend_shapes_backward(&tv,&dv,&wv,&gv,&c,&gtv,NULL,NULL,NULL,&d)==GFFX_STATUS_OK);
    yv.data=t;CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&c,&yv,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT);
    CHECK(t[0]==1&&t[1]==2&&t[2]==3);yv.data=out;
    w[0]=NAN;out[0]=123;
    CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&c,&yv,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT);CHECK(out[0]==123);w[0]=1;
    {
        gffx_buffer overlap={0};void *saved=gtv.data;
        overlap.struct_size=sizeof(overlap);overlap.abi_version=GFFX_ABI_VERSION;
        overlap.device_type=GFFX_DEVICE_CPU;overlap.alignment=8;overlap.capacity_bytes=8;
        overlap.data=t;out[0]=123;
        CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&c,&yv,&overlap,&d)==GFFX_STATUS_INVALID_ARGUMENT);
        CHECK(out[0]==123&&t[0]==1);
        overlap.data=out;
        CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&c,&yv,&overlap,&d)==GFFX_STATUS_INVALID_ARGUMENT);
        CHECK(out[0]==123);
        /* Two requested gradients must not share storage, even with equal shape. */
        gt[0]=73;gd[0]=74;gw[0]=75;gtv.data=gd;
        CHECK(gffx_mesh_blend_shapes_backward(&tv,&dv,&wv,&gv,&c,&gtv,&gdv,&gwv,NULL,&d)==GFFX_STATUS_INVALID_ARGUMENT);
        CHECK(gt[0]==73&&gd[0]==74&&gw[0]==75);gtv.data=saved;
        bytes=93;alignment=94;
        CHECK(gffx_mesh_blend_shapes_workspace(INT64_MAX,1,1,GFFX_DTYPE_FLOAT64,&c,&bytes,&alignment,&d)==GFFX_STATUS_OVERFLOW);
        CHECK(bytes==93&&alignment==94);
        tv.struct_size=8;
        CHECK(gffx_mesh_blend_shapes(&tv,&dv,&wv,&c,&yv,NULL,&d)==GFFX_STATUS_ABI_MISMATCH);
        CHECK(out[0]==123);tv.struct_size=sizeof(tv);
    }
    {
        double r[3]={0,0,0},rotation[9],gr[9]={0,0,0,0,0,-1,0,1,0},dr[3];
        int64_t rs[1]={3},rrs[2]={3,3},st[4][2];
        gffx_tensor_view rv=view(r,GFFX_DTYPE_FLOAT64,1,rs,st[0],0),ov=view(rotation,GFFX_DTYPE_FLOAT64,2,rrs,st[1],1);
        gffx_tensor_view grv=view(gr,GFFX_DTYPE_FLOAT64,2,rrs,st[2],0),drv=view(dr,GFFX_DTYPE_FLOAT64,1,rs,st[3],1);
        CHECK(gffx_transforms_rodrigues(&rv,&c,&ov,NULL,&d)==GFFX_STATUS_OK);
        CHECK(rotation[0]==1&&rotation[4]==1&&rotation[8]==1&&rotation[1]==0);
        CHECK(gffx_transforms_rodrigues_backward(&rv,&grv,&c,&drv,NULL,&d)==GFFX_STATUS_OK);
        CHECK(dr[0]==2&&dr[1]==0&&dr[2]==0);
    }
    {
        double local[32]={0},world[32],scratch_memory[140];int32_t parents[2]={1,-1};
        int64_t ls[4]={1,2,4,4},ps[1]={2},st[3][4];int j;
        gffx_buffer scratch={0};gffx_tensor_view lv,pv,ov;
        for(j=0;j<2;++j){local[j*16]=1;local[j*16+5]=1;local[j*16+10]=1;local[j*16+15]=1;}
        local[3]=2;local[16+7]=3;
        lv=view(local,GFFX_DTYPE_FLOAT64,4,ls,st[0],0);pv=view(parents,GFFX_DTYPE_INT32,1,ps,st[1],0);
        ov=view(world,GFFX_DTYPE_FLOAT64,4,ls,st[2],1);
        CHECK(gffx_skeleton_forward_kinematics_workspace(1,2,GFFX_DTYPE_FLOAT64,&c,&bytes,&alignment,&d)==GFFX_STATUS_OK);
        CHECK(bytes==528&&alignment==8);
        scratch.struct_size=sizeof(scratch);scratch.abi_version=GFFX_ABI_VERSION;scratch.device_type=GFFX_DEVICE_CPU;
        scratch.data=scratch_memory;scratch.capacity_bytes=bytes-1;scratch.alignment=8;
        world[0]=123;
        CHECK(gffx_skeleton_forward_kinematics(&lv,&pv,&c,&ov,&scratch,&d)==GFFX_STATUS_INSUFFICIENT_WORKSPACE);
        CHECK(world[0]==123);
        scratch.capacity_bytes=bytes;scratch.data=(unsigned char *)scratch_memory+1;
        CHECK(gffx_skeleton_forward_kinematics(&lv,&pv,&c,&ov,&scratch,&d)==GFFX_STATUS_INVALID_ARGUMENT);
        CHECK(world[0]==123);scratch.data=scratch_memory;
        scratch.capacity_bytes=bytes;
        CHECK(gffx_skeleton_forward_kinematics(&lv,&pv,&c,&ov,&scratch,&d)==GFFX_STATUS_OK);
        CHECK(world[3]==2&&world[7]==3&&world[16+7]==3);
        parents[1]=0;world[0]=123;
        CHECK(gffx_skeleton_forward_kinematics(&lv,&pv,&c,&ov,&scratch,&d)==GFFX_STATUS_INVALID_ARGUMENT);
        CHECK(world[0]==123);
    }
    {
        /* Null DATA on a valid empty cotangent still means backward. */
        double matrices[32]={0},gradient[32];int q;
        int64_t vs[3]={1,0,3},ms[4]={1,2,4,4},ws0[2]={0,2},st[5][4];
        gffx_tensor_view vv=view(NULL,GFFX_DTYPE_FLOAT64,3,vs,st[0],0);
        gffx_tensor_view mv=view(matrices,GFFX_DTYPE_FLOAT64,4,ms,st[1],0);
        gffx_tensor_view sw=view(NULL,GFFX_DTYPE_FLOAT64,2,ws0,st[2],0);
        gffx_tensor_view cot=view(NULL,GFFX_DTYPE_FLOAT64,3,vs,st[3],0);
        gffx_tensor_view gm=view(gradient,GFFX_DTYPE_FLOAT64,4,ms,st[4],1);
        for(q=0;q<32;++q)gradient[q]=73;
        CHECK(gffx_mesh_skin_lbs_backward(&vv,&mv,&sw,&cot,&c,NULL,&gm,NULL,NULL,&d)==GFFX_STATUS_OK);
        for(q=0;q<32;++q)CHECK(gradient[q]==0);
    }
    {
        double vertices[9]={0},gradient[9];int32_t face[3]={0,1,2};int q;
        int64_t vs[3]={1,3,3},fs[2]={1,3},is[1]={0},bs[2]={0,3},gs[3]={1,0,3},st[6][3];
        gffx_tensor_view vv=view(vertices,GFFX_DTYPE_FLOAT64,3,vs,st[0],0);
        gffx_tensor_view fv=view(face,GFFX_DTYPE_INT32,2,fs,st[1],0);
        gffx_tensor_view iv=view(NULL,GFFX_DTYPE_INT32,1,is,st[2],0);
        gffx_tensor_view bv=view(NULL,GFFX_DTYPE_FLOAT64,2,bs,st[3],0);
        gffx_tensor_view cot=view(NULL,GFFX_DTYPE_FLOAT64,3,gs,st[4],0);
        gffx_tensor_view gv0=view(gradient,GFFX_DTYPE_FLOAT64,3,vs,st[5],1);
        for(q=0;q<9;++q)gradient[q]=73;
        CHECK(gffx_mesh_evaluate_barycentric_backward(&vv,&fv,&iv,&bv,&cot,&c,&gv0,NULL,NULL,&d)==GFFX_STATUS_OK);
        for(q=0;q<9;++q)CHECK(gradient[q]==0);
    }
    return 0;
}
