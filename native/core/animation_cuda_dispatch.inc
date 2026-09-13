/* Core-owned descriptor validation; no device tensor is read by host code. */
static gffx_status an_cuda_view(const gffx_tensor_view *v,uint32_t rank,const int64_t *shape,
    gffx_dtype dtype,int output,const gffx_execution_context *c,gffx_diagnostic_buffer *d) {
    uint32_t i;gffx_status s=gffx_validate_tensor_view(v,d);
    if(s!=GFFX_STATUS_OK)return s;
    if(v->rank!=rank||v->dtype!=dtype||v->device_type!=c->device_type||v->device_index!=c->device_index)
        return an_fail(d,"animation tensor rank, dtype or device mismatch");
    for(i=0;i<rank;++i)if(shape[i]>=0&&v->shape[i]!=shape[i])return an_fail(d,"animation tensor shape mismatch");
    if(((v->flags&GFFX_TENSOR_OUTPUT)!=0)!=(output!=0))return an_fail(d,"animation tensor access flags invalid");
    return GFFX_STATUS_OK;
}

static gffx_status an_cuda_blend_workspace(int64_t B,int64_t V,int64_t K,gffx_dtype dtype,
    const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d) {
    gffx_execution_context cpu;int64_t shape[3]={B,V,K};const gffx_cuda_operations *ops;
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    cpu=*c;cpu.device_type=GFFX_DEVICE_CPU;cpu.device_index=0;cpu.stream=NULL;
    s=gffx_mesh_blend_shapes_workspace(B,V,K,dtype,&cpu,bytes,alignment,d);if(s!=GFFX_STATUS_OK)return s;
    ops=gffx_cuda_loader_operations();
    if(!GFFX_CUDA_HAS_OPERATION(ops,mesh_blend_shapes)||ops->workspace_query==NULL)
        return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks animation blend shapes");
    return ops->workspace_query(GFFX_CUDA_OP_MESH_BLEND_SHAPES,shape,3,dtype,c,bytes,alignment,d);
}

static gffx_status an_cuda_blend(const gffx_tensor_view *t,const gffx_tensor_view *delta,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,
    gffx_tensor_view *gt,gffx_tensor_view *gd,gffx_tensor_view *gw,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    int64_t ts[2]={-1,3},ds[3],ws[2],ys[3];uint64_t bytes,alignment;unsigned i;
    const gffx_tensor_view *inputs[4]={t,delta,w,g},*primal[3]={t,delta,w};
    gffx_tensor_view *outputs[3]={gt,gd,gw};const gffx_cuda_operations *ops;
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_validate_tensor_view(t,d);if(s!=GFFX_STATUS_OK)return s;
    if(t->dtype!=GFFX_DTYPE_FLOAT32&&t->dtype!=GFFX_DTYPE_FLOAT64)return an_fail(d,"animation requires floating template");
    s=an_cuda_view(t,2,ts,t->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ds[0]=-1;ds[1]=t->shape[0];ds[2]=3;s=an_cuda_view(delta,3,ds,t->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ws[0]=-1;ws[1]=delta->shape[0];s=an_cuda_view(w,2,ws,t->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ys[0]=w->shape[0];ys[1]=t->shape[0];ys[2]=3;
    if(g==NULL){outputs[0]=y;s=an_cuda_view(y,3,ys,t->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;}
    else{
        s=an_cuda_view(g,3,ys,t->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
        if(gt==NULL&&gd==NULL&&gw==NULL)return an_fail(d,"at least one animation gradient required");
        for(i=0;i<3;++i)if(outputs[i]!=NULL){s=an_cuda_view(outputs[i],primal[i]->rank,primal[i]->shape,t->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;}
    }
    s=an_cuda_blend_workspace(w->shape[0],t->shape[0],delta->shape[0],t->dtype,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_validate_buffer(scratch,d);if(s!=GFFX_STATUS_OK)return s;
    if(scratch->device_type!=c->device_type||scratch->device_index!=c->device_index)return an_fail(d,"animation scratch device mismatch");
    if(scratch->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"animation CUDA scratch too small");
    if(scratch->alignment<alignment||(uintptr_t)scratch->data%alignment)return an_fail(d,"animation CUDA scratch misaligned");
    s=an_noalias(inputs,g==NULL?3:4,outputs,g==NULL?1:3,scratch,d);if(s!=GFFX_STATUS_OK)return s;
    ops=gffx_cuda_loader_operations();
    if(g==NULL){if(!GFFX_CUDA_HAS_OPERATION(ops,mesh_blend_shapes))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA blend unavailable");return ops->mesh_blend_shapes(t,delta,w,c,y,scratch,d);}
    if(!GFFX_CUDA_HAS_OPERATION(ops,mesh_blend_shapes_backward))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA blend backward unavailable");
    return ops->mesh_blend_shapes_backward(t,delta,w,g,c,gt,gd,gw,scratch,d);
}

static gffx_status an_cuda_deformation_workspace(uint32_t operation,int64_t B,int64_t V,int64_t K,gffx_dtype dtype,
    const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d) {
    gffx_execution_context cpu;int64_t shape[3]={B,V,K};const gffx_cuda_operations *ops;
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    cpu=*c;cpu.device_type=GFFX_DEVICE_CPU;cpu.device_index=0;cpu.stream=NULL;
    if(operation==17)s=gffx_mesh_skin_lbs_workspace(B,V,K,dtype,&cpu,bytes,alignment,d);
    else s=gffx_mesh_evaluate_barycentric_workspace(B,V,K,dtype,&cpu,bytes,alignment,d);
    if(s!=GFFX_STATUS_OK)return s;
    ops=gffx_cuda_loader_operations();
    if((operation==17?!GFFX_CUDA_HAS_OPERATION(ops,mesh_skin_lbs):!GFFX_CUDA_HAS_OPERATION(ops,mesh_evaluate_barycentric))||ops->workspace_query==NULL)
        return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks requested animation operation");
    return ops->workspace_query(operation,shape,3,dtype,c,bytes,alignment,d);
}
static gffx_status an_cuda_storage(const gffx_tensor_view **inputs,unsigned ni,gffx_tensor_view **outputs,unsigned no,
    const gffx_execution_context *c,const gffx_buffer *scratch,uint64_t bytes,uint64_t alignment,gffx_diagnostic_buffer *d) {
    gffx_status s=gffx_validate_buffer(scratch,d);if(s!=GFFX_STATUS_OK)return s;
    if(scratch->device_type!=c->device_type||scratch->device_index!=c->device_index)return an_fail(d,"animation scratch device mismatch");
    if(scratch->capacity_bytes<bytes)return gffx_internal_fail(d,GFFX_STATUS_INSUFFICIENT_WORKSPACE,"animation CUDA scratch too small");
    if(scratch->alignment<alignment||(uintptr_t)scratch->data%alignment)return an_fail(d,"animation CUDA scratch misaligned");
    return an_noalias(inputs,ni,outputs,no,scratch,d);
}
static gffx_status an_cuda_skin(const gffx_tensor_view *v,const gffx_tensor_view *t,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,
    gffx_tensor_view *gv,gffx_tensor_view *gt,gffx_tensor_view *gw,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    int64_t vs[3]={-1,-1,3},ts[4],ws[2];uint64_t bytes,alignment;unsigned i;const gffx_cuda_operations *ops;
    const gffx_tensor_view *inputs[4]={v,t,w,g},*primal[3]={v,t,w};gffx_tensor_view *outputs[3]={gv,gt,gw};
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_validate_tensor_view(v,d);if(s!=GFFX_STATUS_OK)return s;
    if(v->dtype!=GFFX_DTYPE_FLOAT32&&v->dtype!=GFFX_DTYPE_FLOAT64)return an_fail(d,"animation requires floating vertices");
    s=an_cuda_view(v,3,vs,v->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ts[0]=v->shape[0];ts[1]=-1;ts[2]=4;ts[3]=4;s=an_cuda_view(t,4,ts,v->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ws[0]=v->shape[1];ws[1]=t->shape[1];s=an_cuda_view(w,2,ws,v->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    if(g==NULL){outputs[0]=y;s=an_cuda_view(y,3,v->shape,v->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;}
    else{
        s=an_cuda_view(g,3,v->shape,v->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
        if(gv==NULL&&gt==NULL&&gw==NULL)return an_fail(d,"at least one animation gradient required");
        for(i=0;i<3;++i)if(outputs[i]!=NULL){s=an_cuda_view(outputs[i],primal[i]->rank,primal[i]->shape,v->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;}
    }
    s=an_cuda_deformation_workspace(17,v->shape[0],v->shape[1],t->shape[1],v->dtype,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_cuda_storage(inputs,g==NULL?3:4,outputs,g==NULL?1:3,c,scratch,bytes,alignment,d);if(s!=GFFX_STATUS_OK)return s;
    ops=gffx_cuda_loader_operations();
    if(g==NULL){if(!GFFX_CUDA_HAS_OPERATION(ops,mesh_skin_lbs))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA LBS unavailable");return ops->mesh_skin_lbs(v,t,w,c,y,scratch,d);}
    if(!GFFX_CUDA_HAS_OPERATION(ops,mesh_skin_lbs_backward))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA LBS backward unavailable");
    return ops->mesh_skin_lbs_backward(v,t,w,g,c,gv,gt,gw,scratch,d);
}
static gffx_status an_cuda_bary(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *idx,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,gffx_tensor_view *gv,gffx_tensor_view *gw,
    const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    int64_t vs[3]={-1,-1,3},fs[2]={-1,3},is[1]={-1},ws[2],ys[3];uint64_t bytes,alignment;unsigned i;
    const gffx_tensor_view *inputs[5]={v,f,idx,w,g},*primal[2]={v,w};gffx_tensor_view *outputs[2]={gv,gw};const gffx_cuda_operations *ops;
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_validate_tensor_view(v,d);if(s!=GFFX_STATUS_OK)return s;
    if(v->dtype!=GFFX_DTYPE_FLOAT32&&v->dtype!=GFFX_DTYPE_FLOAT64)return an_fail(d,"animation requires floating vertices");
    s=an_cuda_view(v,3,vs,v->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_cuda_view(f,2,fs,GFFX_DTYPE_INT32,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_cuda_view(idx,1,is,GFFX_DTYPE_INT32,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ws[0]=idx->shape[0];ws[1]=3;s=an_cuda_view(w,2,ws,v->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ys[0]=v->shape[0];ys[1]=idx->shape[0];ys[2]=3;
    if(g==NULL){outputs[0]=y;s=an_cuda_view(y,3,ys,v->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;}
    else{
        s=an_cuda_view(g,3,ys,v->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
        if(gv==NULL&&gw==NULL)return an_fail(d,"at least one animation gradient required");
        for(i=0;i<2;++i)if(outputs[i]!=NULL){s=an_cuda_view(outputs[i],primal[i]->rank,primal[i]->shape,v->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;}
    }
    s=an_cuda_deformation_workspace(18,v->shape[0],v->shape[1],idx->shape[0],v->dtype,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_cuda_storage(inputs,g==NULL?4:5,outputs,g==NULL?1:2,c,scratch,bytes,alignment,d);if(s!=GFFX_STATUS_OK)return s;
    ops=gffx_cuda_loader_operations();
    if(g==NULL){if(!GFFX_CUDA_HAS_OPERATION(ops,mesh_evaluate_barycentric))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA barycentric unavailable");return ops->mesh_evaluate_barycentric(v,f,idx,w,c,y,scratch,d);}
    if(!GFFX_CUDA_HAS_OPERATION(ops,mesh_evaluate_barycentric_backward))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA barycentric backward unavailable");
    return ops->mesh_evaluate_barycentric_backward(v,f,idx,w,g,c,gv,gw,scratch,d);
}

static gffx_status an_cuda_pose_workspace(uint32_t operation,int64_t B,int64_t count,gffx_dtype dtype,
    const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment,gffx_diagnostic_buffer *d) {
    gffx_execution_context cpu;int64_t shape[2]={B,count};const gffx_cuda_operations *ops;
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    cpu=*c;cpu.device_type=GFFX_DEVICE_CPU;cpu.device_index=0;cpu.stream=NULL;
    if(operation==15)s=gffx_transforms_rodrigues_workspace(count,dtype,&cpu,bytes,alignment,d);
    else s=gffx_skeleton_forward_kinematics_workspace(B,count,dtype,&cpu,bytes,alignment,d);
    if(s!=GFFX_STATUS_OK)return s;
    if(*bytes>UINT64_MAX-8)return gffx_internal_fail(d,GFFX_STATUS_OVERFLOW,"CUDA skeleton scratch overflow");
    ops=gffx_cuda_loader_operations();
    if((operation==15?!GFFX_CUDA_HAS_OPERATION(ops,transforms_rodrigues):!GFFX_CUDA_HAS_OPERATION(ops,skeleton_forward_kinematics))||ops->workspace_query==NULL)
        return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA provider lacks requested pose operation");
    return ops->workspace_query(operation,shape,2,dtype,c,bytes,alignment,d);
}
static gffx_status an_cuda_rodrigues(const gffx_tensor_view *r,const gffx_tensor_view *g,const gffx_execution_context *c,
    gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    int64_t shape[GFFX_MAX_RANK],count;uint32_t i;uint64_t bytes,alignment;const gffx_cuda_operations *ops;
    const gffx_tensor_view *inputs[2]={r,g};gffx_tensor_view *outputs[1]={out};
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_validate_tensor_view(r,d);if(s!=GFFX_STATUS_OK)return s;
    if(r->dtype!=GFFX_DTYPE_FLOAT32&&r->dtype!=GFFX_DTYPE_FLOAT64)return an_fail(d,"Rodrigues requires floating axis-angle");
    if(r->rank<1||r->rank>=GFFX_MAX_RANK||r->shape[r->rank-1]!=3)return an_fail(d,"axis_angle must have shape [...,3]");
    s=an_cuda_view(r,r->rank,r->shape,r->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    for(i=0;i<r->rank;++i)shape[i]=r->shape[i];shape[r->rank]=3;
    if(g!=NULL){s=an_cuda_view(g,r->rank+1,shape,r->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;}
    s=an_cuda_view(out,g==NULL?r->rank+1:r->rank,g==NULL?shape:r->shape,r->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;
    count=(int64_t)(gffx_mesh_element_count(r)/3);
    s=an_cuda_pose_workspace(15,1,count,r->dtype,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_cuda_storage(inputs,g==NULL?1:2,outputs,1,c,scratch,bytes,alignment,d);if(s!=GFFX_STATUS_OK)return s;
    ops=gffx_cuda_loader_operations();
    if(g==NULL){if(!GFFX_CUDA_HAS_OPERATION(ops,transforms_rodrigues))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA Rodrigues unavailable");return ops->transforms_rodrigues(r,c,out,scratch,d);}
    if(!GFFX_CUDA_HAS_OPERATION(ops,transforms_rodrigues_backward))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA Rodrigues backward unavailable");
    return ops->transforms_rodrigues_backward(r,g,c,out,scratch,d);
}
static gffx_status an_cuda_fk(const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_tensor_view *g,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    int64_t ls[4]={-1,-1,4,4},ps[1];uint64_t bytes,alignment;const gffx_cuda_operations *ops;
    const gffx_tensor_view *inputs[3]={local,parents,g};gffx_tensor_view *outputs[1]={out};
    gffx_status s=gffx_validate_execution_context(c,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_validate_tensor_view(local,d);if(s!=GFFX_STATUS_OK)return s;
    if(local->dtype!=GFFX_DTYPE_FLOAT32&&local->dtype!=GFFX_DTYPE_FLOAT64)return an_fail(d,"FK requires floating local transforms");
    s=an_cuda_view(local,4,ls,local->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    ps[0]=local->shape[1];s=an_cuda_view(parents,1,ps,GFFX_DTYPE_INT32,0,c,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_cuda_view(out,4,local->shape,local->dtype,1,c,d);if(s!=GFFX_STATUS_OK)return s;
    if(g!=NULL){s=an_cuda_view(g,4,local->shape,local->dtype,0,c,d);if(s!=GFFX_STATUS_OK)return s;}
    s=an_cuda_pose_workspace(16,local->shape[0],local->shape[1],local->dtype,c,&bytes,&alignment,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_cuda_storage(inputs,g==NULL?2:3,outputs,1,c,scratch,bytes,alignment,d);if(s!=GFFX_STATUS_OK)return s;
    ops=gffx_cuda_loader_operations();
    if(g==NULL){if(!GFFX_CUDA_HAS_OPERATION(ops,skeleton_forward_kinematics))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA FK unavailable");return ops->skeleton_forward_kinematics(local,parents,c,out,scratch,d);}
    if(!GFFX_CUDA_HAS_OPERATION(ops,skeleton_forward_kinematics_backward))return gffx_internal_fail(d,GFFX_STATUS_UNSUPPORTED,"CUDA FK backward unavailable");
    return ops->skeleton_forward_kinematics_backward(local,parents,g,c,out,scratch,d);
}
