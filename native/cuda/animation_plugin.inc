/* Private entries receive core-validated metadata. Never dereference device data here. */
static CUdeviceptr an_device_data(const gffx_tensor_view *v) {
    return v==NULL?0:(CUdeviceptr)(uintptr_t)v->data+v->byte_offset;
}
static long long an_device_count(const gffx_tensor_view *v) {
    uint32_t i;long long n=1;if(v==NULL)return 0;
    for(i=0;i<v->rank;++i)n*=v->shape[i];return n;
}
static gffx_status an_device_functions(gffx_dtype dtype,int operation,CUfunction *finite,CUfunction *blend,gffx_diagnostic_buffer *d) {
    const char *name=operation==15?(dtype==GFFX_DTYPE_FLOAT32?"gffx_animation_rodrigues_f32":"gffx_animation_rodrigues_f64"):
        operation==16?(dtype==GFFX_DTYPE_FLOAT32?"gffx_animation_fk_f32":"gffx_animation_fk_f64"):
        operation==17?(dtype==GFFX_DTYPE_FLOAT32?"gffx_animation_skin_f32":"gffx_animation_skin_f64"):
        operation==18?(dtype==GFFX_DTYPE_FLOAT32?"gffx_animation_bary_f32":"gffx_animation_bary_f64"):
        (dtype==GFFX_DTYPE_FLOAT32?"gffx_animation_blend_f32":"gffx_animation_blend_f64");
    gffx_status s=gffx_cuda_ensure_module(d);if(s!=GFFX_STATUS_OK)return s;
    if(cuModuleGetFunction(finite,gffx_cuda_module,dtype==GFFX_DTYPE_FLOAT32?"gffx_animation_finite_f32":"gffx_animation_finite_f64")!=CUDA_SUCCESS||
       cuModuleGetFunction(blend,gffx_cuda_module,name)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"animation kernel missing from CUDA module");
    return GFFX_STATUS_OK;
}
static gffx_status an_device_finite(CUfunction fn,const gffx_tensor_view *const *views,unsigned n,
    const gffx_execution_context *c,const gffx_buffer *scratch,int output,gffx_diagnostic_buffer *d) {
    unsigned j;int host_status=0;CUdeviceptr status=(CUdeviceptr)(uintptr_t)scratch->data;
    if(cuMemsetD32Async(status,0,1,(CUstream)c->stream)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"animation validation initialization failed");
    for(j=0;j<n;++j)if(views[j]!=NULL){
        CUdeviceptr data=an_device_data(views[j]);long long count=an_device_count(views[j]);void *args[]={&data,&count,&status};
        gffx_status s=gffx_cuda_launch(fn,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;
    }
    if(cuMemcpyDtoHAsync(&host_status,status,sizeof(host_status),(CUstream)c->stream)!=CUDA_SUCCESS||
       cuStreamSynchronize((CUstream)c->stream)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"animation validation readback failed");
    if(host_status)return gffx_cuda_fail(d,output?GFFX_STATUS_OVERFLOW:GFFX_STATUS_INVALID_ARGUMENT,
        output?"animation arithmetic is not representable":"animation floating inputs must be finite");
    return GFFX_STATUS_OK;
}
static gffx_status an_device_blend(int operation,const gffx_tensor_view *t,const gffx_tensor_view *delta,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,
    gffx_tensor_view *gt,gffx_tensor_view *gd,gffx_tensor_view *gw,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    CUfunction finite,kernel;CUdeviceptr tp,dp,wp,gp;long long B,V,K;unsigned j;
    const gffx_tensor_view *inputs[]={t,delta,w,g};gffx_tensor_view *outputs[]={y,gt,gd,gw};gffx_status s;
    if(t==NULL||delta==NULL||w==NULL||c==NULL||scratch==NULL||scratch->data==NULL||scratch->capacity_bytes<8)
        return gffx_cuda_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"animation requires validated views and scratch");
    s=an_device_functions(t->dtype,operation,&finite,&kernel,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_finite(finite,inputs,g==NULL?3:4,c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    tp=an_device_data(t);dp=an_device_data(delta);wp=an_device_data(w);gp=an_device_data(g);
    B=operation==14?w->shape[0]:t->shape[0];V=operation==14?t->shape[0]:t->shape[1];K=operation==14?delta->shape[0]:delta->shape[1];
    for(j=0;j<4;++j)if(outputs[j]!=NULL){
        CUdeviceptr out=an_device_data(outputs[j]);long long count=an_device_count(outputs[j]);int mode=(int)j;
        void *args[]={&tp,&dp,&wp,&gp,&out,&B,&V,&K,&mode,&count};
        s=gffx_cuda_launch(kernel,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;
    }
    return an_device_finite(finite,(const gffx_tensor_view *const *)outputs,4,c,scratch,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_blend_shapes(const gffx_tensor_view *t,const gffx_tensor_view *delta,const gffx_tensor_view *w,
    const gffx_execution_context *c,gffx_tensor_view *y,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    return an_device_blend(14,t,delta,w,NULL,c,y,NULL,NULL,NULL,scratch,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_blend_shapes_backward(const gffx_tensor_view *t,const gffx_tensor_view *delta,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *gt,gffx_tensor_view *gd,gffx_tensor_view *gw,
    const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    return an_device_blend(14,t,delta,w,g,c,NULL,gt,gd,gw,scratch,d);
}

static gffx_status GFFX_CALL gffx_cuda_op_skin_lbs(const gffx_tensor_view *v,const gffx_tensor_view *t,const gffx_tensor_view *w,
    const gffx_execution_context *c,gffx_tensor_view *y,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    return an_device_blend(17,v,t,w,NULL,c,y,NULL,NULL,NULL,scratch,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_skin_lbs_backward(const gffx_tensor_view *v,const gffx_tensor_view *t,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *gv,gffx_tensor_view *gt,gffx_tensor_view *gw,
    const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    return an_device_blend(17,v,t,w,g,c,NULL,gv,gt,gw,scratch,d);
}

static gffx_status an_device_index(const gffx_tensor_view *v,long long limit,const gffx_execution_context *c,
    const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    CUfunction kernel;int host=0;CUdeviceptr data=an_device_data(v),status=(CUdeviceptr)(uintptr_t)scratch->data;
    long long count=an_device_count(v);void *args[]={&data,&count,&limit,&status};gffx_status s;
    if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_animation_validate_index")!=CUDA_SUCCESS||
        cuMemsetD32Async(status,0,1,(CUstream)c->stream)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"animation index validation setup failed");
    s=gffx_cuda_launch(kernel,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;
    if(cuMemcpyDtoHAsync(&host,status,sizeof(host),(CUstream)c->stream)!=CUDA_SUCCESS||cuStreamSynchronize((CUstream)c->stream)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"animation index validation readback failed");
    return host?gffx_cuda_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"animation index out of range"):GFFX_STATUS_OK;
}
static gffx_status an_device_bary(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *idx,const gffx_tensor_view *w,
    const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,gffx_tensor_view *gv,gffx_tensor_view *gw,
    const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    CUfunction finite,kernel;gffx_status s;unsigned j;long long B=v->shape[0],V=v->shape[1],N=idx->shape[0];
    CUdeviceptr vp=an_device_data(v),fp=an_device_data(f),ip=an_device_data(idx),wp=an_device_data(w),gp=an_device_data(g);
    const gffx_tensor_view *inputs[]={v,w,g};gffx_tensor_view *outputs[]={y,gv,gw};
    s=an_device_functions(v->dtype,18,&finite,&kernel,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_finite(finite,inputs,g==NULL?2:3,c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_index(f,V,c,scratch,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_index(idx,f->shape[0],c,scratch,d);if(s!=GFFX_STATUS_OK)return s;
    for(j=0;j<3;++j)if(outputs[j]!=NULL){CUdeviceptr out=an_device_data(outputs[j]);long long count=an_device_count(outputs[j]);int mode=(int)j;
        void *args[]={&vp,&fp,&ip,&wp,&gp,&out,&B,&V,&N,&mode,&count};s=gffx_cuda_launch(kernel,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
    return an_device_finite(finite,(const gffx_tensor_view *const *)outputs,3,c,scratch,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_bary(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *idx,const gffx_tensor_view *w,
    const gffx_execution_context *c,gffx_tensor_view *y,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    return an_device_bary(v,f,idx,w,NULL,c,y,NULL,NULL,scratch,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_bary_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *idx,const gffx_tensor_view *w,const gffx_tensor_view *g,
    const gffx_execution_context *c,gffx_tensor_view *gv,gffx_tensor_view *gw,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    return an_device_bary(v,f,idx,w,g,c,NULL,gv,gw,scratch,d);
}

static gffx_status an_device_status(const gffx_execution_context *c,const gffx_buffer *scratch,int topology,gffx_diagnostic_buffer *d) {
    int host=0;
    if(cuMemcpyDtoHAsync(&host,(CUdeviceptr)(uintptr_t)scratch->data,sizeof(host),(CUstream)c->stream)!=CUDA_SUCCESS||cuStreamSynchronize((CUstream)c->stream)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"animation status readback failed");
    if(host)return gffx_cuda_fail(d,topology?GFFX_STATUS_INVALID_ARGUMENT:GFFX_STATUS_OVERFLOW,
        topology?"skeleton parent index or cycle invalid":"animation arithmetic is not representable");
    return GFFX_STATUS_OK;
}
static gffx_status an_device_rodrigues(const gffx_tensor_view *r,const gffx_tensor_view *g,const gffx_execution_context *c,
    gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    CUfunction finite,kernel;gffx_status s;CUdeviceptr rp=an_device_data(r),gp=an_device_data(g),yp=an_device_data(out),status=(CUdeviceptr)(uintptr_t)scratch->data;
    long long count=an_device_count(r)/3;const gffx_tensor_view *inputs[]={r,g},*outputs[]={out};void *args[]={&rp,&gp,&yp,&count,&status};
    s=an_device_functions(r->dtype,15,&finite,&kernel,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_finite(finite,inputs,g==NULL?1:2,c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_cuda_launch(kernel,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_status(c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    return an_device_finite(finite,outputs,1,c,scratch,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_rodrigues(const gffx_tensor_view *r,const gffx_execution_context *c,gffx_tensor_view *out,
    const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return an_device_rodrigues(r,NULL,c,out,scratch,d);}
static gffx_status GFFX_CALL gffx_cuda_op_rodrigues_backward(const gffx_tensor_view *r,const gffx_tensor_view *g,const gffx_execution_context *c,
    gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return an_device_rodrigues(r,g,c,out,scratch,d);}
static gffx_status an_device_fk(const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_tensor_view *g,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d) {
    CUfunction finite,kernel,topology;gffx_status s;long long B=local->shape[0],J=local->shape[1];
    CUdeviceptr lp=an_device_data(local),pp=an_device_data(parents),gp=an_device_data(g),yp=an_device_data(out),status=(CUdeviceptr)(uintptr_t)scratch->data;
    CUdeviceptr order=status+8,done=order+J*4,world=done+J*4,adj=world+B*J*16*(local->dtype==GFFX_DTYPE_FLOAT32?4:8);
    const gffx_tensor_view *inputs[]={local,g},*outputs[]={out};void *order_args[]={&pp,&J,&order,&done,&status};
    void *args[]={&lp,&pp,&order,&gp,&yp,&world,&adj,&B,&J,&status};
    s=an_device_functions(local->dtype,16,&finite,&kernel,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_finite(finite,inputs,g==NULL?1:2,c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    if(cuModuleGetFunction(&topology,gffx_cuda_module,"gffx_animation_order")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"animation topology kernel unavailable");
    s=gffx_cuda_launch(topology,1,order_args,c,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_status(c,scratch,1,d);if(s!=GFFX_STATUS_OK)return s;
    s=gffx_cuda_launch(kernel,B,args,c,d);if(s!=GFFX_STATUS_OK)return s;
    s=an_device_status(c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    return an_device_finite(finite,outputs,1,c,scratch,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_fk(const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_execution_context *c,
    gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return an_device_fk(local,parents,NULL,c,out,scratch,d);}
static gffx_status GFFX_CALL gffx_cuda_op_fk_backward(const gffx_tensor_view *local,const gffx_tensor_view *parents,const gffx_tensor_view *g,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return an_device_fk(local,parents,g,c,out,scratch,d);}
