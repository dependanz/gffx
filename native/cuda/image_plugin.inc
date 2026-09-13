/* Private transfer entry: core has validated dense metadata and caller scratch.
 * Validate all input values before launching any output writes. */
static CUdeviceptr im_device_data(const gffx_tensor_view *v){
    return v==NULL?0:(CUdeviceptr)(uintptr_t)v->data+v->byte_offset;
}
static long long im_device_count(const gffx_tensor_view *v){
    uint32_t i;long long n=1;
    for(i=0;i<v->rank;++i)if(v->shape[i]==0)return 0;
    for(i=0;i<v->rank;++i)n*=v->shape[i];return n;
}
static gffx_status im_device_status(const gffx_execution_context *c,const gffx_buffer *scratch,
    int output,gffx_diagnostic_buffer *d){
    int host=0;
    if(cuMemcpyDtoHAsync(&host,(CUdeviceptr)(uintptr_t)scratch->data,sizeof(host),(CUstream)c->stream)!=CUDA_SUCCESS||
       cuStreamSynchronize((CUstream)c->stream)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"image validation/status readback failed");
    if(host)return gffx_cuda_fail(d,output?GFFX_STATUS_OVERFLOW:GFFX_STATUS_INVALID_ARGUMENT,
        output?"image transfer arithmetic is not representable":"image floating inputs must be finite");
    return GFFX_STATUS_OK;
}
static gffx_status im_device_call(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t transfer,
    int backward,const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    CUfunction finite,kernel;CUdeviceptr xp,gp,yp,status;long long count;unsigned j;gffx_status s;
    const gffx_tensor_view *inputs[2]={x,g};const char *finite_name,*kernel_name;
    if(x==NULL||out==NULL||c==NULL||scratch==NULL||scratch->data==NULL||scratch->capacity_bytes<8||
       transfer>1||(backward&&g==NULL))
        return gffx_cuda_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"image requires validated views/mode and caller scratch");
    s=gffx_cuda_ensure_module(d);if(s!=GFFX_STATUS_OK)return s;
    finite_name=x->dtype==GFFX_DTYPE_FLOAT32?"gffx_image_finite_f32":"gffx_image_finite_f64";
    kernel_name=x->dtype==GFFX_DTYPE_FLOAT32?"gffx_image_transfer_f32":"gffx_image_transfer_f64";
    if(cuModuleGetFunction(&finite,gffx_cuda_module,finite_name)!=CUDA_SUCCESS||
       cuModuleGetFunction(&kernel,gffx_cuda_module,kernel_name)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"image kernel missing from CUDA module");
    status=(CUdeviceptr)(uintptr_t)scratch->data;
    if(cuMemsetD32Async(status,0,1,(CUstream)c->stream)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"image validation initialization failed");
    for(j=0;j<(backward?2u:1u);++j){
        CUdeviceptr data=im_device_data(inputs[j]);long long length=im_device_count(inputs[j]);void *args[]={&data,&length,&status};
        s=gffx_cuda_launch(finite,length,args,c,d);if(s!=GFFX_STATUS_OK)return s;
    }
    s=im_device_status(c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    xp=im_device_data(x);gp=im_device_data(g);yp=im_device_data(out);count=im_device_count(x);
    {void *args[]={&xp,&gp,&yp,&count,&transfer,&backward,&status};
     s=gffx_cuda_launch(kernel,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
    return im_device_status(c,scratch,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_image_transfer(const gffx_tensor_view *x,uint32_t transfer,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    return im_device_call(x,NULL,transfer,0,c,out,scratch,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_image_transfer_backward(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t transfer,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    return im_device_call(x,g,transfer,1,c,out,scratch,d);
}
