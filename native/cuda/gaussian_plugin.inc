/* Core-validated Gaussian descriptors; no output writes before all domains pass. */
static gffx_status gw_device_call(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t radius,
    int backward,const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    CUfunction finite,kernel;CUdeviceptr xp,gp,yp,status;long long count;unsigned j;gffx_status s;
    const gffx_tensor_view *inputs[2]={x,g};const char *finite_name,*kernel_name;
    if(x==NULL||out==NULL||c==NULL||scratch==NULL||scratch->data==NULL||scratch->capacity_bytes<8||
       radius>32||(backward&&g==NULL))
        return gffx_cuda_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"Gaussian requires validated views/mode and caller scratch");
    s=gffx_cuda_ensure_module(d);if(s!=GFFX_STATUS_OK)return s;
    finite_name=x->dtype==GFFX_DTYPE_FLOAT32?"gffx_gaussian_domain_f32":"gffx_gaussian_domain_f64";
    kernel_name=x->dtype==GFFX_DTYPE_FLOAT32?"gffx_gaussian_f32":"gffx_gaussian_f64";
    if(cuModuleGetFunction(&finite,gffx_cuda_module,finite_name)!=CUDA_SUCCESS||
       cuModuleGetFunction(&kernel,gffx_cuda_module,kernel_name)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"Gaussian kernel missing from CUDA module");
    status=(CUdeviceptr)(uintptr_t)scratch->data;
    if(cuMemsetD32Async(status,0,1,(CUstream)c->stream)!=CUDA_SUCCESS)
        return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"Gaussian validation initialization failed");
    for(j=0;j<(backward?2u:1u);++j){
        if(j==1&&cuModuleGetFunction(&finite,gffx_cuda_module,x->dtype==GFFX_DTYPE_FLOAT32?"gffx_image_finite_f32":"gffx_image_finite_f64")!=CUDA_SUCCESS)
            return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"Gaussian cotangent validation kernel missing");
        CUdeviceptr data=im_device_data(inputs[j]);long long length=im_device_count(inputs[j]);void *args[]={&data,&length,&status};
        s=gffx_cuda_launch(finite,length,args,c,d);if(s!=GFFX_STATUS_OK)return s;
    }
    s=im_device_status(c,scratch,0,d);if(s!=GFFX_STATUS_OK)return s;
    xp=im_device_data(x);gp=im_device_data(g);yp=im_device_data(out);count=im_device_count(x);
    {void *args[]={&xp,&gp,&yp,&count,&radius,&backward,&status};
     s=gffx_cuda_launch(kernel,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
    return im_device_status(c,scratch,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_image_gaussian_weights(const gffx_tensor_view *x,uint32_t radius,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    return gw_device_call(x,NULL,radius,0,c,out,scratch,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_image_gaussian_weights_backward(const gffx_tensor_view *x,const gffx_tensor_view *g,uint32_t radius,
    const gffx_execution_context *c,gffx_tensor_view *out,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
    return gw_device_call(x,g,radius,1,c,out,scratch,d);
}
