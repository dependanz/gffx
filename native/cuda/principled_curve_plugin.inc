/* Core checks metadata/storage; provider validates values on caller stream. */
static gffx_status pc_device_init(const gffx_execution_context *c,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 gffx_status s=gffx_cuda_ensure_module(d);if(s!=GFFX_STATUS_OK)return s;
 if(!c||!w||!w->data||w->capacity_bytes<8)return gffx_cuda_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"curve caller scratch missing");
 if(cuMemsetD32Async((CUdeviceptr)(uintptr_t)w->data,0,1,(CUstream)c->stream)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"curve status initialization failed");return GFFX_STATUS_OK;
}
static gffx_status GFFX_CALL gffx_cuda_op_shading_loss_curve(const gffx_tensor_view *x,const gffx_tensor_view *b,const gffx_tensor_view *u,const gffx_execution_context *c,gffx_tensor_view *y,gffx_tensor_view *dy,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction finite,values,slopes,controls_kernel;CUdeviceptr xp=im_device_data(x),bp=im_device_data(b),up=im_device_data(u),yp=im_device_data(y),dp=im_device_data(dy),status=(CUdeviceptr)(uintptr_t)w->data,scratch=status+8;
 long long batch=u->shape[0];int n=(int)x->shape[0];gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;
 if(cuModuleGetFunction(&finite,gffx_cuda_module,"gffx_pc_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&values,gffx_cuda_module,"gffx_pc_values")!=CUDA_SUCCESS||cuModuleGetFunction(&slopes,gffx_cuda_module,"gffx_pc_slopes")!=CUDA_SUCCESS||cuModuleGetFunction(&controls_kernel,gffx_cuda_module,"gffx_pc_controls")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"curve kernels missing");
 {void *args[]={&xp,&bp,&up,&batch,&n,&status};s=gffx_cuda_launch(finite,batch*(2*n+1)*4>n?batch*(2*n+1)*4:n,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&xp,&bp,&up,&yp,&dp,&scratch,&batch,&n,&status};s=gffx_cuda_launch(values,batch*(2*n+1),args,c,d);if(s!=GFFX_STATUS_OK)return s;
  s=gffx_cuda_launch(slopes,batch*n,args,c,d);if(s!=GFFX_STATUS_OK)return s;
  s=gffx_cuda_launch(controls_kernel,batch*(n-1)*2,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 return im_device_status(c,w,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_shading_loss_curve_backward(const gffx_tensor_view *p,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *y,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction finite,kernel;CUdeviceptr pp=im_device_data(p),gp=im_device_data(g),yp=im_device_data(y),status=(CUdeviceptr)(uintptr_t)w->data,scratch=status+8;
 long long batch=p->shape[0],count=im_device_count(p);int size=(int)p->shape[1];gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;
 if(cuModuleGetFunction(&finite,gffx_cuda_module,"gffx_image_finite_f64")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_pc_contract")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"curve backward kernels missing");
 {void *args[]={&pp,&count,&status};s=gffx_cuda_launch(finite,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;args[0]=&gp;s=gffx_cuda_launch(finite,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&pp,&gp,&yp,&scratch,&batch,&size,&status};s=gffx_cuda_launch(kernel,batch,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 return im_device_status(c,w,1,d);
}

static gffx_status GFFX_CALL gffx_cuda_op_shading_loss_query(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *weights,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *cosine,const gffx_tensor_view *owners,const gffx_execution_context *c,gffx_tensor_view *query,gffx_tensor_view *mean,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction domain,kernel;CUdeviceptr xp=im_device_data(cn),lp=im_device_data(logs),wp=im_device_data(weights),vp=im_device_data(curve),pp=im_device_data(partial),cp=im_device_data(cosine),op=im_device_data(owners),yp=im_device_data(query),mp=im_device_data(mean),status=(CUdeviceptr)(uintptr_t)w->data,scratch=status+8;
 int b=(int)curve->shape[0],n=(int)cn->shape[0],q=(int)cosine->shape[0],count=b*(10*n-6);gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;if(count<4*n)count=4*n;if(count<q)count=q;
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_pq_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_pq_evaluate")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"query kernels missing");
 {void *args[]={&xp,&lp,&wp,&vp,&pp,&cp,&op,&b,&n,&q,&status};s=gffx_cuda_launch(domain,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&xp,&lp,&wp,&vp,&pp,&cp,&op,&yp,&mp,&scratch,&b,&n,&q,&status};s=gffx_cuda_launch(kernel,q>b?q:b,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_shading_loss_query_backward(const gffx_tensor_view *query,const gffx_tensor_view *mean,const gffx_tensor_view *owners,const gffx_tensor_view *gradient,const gffx_tensor_view *mean_gradient,const gffx_execution_context *c,gffx_tensor_view *grad_u,gffx_tensor_view *grad_cosine,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction domain,kernel;CUdeviceptr vp=im_device_data(query),mp=im_device_data(mean),op=im_device_data(owners),gp=im_device_data(gradient),hp=im_device_data(mean_gradient),up=im_device_data(grad_u),cp=im_device_data(grad_cosine),status=(CUdeviceptr)(uintptr_t)w->data,scratch=status+8;
 int b=(int)mean->shape[0],q=(int)query->shape[0];gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_pq_backward_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_pq_backward")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"query backward kernels missing");
 {void *args[]={&vp,&mp,&op,&gp,&hp,&b,&q,&status};s=gffx_cuda_launch(domain,6*q>4*b?6*q:4*b,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&vp,&mp,&op,&gp,&hp,&up,&cp,&scratch,&b,&q,&status};s=gffx_cuda_launch(kernel,q>b?q:b,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}
