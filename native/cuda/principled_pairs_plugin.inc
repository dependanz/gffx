static gffx_status GFFX_CALL gffx_cuda_op_shading_principled_pairs(const gffx_tensor_view *cn,const gffx_tensor_view *logs,const gffx_tensor_view *curve,const gffx_tensor_view *partial,const gffx_tensor_view *mean,const gffx_tensor_view *width,const gffx_tensor_view *normal,const gffx_tensor_view *view,const gffx_tensor_view *light,const gffx_tensor_view *albedo,const gffx_tensor_view *roughness,const gffx_tensor_view *f0,uint32_t options,const gffx_execution_context *c,gffx_tensor_view *value,gffx_tensor_view *jacobian,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction domain,kernel;CUdeviceptr xp=im_device_data(cn),lp=im_device_data(logs),vp=im_device_data(curve),dp=im_device_data(partial),mp=im_device_data(mean),wp=im_device_data(width),np=im_device_data(normal),vv=im_device_data(view),ll=im_device_data(light),ap=im_device_data(albedo),rp=im_device_data(roughness),fp=im_device_data(f0),yp=im_device_data(value),jp=im_device_data(jacobian),status=(CUdeviceptr)(uintptr_t)w->data;
 int p=(int)normal->shape[0],q=(int)light->shape[1],n=(int)cn->shape[0],flags=(int)options,curve_rows=(int)curve->shape[0],count=curve_rows*(10*n-6);gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;if(count<p*q*3)count=p*q*3;if(count<n)count=n;if(count<p*3)count=p*3;
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_pp_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_pp_evaluate")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"Principled kernels missing");
 {void *args[]={&xp,&lp,&vp,&dp,&mp,&wp,&np,&vv,&ll,&ap,&rp,&fp,&p,&q,&n,&curve_rows,&status};s=gffx_cuda_launch(domain,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&xp,&lp,&vp,&dp,&mp,&wp,&np,&vv,&ll,&ap,&rp,&fp,&yp,&jp,&p,&q,&n,&curve_rows,&flags,&status};s=gffx_cuda_launch(kernel,p*q,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_shading_principled_pairs_backward(const gffx_tensor_view *jacobian,const gffx_tensor_view *gradient,const gffx_execution_context *c,gffx_tensor_view *point_gradient,gffx_tensor_view *light_gradient,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction finite,kernel;CUdeviceptr jp=im_device_data(jacobian),gp=im_device_data(gradient),pp=im_device_data(point_gradient),lp=im_device_data(light_gradient),status=(CUdeviceptr)(uintptr_t)w->data,scratch=status+8;int p=(int)jacobian->shape[0],q=(int)jacobian->shape[1];long long count=im_device_count(jacobian);gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;
 if(cuModuleGetFunction(&finite,gffx_cuda_module,"gffx_image_finite_f64")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_pp_backward")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"Principled backward kernels missing");
 {void *args[]={&jp,&count,&status};s=gffx_cuda_launch(finite,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;args[0]=&gp;count=im_device_count(gradient);s=gffx_cuda_launch(finite,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&jp,&gp,&pp,&lp,&scratch,&p,&q,&status};s=gffx_cuda_launch(kernel,p*q>p?p*q:p,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}

static gffx_status GFFX_CALL gffx_cuda_op_shading_ordered_rgba(const gffx_tensor_view *carry,const gffx_tensor_view *values,const gffx_execution_context *c,gffx_tensor_view *output,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction finite,kernel;CUdeviceptr ap=im_device_data(carry),vp=im_device_data(values),yp=im_device_data(output),status=(CUdeviceptr)(uintptr_t)w->data;int n=(int)values->shape[0];long long count=4;gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;
 if(cuModuleGetFunction(&finite,gffx_cuda_module,"gffx_image_finite_f64")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_ordered_rgba")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"ordered RGBA kernels missing");
 {void *args[]={&ap,&count,&status};s=gffx_cuda_launch(finite,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;if(n){args[0]=&vp;count=4*n;s=gffx_cuda_launch(finite,count,args,c,d);if(s!=GFFX_STATUS_OK)return s;}}
 s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&ap,&vp,&yp,&n,&status};s=gffx_cuda_launch(kernel,4,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}

static gffx_status GFFX_CALL gffx_cuda_op_shading_unit_newton(const gffx_tensor_view *input,const gffx_execution_context *c,gffx_tensor_view *output,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction domain,kernel;CUdeviceptr xp=im_device_data(input),yp=im_device_data(output),status=(CUdeviceptr)(uintptr_t)w->data;int n=(int)input->shape[0];gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;if(!n)return GFFX_STATUS_OK;
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_unit_newton_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_unit_newton_evaluate")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"unit Newton kernels missing");
 {void *args[]={&xp,&n,&status};s=gffx_cuda_launch(domain,n,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&xp,&yp,&n};s=gffx_cuda_launch(kernel,n,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}
static gffx_status GFFX_CALL gffx_cuda_op_shading_unit_vectors(const gffx_tensor_view *input,const gffx_execution_context *c,gffx_tensor_view *output,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction domain,kernel;CUdeviceptr xp=im_device_data(input),yp=im_device_data(output),status=(CUdeviceptr)(uintptr_t)w->data;int n=(int)input->shape[0];gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;if(!n)return GFFX_STATUS_OK;
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_unit_vectors_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_unit_vectors_evaluate")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"unit vectors kernels missing");
 {void *args[]={&xp,&n,&status};s=gffx_cuda_launch(domain,n,args,c,d);if(s!=GFFX_STATUS_OK)return s;}
 s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&xp,&yp,&n};s=gffx_cuda_launch(kernel,n,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}

static gffx_status GFFX_CALL gffx_cuda_op_shading_delta_brdf(const gffx_tensor_view *normal,const gffx_tensor_view *brdf,const gffx_tensor_view *direction,const gffx_tensor_view *incoming,const gffx_tensor_view *cotangent,const gffx_execution_context *c,gffx_tensor_view *value,gffx_tensor_view *gradient,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 CUfunction domain,kernel;CUdeviceptr np=im_device_data(normal),bp=im_device_data(brdf),lp=im_device_data(direction),ip=im_device_data(incoming),gp=im_device_data(cotangent),vp=im_device_data(value),op=im_device_data(gradient),status=(CUdeviceptr)(uintptr_t)w->data;int p=(int)brdf->shape[0],q=(int)brdf->shape[1];gffx_status s=pc_device_init(c,w,d);if(s!=GFFX_STATUS_OK)return s;if(!p)return GFFX_STATUS_OK;
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_delta_brdf_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_delta_brdf_evaluate")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"delta BRDF kernels missing");
 {void *args[]={&np,&bp,&lp,&ip,&gp,&p,&q,&status};s=gffx_cuda_launch(domain,p,args,c,d);if(s!=GFFX_STATUS_OK)return s;}s=im_device_status(c,w,0,d);if(s!=GFFX_STATUS_OK)return s;
 {void *args[]={&np,&bp,&lp,&ip,&gp,&vp,&op,&p,&q,&status};s=gffx_cuda_launch(kernel,p,args,c,d);if(s!=GFFX_STATUS_OK)return s;}return im_device_status(c,w,1,d);
}
