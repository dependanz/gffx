/* Independent checks and exact padding precede the unchanged ordered worker. */
static gffx_status gffx_cuda_opaque_prepare(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,const gffx_tensor_view *g,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_execution_context *c,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
 CUfunction domain,clear;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),gp=im_device_data(g),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v->shape[0],nf=f->shape[0],cc=im_device_count(cuts),gc=g?im_device_count(g):0,ac=im_device_count(out),bc=im_device_count(other);int real64=v->dtype==GFFX_DTYPE_FLOAT64,backward=g!=NULL;gffx_status s;
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_opaque_domain_parallel")!=CUDA_SUCCESS||cuModuleGetFunction(&clear,gffx_cuda_module,"gffx_opaque_clear_parallel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque preparation kernels missing");
 {void *args[]={&vp,&fp,&cp,&gp,&nv,&nf,&cc,&gc,&real64,&sp};if(cuLaunchKernel(domain,32,1,1,128,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque parallel domain launch");}
 s=im_device_status(c,scratch,0,d);if(s)return s;
 {void *args[]={&ap,&bp,&ac,&bc,&real64,&backward};if(cuLaunchKernel(clear,32,1,1,128,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque parallel clear launch");}
 return GFFX_STATUS_OK;
}
/* A single ordered device worker; no floating atomics or host geometry copy. */
static gffx_status GFFX_CALL gffx_cuda_opaque_dispatch(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int edge){
    CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),gp=im_device_data(g),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
    long long nv=v->shape[0],nf=f->shape[0];int rows=(int)cuts->shape[1],nc=(int)cuts->shape[2],real64=v->dtype==GFFX_DTYPE_FLOAT64,backward=g!=NULL,domain=1;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
    if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_quadrature_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque quadrature kernel missing");
    {void *args[]={&vp,&fp,&cp,&gp,&ap,&bp,&nv,&nf,&nc,&rows,&h,&w,&ox,&oy,&real64,&backward,&domain,&edge,&sp};
        s=gffx_cuda_opaque_prepare(v,f,cuts,g,out,other,c,scratch,d);if(s)return s;domain=0;
        if(!backward && edge<0 && h*w>1 && scratch->capacity_bytes>=2*1048576){
            unsigned lanes=scratch->capacity_bytes>=4*1048576?4:(unsigned)(scratch->capacity_bytes/1048576);if((long long)lanes>h*w)lanes=(unsigned)(h*w);
            int material=edge==-2;
            if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_quadrature_pixel_lanes")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque pixel lanes missing");
            void *parallel[]={&vp,&fp,&cp,&ap,&bp,&nv,&nf,&nc,&rows,&h,&w,&ox,&oy,&real64,&material,&sp};
            if(cuLaunchKernel(kernel,lanes,1,1,1,1,1,0,(CUstream)c->stream,parallel,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque pixel lanes launch failed");
        }else if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque quadrature launch failed");}
    return im_device_status(c,scratch,1,d);
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_quadrature(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_cuda_opaque_dispatch(v,f,cuts,h,w,ox,oy,g,c,out,other,scratch,d,-1);}
static gffx_status GFFX_CALL gffx_cuda_op_opaque_boundary(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,int32_t edge,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_cuda_opaque_dispatch(v,f,cuts,h,w,ox,oy,NULL,c,out,other,scratch,d,edge);}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_material(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,int64_t h,int64_t w,double ox,double oy,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){return gffx_cuda_opaque_dispatch(v,f,cuts,h,w,ox,oy,g,c,out,other,scratch,d,-2);}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_batch_rule(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t rule,int32_t derivatives){
 CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),gp=im_device_data(g),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v->shape[0],nf=f->shape[0];int rows=(int)cuts->shape[1],nc=(int)cuts->shape[2],begin=cursor[0],batch=cursor[1],slots=cursor[2],blocks=cursor[3],real64=v->dtype==GFFX_DTYPE_FLOAT64,backward=g!=NULL,domain=1;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
 if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_batch_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch kernel missing");
 {void *args[]={&vp,&fp,&cp,&gp,&ap,&bp,&nv,&nf,&nc,&rows,&ox,&oy,&begin,&batch,&slots,&blocks,&real64,&backward,&domain,&sp,&rule,&derivatives};
  s=gffx_cuda_opaque_prepare(v,f,cuts,g,out,other,c,scratch,d);if(s)return s;domain=0;
  if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch launch");}
 s=im_device_status(c,scratch,1,d);if(s)return s;
 if(cuMemcpyDtoH(receipt,sp+sizeof(int32_t),8*sizeof(int32_t))!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch receipt copy");return GFFX_STATUS_OK;
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_batch(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
 return gffx_cuda_op_opaque_batch_rule(v,f,cuts,ox,oy,cursor,receipt,g,c,out,other,scratch,d,0,3);
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_partition(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,gffx_tensor_view *rows,int32_t *result,int count,int begin,gffx_tensor_view *ids,gffx_tensor_view *samples,const gffx_execution_context *c,const gffx_buffer *scratch,gffx_diagnostic_buffer *d){
 int capacity=(int)rows->shape[0];CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),rp=im_device_data(rows),ap=im_device_data(ids),bp=im_device_data(samples),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v?v->shape[0]:0,nf=f?f->shape[0]:0;int nc=cuts?(int)cuts->shape[2]:0;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
 if(v){CUfunction domain;CUdeviceptr gp=0;long long cc=im_device_count(cuts),gc=0;int real64=1;
  if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_opaque_domain_parallel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"partition domain kernel missing");
  void *args[]={&vp,&fp,&cp,&gp,&nv,&nf,&cc,&gc,&real64,&sp};if(cuLaunchKernel(domain,32,1,1,128,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"partition domain launch");s=im_device_status(c,scratch,0,d);if(s)return s;
 }
 if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_partition_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"partition kernel missing");
 void *args[]={&vp,&fp,&cp,&rp,&ap,&bp,&nv,&nf,&nc,&ox,&oy,&count,&begin,&capacity,&sp};if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"partition launch");s=im_device_status(c,scratch,1,d);if(s)return s;
 if(result&&cuMemcpyDtoH(result,sp+4,8)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"partition receipt copy");return GFFX_STATUS_OK;
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_prefix_backward(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t *prefix){
 int rule=2,face=prefix[0],seen=prefix[1];int32_t result[10];
 CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),gp=im_device_data(g),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v->shape[0],nf=f->shape[0];int rows=(int)cuts->shape[1],nc=(int)cuts->shape[2],begin=cursor[0],batch=cursor[1],slots=cursor[2],blocks=cursor[3],real64=v->dtype==GFFX_DTYPE_FLOAT64,backward=g!=NULL,domain=1;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
 if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_prefix_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch kernel missing");
 {void *args[]={&vp,&fp,&cp,&gp,&ap,&bp,&nv,&nf,&nc,&rows,&ox,&oy,&begin,&batch,&slots,&blocks,&real64,&backward,&domain,&sp,&rule,&derivatives,&face,&seen};
  s=gffx_cuda_opaque_prepare(v,f,cuts,g,out,other,c,scratch,d);if(s)return s;domain=0;
  if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch launch");}
 s=im_device_status(c,scratch,1,d);if(s)return s;
 if(cuMemcpyDtoH(result,sp+sizeof(int32_t),10*sizeof(int32_t))!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch receipt copy");memcpy(receipt,result,8*sizeof(int32_t));prefix[0]=result[8];prefix[1]=result[9];return GFFX_STATUS_OK;
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_prefix_backward64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t *prefix){
 int rule=3,face=prefix[0],seen=prefix[1];int32_t result[10];
 CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),gp=im_device_data(g),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v->shape[0],nf=f->shape[0];int rows=(int)cuts->shape[1],nc=(int)cuts->shape[2],begin=cursor[0],batch=cursor[1],slots=cursor[2],blocks=cursor[3],real64=v->dtype==GFFX_DTYPE_FLOAT64,backward=g!=NULL,domain=1;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
 if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_prefix_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch kernel missing");
 {void *args[]={&vp,&fp,&cp,&gp,&ap,&bp,&nv,&nf,&nc,&rows,&ox,&oy,&begin,&batch,&slots,&blocks,&real64,&backward,&domain,&sp,&rule,&derivatives,&face,&seen};
  s=gffx_cuda_opaque_prepare(v,f,cuts,g,out,other,c,scratch,d);if(s)return s;domain=0;
  if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch launch");}
 s=im_device_status(c,scratch,1,d);if(s)return s;
 if(cuMemcpyDtoH(result,sp+sizeof(int32_t),10*sizeof(int32_t))!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch receipt copy");memcpy(receipt,result,8*sizeof(int32_t));prefix[0]=result[8];prefix[1]=result[9];return GFFX_STATUS_OK;
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_boundary_batch64(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t edge,int32_t unused){
 CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v->shape[0],nf=f->shape[0];int nc=(int)cuts->shape[2],begin=cursor[0],batch=cursor[1],slots=cursor[2],blocks=cursor[3],real64=v->dtype==GFFX_DTYPE_FLOAT64;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
 if(g||edge<0||edge>3||cuts->shape[1]!=3)return gffx_cuda_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"boundary stream metadata");
 if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_boundary_batch64_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"boundary stream kernel missing");
 s=gffx_cuda_opaque_prepare(v,f,cuts,NULL,out,other,c,scratch,d);if(s)return s;
 {void *args[]={&vp,&fp,&cp,&ap,&bp,&nv,&nf,&nc,&ox,&oy,&begin,&batch,&slots,&blocks,&real64,&edge,&sp};
  if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"boundary stream launch");}
 s=im_device_status(c,scratch,1,d);if(s)return s;
 if(cuMemcpyDtoH(receipt,sp+sizeof(int32_t),8*sizeof(int32_t))!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"boundary stream receipt copy");return GFFX_STATUS_OK;
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_prefix_backward80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t derivatives,int32_t *prefix){
 int rule=4,face=prefix[0],seen=prefix[1];int32_t result[10];
 CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),gp=im_device_data(g),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v->shape[0],nf=f->shape[0];int rows=(int)cuts->shape[1],nc=(int)cuts->shape[2],begin=cursor[0],batch=cursor[1],slots=cursor[2],blocks=cursor[3],real64=v->dtype==GFFX_DTYPE_FLOAT64,backward=g!=NULL,domain=1;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
 if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_prefix_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch kernel missing");
 {void *args[]={&vp,&fp,&cp,&gp,&ap,&bp,&nv,&nf,&nc,&rows,&ox,&oy,&begin,&batch,&slots,&blocks,&real64,&backward,&domain,&sp,&rule,&derivatives,&face,&seen};
  s=gffx_cuda_opaque_prepare(v,f,cuts,g,out,other,c,scratch,d);if(s)return s;domain=0;
  if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch launch");}
 s=im_device_status(c,scratch,1,d);if(s)return s;
 if(cuMemcpyDtoH(result,sp+sizeof(int32_t),10*sizeof(int32_t))!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque batch receipt copy");memcpy(receipt,result,8*sizeof(int32_t));prefix[0]=result[8];prefix[1]=result[9];return GFFX_STATUS_OK;
}

static gffx_status GFFX_CALL gffx_cuda_op_opaque_boundary_batch80(const gffx_tensor_view *v,const gffx_tensor_view *f,const gffx_tensor_view *cuts,double ox,double oy,const int32_t *cursor,int32_t *receipt,const gffx_tensor_view *g,const gffx_execution_context *c,gffx_tensor_view *out,gffx_tensor_view *other,const gffx_buffer *scratch,gffx_diagnostic_buffer *d,int32_t edge,int32_t unused){
 CUfunction kernel;CUdeviceptr vp=im_device_data(v),fp=im_device_data(f),cp=im_device_data(cuts),ap=im_device_data(out),bp=im_device_data(other),sp=(CUdeviceptr)(uintptr_t)scratch->data;
 long long nv=v->shape[0],nf=f->shape[0];int nc=(int)cuts->shape[2],begin=cursor[0],batch=cursor[1],slots=cursor[2],blocks=cursor[3],real64=v->dtype==GFFX_DTYPE_FLOAT64;gffx_status s=pc_device_init(c,scratch,d);if(s)return s;
 if(g||edge<0||edge>3||cuts->shape[1]!=3)return gffx_cuda_fail(d,GFFX_STATUS_INVALID_ARGUMENT,"boundary stream metadata");
 if(cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_boundary_batch80_kernel")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"boundary stream kernel missing");
 s=gffx_cuda_opaque_prepare(v,f,cuts,NULL,out,other,c,scratch,d);if(s)return s;
 {void *args[]={&vp,&fp,&cp,&ap,&bp,&nv,&nf,&nc,&ox,&oy,&begin,&batch,&slots,&blocks,&real64,&edge,&sp};
  if(cuLaunchKernel(kernel,1,1,1,1,1,1,0,(CUstream)c->stream,args,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"boundary stream launch");}
 s=im_device_status(c,scratch,1,d);if(s)return s;
 if(cuMemcpyDtoH(receipt,sp+sizeof(int32_t),8*sizeof(int32_t))!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"boundary stream receipt copy");return GFFX_STATUS_OK;
}
