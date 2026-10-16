#include "../core/opaque_samples_args.h"
static gffx_status GFFX_CALL gffx_cuda_op_shading_opaque_samples(const gffx_tensor_view *const *v,const gffx_execution_context *c,gffx_tensor_view *weighted,gffx_tensor_view *rgba,const gffx_buffer *w,gffx_diagnostic_buffer *d){
 GffxOpaqueSamples x;CUfunction domain,kernel;CUdeviceptr status=(CUdeviceptr)(uintptr_t)w->data;long long largest=1;int i;gffx_status s=pc_device_init(c,w,d);if(s)return s;
 for(i=0;i<18;++i){x.data[i]=(const void*)(uintptr_t)im_device_data(v[i]);x.count[i]=im_device_count(v[i]);if(x.count[i]>largest)largest=x.count[i];}
 x.pixels=(int)v[1]->shape[0];x.nv=(int)v[2]->shape[0];x.nf=(int)v[3]->shape[0];x.nodes=(int)v[7]->shape[0];x.weighted=(double*)(uintptr_t)im_device_data(weighted);x.rgba=(double*)(uintptr_t)im_device_data(rgba);
 if(cuModuleGetFunction(&domain,gffx_cuda_module,"gffx_opaque_samples_domain")!=CUDA_SUCCESS||cuModuleGetFunction(&kernel,gffx_cuda_module,"gffx_opaque_samples_evaluate")!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque sample kernels missing");
 {void *a[]={&x,&largest,&status};if(cuLaunchKernel(domain,32,1,1,128,1,1,0,(CUstream)c->stream,a,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque sample domain launch");}
 s=im_device_status(c,w,0,d);if(s)return s;
 {void *a[]={&x,&status};if(cuLaunchKernel(kernel,x.pixels,1,1,128,1,1,0,(CUstream)c->stream,a,NULL)!=CUDA_SUCCESS)return gffx_cuda_fail(d,GFFX_STATUS_BACKEND_FAILURE,"opaque sample composition launch");}
 return im_device_status(c,w,1,d);
}
