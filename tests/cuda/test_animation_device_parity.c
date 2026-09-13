/* Real-device public C parity. No framework or research assets are involved. */
#include <gffx/animation.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef union storage { double f64[256]; float f32[512]; int32_t i32[512]; } storage;
typedef struct descriptor { gffx_tensor_view view; int64_t shape[4],strides[4]; } descriptor;
#define CHECK(x) do { if(!(x)){fprintf(stderr,"animation device parity line %d: %s\n",__LINE__,#x);return 1;} } while(0)
static uint64_t count(const gffx_tensor_view *v){uint64_t n=1;uint32_t i;for(i=0;i<v->rank;++i)n*=(uint64_t)v->shape[i];return n;}
static uint64_t extent(const gffx_tensor_view *v){return count(v)*(v->dtype==GFFX_DTYPE_FLOAT64?8:4);}
static void init(descriptor *v,void *data,gffx_dtype type,int rank,int64_t a,int64_t b,int64_t c,int64_t d){
    int i;int64_t stride=1;memset(v,0,sizeof(*v));v->shape[0]=a;v->shape[1]=b;v->shape[2]=c;v->shape[3]=d;
    for(i=rank;i-->0;){v->strides[i]=stride;stride*=v->shape[i]?v->shape[i]:1;}
    v->view.struct_size=sizeof(v->view);v->view.abi_version=GFFX_ABI_VERSION;v->view.data=data;
    v->view.rank=(uint32_t)rank;v->view.shape=v->shape;v->view.strides=v->strides;v->view.dtype=type;v->view.flags=GFFX_TENSOR_READ_ONLY;v->view.device_type=GFFX_DEVICE_CPU;
}
static gffx_status query(int op,gffx_dtype dtype,const gffx_execution_context *c,uint64_t *bytes,uint64_t *alignment){
    switch(op){
    case 0:return gffx_mesh_blend_shapes_workspace(2,4,2,dtype,c,bytes,alignment,NULL);
    case 1:return gffx_transforms_rodrigues_workspace(2,dtype,c,bytes,alignment,NULL);
    case 2:return gffx_skeleton_forward_kinematics_workspace(2,3,dtype,c,bytes,alignment,NULL);
    case 3:return gffx_mesh_skin_lbs_workspace(2,4,3,dtype,c,bytes,alignment,NULL);
    default:return gffx_mesh_evaluate_barycentric_workspace(2,4,3,dtype,c,bytes,alignment,NULL);}
}
static gffx_status call(int op,int backward,gffx_tensor_view *v,gffx_tensor_view *out,const gffx_execution_context *c,const gffx_buffer *scratch){
    switch(op){
    case 0:return backward?gffx_mesh_blend_shapes_backward(v,v+1,v+2,v+3,c,out,out+1,out+2,scratch,NULL):gffx_mesh_blend_shapes(v,v+1,v+2,c,out,scratch,NULL);
    case 1:return backward?gffx_transforms_rodrigues_backward(v,v+1,c,out,scratch,NULL):gffx_transforms_rodrigues(v,c,out,scratch,NULL);
    case 2:return backward?gffx_skeleton_forward_kinematics_backward(v,v+1,v+2,c,out,scratch,NULL):gffx_skeleton_forward_kinematics(v,v+1,c,out,scratch,NULL);
    case 3:return backward?gffx_mesh_skin_lbs_backward(v,v+1,v+2,v+3,c,out,out+1,out+2,scratch,NULL):gffx_mesh_skin_lbs(v,v+1,v+2,c,out,scratch,NULL);
    default:return backward?gffx_mesh_evaluate_barycentric_backward(v,v+1,v+2,v+3,v+4,c,out,out+1,scratch,NULL):gffx_mesh_evaluate_barycentric(v,v+1,v+2,v+3,c,out,scratch,NULL);}
}
static int fixture(int op,gffx_dtype dtype,CUstream stream){
    storage data[5],expected[3],actual[3],host_scratch;
    descriptor descriptors[5];gffx_tensor_view host[5],device[5],cpu_output[3],gpu_output[3];
    CUdeviceptr device_input[5]={0},device_output[3]={0},device_scratch=0;
    gffx_execution_context cpu={0},gpu={0};gffx_buffer cpu_buffer={0},gpu_buffer={0};
    uint64_t cpu_bytes,gpu_bytes,alignment;int i,j,backward,ninputs=0,gradient_index=0,output_count;int primal[3]={0,1,2};
    for(i=0;i<5;++i)for(j=0;j<256;++j){double value=((j*5+i*3)%17-8)/32.;if(dtype==GFFX_DTYPE_FLOAT64)data[i].f64[j]=value;else data[i].f32[j]=(float)value;}
    if(op==0){ninputs=4;gradient_index=3;
        init(descriptors,data,dtype,2,4,3,0,0);init(descriptors+1,data+1,dtype,3,2,4,3,0);
        init(descriptors+2,data+2,dtype,2,2,2,0,0);init(descriptors+3,data+3,dtype,3,2,4,3,0);
    }else if(op==1){ninputs=2;gradient_index=1;
        init(descriptors,data,dtype,2,2,3,0,0);init(descriptors+1,data+1,dtype,3,2,3,3,0);
    }else if(op==2){ninputs=3;gradient_index=2;
        init(descriptors,data,dtype,4,2,3,4,4);init(descriptors+1,data+1,GFFX_DTYPE_INT32,1,3,0,0,0);
        data[1].i32[0]=1;data[1].i32[1]=-1;data[1].i32[2]=1;init(descriptors+2,data+2,dtype,4,2,3,4,4);
    }else if(op==3){ninputs=4;gradient_index=3;
        init(descriptors,data,dtype,3,2,4,3,0);init(descriptors+1,data+1,dtype,4,2,3,4,4);
        init(descriptors+2,data+2,dtype,2,4,3,0,0);init(descriptors+3,data+3,dtype,3,2,4,3,0);
    }else{ninputs=5;gradient_index=4;primal[1]=3;
        init(descriptors,data,dtype,3,2,4,3,0);init(descriptors+1,data+1,GFFX_DTYPE_INT32,2,2,3,0,0);
        init(descriptors+2,data+2,GFFX_DTYPE_INT32,1,3,0,0,0);init(descriptors+3,data+3,dtype,2,3,3,0,0);
        init(descriptors+4,data+4,dtype,3,2,3,3,0);
        {int32_t faces[6]={0,1,2,1,2,3};memcpy(data[1].i32,faces,sizeof(faces));}
        data[2].i32[0]=0;data[2].i32[1]=1;data[2].i32[2]=0;
    }
    cpu.struct_size=sizeof(cpu);cpu.abi_version=GFFX_ABI_VERSION;cpu.device_type=GFFX_DEVICE_CPU;gpu=cpu;gpu.device_type=GFFX_DEVICE_CUDA;gpu.stream=(void *)stream;
    CHECK(query(op,dtype,&cpu,&cpu_bytes,&alignment)==GFFX_STATUS_OK);CHECK(cpu_bytes<=sizeof(host_scratch));
    cpu_buffer.struct_size=sizeof(cpu_buffer);cpu_buffer.abi_version=GFFX_ABI_VERSION;cpu_buffer.data=&host_scratch;cpu_buffer.capacity_bytes=cpu_bytes;cpu_buffer.alignment=8;cpu_buffer.device_type=GFFX_DEVICE_CPU;
    CHECK(query(op,dtype,&gpu,&gpu_bytes,&alignment)==GFFX_STATUS_OK);CHECK(gpu_bytes<=4096);
    CHECK(cuMemAlloc(&device_scratch,gpu_bytes)==CUDA_SUCCESS);gpu_buffer=cpu_buffer;gpu_buffer.data=(void *)(uintptr_t)device_scratch;gpu_buffer.capacity_bytes=gpu_bytes;gpu_buffer.device_type=GFFX_DEVICE_CUDA;
    for(i=0;i<ninputs;++i){host[i]=descriptors[i].view;device[i]=host[i];CHECK(extent(host+i)<=sizeof(storage));
        CHECK(cuMemAlloc(device_input+i,extent(host+i))==CUDA_SUCCESS);device[i].data=(void *)(uintptr_t)device_input[i];device[i].device_type=GFFX_DEVICE_CUDA;
        CHECK(cuMemcpyHtoD(device_input[i],host[i].data,extent(host+i))==CUDA_SUCCESS);}
    for(i=0;i<3;++i)CHECK(cuMemAlloc(device_output+i,sizeof(storage))==CUDA_SUCCESS);
    for(backward=0;backward<2;++backward){output_count=backward?(op==0||op==3?3:op==4?2:1):1;
        for(i=0;i<output_count;++i){cpu_output[i]=host[backward?primal[i]:gradient_index];cpu_output[i].data=expected+i;cpu_output[i].flags=GFFX_TENSOR_OUTPUT;
            gpu_output[i]=cpu_output[i];gpu_output[i].data=(void *)(uintptr_t)device_output[i];gpu_output[i].device_type=GFFX_DEVICE_CUDA;}
        CHECK(call(op,backward,host,cpu_output,&cpu,&cpu_buffer)==GFFX_STATUS_OK);
        CHECK(call(op,backward,device,gpu_output,&gpu,&gpu_buffer)==GFFX_STATUS_OK);
        CHECK(cuStreamSynchronize(stream)==CUDA_SUCCESS);
        for(i=0;i<output_count;++i)CHECK(cuMemcpyDtoH(actual+i,device_output[i],extent(cpu_output+i))==CUDA_SUCCESS);
        for(i=0;i<output_count;++i)CHECK(memcmp(expected+i,actual+i,(size_t)extent(cpu_output+i))==0);
    }
    for(i=0;i<ninputs;++i)CHECK(cuMemFree(device_input[i])==CUDA_SUCCESS);
    for(i=0;i<3;++i)CHECK(cuMemFree(device_output[i])==CUDA_SUCCESS);
    CHECK(cuMemFree(device_scratch)==CUDA_SUCCESS);return 0;
}
int main(int argc,char **argv){
    CUdevice device;CUcontext context;CUstream stream;int devices=0,op;
    CHECK(argc==3);op=atoi(argv[2]);CHECK(op>=0&&op<5);
#if defined(_WIN32)
    CHECK(_putenv_s("GFFX_CUDA_PLUGIN_PATH",argv[1])==0);
#else
    CHECK(setenv("GFFX_CUDA_PLUGIN_PATH",argv[1],1)==0);
#endif
    if(cuInit(0)!=CUDA_SUCCESS||cuDeviceGetCount(&devices)!=CUDA_SUCCESS||devices==0)return 77;
    CHECK(cuDeviceGet(&device,0)==CUDA_SUCCESS);CHECK(cuDevicePrimaryCtxRetain(&context,device)==CUDA_SUCCESS);
    CHECK(cuCtxSetCurrent(context)==CUDA_SUCCESS);CHECK(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING)==CUDA_SUCCESS);
    CHECK(fixture(op,GFFX_DTYPE_FLOAT32,stream)==0);CHECK(fixture(op,GFFX_DTYPE_FLOAT64,stream)==0);
    CHECK(cuStreamDestroy(stream)==CUDA_SUCCESS);CHECK(cuCtxSetCurrent(NULL)==CUDA_SUCCESS);CHECK(cuDevicePrimaryCtxRelease(device)==CUDA_SUCCESS);
    return 0;
}
