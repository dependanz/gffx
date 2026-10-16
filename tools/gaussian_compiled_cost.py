"""Inspect exact MSVC object prologues and driver-JIT Gaussian resource attributes."""
import argparse,ctypes as C,json,os,re,subprocess,time
from pathlib import Path
from tools import material_loss_cost as cost,shading_cuda_closeout as artifact

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    for name in ('fixtures','output','wheel','build','dumpbin'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True);a=p.parse_args()
    if not a.worker:
        cost.PROFILE['sampled_working_set_limit_bytes']=2*1024**3
        return cost.supervise(a.fixtures,a.output,worker_module='tools.gaussian_compiled_cost',worker_args=['--wheel',str(a.wheel),'--sha256',a.sha256,'--build',str(a.build),'--dumpbin',str(a.dumpbin)])
    row={'status':'working','worker_pid':os.getpid()};cost.write_report(a.output,row)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        row['wheel']=artifact.identity(a.wheel,a.sha256);row['driver_sha256']=artifact.digest(__file__)
        obj=a.build/'CMakeFiles/gffx_core.dir/native/core/image.c.obj';ptx=a.build/'gffx_kernels.ptx'
        row['inputs']={str(x):artifact.digest(x) for x in (obj,ptx,a.dumpbin)}
        text=subprocess.run([str(a.dumpbin),'/disasm',str(obj)],check=True,capture_output=True,text=True,timeout=20).stdout
        dis=a.output.with_suffix('.disasm.txt')
        with dis.open('x',encoding='utf-8') as f:f.write(text)
        row['disassembly_sha256']=artifact.digest(dis);row['cpu_functions']={}
        parts=re.split(r'(?m)^([A-Za-z_][A-Za-z_0-9]*):\s*$',text)
        for name,body in zip(parts[1::2],parts[2::2]):
            if name=='gw_call' or name.startswith('gffx_gw_') or name.startswith('gffx_image_gaussian'):
                before_call=body.split('call        ')[0]
                # Static prologue only: include pushes, exclude caller's return address.
                amounts=re.findall(r'sub\s+rsp,([0-9A-F]+)h',before_call)
                pushes=len(re.findall(r'\bpush\s+',before_call))
                row['cpu_functions'][name]={'stack_subtraction_bytes':sum(int(x,16) for x in amounts),'push_bytes':8*pushes,'calls':sorted(set(re.findall(r'call\s+([A-Za-z_][A-Za-z_0-9]*)',body)))}
        assert all(n in row['cpu_functions'] for n in ('gw_call','gffx_gw_evaluate','gffx_image_gaussian_weights'))
        torch.cuda.init();token=torch.zeros(1,device='cuda');torch.cuda.synchronize()
        row['gpu']={'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))}
        lib=C.WinDLL('nvcuda.dll');P=C.c_void_p;I=C.c_int
        lib.cuModuleLoadData.argtypes=[C.POINTER(P),P];lib.cuModuleUnload.argtypes=[P]
        lib.cuModuleGetFunction.argtypes=[C.POINTER(P),P,C.c_char_p]
        lib.cuFuncGetAttribute.argtypes=[C.POINTER(I),I,P]
        def check(code):
            if code:raise RuntimeError('CUDA driver status '+str(code))
        version=I();check(lib.cuDriverGetVersion(C.byref(version)));row['cuda_driver_api_version']=version.value
        module=P();blob=C.create_string_buffer(ptx.read_bytes());check(lib.cuModuleLoadData(C.byref(module),blob))
        row['cuda_functions']={}
        try:
            for name in ('gffx_gaussian_f32','gffx_gaussian_f64','gffx_gaussian_domain_f32','gffx_gaussian_domain_f64'):
                fun=P();check(lib.cuModuleGetFunction(C.byref(fun),module,name.encode()));attrs={}
                for label,code in (('max_threads_per_block',0),('static_shared_bytes_per_block',1),('local_bytes_per_thread',3),('registers_per_thread',4),('ptx_version',5),('binary_version',6)):
                    value=I();check(lib.cuFuncGetAttribute(C.byref(value),code,fun));attrs[label]=value.value
                row['cuda_functions'][name]=attrs
        finally:check(lib.cuModuleUnload(module))
        del token;torch.cuda.synchronize();row['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        assert row['torch_peak_allocated_bytes']<=64*1024**2
        row['scope']='Exact build object static stack prologues and same PTX driver JIT in existing Torch context. No kernel launched; attributes are per thread/block, not whole-process or whole-launch memory. CPU rows exclude external validation/libm frames and caller return addresses; no total-stack claim.'
        row['status']='passed'
    except Exception as e:row.update(status='stopped',error=str(e));raise
    finally:cost.write_report(a.output,row)
    return 0
if __name__=='__main__':raise SystemExit(main())
