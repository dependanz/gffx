"""Research-only CUDA Driver invocation for the bounded geometry prototype."""
import ctypes,os
from pathlib import Path
import torch

class Probe:
    def __init__(self):
        torch.cuda.init();self.device=torch.device('cuda:0');torch.empty(1,device=self.device)
        self.driver=ctypes.WinDLL('nvcuda.dll');d=self.driver;P=ctypes.c_void_p;I=ctypes.c_int;U=ctypes.c_uint
        signatures={'cuInit':[U],'cuCtxSynchronize':[],'cuModuleLoadData':[ctypes.POINTER(P),P],'cuModuleGetFunction':[ctypes.POINTER(P),P,ctypes.c_char_p],'cuModuleUnload':[P],'cuLaunchKernel':[P,U,U,U,U,U,U,U,P,ctypes.POINTER(P),P]}
        for name,args in signatures.items():fn=getattr(d,name);fn.argtypes=args;fn.restype=I
        self.check(d.cuInit(0));self.module=P();ptx=ctypes.create_string_buffer(Path(os.environ['GFFX_BATCH_CUDA_PTX']).read_bytes())
        self.check(d.cuModuleLoadData(ctypes.byref(self.module),ptx));self.fn=P();self.check(d.cuModuleGetFunction(ctypes.byref(self.fn),self.module,b'batch_probe_cuda'))
        self.scratch=torch.empty(1048576,dtype=torch.uint8,device=self.device)
    @staticmethod
    def check(code):
        if code:raise RuntimeError('CUDA Driver status '+str(code))
    def call(self,v,f,c,cursor,*,seed=-1,cut_seed=-1,gradient=None):
        tensors=[torch.as_tensor(x,device=self.device).contiguous() for x in (v,f,c)]
        v,f,c=tensors;state=torch.tensor(cursor,dtype=torch.int32,device=self.device)
        ids=torch.empty((1,800),dtype=torch.int32,device=self.device);q=torch.empty((1,800,4),dtype=torch.float64,device=self.device)
        receipt=torch.empty(9,dtype=torch.int32,device=self.device);answer=torch.empty(1,dtype=torch.float64,device=self.device)
        g=None if gradient is None else torch.as_tensor(gradient,device=self.device).contiguous()
        P,I,L=ctypes.c_void_p,ctypes.c_int,ctypes.c_int64
        values=[P(v.data_ptr()),I(len(v)),P(f.data_ptr()),I(len(f)),P(c.data_ptr()),I(c.shape[1]),P(state.data_ptr()),L(seed),L(cut_seed),P(0 if g is None else g.data_ptr()),P(ids.data_ptr()),P(q.data_ptr()),P(receipt.data_ptr()),P(answer.data_ptr()),P(self.scratch.data_ptr())]
        args=(P*len(values))(*(ctypes.cast(ctypes.byref(x),P) for x in values))
        torch.cuda.synchronize();self.check(self.driver.cuLaunchKernel(self.fn,1,1,1,1,1,1,0,None,args,None));self.check(self.driver.cuCtxSynchronize())
        return ids.cpu().numpy(),q.cpu().numpy(),receipt.cpu().numpy(),answer.cpu().numpy()
    def close(self):
        self.check(self.driver.cuModuleUnload(self.module))
