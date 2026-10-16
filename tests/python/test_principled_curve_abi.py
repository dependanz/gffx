"""Installed native curve: C ABI atomic domains, aliases and resource limits."""
import ctypes as C
from pathlib import Path
import pytest,torch
from test_gaussian_abi_failures import View,Context,Buffer,view,L,Q,P

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_curve_abi_domains_storage_and_overflow(device):
    import gffx,gffx.torch.image
    lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
    lib.gffx_shading_loss_curve.argtypes=[C.POINTER(View)]*3+[C.POINTER(Context)]+[C.POINTER(View)]*2+[C.POINTER(Buffer),P]
    lib.gffx_shading_loss_curve_backward.argtypes=[C.POINTER(View)]*2+[C.POINTER(Context),C.POINTER(View),C.POINTER(Buffer),P]
    lib.gffx_shading_loss_curve_workspace.argtypes=[L,L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
    cuda=device=='cuda';stream=torch.cuda.Stream() if cuda else None
    ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1;ctx.stream=stream.cuda_stream if cuda else None
    x=torch.tensor([0.],dtype=torch.float64,device=device);b=torch.full((2,3,4),-.5,dtype=torch.float64,device=device);u=torch.full((2,),.37,dtype=torch.float64,device=device)
    y=torch.full((2,4),91.,dtype=torch.float64,device=device);dy=torch.full_like(y,92.);cot=torch.ones_like(y);out=torch.full((2,),93.,dtype=torch.float64,device=device)
    storage=torch.empty(72,dtype=torch.uint8,device=device);w=Buffer();w.size=C.sizeof(w);w.abi=65536;w.data=storage.data_ptr();w.capacity=72;w.alignment=8;w.device=ctx.device
    xv,bv,uv,yv,dv,gv,ov=view(x),view(b),view(u),view(y,True),view(dy,True),view(cot),view(out,True)
    def sync():
        if cuda:torch.cuda.synchronize()
    def f():
        sync();result=lib.gffx_shading_loss_curve(C.byref(xv),C.byref(bv),C.byref(uv),C.byref(ctx),C.byref(yv),C.byref(dv),C.byref(w),None);sync();return result
    def back():
        pv=view(dy);sync();result=lib.gffx_shading_loss_curve_backward(C.byref(pv),C.byref(gv),C.byref(ctx),C.byref(ov),C.byref(w),None);sync();return result
    for batch,n in ((33,1),(1,513),(-1,1),(2**63-1,512)):
        size,align=Q(91),Q(92);assert lib.gffx_shading_loss_curve_workspace(batch,n,C.byref(ctx),C.byref(size),C.byref(align),None)==1
        assert (size.value,align.value)==(91,92)
    for t,idx,bad in ((x,0,float('nan')),(u,1,1.1),(b,(1,2,3),-746.)):
        old=t[idx].item();t[idx]=bad;assert f()==1 and bool((y==91).all()) and bool((dy==92).all());t[idx]=old
    w.capacity=71;assert f()==3;w.capacity=72
    saved=dv.data;dv.data=yv.data;assert f()==1;dv.data=saved
    saved=yv.data;yv.data=bv.data;assert f()==1;yv.data=saved
    saved=w.data;w.data=bv.data;assert f()==1;w.data=saved
    assert f()==0 and bool(torch.isfinite(y).all()) and bool(torch.isfinite(dy).all())
    cot[-1,-1]=float('nan');assert back()==1 and bool((out==93).all());cot.fill_(1.)
    assert back()==0
    dy.fill_(2.);cot.fill_(torch.finfo(torch.float64).max);assert back()==4
