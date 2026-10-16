"""Installed material ABI: atomic domains, disjoint storage and typed overflow."""
import ctypes as C
from pathlib import Path
import pytest,torch
from test_gaussian_abi_failures import View,Context,Buffer,view,L,Q,P,U
from test_principled_pairs_operation import fixture

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('family',['pairs','groups'])
def test_principled_pairs_abi_domains_storage_and_overflow(device,family):
 import gffx,gffx.torch.image
 lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
 forward=getattr(lib,'gffx_shading_principled_'+family);backward=getattr(lib,'gffx_shading_principled_'+family+'_backward');workspace=getattr(lib,'gffx_shading_principled_'+family+'_workspace')
 forward.argtypes=[C.POINTER(View)]*12+[U,C.POINTER(Context)]+[C.POINTER(View)]*2+[C.POINTER(Buffer),P]
 backward.argtypes=[C.POINTER(View)]*2+[C.POINTER(Context)]+[C.POINTER(View)]*2+[C.POINTER(Buffer),P]
 workspace.argtypes=[L,L,U,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
 budget,parameters,inputs=fixture();inputs=[x.to(device) for x in inputs];cuda=device=='cuda';stream=torch.cuda.Stream() if cuda else None
 ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1;ctx.stream=stream.cuda_stream if cuda else None
 value=torch.full((2,1,3),91.,dtype=torch.float64,device=device);jac=torch.full((2,1,3,16),92.,dtype=torch.float64,device=device);vv=[view(x) for x in inputs];yv,jv=view(value,True),view(jac,True)
 storage=torch.empty(24,dtype=torch.uint8,device=device);w=Buffer();w.size=C.sizeof(w);w.abi=65536;w.data=storage.data_ptr();w.capacity=24;w.alignment=8;w.device=ctx.device
 def sync():
  if cuda:torch.cuda.synchronize()
 def f():
  sync();result=forward(*[C.byref(x) for x in vv],2,C.byref(ctx),C.byref(yv),C.byref(jv),C.byref(w),None);sync();return result
 for p,q,d in (((9,1,0),(5,1,1)) if family=='pairs' else ((129,1,0),(129,1,1)))+((4,257,1),(1,1,2),(-1,0,0)):
  size,align=Q(91),Q(92);assert workspace(p,q,d,C.byref(ctx),C.byref(size),C.byref(align),None)==1;assert (size.value,align.value)==(91,92)
 for slot,index,bad in ((0,0,0.),(1,0,float('nan')),(5,0,0.),(9,(0,0),2.),(10,0,0.),(11,(0,0),-1.),(3,(0,0),float('inf'))):
  old=inputs[slot][index].item();inputs[slot][index]=bad;assert f()==1 and bool((value==91).all()) and bool((jac==92).all());inputs[slot][index]=old
 w.capacity=23;assert f()==3;w.capacity=24
 saved=jv.data;jv.data=yv.data;assert f()==1;jv.data=saved
 saved=yv.data;yv.data=vv[6].data;assert f()==1;yv.data=saved
 saved=w.data;w.data=vv[2].data;assert f()==1;w.data=saved
 assert f()==0
 gradient=torch.ones_like(value);point=torch.full((2,16),93.,dtype=torch.float64,device=device);light=torch.full((2,1,3),94.,dtype=torch.float64,device=device)
 def back():
  a,b,c,d=view(jac),view(gradient),view(point,True),view(light,True);sync();result=backward(C.byref(a),C.byref(b),C.byref(ctx),C.byref(c),C.byref(d),C.byref(w),None);sync();return result
 gradient[0,0,0]=float('nan');assert back()==1 and bool((point==93).all()) and bool((light==94).all());gradient.fill_(1.);assert back()==0
 jac.fill_(2.);gradient.fill_(torch.finfo(torch.float64).max);assert back()==4
