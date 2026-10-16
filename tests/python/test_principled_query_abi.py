"""Raw query ABI pre-write rejection and typed overflow, on both devices."""
import ctypes as C
from pathlib import Path
import pytest,torch
from test_gaussian_abi_failures import View,Context,Buffer,view as floating_view,L,Q,P

def view(t,output=False):
 v=floating_view(t,output)
 if t.dtype==torch.int32:v.dtype=3
 return v

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_query_abi_atomic_domains_aliases_scratch_and_overflow(device):
 import gffx,gffx.torch.image
 lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
 lib.gffx_shading_loss_query.argtypes=[C.POINTER(View)]*7+[C.POINTER(Context)]+[C.POINTER(View)]*2+[C.POINTER(Buffer),P]
 lib.gffx_shading_loss_query_backward.argtypes=[C.POINTER(View)]*5+[C.POINTER(Context)]+[C.POINTER(View)]*2+[C.POINTER(Buffer),P]
 lib.gffx_shading_loss_query_workspace.argtypes=[L,L,L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
 ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if device=='cuda' else 1
 stream=torch.cuda.Stream() if device=='cuda' else None;ctx.stream=stream.cuda_stream if stream else None
 def t(x):return torch.tensor(x,dtype=torch.float64,device=device)
 cn=t([1.]);logs=t([0.]);weights=t([]).reshape(0,4);curve=t([[.8,.3,.9,0.],[.7,.2,.8,0.]]);partial=t([[.1,.2,.3,0.],[.2,.3,.4,0.]])
 cosine=t([0.,.3,1.]);owners=torch.tensor([0,0,1],dtype=torch.int32,device=device)
 query=torch.full((3,6),91.,dtype=torch.float64,device=device);mean=torch.full((2,4),92.,dtype=torch.float64,device=device)
 inputs=[cn,logs,weights,curve,partial,cosine,owners];vv=[view(x) for x in inputs];yv,mv=view(query,True),view(mean,True)
 storage=torch.empty(72,dtype=torch.uint8,device=device);w=Buffer();w.size=C.sizeof(w);w.abi=65536;w.data=storage.data_ptr();w.capacity=72;w.alignment=8;w.device=ctx.device
 def sync():
  if stream:torch.cuda.synchronize()
 def f():
  sync();s=lib.gffx_shading_loss_query(*[C.byref(x) for x in vv],C.byref(ctx),C.byref(yv),C.byref(mv),C.byref(w),None);sync();return s
 for b,n,q in ((33,1,1),(1,513,1),(1,1,1057),(0,1,1),(-1,1,0)):
  size,align=Q(91),Q(92);assert lib.gffx_shading_loss_query_workspace(b,n,q,C.byref(ctx),C.byref(size),C.byref(align),None)==1;assert(size.value,align.value)==(91,92)
 for target,index,bad in ((cn,0,0.),(logs,0,float('nan')),(curve,(1,0),float('inf')),(cosine,1,-.1),(owners,1,2)):
  old=target[index].item();target[index]=bad;assert f()==1 and bool((query==91).all()) and bool((mean==92).all());target[index]=old
 w.capacity=71;assert f()==3;w.capacity=72
 saved=mv.data;mv.data=yv.data;assert f()==1;mv.data=saved
 saved=yv.data;yv.data=vv[3].data;assert f()==1;yv.data=saved
 saved=w.data;w.data=vv[3].data;assert f()==1;w.data=saved
 assert f()==0
 gradient=t([[1.,1.]]*3);mg=t([[1.,1.]]*2);gu=t([93.,93.]);gc=t([94.]*3)
 def back():
  ins=[view(x) for x in (query,mean,owners,gradient,mg)];a,b=view(gu,True),view(gc,True);sync();s=lib.gffx_shading_loss_query_backward(*[C.byref(x) for x in ins],C.byref(ctx),C.byref(a),C.byref(b),C.byref(w),None);sync();return s
 gradient[0,0]=float('nan');assert back()==1 and bool((gu==93).all()) and bool((gc==94).all());gradient.fill_(1.);assert back()==0
 query[:,2:].fill_(2.);gradient.fill_(torch.finfo(torch.float64).max);assert back()==4
