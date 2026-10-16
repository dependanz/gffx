"""Bounded native delta lighting must match the independent eager value and BRDF VJP."""
import os,pytest,torch
import gffx.torch

def bits(x):return x.detach().cpu().contiguous().numpy().tobytes()

def inputs(p,q):
 dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');t=lambda x:torch.tensor(x,dtype=torch.float64,device=dev)
 n=t([[.2,-.3,1.],[-0.,0.,-1.],[1e-280,2e-280,-1e-280]]).repeat((p+2)//3,1)[:p].contiguous()
 d=t([[.1,.2,1.],[1.,0.,0.],[-.1,-.2,-1.]]).repeat((q+2)//3,1)[:q].expand(p,q,3).contiguous()
 b=t([.2,.4,.7]).expand(p,q,3).contiguous();i=t([.7,.3,1.]).expand(p,q,3).contiguous();g=t([.2,-.3,1e-280]).expand(p,3).contiguous()
 return tuple(x.clone(memory_format=torch.contiguous_format) for x in (n,b,d,i,g))

def eager(n,b,d,i,g):
 from gffx.torch.light import integrate
 with torch.enable_grad():
  leaf=b.detach().requires_grad_();out=integrate(n,leaf,dict(measure='delta',direction=d,irradiance=i,valid=torch.ones(d.shape[:-1],dtype=torch.bool,device=d.device)))
  grad,=torch.autograd.grad(out,leaf,g)
 return out,grad

@pytest.mark.parametrize('p,q',[(1,1),(3,3),(128,8),(0,2),(3,0)])
def test_delta_brdf_native_preserves_values_and_first_vjp(p,q):
 args=inputs(p,q);expected=eager(*args);actual=torch.ops.gffx.shading_delta_brdf(*args)
 assert all(bits(a)==bits(b) for a,b in zip(actual,expected))

def test_delta_brdf_native_retains_subnormal_and_signed_zero_terms():
 args=list(inputs(3,3));args[1][0,0,0]=-0.;args[3][0,0,1]=-0.;args[4][1,0]=-0.;args[4][0,2]=float.fromhex('0x0.0000000000001p-1022')
 expected=eager(*args);actual=torch.ops.gffx.shading_delta_brdf(*args)
 assert all(bits(a)==bits(b) for a,b in zip(actual,expected))

@pytest.mark.parametrize('bad',['normal','brdf','direction','irradiance','cotangent','zero_normal','zero_direction','negative_brdf','negative_irradiance','points','samples','dtype','shape','stride'])
def test_delta_brdf_native_refuses_invalid_inputs(bad):
 args=list(inputs(3,3));keys=['normal','brdf','direction','irradiance','cotangent']
 if bad in keys:args[keys.index(bad)].reshape(-1)[-1]=float('nan')
 elif bad.startswith('zero'):args[0 if bad=='zero_normal' else 2].zero_()
 elif bad.startswith('negative'):args[1 if bad=='negative_brdf' else 3].fill_(-1)
 elif bad=='points':args=list(inputs(129,1))
 elif bad=='samples':args=list(inputs(1,9))
 elif bad=='dtype':args[1]=args[1].float()
 elif bad=='shape':args[4]=args[4][:,:2].contiguous()
 elif bad=='stride':args[0]=args[0].T
 with pytest.raises(RuntimeError):torch.ops.gffx.shading_delta_brdf(*args)

def test_delta_brdf_raw_atomic_domains_aliases_workspace_and_stream():
 import ctypes as C,gffx
 from pathlib import Path
 from test_gaussian_abi_failures import View,Context,Buffer,view,Q,L,P
 dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');cuda=dev=='cuda';args=list(inputs(3,3))
 torch.ops.gffx.shading_delta_brdf(*args)
 lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
 fn=lib.gffx_shading_delta_brdf;fn.argtypes=[C.POINTER(View)]*5+[C.POINTER(Context)]+[C.POINTER(View)]*2+[C.POINTER(Buffer),P]
 query=lib.gffx_shading_delta_brdf_workspace;query.argtypes=[L,L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
 stream=torch.cuda.Stream() if cuda else None;ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1;ctx.stream=stream.cuda_stream if cuda else None
 value=torch.full_like(args[0],91.);grad=torch.full_like(args[1],92.);storage=torch.empty(8,dtype=torch.uint8,device=dev)
 ws=Buffer();ws.size=C.sizeof(ws);ws.abi=65536;ws.device=ctx.device;ws.data=storage.data_ptr();ws.capacity=8;ws.alignment=8
 views=[view(x) for x in args];vv,gv=view(value,True),view(grad,True)
 def sync():
  if cuda:torch.cuda.synchronize()
 def call():
  sync();result=fn(*(C.byref(v) for v in views),C.byref(ctx),C.byref(vv),C.byref(gv),C.byref(ws),None);sync();return result
 def unchanged():assert bool((value==91).all()) and bool((grad==92).all())
 n,a=Q(71),Q(72);assert query(128,8,C.byref(ctx),C.byref(n),C.byref(a),None)==0;assert(n.value,a.value)==(8,8)
 for points,samples in ((129,8),(128,9),(-1,1),(1,-1)):
  n.value=71;a.value=72;assert query(points,samples,C.byref(ctx),C.byref(n),C.byref(a),None)!=0;assert(n.value,a.value)==(71,72)
 for index in range(5):
  old=args[index].reshape(-1)[-1].clone();args[index].reshape(-1)[-1]=float('nan');assert call()!=0;unchanged();args[index].reshape(-1)[-1]=old
 ws.capacity=7;assert call()!=0;unchanged();ws.capacity=8
 old=vv.data;vv.data=views[0].data;assert call()!=0;unchanged();vv.data=old
 old=gv.data;gv.data=vv.data;assert call()!=0;unchanged();gv.data=old
 ws.data=views[0].data;assert call()!=0;unchanged();ws.data=storage.data_ptr()
 ctx.reserved[0]=1;assert call()!=0;unchanged();ctx.reserved[0]=0
 assert call()==0;expected=eager(*args);assert bits(value)==bits(expected[0]) and bits(grad)==bits(expected[1])

def test_delta_brdf_old_provider_refuses_additive_operation():
 if os.environ.get('GFFX_OLD_PROVIDER_CONTROL')!='1':pytest.skip('Preserved provider requires separate fresh process')
 with pytest.raises(RuntimeError,match='provider lacks delta BRDF'):torch.ops.gffx.shading_delta_brdf(*inputs(1,1))
