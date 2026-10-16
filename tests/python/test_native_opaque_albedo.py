"""Native sample albedo VJP retains full derivatives and original reduction order."""
import os,pytest,torch
from dataclasses import replace
from test_native_opaque_samples import sample,bits
from test_opaque_full_original import original


@pytest.fixture(scope='module')
def context():
 from gffx.torch import render,_opaque_scene as scene
 s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));captured=[];old=scene._sample_integral
 def capture(*a,**kw):
  out=old(*a,**kw)
  if not captured:captured.append(a)
  return out
 scene._sample_integral=capture
 try:
  with torch.no_grad():render.opaque_window(s,c,(10,59,1,1),cull_mode=2)
 finally:scene._sample_integral=old
 return captured[0]


def eager(context,owners,q,cot):
 from gffx.torch._opaque_scene import _sample_integral_eager
 packed,camera,clip,normals,oldowners,oldq,workload=context
 a=packed.materials[0].parameters['albedo'].detach().clone().requires_grad_()
 packed=replace(packed,materials=(replace(packed.materials[0],parameters=dict(packed.materials[0].parameters,albedo=a)),))
 workload=dict(workload,principled_point_block_limit=4,principled_group_point_limit=128)
 with torch.enable_grad():
  rgb,mask,weighted=_sample_integral_eager(packed,camera,clip,normals,owners.reshape_as(oldowners),q.reshape_as(oldq),workload,contributions=True)
  gradient=torch.autograd.grad(rgb,a,cot.reshape_as(rgb))[0] if rgb.requires_grad else torch.zeros_like(a)
 return weighted.reshape(1,800,4),torch.cat((rgb.reshape(1,3),mask.reshape(1,1)),-1),gradient.reshape(1,3)


@pytest.mark.parametrize('points',[0,1,2,3,4,5,127,128,129,800])
@pytest.mark.parametrize('cotangent',[(.2,-.3,.7),(-0.,5e-324,-1e-280)])
def test_native_sample_albedo_exact_eager_order(sample,context,points,cotangent):
 args,_=sample;args=list(args);args[0]=args[0].clone();valid=torch.nonzero(args[0][0]>=0).flatten();keep=valid[:points]
 args[0].fill_(-1);args[0][0,keep]=sample[0][0][0,keep]
 g=args[1].new_tensor([cotangent]);expected=eager(context,args[0],args[1],g)
 actual=torch.ops.gffx.shading_opaque_albedo(*args,g)
 assert all(bits(a)==bits(b) for a,b in zip(actual,expected))


@pytest.mark.parametrize('pixels',[1,2,16])
def test_native_sample_albedo_pixel_partition(sample,pixels):
 args,_=sample;g=args[1].new_tensor([[.2,-.3,.7]]);one=torch.ops.gffx.shading_opaque_albedo(*args,g)
 args=list(args);args[0]=args[0].repeat(pixels,1);args[1]=args[1].repeat(pixels,1,1)
 actual=torch.ops.gffx.shading_opaque_albedo(*args,g.repeat(pixels,1))
 for rows,expected in zip(actual,one):assert all(bits(row)==bits(expected[0]) for row in rows)


@pytest.mark.parametrize('field',list(range(1,19)))
def test_native_sample_albedo_nonfinite_refusal(sample,field):
 args,_=sample;args=list(args)+[args[1].new_ones((1,3))]
 if field==3:args[field]=args[field].clone();args[field].reshape(-1)[-1]=-1
 else:args[field]=args[field].clone();args[field].reshape(-1)[-1]=float('nan')
 with pytest.raises(RuntimeError):torch.ops.gffx.shading_opaque_albedo(*args)


@pytest.mark.parametrize('kind',['count','shape','stride','negative_albedo','zero_direction','zero_width'])
def test_native_sample_albedo_domain_refusal(sample,kind):
 args,_=sample;args=list(args);g=args[1].new_ones((1,3))
 if kind=='count':args[0]=args[0].repeat(17,1);args[1]=args[1].repeat(17,1,1);g=g.repeat(17,1)
 if kind=='shape':g=g[:,:2].contiguous()
 if kind=='stride':g=g.repeat(1,2)[:,::2]
 if kind=='negative_albedo':args[13]=-torch.ones_like(args[13])
 if kind=='zero_direction':args[16]=torch.zeros_like(args[16])
 if kind=='zero_width':args[12]=torch.zeros_like(args[12])
 with pytest.raises(RuntimeError):torch.ops.gffx.shading_opaque_albedo(*args,g)


def test_native_albedo_raw_atomic_domains_workspace_aliases_stream(sample):
    import ctypes as C,gffx
    from pathlib import Path
    from test_gaussian_abi_failures import View,Context,Buffer,view,Q,L,P
    args,_=sample;args=[a.clone() for a in args]+[args[1].new_ones((1,3))];dev=args[1].device;cuda=dev.type=='cuda'
    expected=torch.ops.gffx.shading_opaque_albedo(*args)
    lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
    fn=lib.gffx_shading_opaque_albedo;fn.argtypes=[C.POINTER(View)]*19+[C.POINTER(Context)]+[C.POINTER(View)]*3+[C.POINTER(Buffer),P]
    query=lib.gffx_shading_opaque_albedo_workspace;query.argtypes=[L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
    stream=torch.cuda.Stream() if cuda else None;ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1;ctx.stream=stream.cuda_stream if cuda else None
    size,alignment=Q(),Q();assert query(1,C.byref(ctx),C.byref(size),C.byref(alignment),None)==0 and size.value==44808
    weighted=torch.full_like(expected[0],91.);rgba=torch.full_like(expected[1],92.);gradient=torch.full_like(expected[2],93.);storage=torch.empty(size.value,dtype=torch.uint8,device=dev)
    ws=Buffer();ws.size=C.sizeof(ws);ws.abi=65536;ws.device=ctx.device;ws.data=storage.data_ptr();ws.capacity=size.value;ws.alignment=8
    views=[view(a) for a in args];wv,yv,gv=view(weighted,True),view(rgba,True),view(gradient,True)
    views[0].dtype=3;views[3].dtype=3
    def call():
        if cuda:torch.cuda.synchronize()
        result=fn(*(C.byref(v) for v in views),C.byref(ctx),C.byref(wv),C.byref(yv),C.byref(gv),C.byref(ws),None)
        if cuda:torch.cuda.synchronize()
        return result
    def unchanged():assert bool((weighted==91).all()) and bool((rgba==92).all()) and bool((gradient==93).all())
    for i in (1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18):
        old=args[i].reshape(-1)[-1].clone();args[i].reshape(-1)[-1]=float('nan');assert call()!=0;unchanged();args[i].reshape(-1)[-1]=old
    ws.capacity-=1;assert call()!=0;unchanged();ws.capacity+=1
    old=wv.data;wv.data=views[1].data;assert call()!=0;unchanged();wv.data=old
    old=yv.data;yv.data=wv.data;assert call()!=0;unchanged();yv.data=old
    old=gv.data;gv.data=views[18].data;assert call()!=0;unchanged();gv.data=old
    ws.data=views[1].data;assert call()!=0;unchanged();ws.data=storage.data_ptr()
    ctx.reserved[0]=1;assert call()!=0;unchanged();ctx.reserved[0]=0
    for p in (0,17,-1):
        size.value=71;alignment.value=72;assert query(p,C.byref(ctx),C.byref(size),C.byref(alignment),None)!=0
        assert (size.value,alignment.value)==(71,72)
    assert call()==0
    assert all(bits(a)==bits(b) for a,b in zip((weighted,rgba,gradient),expected))


@pytest.mark.parametrize('weight',[0.,-0.,5e-324,-5e-324])
def test_native_albedo_retains_tiny_weights(sample,context,weight):
 args,_=sample;args=list(args);args[1]=args[1].clone();args[1][...,3]=weight;g=args[1].new_tensor([[.2,-.3,.7]])
 expected=eager(context,args[0],args[1],g);actual=torch.ops.gffx.shading_opaque_albedo(*args,g)
 assert all(bits(a)==bits(b) for a,b in zip(actual,expected))


def test_native_albedo_old_provider_query_refuses():
 if os.environ.get('GFFX_OLD_PROVIDER_CONTROL')!='1':pytest.skip('Preserved provider requires a fresh process')
 import ctypes as C,gffx
 from pathlib import Path
 from test_gaussian_abi_failures import Context,Q,L,P
 lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'));fn=lib.gffx_shading_opaque_albedo_workspace
 fn.argtypes=[L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
 ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2;size,alignment=Q(71),Q(72)
 assert fn(1,C.byref(ctx),C.byref(size),C.byref(alignment),None)!=0 and (size.value,alignment.value)==(71,72)
