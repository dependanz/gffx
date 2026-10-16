"""Exact forward normalization fusion with unchanged differentiated fallback."""
import os,pytest,torch
import gffx.torch
from test_native_unit_newton import bits,reference as root

def reference(x):
    scale=x.abs().amax(-1,keepdim=True);v=x/scale
    square=v.square();s=(square[...,0]+square[...,1])+square[...,2]
    return v/root(s)[...,None]

def values(device):
    small=float.fromhex('0x0.0000000000001p-1022');large=torch.finfo(torch.float64).max
    return torch.tensor([[1.,2.,3.],[-0.,-1.,0.],[small,2*small,-small],[large,-large,large],[-1e-300,1e-300,0.]],dtype=torch.float64,device=device)

@pytest.mark.parametrize('copies',[0,1,27,821])
def test_unit_vectors_preserve_extremes_signed_zero_and_rounding(copies):
    x=values(os.environ.get('GFFX_KNOT_DEVICE','cpu')).repeat(copies,1)
    y=torch.ops.gffx.shading_unit_vectors(x)
    assert bits(y)==bits(reference(x))

@pytest.mark.parametrize('module',['light','_deferred'])
def test_unit_vectors_forward_uses_fused_path_with_exact_bytes(module,monkeypatch):
    import importlib
    m=importlib.import_module('gffx.torch.'+module);x=values(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    def forbidden(*a,**kw):raise AssertionError('unfused per-step square-root dispatch')
    monkeypatch.setattr(m,'sqrt_unit',forbidden)
    with torch.no_grad():y=m._unit(x,'test vectors')
    assert bits(y)==bits(reference(x))

def test_unit_vectors_differentiated_path_remains_original(monkeypatch):
    from gffx.torch import light
    x=values(os.environ.get('GFFX_KNOT_DEVICE','cpu'))[:1].clone().requires_grad_();v=x.detach().clone().requires_grad_()
    def forbidden(*a,**kw):raise AssertionError('forward-only native path entered differentiation')
    monkeypatch.setattr(torch.ops.gffx,'shading_unit_vectors',forbidden)
    y=light._unit(x,'test vectors');expected=reference(v);cot=x.new_tensor([[.3,-.5,.7]])
    g,=torch.autograd.grad(y,x,cot)
    # First VJP of the qualified normalization uses 1/(2*y),not Newton-loop AD.
    from gffx.torch._reference_channels import sqrt_unit,divide
    z=divide(v,v.abs().amax(-1,keepdim=True));s=z.square();old=divide(z,sqrt_unit((s[...,0]+s[...,1])+s[...,2])[...,None])
    expected_grad,=torch.autograd.grad(old,v,cot)
    assert bits(y)==bits(expected) and bits(g)==bits(expected_grad)

def test_unit_vectors_refuse_bad_shapes_domains_and_types():
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');x=values(dev)
    for v in (x.float(),x[:,0],x[:,:2].contiguous(),x[::2],x.new_empty(1048577,3)):
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_unit_vectors(v)
    for bad in (float('nan'),float('inf')):
        v=x.clone();v[-1,-1]=bad
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_unit_vectors(v)
    with pytest.raises(RuntimeError):torch.ops.gffx.shading_unit_vectors(x.new_zeros(1,3))

def test_unit_vectors_old_provider_refuses_additive_operation():
    if os.environ.get('GFFX_OLD_PROVIDER_CONTROL')!='1':pytest.skip('Preserved provider requires separate fresh process')
    with pytest.raises(RuntimeError,match='provider lacks unit vectors'):
        torch.ops.gffx.shading_unit_vectors(torch.ones((1,3),dtype=torch.float64,device='cuda'))

def test_unit_vectors_raw_failure_atomicity_aliases_workspace_and_stream():
    import ctypes as C,gffx
    from test_gaussian_abi_failures import View,Context,Buffer,view,Q,L,P
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');cuda=dev=='cuda'
    # Initialize the selected provider before querying its appended operation.
    torch.ops.gffx.shading_unit_vectors(torch.ones((1,3),dtype=torch.float64,device=dev))
    lib=C.CDLL(str(__import__('pathlib').Path(gffx.__file__).parent/'gffx_core.dll'))
    fn=lib.gffx_shading_unit_vectors;fn.argtypes=[C.POINTER(View),C.POINTER(Context),C.POINTER(View),C.POINTER(Buffer),P]
    query=lib.gffx_shading_unit_vectors_workspace;query.argtypes=[L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
    stream=torch.cuda.Stream() if cuda else None;ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1;ctx.stream=stream.cuda_stream if cuda else None
    x=torch.ones((129,3),dtype=torch.float64,device=dev);y=torch.full_like(x,91.);storage=torch.empty(8,dtype=torch.uint8,device=dev)
    ws=Buffer();ws.size=C.sizeof(ws);ws.abi=65536;ws.device=ctx.device;ws.data=storage.data_ptr();ws.capacity=8;ws.alignment=8
    xv,yv=view(x),view(y,True)
    def sync():
        if cuda:torch.cuda.synchronize()
    def call():
        sync();result=fn(C.byref(xv),C.byref(ctx),C.byref(yv),C.byref(ws),None);sync();return result
    n,a=Q(91),Q(92);assert query(129,C.byref(ctx),C.byref(n),C.byref(a),None)==0;assert(n.value,a.value)==(8,8)
    for count in (-1,1048577):
        n.value=91;a.value=92;assert query(count,C.byref(ctx),C.byref(n),C.byref(a),None)!=0;assert(n.value,a.value)==(91,92)
    for bad in (0.,float('nan'),float('inf')):
        x[-1]=bad;assert call()!=0;assert bool((y==91).all())
    x[-1]=1.;saved=yv.data;yv.data=xv.data;assert call()!=0;assert bool((x==1).all());yv.data=saved
    ws.capacity=7;assert call()!=0;assert bool((y==91).all());ws.capacity=8
    ws.data=xv.data;assert call()!=0;assert bool((y==91).all());ws.data=storage.data_ptr()
    ctx.reserved[0]=1;assert call()!=0;assert bool((y==91).all());ctx.reserved[0]=0
    assert call()==0;assert bits(y)==bits(reference(x))
