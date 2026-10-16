"""Exact six-step normalization arithmetic; not a general square-root claim."""
import os,pytest,torch
import gffx.torch

def reference(x):
    y=torch.full_like(x,1.5)
    for _ in range(6):y=(y+x/y)*.5
    return y

def bits(x):return x.detach().cpu().contiguous().numpy().tobytes()

@pytest.mark.parametrize('n',[0,1,129,4097])
def test_unit_newton_matches_each_rounded_step(n):
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');x=torch.linspace(.5,4,n,dtype=torch.float64,device=dev)
    y=torch.ops.gffx.shading_unit_newton(x)
    assert bits(y)==bits(reference(x))

def test_unit_newton_keeps_endpoint_neighbours_and_first_vjp():
    from gffx.torch._reference_channels import sqrt_unit
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');base=torch.tensor([1.,1.125,1.73,2.3,3.],dtype=torch.float64,device=dev)
    x=torch.cat([base,torch.nextafter(base,base.new_tensor(0.)),torch.nextafter(base,base.new_tensor(4.))]).requires_grad_();cot=torch.arange(len(x),device=dev,dtype=x.dtype)*.125-.5
    y=sqrt_unit(x);g,=torch.autograd.grad(y,x,cot)
    assert bits(y)==bits(reference(x.detach()))
    assert bits(g)==bits(cot/(2*reference(x.detach())))
    assert bits(torch.ops.gffx.shading_unit_newton(x.detach()))==bits(y)

def test_unit_newton_refuses_invalid_inputs():
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');x=torch.ones(10,dtype=torch.float64,device=dev)
    for v in (x.float(),x.reshape(2,5),x[::2],x.new_empty(1048577)):
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_unit_newton(v)
    for value in (0.,-1.,float('nan'),float('inf'),4.00001):
        v=x.clone();v[-1]=value
        with pytest.raises(RuntimeError):torch.ops.gffx.shading_unit_newton(v)

def test_unit_newton_old_provider_refuses_additive_operation():
    if os.environ.get('GFFX_OLD_PROVIDER_CONTROL')!='1':pytest.skip('Preserved provider requires separate fresh process')
    with pytest.raises(RuntimeError,match='provider lacks unit Newton'):
        torch.ops.gffx.shading_unit_newton(torch.ones(1,dtype=torch.float64,device='cuda'))


def test_unit_newton_raw_failure_atomicity_aliases_workspace_and_stream():
    import ctypes as C,gffx
    from test_gaussian_abi_failures import View,Context,Buffer,view,Q,L,P
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');cuda=dev=='cuda'
    # Initialize the selected provider before querying its appended operation.
    torch.ops.gffx.shading_unit_newton(torch.ones(1,dtype=torch.float64,device=dev))
    lib=C.CDLL(str(__import__('pathlib').Path(gffx.__file__).parent/'gffx_core.dll'))
    fn=lib.gffx_shading_unit_newton;fn.argtypes=[C.POINTER(View),C.POINTER(Context),C.POINTER(View),C.POINTER(Buffer),P]
    query=lib.gffx_shading_unit_newton_workspace;query.argtypes=[L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
    stream=torch.cuda.Stream() if cuda else None;ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1;ctx.stream=stream.cuda_stream if cuda else None
    x=torch.ones(129,dtype=torch.float64,device=dev);y=torch.full_like(x,91.);storage=torch.empty(8,dtype=torch.uint8,device=dev)
    ws=Buffer();ws.size=C.sizeof(ws);ws.abi=65536;ws.device=ctx.device;ws.data=storage.data_ptr();ws.capacity=8;ws.alignment=8
    xv,yv=view(x),view(y,True)
    def sync():
        if cuda:torch.cuda.synchronize()
    def call():
        sync();result=fn(C.byref(xv),C.byref(ctx),C.byref(yv),C.byref(ws),None);sync();return result
    n,a=Q(91),Q(92);assert query(129,C.byref(ctx),C.byref(n),C.byref(a),None)==0;assert(n.value,a.value)==(8,8)
    for count in (-1,1048577):
        n.value=91;a.value=92;assert query(count,C.byref(ctx),C.byref(n),C.byref(a),None)!=0;assert(n.value,a.value)==(91,92)
    for bad in (0.,float('nan'),float('inf'),4.01):
        x[-1]=bad;assert call()!=0;assert bool((y==91).all())
    x[-1]=1.;saved=yv.data;yv.data=xv.data;assert call()!=0;assert bool((x==1).all());yv.data=saved
    ws.capacity=7;assert call()!=0;assert bool((y==91).all());ws.capacity=8
    ws.data=xv.data;assert call()!=0;assert bool((y==91).all());ws.data=storage.data_ptr()
    ctx.reserved[0]=1;assert call()!=0;assert bool((y==91).all());ctx.reserved[0]=0
    assert call()==0;assert bits(y)==bits(reference(x))
