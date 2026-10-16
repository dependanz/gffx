"""Independent CUDA pixel lanes preserve scalar arithmetic and bounded scratch."""
import ctypes as C
import pytest,torch
from gffx.torch import render


def data(dtype=torch.float64,device='cuda'):
    v=torch.tensor([[-.2,-.1,.1],[4.2,.1,.7],[.2,4.8,.3],[-.1,.2,.6],[4.,-.2,.2],[3.8,4.7,.4]],dtype=dtype,device=device)
    f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32,device=device)
    c=torch.tensor([[[-.4],[.6],[.1]],[[-.3],[.7],[.2]]],dtype=dtype,device=device)
    return v,f,c


def bits(x):return x.detach().cpu().contiguous().numpy().tobytes()


def test_adapter_charges_and_allocates_four_disjoint_slabs():
    v,f,c=data();render.opaque_material_quadrature(v,f,c,1,1)
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    result=render.opaque_material_quadrature(v,f,c,1,16)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-baseline
    assert 4*1048576<=peak<=67108864,peak
    assert result[1].shape==(16,800,4)
    with pytest.raises(ValueError,match='budget'):render.opaque_material_quadrature(v,f,c,1,16,workspace_limit_bytes=1048576)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape',[(1,2),(1,16),(4,4)])
def test_parallel_forward_keeps_scalar_pixel_bytes_on_nondefault_stream(dtype,shape):
    v,f,c=data(dtype,'cpu');h,w=shape
    expected=render.opaque_material_quadrature(v,f,c,h,w,origin=(.125,-.25))
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):actual=render.opaque_material_quadrature(v.cuda(),f.cuda(),c.cuda(),h,w,origin=(.125,-.25))
    stream.synchronize()
    assert all(bits(a)==bits(b) for a,b in zip(expected,actual))


@pytest.mark.parametrize('slabs',[1,2,4])
def test_native_caller_owned_scratch_keeps_minimum_fallback_and_padding(slabs):
    from tools.animation_cost import Context,Buffer,View,ABI,tensor_view,bind_native
    v,f,c=data();expected=render.opaque_material_quadrature(v,f,c,4,4)
    lib,_=bind_native();ctx=Context();ctx.struct_size=C.sizeof(ctx);ctx.abi_version=ABI;ctx.device_type=2;ctx.device_index=torch.cuda.current_device();ctx.stream=torch.cuda.current_stream().cuda_stream
    storage=torch.empty(slabs*1048576,device='cuda',dtype=torch.uint8);work=Buffer();work.struct_size=C.sizeof(work);work.abi_version=ABI;work.data=storage.data_ptr();work.capacity_bytes=storage.numel();work.alignment=8;work.device_type=2;work.device_index=ctx.device_index
    ids=torch.full_like(expected[0],91);q=torch.full_like(expected[1],91)
    vv,ff,cc,ii,qq=[tensor_view(t,output=k>=3) for k,t in enumerate((v,f,c,ids,q))];ff.dtype=ii.dtype=3
    op=lib.gffx_render_opaque_material_quadrature;op.argtypes=[C.POINTER(View)]*3+[C.c_int64,C.c_int64,C.c_double,C.c_double,C.POINTER(Context),C.POINTER(View),C.POINTER(View),C.POINTER(Buffer),C.c_void_p];op.restype=C.c_uint32
    def call():return op(C.byref(vv),C.byref(ff),C.byref(cc),4,4,0.,0.,C.byref(ctx),C.byref(ii),C.byref(qq),C.byref(work),None)
    for pattern in (0,165,255):
        storage.fill_(pattern);assert call()==0;assert bits(ids)==bits(expected[0]) and bits(q)==bits(expected[1])
    q.fill_(91);ids.fill_(91);f[-1,-1]=len(v)
    assert call()==1 and bool((q==91).all()) and bool((ids==91).all())


def test_tight_public_budget_retains_serial_forward_and_first_vjp():
    values=[]
    for device in ('cpu','cuda'):
        v,f,c=data(device=device);v.requires_grad_();c.requires_grad_()
        owners,q=render.opaque_quadrature(v,f,c,1,2,workspace_limit_bytes=2*1048576)
        g=(torch.arange(q.numel(),dtype=q.dtype).reshape(q.shape)%17/17).to(device)
        gradients=torch.autograd.grad((q*g).sum(),(v,c))
        values.append([bits(x) for x in (owners,q,*gradients)])
    assert values[0]==values[1]


def test_parallel_curved_pixels_and_ordered_first_vjp_match_cpu():
    rows=[]
    for device in ('cpu','cuda'):
        vertices=[[x+.125,.125,0.] for x in range(4) for _ in range(3)]
        for x in range(4):vertices[3*x+1][0]=x+.75;vertices[3*x+2][1]=.75
        v=torch.tensor(vertices,device=device,dtype=torch.float64,requires_grad=True)
        f=torch.arange(12,device=device,dtype=torch.int32).reshape(4,3)
        c=v.new_tensor([[-.25,.75,.65,.5,.5,1.5]]).repeat(4,1).requires_grad_()
        owners,q=render.opaque_quadrature(v,f,v.new_empty((4,3,0)),1,4,curves=c)
        g=torch.arange(q.numel(),device=device,dtype=q.dtype).reshape(q.shape)/4096
        derivatives=torch.autograd.grad(q,(v,c),g)
        rows.append([bits(x) for x in (owners,q,*derivatives)])
    assert rows[0]==rows[1]


def test_successful_neighbour_lanes_cannot_erase_capacity_failure():
    v=torch.tensor([[.125,.125,0.],[.75,.125,0.],[.125,.75,0.]],device='cuda',dtype=torch.float64)
    f=torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32).repeat(33,1)
    with pytest.raises(OverflowError):render.opaque_material_quadrature(v,f,v.new_empty((33,3,0)),1,16)
