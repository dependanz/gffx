"""Device-owned opaque geometry: values, first VJP, validation and streams."""
import torch,pytest
from gffx.torch.render import opaque_quadrature

def raw(value):return value.detach().cpu().contiguous().flatten().view(torch.uint8)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_opaque_cuda_quadrature_matches_cpu_with_material_and_depth_boundaries(dtype):
    v=torch.tensor([[-.2,-.1,.1],[2.2,.1,.7],[.2,1.8,.3],[-.1,.2,.6],[2.,-.2,.2],[1.8,1.7,.4]],dtype=dtype,requires_grad=True)
    f=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.int32)
    c=torch.tensor([[[-.4],[.6],[.1]],[[-.3],[.7],[.2]]],dtype=dtype,requires_grad=True)
    ids,q=opaque_quadrature(v,f,c,1,2);g=torch.arange(q.numel(),dtype=dtype).reshape(q.shape)%17/17
    a=torch.autograd.grad((q*g).sum(),(v,c))
    vv=v.detach().cuda().requires_grad_();cc=c.detach().cuda().requires_grad_()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ii,qq=opaque_quadrature(vv,f.cuda(),cc,1,2);bb=torch.autograd.grad((qq*g.cuda()).sum(),(vv,cc))
    stream.synchronize()
    assert ii.device.type=='cuda' and qq.device.type=='cuda'
    assert torch.equal(raw(ids),raw(ii)) and torch.equal(raw(q),raw(qq))
    for x,y in zip(a,bb):assert torch.equal(raw(x),raw(y))

@pytest.mark.parametrize('bad',['index','nan','cuts'])
def test_opaque_cuda_rejects_invalid_device_data(bad):
    v=torch.tensor([[0.,0.,.1],[1.,0.,.2],[0.,1.,.3]],dtype=torch.float64,device='cuda')
    f=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda');c=torch.empty((1,3,0),dtype=v.dtype,device=v.device)
    if bad=='index':f[0,2]=99
    elif bad=='nan':v[0,0]=float('nan')
    else:c=torch.full((1,3,1),float('nan'),dtype=v.dtype,device=v.device)
    with pytest.raises(ValueError):opaque_quadrature(v,f,c,1,1)


@pytest.mark.parametrize('size,cap',[(1,1),(2,1),(2,2)])
def test_opaque_cuda_scene_keeps_device_ownership_and_matches_independent_cpu_anchor(size,cap,record_property):
    import importlib.util
    from pathlib import Path
    from gffx.torch import render
    spec=importlib.util.spec_from_file_location('oc_fixture',Path(__file__).with_name('test_opaque_nonlinear_scene.py'));fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from torch.utils._python_dispatch import TorchDispatchMode
    class NoHostGeometry(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {}
            if func==torch.ops.aten._to_copy.default and args[0].device.type=='cuda' and torch.device(kwargs.get('device',args[0].device)).type=='cpu':raise AssertionError('hidden device-to-host tensor transfer')
            return func(*args,**kwargs)
    values=[];grads=[];peak=0
    for device in ('cpu','cuda'):
        model,cam,v,f,b=fixture.fixture(texture=True,rough=True,device=device)
        if size>1:
            from gffx.torch.camera import Camera
            t=v.new_tensor;cam=Camera.from_intrinsics(t(128.*size),t(128.*size),t(size/2-.03),t(size/2-.02),size,size,.1,10.)
        v.requires_grad_()
        if device=='cuda':torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        with NoHostGeometry():
            o=render.render(model,cam,coverage='opaque',cull_mode=render.CULL_NONE,opaque_tile_pixels=cap)
            gradient=torch.autograd.grad(o['colour'].sum(),v)[0]
        assert all(a.device.type==device for a in o.values())
        values.append(o);grads.append(gradient)
        if device=='cuda':
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-baseline;assert peak<=67108864
    record_property('cuda_incremental_peak_bytes',peak)
    for k in values[0]:assert torch.equal(raw(values[0][k]),raw(values[1][k]))
    assert torch.equal(raw(grads[0]),raw(grads[1]))


def test_opaque_cuda_preallocated_scratch_alias_and_validation_contract():
    import ctypes as C
    from tools.animation_cost import Context,Buffer,View,ABI,tensor_view,bind_native
    v=torch.tensor([[.1,.1,.2],[.8,.1,.2],[.1,.8,.2]],device='cuda',dtype=torch.float64)
    f=torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32);cuts=torch.tensor([[[-.2],[.5],[-.2]]],device='cuda',dtype=v.dtype)
    expected_id,expected_q=opaque_quadrature(v,f,cuts,1,1) # Explicit adapter initializes the provider.
    lib,_=bind_native();ctx=Context();ctx.struct_size=C.sizeof(ctx);ctx.abi_version=ABI;ctx.device_type=2;ctx.device_index=torch.cuda.current_device();ctx.stream=torch.cuda.current_stream().cuda_stream
    query=lib.gffx_render_opaque_quadrature_workspace;query.argtypes=[C.c_uint32,C.POINTER(Context),C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.c_void_p];query.restype=C.c_uint32
    size=C.c_uint64();alignment=C.c_uint64();assert query(2,C.byref(ctx),C.byref(size),C.byref(alignment),None)==0;assert size.value==1048576 and alignment.value==8
    storage=torch.empty(size.value,device='cuda',dtype=torch.uint8);work=Buffer();work.struct_size=C.sizeof(work);work.abi_version=ABI;work.data=storage.data_ptr();work.capacity_bytes=size.value;work.alignment=alignment.value;work.device_type=2;work.device_index=ctx.device_index
    ids=torch.full_like(expected_id,91);q=torch.full_like(expected_q,91);g=torch.zeros_like(q);g[...,3]=1;gv=torch.empty_like(v);gc=torch.empty_like(cuts)
    vv,ff,cc,ii,qq,gg,vgrad,cgrad=[tensor_view(t,output=k in (3,4,6,7)) for k,t in enumerate((v,f,cuts,ids,q,g,gv,gc))];ff.dtype=ii.dtype=3
    forward=lib.gffx_render_opaque_quadrature;forward.argtypes=[C.POINTER(View)]*3+[C.c_int64,C.c_int64,C.c_double,C.c_double,C.POINTER(Context),C.POINTER(View),C.POINTER(View),C.POINTER(Buffer),C.c_void_p];forward.restype=C.c_uint32
    backward=lib.gffx_render_opaque_quadrature_backward;backward.argtypes=[C.POINTER(View)]*3+[C.c_int64,C.c_int64,C.c_double,C.c_double,C.POINTER(View),C.POINTER(Context),C.POINTER(View),C.POINTER(View),C.POINTER(Buffer),C.c_void_p];backward.restype=C.c_uint32
    def call():return forward(C.byref(vv),C.byref(ff),C.byref(cc),1,1,0.,0.,C.byref(ctx),C.byref(ii),C.byref(qq),C.byref(work),None)
    derivatives=[]
    for pattern in (0,165,255):
        storage.fill_(pattern);assert call()==0;assert torch.equal(raw(q),raw(expected_q)) and torch.equal(raw(ids),raw(expected_id))
        assert backward(C.byref(vv),C.byref(ff),C.byref(cc),1,1,0.,0.,C.byref(gg),C.byref(ctx),C.byref(vgrad),C.byref(cgrad),C.byref(work),None)==0
        derivatives.append((raw(gv).clone(),raw(gc).clone()))
    assert all(torch.equal(a,b) for row in derivatives[1:] for a,b in zip(derivatives[0],row))
    torch.testing.assert_close(gv,torch.tensor([[-.35,-.35,0],[.35,0,0],[0,.35,0]],device='cuda',dtype=v.dtype),atol=1e-12,rtol=0)
    q.fill_(91);ids.fill_(91)
    work.capacity_bytes=size.value-1;assert call()==3;work.capacity_bytes=size.value
    work.data=v.data_ptr();assert call()==1;work.data=storage.data_ptr()
    qq.data=v.data_ptr();assert call()==1;qq.data=q.data_ptr()
    qq.flags=1;assert call()==1;qq.flags=2
    vv.device_type=1;assert call()==1;vv.device_type=2
    f[0,2]=99;assert call()==1;f[0,2]=2
    assert bool((q==91).all()) and bool((ids==91).all())
