"""Real installed C ABI sentinel/alias/domain tests on CPU and the caller CUDA stream."""
import ctypes as C
import pytest
import torch
from pathlib import Path
U=C.c_uint32;L=C.c_int64;Q=C.c_uint64;P=C.c_void_p
class View(C.Structure):
    _fields_=[('size',U),('abi',U),('data',P),('offset',Q),('shape',C.POINTER(L)),('strides',C.POINTER(L)),('rank',U),('dtype',U),('device',U),('index',C.c_int32),('flags',U),('reserved0',U),('reserved',Q*4)]
class Context(C.Structure):
    _fields_=[('size',U),('abi',U),('device',U),('index',C.c_int32),('stream',P),('flags',U),('reserved0',U),('reserved',Q*4)]
class Buffer(C.Structure):
    _fields_=[('size',U),('abi',U),('data',P),('capacity',Q),('alignment',Q),('device',U),('index',C.c_int32),('flags',U),('reserved0',U),('reserved',Q*2)]
def view(t,output=False):
    v=View();v.size=C.sizeof(v);v.abi=65536;v.data=t.data_ptr();v.rank=t.ndim;v.dtype=1 if t.dtype==torch.float32 else 2;v.device=2 if t.is_cuda else 1;v.flags=2 if output else 1
    v._shape=(L*t.ndim)(*t.shape);v._strides=(L*t.ndim)(*t.stride());v.shape=v._shape if t.ndim else None;v.strides=v._strides if t.ndim else None
    return v

def library():
    import gffx, gffx.torch.image
    lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
    lib.gffx_image_gaussian_weights.argtypes=[C.POINTER(View),U,C.POINTER(Context),C.POINTER(View),C.POINTER(Buffer),P]
    lib.gffx_image_gaussian_weights_backward.argtypes=[C.POINTER(View),C.POINTER(View),U,C.POINTER(Context),C.POINTER(View),C.POINTER(Buffer),P]
    lib.gffx_image_gaussian_weights_workspace.argtypes=[L,U,U,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
    return lib

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_gaussian_c_abi_failure_atomicity_aliases_and_workspace(device,dtype):
    lib=library();cuda=device=='cuda'
    if cuda and not torch.cuda.is_available():pytest.fail('CUDA required')
    stream=torch.cuda.Stream() if cuda else None
    context=Context();context.size=C.sizeof(context);context.abi=65536;context.device=2 if cuda else 1;context.stream=stream.cuda_stream if cuda else None
    storage=torch.empty(8,dtype=torch.uint8,device=device);workspace=Buffer();workspace.size=C.sizeof(workspace);workspace.abi=65536;workspace.data=storage.data_ptr();workspace.capacity=8;workspace.alignment=8;workspace.device=context.device
    scratch=C.byref(workspace) if cuda else None
    def sync():
        if cuda:torch.cuda.synchronize()
    sigma=torch.tensor([.83,.91],dtype=dtype,device=device);out=torch.full((2,5),91.,dtype=dtype,device=device);cot=torch.ones_like(out);grad=torch.full_like(sigma,92.)
    x,y,g,z=view(sigma),view(out,True),view(cot),view(grad,True)
    def forward():sync();code=lib.gffx_image_gaussian_weights(C.byref(x),2,C.byref(context),C.byref(y),scratch,None);sync();return code
    def backward():sync();code=lib.gffx_image_gaussian_weights_backward(C.byref(x),C.byref(g),2,C.byref(context),C.byref(z),scratch,None);sync();return code
    n,a=Q(91),Q(92);assert lib.gffx_image_gaussian_weights_workspace(2,x.dtype,2,C.byref(context),C.byref(n),C.byref(a),None)==0
    assert (n.value,a.value)==((8,8) if cuda else (0,1))
    for bad in (float('nan'),float('inf'),.01,33.):
        sigma[1]=bad;assert forward()==1 and bool((out==91).all())
        assert backward()==1 and bool((grad==92).all())
    sigma[1]=.91;cot[-1,-1]=float('nan');assert backward()==1 and bool((grad==92).all());cot.fill_(1.)
    y.dtype=2 if x.dtype==1 else 1;assert forward()==1 and bool((out==91).all());y.dtype=x.dtype
    y._strides[0]=6;assert forward()==2 and bool((out==91).all());y._strides[0]=5
    saved=y.data;y.data=x.data;assert forward()==1 and torch.equal(sigma,sigma.new_tensor([.83,.91]));y.data=saved
    saved=z.data;z.data=g.data;assert backward()==1 and bool((cot==1).all());z.data=saved
    if cuda:
        workspace.capacity=7;assert forward()==3 and bool((out==91).all());workspace.capacity=8
        workspace.data=x.data;assert forward()==1 and bool((out==91).all());workspace.data=storage.data_ptr()
    assert forward()==0 and bool(torch.isfinite(out).all())
    assert backward()==0 and bool(torch.isfinite(grad).all())
    # Radius zero still validates cotangents, accepts input-to-input aliasing and returns +0.
    one=torch.tensor(.83,dtype=dtype,device=device);positive=torch.full((),92.,dtype=dtype,device=device);vx,vg,vz=view(one),view(one.reshape(1)),view(positive,True)
    sync();assert lib.gffx_image_gaussian_weights_backward(C.byref(vx),C.byref(vg),0,C.byref(context),C.byref(vz),scratch,None)==0;sync()
    assert positive==0 and not torch.signbit(positive)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_gaussian_c_abi_typed_overflow_without_clipping(device,dtype):
    lib=library();cuda=device=='cuda';ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1
    stream=torch.cuda.Stream() if cuda else None;ctx.stream=stream.cuda_stream if cuda else None
    storage=torch.empty(8,dtype=torch.uint8,device=device);ws=Buffer();ws.size=C.sizeof(ws);ws.abi=65536;ws.data=storage.data_ptr();ws.capacity=8;ws.alignment=8;ws.device=ctx.device
    sigma=torch.tensor(.5,dtype=dtype,device=device);cot=torch.full((3,),torch.finfo(dtype).max,dtype=dtype,device=device);out=torch.tensor(92.,dtype=dtype,device=device)
    x,g,z=view(sigma),view(cot),view(out,True)
    if cuda:torch.cuda.synchronize()
    code=lib.gffx_image_gaussian_weights_backward(C.byref(x),C.byref(g),1,C.byref(ctx),C.byref(z),C.byref(ws) if cuda else None,None)
    if cuda:torch.cuda.synchronize()
    # Central derivative magnitude exceeds one: finite cotangent causes a product
    # overflow before an eventual cancellation. Output is invalid, never clipped.
    assert code==4 and sigma==.5 and bool((cot==torch.finfo(dtype).max).all())
    n,a=Q(91),Q(92)
    assert lib.gffx_image_gaussian_weights_workspace(2**63-1,x.dtype,32,C.byref(ctx),C.byref(n),C.byref(a),None)==4
    assert (n.value,a.value)==(91,92)
    if not cuda:
        # Even optional CPU scratch cannot overlap a read-only tensor.
        y=torch.full((3,),91.,dtype=dtype);vy=view(y,True);ws.data=x.data
        assert lib.gffx_image_gaussian_weights(C.byref(x),1,C.byref(ctx),C.byref(vy),C.byref(ws),None)==1
        assert bool((y==91).all()) and sigma==.5
