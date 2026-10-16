"""The bounded native composition must preserve the existing sample integral."""
import os,torch,pytest
from test_opaque_full_original import original

@pytest.fixture(scope='module')
def sample():
    from gffx.torch import render,_opaque_scene as scene
    from gffx.torch._principled_groups import _prepare_material_curve
    from gffx.torch._deferred import _light_packet
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));captured=[];base=scene._sample_integral
    def capture(*a,**kw):
        out=base(*a,**kw)
        if not captured:captured.append((a,out))
        return out
    scene._sample_integral=capture
    try:
        with torch.no_grad():render.opaque_window(s,c,(10,59,1,1),cull_mode=2)
    finally:scene._sample_integral=base
    a,expected=captured[0];packed,camera,clip,normals,owners,q,_=a
    p=packed.materials[0].parameters;cn,cv,dp,mn,wd=_prepare_material_curve(p['budget'],p['roughness'].reshape(1))
    packet=_light_packet(packed.lights[0],packed,0,packed.vertices[:1])
    args=[owners.reshape(-1,800),q.reshape(-1,800,4),packed.vertices,packed.faces,clip,normals,camera.pose[0,:3,3],cn[1:],p['budget']._data['_clog'],cv,dp,mn,wd,p['albedo'],p['roughness'].reshape(1),p['f0'],packet['direction'].reshape(3),packet['irradiance'].reshape(3)]
    return [x.contiguous() for x in args],expected

def bits(x):return x.detach().cpu().contiguous().numpy().tobytes()

def test_native_original_sample_composition_exact(sample):
    args,expected=sample;weighted,rgba=torch.ops.gffx.shading_opaque_samples(*args)
    assert bits(rgba[:,:3])==bits(expected[0])
    assert bits(rgba[:,3])==bits(expected[1])
    if len(expected)==3:assert bits(weighted)==bits(expected[2])

@pytest.mark.parametrize('field',[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
def test_native_sample_nonfinite_refuses(sample,field):
    args,_=sample;args=list(args);args[field]=args[field].clone();args[field].reshape(-1)[-1]=float('nan')
    with pytest.raises(RuntimeError):torch.ops.gffx.shading_opaque_samples(*args)

@pytest.mark.parametrize('field,value',[(0,-2),(0,10000),(3,-1),(3,5002),(12,0.),(13,-.1),(14,0.),(15,1.1),(17,-1.)])
def test_native_sample_domain_refuses(sample,field,value):
    args,_=sample;args=list(args);args[field]=args[field].clone();args[field].reshape(-1)[-1]=value
    with pytest.raises(RuntimeError):torch.ops.gffx.shading_opaque_samples(*args)

@pytest.mark.parametrize('pixels',[1,2,16])
def test_native_sample_pixel_partition_and_capacity(sample,pixels):
    args,_=sample;expected=torch.ops.gffx.shading_opaque_samples(*args);args=list(args)
    args[0]=args[0].repeat(pixels,1);args[1]=args[1].repeat(pixels,1,1)
    actual=torch.ops.gffx.shading_opaque_samples(*args)
    for got,old in zip(actual,expected):
        for row in got:assert bits(row)==bits(old[0])

@pytest.mark.parametrize('kind',['zero','over','dtype','strided','shape','zero_direction'])
def test_native_sample_invalid_layout_refuses(sample,kind):
    args,_=sample;args=list(args)
    if kind in ('zero','over'):
        p=0 if kind=='zero' else 17;args[0]=args[0].repeat(p,1);args[1]=args[1].repeat(p,1,1)
    elif kind=='dtype':args[1]=args[1].float()
    elif kind=='strided':args[1]=args[1].transpose(1,2)
    elif kind=='shape':args[13]=args[13][:2]
    else:args[16]=torch.zeros_like(args[16])
    with pytest.raises(RuntimeError):torch.ops.gffx.shading_opaque_samples(*args)

def test_native_sample_raw_atomic_domains_workspace_aliases_stream(sample):
    import ctypes as C,gffx
    from pathlib import Path
    from test_gaussian_abi_failures import View,Context,Buffer,view,Q,L,P
    args,_=sample;args=[a.clone() for a in args];dev=args[1].device;cuda=dev.type=='cuda'
    expected=torch.ops.gffx.shading_opaque_samples(*args)
    lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
    fn=lib.gffx_shading_opaque_samples;fn.argtypes=[C.POINTER(View)]*18+[C.POINTER(Context)]+[C.POINTER(View)]*2+[C.POINTER(Buffer),P]
    query=lib.gffx_shading_opaque_samples_workspace;query.argtypes=[L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
    stream=torch.cuda.Stream() if cuda else None;ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2 if cuda else 1;ctx.stream=stream.cuda_stream if cuda else None
    size,alignment=Q(),Q();assert query(1,C.byref(ctx),C.byref(size),C.byref(alignment),None)==0 and size.value==25608
    weighted=torch.full_like(expected[0],91.);rgba=torch.full_like(expected[1],92.);storage=torch.empty(size.value,dtype=torch.uint8,device=dev)
    ws=Buffer();ws.size=C.sizeof(ws);ws.abi=65536;ws.device=ctx.device;ws.data=storage.data_ptr();ws.capacity=size.value;ws.alignment=8
    views=[view(a) for a in args];wv,yv=view(weighted,True),view(rgba,True)
    views[0].dtype=3;views[3].dtype=3
    def call():
        if cuda:torch.cuda.synchronize()
        result=fn(*(C.byref(v) for v in views),C.byref(ctx),C.byref(wv),C.byref(yv),C.byref(ws),None)
        if cuda:torch.cuda.synchronize()
        return result
    def unchanged():assert bool((weighted==91).all()) and bool((rgba==92).all())
    for i in (1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17):
        old=args[i].reshape(-1)[-1].clone();args[i].reshape(-1)[-1]=float('nan');assert call()!=0;unchanged();args[i].reshape(-1)[-1]=old
    ws.capacity-=1;assert call()!=0;unchanged();ws.capacity+=1
    old=wv.data;wv.data=views[1].data;assert call()!=0;unchanged();wv.data=old
    old=yv.data;yv.data=wv.data;assert call()!=0;unchanged();yv.data=old
    ws.data=views[1].data;assert call()!=0;unchanged();ws.data=storage.data_ptr()
    ctx.reserved[0]=1;assert call()!=0;unchanged();ctx.reserved[0]=0
    for p in (0,17,-1):
        size.value=71;alignment.value=72;assert query(p,C.byref(ctx),C.byref(size),C.byref(alignment),None)!=0
        assert (size.value,alignment.value)==(71,72)
    assert call()==0
    assert all(bits(a)==bits(b) for a,b in zip((weighted,rgba),expected))

@pytest.mark.parametrize('value',[0.,-0.,5e-324,-5e-324])
def test_native_empty_signed_zero_and_small_weights(sample,value):
    from gffx.torch._curve_math import sum_last
    args,_=sample;args=list(args);args[0]=torch.full_like(args[0],-1);args[1]=torch.zeros_like(args[1]);args[1][...,3]=value
    weighted,rgba=torch.ops.gffx.shading_opaque_samples(*args)
    weights=args[1][...,3];rgb=args[1][...,:3]*weights[...,None]
    expected=torch.cat((rgb,weights[...,None]),-1)
    assert bits(weighted)==bits(expected)
    assert bits(rgba)==bits(sum_last(expected.transpose(-1,-2)))

def test_native_sample_old_provider_query_refuses():
    if os.environ.get('GFFX_OLD_PROVIDER_CONTROL')!='1':pytest.skip('Preserved provider requires a fresh process')
    import ctypes as C,gffx
    from pathlib import Path
    from test_gaussian_abi_failures import Context,Q,L,P
    lib=C.CDLL(str(Path(gffx.__file__).parent/'gffx_core.dll'))
    fn=lib.gffx_shading_opaque_samples_workspace;fn.argtypes=[L,C.POINTER(Context),C.POINTER(Q),C.POINTER(Q),P]
    ctx=Context();ctx.size=C.sizeof(ctx);ctx.abi=65536;ctx.device=2
    size,alignment=Q(71),Q(72)
    assert fn(1,C.byref(ctx),C.byref(size),C.byref(alignment),None)!=0
    assert (size.value,alignment.value)==(71,72)
