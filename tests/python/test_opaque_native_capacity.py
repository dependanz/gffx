"""A native window reuses the admitted storage and falls back before eager allocation."""
import os,torch,pytest
from test_opaque_full_original import original

def test_native_forward_admits_sixteen_inside_existing_reserve(monkeypatch):
    from gffx.torch import render,_opaque_stream as stream
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));seen=[];old=stream._evaluate
    def track(config,*args,**kw):seen.append(config);return old(config,*args,**kw)
    monkeypatch.setattr(stream,'_evaluate',track)
    with torch.no_grad():out=render.opaque_window(s,c,(64,0,1,128),cull_mode=2,tile_pixels=16)
    assert seen and seen[0]['cap']==16
    cfg=seen[0];assert cfg['workload']['opaque_native_window']
    assert cfg['reserve_bytes']+cfg['workload']['estimated_bytes']<=67108864
    assert torch.isfinite(out['colour']).all()

def test_unsupported_native_never_allocates_multi_pixel_eager(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene,_opaque_samples as native
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    with torch.no_grad():expected=render.opaque_window(s,c,(64,55,1,8),cull_mode=2,tile_pixels=1)
    old=scene._sample_integral_eager;seen=[]
    def eager(*args,**kw):
        pixels=args[5].numel()//3200;seen.append(pixels)
        assert pixels==1,'multi-pixel eager fallback breached its reserved layout'
        return old(*args,**kw)
    monkeypatch.setattr(native,'compose',lambda *a,**kw:None);monkeypatch.setattr(scene,'_sample_integral_eager',eager)
    with torch.no_grad():actual=render.opaque_window(s,c,(64,55,1,8),cull_mode=2,tile_pixels=16)
    assert seen and max(seen)==1
    for k in actual:assert actual[k].cpu().numpy().tobytes()==expected[k].cpu().numpy().tobytes()

def test_differentiated_plan_keeps_original_storage(monkeypatch):
    from gffx.torch import render,_opaque_stream as stream
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters['albedo'].requires_grad_();seen=[];old=stream._evaluate
    def track(config,*args,**kw):seen.append(config);return old(config,*args,**kw)
    monkeypatch.setattr(stream,'_evaluate',track)
    render.opaque_window(s,c,(64,55,1,2),cull_mode=2,tile_pixels=16)
    assert seen and seen[0]['workload']['backward']
    assert not seen[0]['workload'].get('opaque_native_window',False)

@pytest.mark.parametrize('inventory',[{}, {'backward':False}, {'backward':False,'components':{'buffers':1}}])
def test_missing_native_inventory_keeps_legacy_plan(inventory):
    from gffx.torch._opaque_samples import window_capacity
    assert window_capacity(inventory,1,16)==0
