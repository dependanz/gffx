"""Native primal selection must retain the reference image, errors and replay."""
import os,torch,pytest
from test_opaque_full_original import original

def test_constant_forward_selects_native_and_backward_keeps_eager(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    from gffx.torch import _opaque_samples as sample
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];a.requires_grad_()
    selected=[];eager=[];base=sample.compose;old=scene._sample_integral_eager
    def track(*args,**kw):
        result=base(*args,**kw)
        if result is not None:selected.append(torch.is_grad_enabled())
        return result
    def reference(*args,**kw):eager.append(torch.is_grad_enabled());return old(*args,**kw)
    monkeypatch.setattr(sample,'compose',track);monkeypatch.setattr(scene,'_sample_integral_eager',reference)
    out=render.opaque_window(s,c,(10,59,1,1),cull_mode=2);assert selected and not any(selected)
    out['colour'].sum().backward();assert any(eager) and torch.isfinite(a.grad).all()

@pytest.mark.parametrize('change',['two_sided','extra_parameter','multiple_lights'])
def test_unsupported_native_profile_returns_to_reference(monkeypatch,change):
    from gffx.torch import render,_opaque_samples as sample
    from dataclasses import replace
    from gffx.torch.scene import Scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));m=s.prims[1]
    if change=='two_sided':m=replace(m,parameters=dict(m.parameters,two_sided=True))
    if change=='extra_parameter':m=replace(m,parameters=dict(m.parameters,allow_experimental_budget=False))
    prims=[s.prims[0],m,s.prims[2]]
    if change=='multiple_lights':prims.append(replace(s.prims[2],name='second-light'))
    s=Scene(prims);old=sample.compose;results=[]
    def wrapped(*a,**kw):
        value=old(*a,**kw);results.append(value is None);return value
    monkeypatch.setattr(sample,'compose',wrapped)
    with torch.no_grad():out=render.opaque_window(s,c,(10,59,1,1),cull_mode=2)
    assert results and all(results) and torch.isfinite(out['colour']).all()

@pytest.mark.parametrize('field,value',[('albedo',-1.),('roughness',0.),('f0',1.1)])
def test_native_invalid_material_preserves_reference_error(monkeypatch,field,value):
    from gffx.torch import render,_opaque_samples as sample
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters[field].fill_(value)
    def error():
        with pytest.raises((ValueError,RuntimeError,TypeError)) as caught:
            with torch.no_grad():render.opaque_window(s,c,(10,59,1,1),cull_mode=2)
        return type(caught.value),str(caught.value)
    actual=error();monkeypatch.setattr(sample,'compose',lambda *a,**kw:None);assert error()==actual

def test_native_storage_is_charged_inside_existing_allowance():
    from gffx.torch._opaque_scene import _extra_geometry_bytes
    from gffx.torch._opaque_samples import RESERVE_BYTES
    s,c,_=original();n=len(s.prims[1].parameters['budget']._data['cosine_nodes'])
    prior=8*64*n+10000*((384+12*2+3*2+6+4)*8+2+1)
    assert RESERVE_BYTES==1048576 and _extra_geometry_bytes(s,c)==prior+RESERVE_BYTES
    assert 8+16*800*4*8+16*800*4*8+16*4*8+4096<RESERVE_BYTES
