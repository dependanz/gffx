"""Closed face bounds can prove a complete forward window is empty."""
import os,torch,pytest
from test_opaque_full_original import original

@pytest.mark.parametrize('background',[False,True])
def test_validated_empty_request_omits_quadrature(monkeypatch,background):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));calls=[];old=render.opaque_material_quadrature
    def count(*a,**kw):calls.append(1);return old(*a,**kw)
    monkeypatch.setattr(render,'opaque_material_quadrature',count)
    bg=c.fx.new_tensor([-0.,-.5,5e-324]) if background else None
    out=render.opaque_window(s,c,(0,0,1,128),cull_mode=2,background=bg,tile_pixels=1)
    assert not calls,'empty request still launches quadrature'
    expected=torch.zeros_like(out['colour'])
    if bg is not None:expected=expected+(1-torch.zeros_like(out['mask'])[...,None])*bg
    assert out['colour'].detach().cpu().numpy().tobytes()==expected.cpu().numpy().tobytes()
    assert out['mask'].detach().cpu().numpy().tobytes()==bytes(out['mask'].numel()*8)

@pytest.mark.parametrize('bad',['background','vertices','faces','material'])
def test_empty_request_still_validates_inputs(bad):
    from gffx.torch import render
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));bg=None
    if bad=='background':bg=c.fx.new_tensor([0.,float('nan'),0.])
    if bad=='vertices':s.prims[0].vertices.reshape(-1)[-1]=float('nan')
    if bad=='faces':s.prims[0].faces.reshape(-1)[-1]=-1
    if bad=='material':s.prims[1].parameters['albedo'][0]=float('nan')
    with pytest.raises((ValueError,RuntimeError,IndexError)):
        render.opaque_window(s,c,(0,0,1,128),cull_mode=2,background=bg,tile_pixels=1)

def test_empty_preflight_refuses_mutation_before_return(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];old=scene.shade;calls=[]
    def changed(*args,**kw):
        result=old(*args,**kw)
        if kw.get('_empty_only'):
            assert result is not None
            with torch.no_grad():a.add_(.001)
            calls.append(1)
        return result
    monkeypatch.setattr(scene,'shade',changed)
    with pytest.raises(RuntimeError,match='modified during forward window'):
        render.opaque_window(s,c,(0,0,1,128),cull_mode=2,tile_pixels=1)
    assert len(calls)==1
