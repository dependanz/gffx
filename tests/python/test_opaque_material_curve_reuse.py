"""Constant material setup may live for one forward request, never its VJP."""
import os,gc,weakref,json,hashlib
from pathlib import Path
import torch
from test_opaque_full_original import original

def test_original_request_shares_curve_releases_it_and_refreshes_next_call(monkeypatch):
    from gffx.torch import render,_principled_groups as groups
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters['albedo'].requires_grad_()
    base=groups._prepare_material_curve;calls=[];refs=[]
    def observed(*args,**kw):
        result=base(*args,**kw);calls.append(1);refs.extend(weakref.ref(t) for t in result[1:]);return result
    monkeypatch.setattr(groups,'_prepare_material_curve',observed)
    out=render.opaque_window(s,c,(64,0,1,128),cull_mode=2,tile_pixels=16)
    assert len(calls)==1,len(calls)
    gc.collect();assert all(r() is None for r in refs),'curve retained beyond public forward'
    ref=json.loads(Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque185-20261002/request-cpu2.json').read_bytes())['cases'][3]['hashes']
    assert {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in out.items()}==ref
    s.prims[1].parameters['roughness'].fill_(.9)
    changed=render.opaque_window(s,c,(64,0,1,128),cull_mode=2,tile_pixels=16)
    assert len(calls)==2 and not torch.equal(out['colour'],changed['colour'])

def test_shared_curve_refuses_changed_resource_or_roughness():
    from gffx.torch._principled_groups import _forward_curve_state
    import pytest
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));p=s.prims[1].parameters;state={}
    _forward_curve_state(state,p['budget'],p['roughness'])
    p['roughness'].add_(.001)
    with pytest.raises(RuntimeError,match='modified'):_forward_curve_state(state,p['budget'],p['roughness'])
    state={};_forward_curve_state(state,p['budget'],p['roughness'])
    p['budget']._data['_weights'].add_(0)
    with pytest.raises(RuntimeError,match='modified'):_forward_curve_state(state,p['budget'],p['roughness'])
