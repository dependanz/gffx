"""Forward horizon values are shared but per-pixel layout eligibility is exact."""
import os,gc,weakref,json,hashlib
from pathlib import Path
import torch
from test_opaque_full_original import original


def test_request_prepares_horizons_once_and_releases_them(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters['albedo'].requires_grad_();base=scene._horizon_cuts;calls=[];refs=[]
    def observed(*a,**kw):
        result=base(*a,**kw)
        if kw.get('_constant') is None:
            calls.append(1);refs.extend(weakref.ref(t) for t in result if t is not None)
        return result
    monkeypatch.setattr(scene,'_horizon_cuts',observed)
    out=render.opaque_window(s,c,(64,0,1,128),cull_mode=2,tile_pixels=16)
    assert len(calls)==1,len(calls)
    gc.collect();assert all(p() is None for p in refs),'horizon tensors retained by autograd context'
    B=Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque185-20261002')
    ref=json.loads((B/'request-cpu2.json').read_bytes())['cases'][3]['hashes']
    assert {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in out.items()}==ref


def test_active_zero_coefficients_keep_their_horizon_column():
    from gffx.torch._opaque_scene import _select_horizons
    d=os.environ.get('GFFX_KNOT_DEVICE','cpu');linear=torch.zeros((2,3,2),dtype=torch.float64,device=d);curved=torch.zeros((2,6),dtype=torch.float64,device=d)
    activity=dict(linear=[torch.tensor([True,False],device=d),torch.tensor([False,True],device=d)],curved=torch.tensor([False,True],device=d))
    a,b=_select_horizons(linear,curved,activity,torch.tensor([True,False],device=d));assert a.shape==(2,3,1) and b is None
    a,b=_select_horizons(linear,curved,activity,torch.tensor([False,True],device=d));assert a.shape==(2,3,1) and b is curved
    a,b=_select_horizons(linear,curved,activity,torch.tensor([False,False],device=d));assert a.shape==(2,3,0) and b is None


def test_union_cut_capacity_falls_back_without_expanding_limits(monkeypatch):
    from gffx.torch import render,_opaque_scene as scene
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));base=scene._horizon_cuts;attempts=[]
    def limited(*args,**kw):
        if kw.get('_activity') is not None:
            attempts.append(True);raise OverflowError('opaque scene shared material/horizon-cut capacity16 exceeded')
        return base(*args,**kw)
    monkeypatch.setattr(scene,'_horizon_cuts',limited)
    out=render.opaque_window(s,c,(64,23,1,7),cull_mode=2,tile_pixels=7)
    assert len(attempts)==1
    monkeypatch.setattr(scene,'_horizon_cuts',base)
    one=[render.opaque_window(s,c,(64,x,1,1),cull_mode=2) for x in range(23,30)]
    for key in out:assert torch.equal(out[key].view(torch.uint8),torch.cat([a[key] for a in one],2).view(torch.uint8))
