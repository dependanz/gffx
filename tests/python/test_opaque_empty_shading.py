"""Native uncovered forward samples need no interpolation or shader setup."""
import os,json,hashlib
from pathlib import Path
import torch,pytest
from test_opaque_full_original import original

def test_uncovered_forward_omits_interpolation_with_exact_row(monkeypatch):
    from gffx.torch import render
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters['albedo'].requires_grad_()
    calls=[];base=render.interpolate
    def observed(*a,**kw):calls.append(1);return base(*a,**kw)
    monkeypatch.setattr(render,'interpolate',observed)
    out=render.opaque_window(s,c,(0,0,1,128),cull_mode=2,tile_pixels=16)
    assert not calls,len(calls)
    for k,v in out.items():assert v.detach().cpu().numpy().tobytes()==bytes(v.numel()*8),k

def test_uncovered_background_retains_exact_signed_bytes():
    from gffx.torch import render
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));bg=torch.tensor([-0.,-0.5,0.5],dtype=torch.float64,device=c.fx.device)
    out=render.opaque_window(s,c,(0,0,1,16),background=bg,cull_mode=2,tile_pixels=16)
    expected=torch.zeros_like(out['colour'])+(1-torch.zeros_like(out['mask'])[...,None])*bg
    assert out['colour'].detach().cpu().numpy().tobytes()==expected.cpu().numpy().tobytes()

def test_uncovered_invalid_binding_still_rejected():
    from gffx.torch import render
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters['albedo'][0]=float('nan')
    with pytest.raises(ValueError):render.opaque_window(s,c,(0,0,1,16),cull_mode=2)

def test_uncovered_backward_retains_fresh_interpolation(monkeypatch):
    from gffx.torch import render
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));v=s.prims[0].vertices;v.requires_grad_();calls=[];base=render.interpolate
    def observed(*a,**kw):
        if torch.is_grad_enabled():calls.append(1)
        return base(*a,**kw)
    monkeypatch.setattr(render,'interpolate',observed)
    out=render.opaque_window(s,c,(0,0,1,1),cull_mode=2)
    (out['colour'].sum()+out['mask'].sum()).backward()
    assert calls
    assert v.grad is None or torch.count_nonzero(v.grad)==0
