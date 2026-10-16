"""Restore borrowed camera/scene bindings once per guarded forward request."""
import os,json,hashlib
from pathlib import Path
import torch,pytest
from test_opaque_full_original import original

def test_forward_restores_camera_once_with_exact_original_row(monkeypatch):
    from gffx.torch import render,_opaque_stream as stream
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));s.prims[1].parameters['albedo'].requires_grad_();base=stream._restore;calls=[]
    def observed(node,tensors):
        if node[0]=='camera':calls.append(1)
        return base(node,tensors)
    monkeypatch.setattr(stream,'_restore',observed)
    out=render.opaque_window(s,c,(64,0,1,128),cull_mode=2,tile_pixels=16)
    assert len(calls)==1,len(calls)
    ref=json.loads(Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque185-20261002/request-cpu2.json').read_bytes())['cases'][3]['hashes']
    assert {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in out.items()}==ref

def test_preplanned_window_omits_unused_sample_camera(monkeypatch):
    from gffx.torch import render
    from gffx.torch.camera import Camera
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));base=Camera.from_intrinsics
    def observed(*a,**kw):
        assert a[5]!=102400,'unused sample-budget camera rebuilt after planning'
        return base(*a,**kw)
    monkeypatch.setattr(Camera,'from_intrinsics',observed)
    render.opaque_window(s,c,(64,64,1,1),cull_mode=2)

def test_next_request_revalidates_mutated_camera():
    from gffx.torch import render
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    render.opaque_window(s,c,(0,0,1,1),cull_mode=2)
    with torch.no_grad():c.pose.zero_()
    with pytest.raises(ValueError):render.opaque_window(s,c,(0,0,1,1),cull_mode=2)
