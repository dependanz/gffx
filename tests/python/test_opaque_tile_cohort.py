"""Tile growth must not sacrifice already-admitted shader batching."""
import os,json,hashlib
from pathlib import Path
import torch
from test_opaque_full_original import original

def test_original_row_keeps_grouped_shader_and_exact_pixels(monkeypatch):
    from gffx.torch import render,_deferred
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    base=_deferred._principled_recompute
    def observed(*a,**kw):
        raise AssertionError('tile growth discarded grouped shading on the original row')
    monkeypatch.setattr(_deferred,'_principled_recompute',observed)
    out=render.opaque_window(s,c,(64,0,1,128),cull_mode=2,tile_pixels=16)
    ref=json.loads(Path('D:/devspace/gffx/qualification-20260913/m1-continuation-20260913/opaque185-20261002/request-cpu2.json').read_bytes())['cases'][3]['hashes']
    assert {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in out.items()}==ref

def test_small_budget_single_point_plan_remains_available(monkeypatch):
    from gffx.torch import render,_deferred,_opaque_stream
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));seen=[]
    def bounded(*a,**kw):
        return dict(principled_point_block_limit=1,principled_group_point_limit=128)
    def capture(config,*tensors):
        seen.append(config);return c.fx.new_zeros((1,1,3,3)),c.fx.new_zeros((1,1,3))
    monkeypatch.setattr(_deferred,'_plan_workload',bounded)
    monkeypatch.setattr(_opaque_stream._Replay,'apply',capture)
    render.opaque_window(s,c,(64,23,1,3),cull_mode=2,tile_pixels=3)
    assert seen[0]['cap']==3 and seen[0]['workload']['principled_point_block_limit']==1
