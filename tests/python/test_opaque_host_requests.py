"""Bound host requests by existing storage blocks without enlarging native tiles."""
import copy
from pathlib import Path
import pytest,torch
from test_opaque_frame_windows import fixture
from test_opaque_original_frame_admission import current,anchor


def test_host_request_profile_preserves_native_cap_and_block_storage(tmp_path,monkeypatch):
    from tools.opaque_frame_execution import execute
    from tools.opaque_frame_store import Reader
    from gffx.torch import render
    s,c,_=fixture(1);c.image_width=259;calls=[]
    def fake(scene,camera,region,**kw):
        y,x,h,w=region;calls.append((region,kw.get('tile_pixels')))
        return dict(colour=c.fx.new_zeros((1,h,w,3)),mask=c.fx.new_zeros((1,h,w)))
    monkeypatch.setattr(render,'opaque_window',fake)
    r=execute(s,c,tmp_path/'frame',identity={},window_pixels=16,request_pixels=256)
    assert calls==[((0,0,1,256),16),((0,256,1,3),16)]
    assert r['identity']['profile']=='public-window-pixel-replay-v4'
    rd=Reader(r['root'],r['manifest_sha256'],r['identity'])
    assert [n for _,n,_ in rd.blocks()]==[256,3]


def test_host_request_profile_preserves_actual_pixels_and_ordered_replay(tmp_path):
    from tools.opaque_frame_execution import execute,replay
    s,c,_=fixture(3)
    old=execute(s,c,tmp_path/'old',identity={},window_pixels=16);g0=replay(s,c,old)
    new=execute(s,c,tmp_path/'new',identity={},window_pixels=16,request_pixels=256);g1=replay(s,c,new)
    for k in ('colour','mask','gcolour','gmask'):
        assert (Path(old['root'])/(k+'.f64')).read_bytes()==(Path(new['root'])/(k+'.f64')).read_bytes()
    assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(g0,g1))
    bad=copy.deepcopy(new);bad['identity']['request_pixels']=257
    with pytest.raises(ValueError):replay(s,c,bad)


def test_host_request_invalid_bounds_refuse_before_storage(tmp_path):
    from tools.opaque_frame_execution import execute
    s,c,_=fixture(1)
    for n in (True,0,-1,257,256.0):
        with pytest.raises(ValueError):execute(s,c,tmp_path/'bad',identity={},window_pixels=16,request_pixels=n)
        assert not (tmp_path/'bad').exists()


def test_host_request_admission_preserves_full_anchor_identity():
    from tools.opaque_original_frame_admission import time_admission
    r=current();r['ownership_profile']='public-window-pixel-replay-v4';r['profile'].update(window_pixels=16,request_pixels=256)
    assert time_admission([],r,remaining_seconds=110)['startup']
    for k,v in [('request_pixels',257),('request_pixels',True),('window_pixels',17),('window_pixels',True)]:
        bad=copy.deepcopy(r);bad['profile'][k]=v
        with pytest.raises(ValueError):time_admission([],bad,remaining_seconds=110)
    r['profile'].update(width=512,height=512)
    with pytest.raises(ValueError):time_admission([anchor(128,1),anchor(256,3)],r,remaining_seconds=110)
