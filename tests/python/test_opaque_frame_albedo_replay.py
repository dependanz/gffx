"""Explicit frame replay uses ordered carries and validates every stored pixel."""
import os,torch,pytest
from pathlib import Path
from test_opaque_nonlinear_scene import fixture as base
from gffx.torch.camera import Camera


def fixture(size=3):
    s,c,v,_,_=base(device=os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];a.requires_grad_();t=v.new_tensor
    c=Camera.from_intrinsics(t(128.*size),t(128.*size),t(size/2-.03),t(size/2-.02),size,size,.1,10)
    return s,c,a


def test_frame_albedo_replay_preserves_exact_carry_and_checks_pixels(tmp_path,monkeypatch):
    from tools.opaque_frame_execution import execute,replay
    from gffx.torch import render
    s,c,a=fixture();old=execute(s,c,tmp_path/'old',identity={},cull_mode=render.CULL_NONE,window_pixels=3,request_pixels=3);expected=replay(s,c,old)
    new=execute(s,c,tmp_path/'new',identity={},cull_mode=render.CULL_NONE,window_pixels=3,request_pixels=3,albedo_replay=True)
    assert new['identity']['profile']=='public-window-albedo-replay-v5'
    calls=[];original=render.opaque_albedo_replay
    def traced(*args,**kw):calls.append(args[2]);return original(*args,**kw)
    monkeypatch.setattr(render,'opaque_albedo_replay',traced);actual=replay(s,c,new)
    assert len(calls)==3 and all(r[2:]==(1,3) for r in calls)
    assert expected[0].cpu().numpy().tobytes()==actual[0].cpu().numpy().tobytes()
    def wrong(*args,**kw):
        result=original(*args,**kw);result['colour'][0,0,0,0]+=1e-12;return result
    monkeypatch.setattr(render,'opaque_albedo_replay',wrong)
    with pytest.raises(ValueError,match='pixels differ'):replay(s,c,new)


def test_frame_albedo_profile_refuses_wrong_targets_before_storage(tmp_path):
    from tools.opaque_frame_execution import execute
    for bad in ('no_target','geometry','bool'):
        s,c,a=fixture(1)
        if bad=='no_target':a.requires_grad_(False)
        elif bad=='geometry':s.prims[0].vertices.requires_grad_()
        with pytest.raises((ValueError,NotImplementedError)):execute(s,c,tmp_path/bad,identity={},albedo_replay=1 if bad=='bool' else True)
        assert not (tmp_path/bad).exists()
