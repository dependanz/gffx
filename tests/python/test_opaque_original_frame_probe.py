"""Completion requires all stored bytes, finite outputs and an intact manifest."""
import struct
import pytest
from tools.opaque_frame_store import Writer

def store(tmp_path,value=0.2):
    w=Writer(tmp_path/'frame',2,1,{'test':1});v=dict(colour=struct.pack('<6d',*[value]*6),mask=struct.pack('<2d',0,1),gcolour=struct.pack('<6d',*[1]*6),gmask=struct.pack('<2d',0,0));w.append(0,2,v);r=w.finish();r['identity']={'test':1};return r

def test_frame_inspection_requires_complete_finite_hashed_streams(tmp_path):
    from tools.opaque_original_frame_probe import inspect_frame
    r=store(tmp_path);v=inspect_frame(r);assert v['pixels']==2 and v['covered_pixels']==1
    assert v['arrays']['colour']['bytes']==48 and v['arrays']['mask']['shape']==[1,1,2]

def test_nonfinite_complete_output_is_not_a_pass(tmp_path):
    from tools.opaque_original_frame_probe import inspect_frame
    with pytest.raises(ValueError):inspect_frame(store(tmp_path,float('nan')))

def test_corrupt_or_incomplete_frame_is_not_an_anchor(tmp_path):
    from tools.opaque_original_frame_probe import inspect_frame
    r=store(tmp_path);p=tmp_path/'frame'/'colour.f64';p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError):inspect_frame(r)


@pytest.mark.parametrize('explicit',[False,True])
def test_frame_probe_forwards_replay_mode_and_verifies_receipt(explicit):
    from types import SimpleNamespace
    from tools.opaque_original_frame_probe import _execute_frame
    seen=[];profile='public-window-albedo-replay-v5' if explicit else 'public-window-pixel-replay-v4'
    args=SimpleNamespace(width=128,height=128,sha256='w'*64,window_pixels=16,request_pixels=256,albedo_replay=explicit)
    row=dict(scene_sha256='s'*64,profile={},ownership_profile=profile)
    def runner(*a,**kw):seen.append(kw);return {'identity':{'profile':profile}}
    _execute_frame(runner,None,None,None,row,args,90)
    assert seen[0]['albedo_replay'] is explicit and seen[0]['request_pixels']==256
    assert seen[0]['seconds']==90 and seen[0]['cotangent']==(1/(128*128*3),)*3+(0.,)
    with pytest.raises(ValueError,match='ownership'):_execute_frame(lambda *a,**kw:{'identity':{'profile':'wrong'}},None,None,None,row,args,90)
