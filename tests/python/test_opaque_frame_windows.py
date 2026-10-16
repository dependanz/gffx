"""Forward window scheduling must retain exact stored bytes and canonical VJP."""
import importlib.util,os,copy
from pathlib import Path
import torch,pytest

def fixture(size=3):
    spec=importlib.util.spec_from_file_location('frame_window_fixture',Path(__file__).with_name('test_opaque_window.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.fixture(size,os.environ.get('GFFX_KNOT_DEVICE','cpu'))

def test_forward_windows_preserve_pixels_and_canonical_replay(tmp_path,monkeypatch):
    from tools.opaque_frame_execution import execute,replay
    from gffx.torch import render
    s,c,v=fixture();base=execute(s,c,tmp_path/'base',identity={},cull_mode=render.CULL_NONE);expected=replay(s,c,base);seen=[];original=render.opaque_window
    def traced(*args,**kw):seen.append(args[2]);return original(*args,**kw)
    monkeypatch.setattr(render,'opaque_window',traced)
    for cap in (2,16):
        receipt=execute(s,c,tmp_path/str(cap),identity={},cull_mode=render.CULL_NONE,window_pixels=cap)
        assert receipt['identity']['profile']=='public-window-pixel-replay-v3' and receipt['identity']['window_pixels']==cap
        for k in ('colour','mask','gcolour','gmask'):assert (Path(receipt['root'])/(k+'.f64')).read_bytes()==(Path(base['root'])/(k+'.f64')).read_bytes()
        seen.clear();actual=replay(s,c,receipt);assert len(seen)==9 and all(r[2:]==(1,1) for r in seen)
        assert all(a.cpu().numpy().tobytes()==b.cpu().numpy().tobytes() for a,b in zip(actual,expected))
        bad=copy.deepcopy(receipt);bad['identity']['window_pixels']=1
        with pytest.raises(ValueError):replay(s,c,bad)

def test_invalid_window_limits_refuse_before_storage(tmp_path):
    from tools.opaque_frame_execution import execute
    s,c,_=fixture(1)
    for cap in (True,0,-1,17,1.0,'2'):
        with pytest.raises(ValueError):execute(s,c,tmp_path/'bad',identity={},window_pixels=cap)
        assert not (tmp_path/'bad').exists()
