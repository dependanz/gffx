"""Nonsquare qualification profiles must reach the camera without transposition."""
import importlib
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('module',['tools.deferred_flame_probe','tools.deferred_stream_probe'])
def test_nonsquare_profile_matches_camera_extent_before_render(monkeypatch,tmp_path,module):
 import sys,torch
 from gffx.torch.camera import Camera
 from tools import corpus_registry,deferred_flame
 driver=importlib.import_module(module)
 class StopBeforeRendering(Exception):pass
 def camera(fx,fy,cx,cy,image_height,image_width,near,far,pose=None):
  assert (image_height,image_width)==(1080,1920)
  raise StopBeforeRendering
 monkeypatch.setattr(Camera,'from_intrinsics',camera)
 monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
 monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
 monkeypatch.setattr(driver.cost,'write_report',lambda *args:None)
 monkeypatch.setattr(driver.artifact,'identity',lambda *args:{})
 monkeypatch.setattr(corpus_registry,'load_entry',lambda *args,**kw:{'files':{'scene.json':b'{}'}})
 monkeypatch.setattr(deferred_flame,'create_scene',lambda *args,**kw:(None,SimpleNamespace(fx=1,fy=1,cx=64,cy=64,pose=None)))
 monkeypatch.setattr(sys,'argv',['probe','--worker','--fixtures',str(tmp_path),'--output',str(tmp_path/'out.json'),'--wheel',str(tmp_path/'wheel'),'--sha256','0'*64,'--width','1920','--height','1080','--device','cpu','--material','principled'])
 with pytest.raises(StopBeforeRendering):driver.main()
