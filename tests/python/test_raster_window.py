"""Global pixel windows must preserve full-frame raster arithmetic exactly."""
import pytest,torch
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('blur',[0.,.6])
def test_raster_windows_match_full_frame_bytes(device,dtype,blur):
 import gffx.torch
 v=torch.tensor([[-.87,-.72,.3],[.81,-.63,.3],[-.11,.91,.3],[.63,.78,.5]],device=device,dtype=dtype)
 f=torch.tensor([[0,1,2],[0,1,2],[1,3,2]],device=device,dtype=torch.int32)
 vo=torch.tensor([0,4],device=device,dtype=torch.int32);fo=torch.tensor([0,3],device=device,dtype=torch.int32)
 assert hasattr(torch.ops.gffx,'rasterize_window'),'native global window operation is missing'
 for cull in (1,2,3):
  full=torch.ops.gffx.rasterize(v,f,vo,fo,7,11,2,blur,cull,1e-12)
  for row,col,h,w in ((0,0,7,11),(1,2,5,6),(6,10,1,1),(0,0,1,3)):
   win=torch.ops.gffx.rasterize_window(v,f,vo,fo,7,11,row,col,h,w,2,blur,cull,1e-12)
   for x,y in zip(win,full):assert torch.equal(raw(x),raw(y[:,row:row+h,col:col+w].contiguous()))

@pytest.mark.parametrize('region',[(-1,0,1,1),(0,-1,1,1),(0,0,0,1),(6,0,2,1),(0,10,1,2),(0,0,2**62,1)])
def test_raster_window_rejects_invalid_bounds_before_allocation(region):
 import gffx.torch
 v=torch.empty((0,3),dtype=torch.float64);f=torch.empty((0,3),dtype=torch.int32);o=torch.tensor([0,0],dtype=torch.int32)
 with pytest.raises((ValueError,RuntimeError),match='window'):
  torch.ops.gffx.rasterize_window(v,f,o,o,7,11,*region,1,0.,1,1e-12)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('soft',[0.,.2])
def test_deferred_window_preserves_interpolation_and_colour(device,soft):
 from test_deferred_native_storage import scale
 from gffx.torch import _deferred as d
 model,camera,albedo=scale(device,16)
 with torch.no_grad():
  packed,camera=d._prepare(model,camera)
  full=d._buffers(packed,camera,soft,2)
  colour=d._shade(packed,camera,full)
  for row,col,h,w in ((5,3,7,8),(0,0,2,2),(15,15,1,1)):
   win=d._buffers(packed,camera,soft,2,window=(row,col,h,w))
   for key in full:
    if key=='_projection':continue
    assert torch.equal(raw(win[key]),raw(full[key][:,row:row+h,col:col+w].contiguous())),key
   actual=d._shade(packed,camera,win)
   assert torch.equal(raw(actual),raw(colour[:,row:row+h,col:col+w].contiguous()))
