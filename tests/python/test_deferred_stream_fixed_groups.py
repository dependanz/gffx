"""Pinned existing group capacity across differently sized streamed windows."""
import copy
import pytest,torch
from test_deferred_native_storage import scale
from test_principled_native_operation import raw

@pytest.mark.parametrize('size',[32,64,128,256,512])
def test_fixed_groups_choose_largest_admissible_row_window(size):
 from tools.deferred_stream import plan_windows
 model,camera,_=scale('cpu',size)
 plan=plan_windows(model,camera,backward=True,group_point_limit=128)
 assert plan['estimate']['principled_group_point_limit']==128
 assert plan['estimate']['principled_point_block_limit']==4
 assert plan['estimate']['estimated_bytes']+plan['carry_reserve_bytes']<=67108864
 assert plan['caller_output_bytes']==size*size*32
 assert camera.image_height==size and camera.image_width==size
 rows=plan['tile_rows']
 if rows<size:
  with pytest.raises(ValueError,match='workspace'):
   plan_windows(model,camera,backward=True,tile_rows=rows+1,group_point_limit=128)

@pytest.mark.parametrize('invalid',[True,0,64,132,128.0])
def test_fixed_group_profile_rejects_unqualified_partition(invalid):
 from tools.deferred_stream import plan_windows
 model,camera,_=scale('cpu',16)
 with pytest.raises(ValueError,match='group'):
  plan_windows(model,camera,group_point_limit=invalid)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('soft',[0.,.2])
def test_fixed_group_stream_keeps_every_pixel_and_original_gradient(device,soft):
 from tools.deferred_stream import stream_into
 from gffx.torch import render
 model,camera,albedo=scale(device,32)
 expected=render(model,camera,soft=soft)
 eg,=torch.autograd.grad(expected['colour'].mean(),albedo)
 out={k:torch.empty_like(v,device='cpu') for k,v in expected.items()}
 for rows in (3,7,32):
  report,g=stream_into(model,camera,out,soft=soft,backward=True,tile_rows=rows,group_point_limit=128)
  assert report['complete'] and report['plan']['estimate']['principled_group_point_limit']==128
  for k in out:assert torch.equal(raw(out[k]),raw(expected[k]))
  assert torch.equal(raw(g),raw(eg))

def test_fixed_group_refusal_leaves_destinations_untouched():
 from tools.deferred_stream import stream_into,StreamFailure,plan_windows
 model,camera,_=scale('cpu',64)
 out={'colour':torch.full((1,64,64,3),-123.,dtype=torch.float64),'mask':torch.full((1,64,64),-123.,dtype=torch.float64)}
 before={k:v.clone() for k,v in out.items()}
 # Shared native curves now let this extent fit. Exercise an actual fixed-group
 # storage boundary instead of assuming a particular resolution must fail.
 plan=plan_windows(model,camera,backward=True,tile_rows=64,group_point_limit=128)
 limit=plan['estimate']['estimated_bytes']+plan['carry_reserve_bytes']-1
 assert 32768<limit<67108864
 with pytest.raises(StreamFailure,match='workspace') as error:
  stream_into(model,camera,out,backward=True,tile_rows=64,group_point_limit=128,workspace_limit_bytes=limit)
 assert not error.value.report['complete'] and error.value.report['completed_windows']==0
 for k in out:assert torch.equal(raw(out[k]),raw(before[k]))
