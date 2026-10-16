"""Borrow records inside a live window; own every remainder before releasing it."""
import pytest,torch
from test_deferred_native_storage import scale

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_group_views_avoid_a_copy_for_every_four_points(device,monkeypatch):
 from tools.deferred_stream import stream_into
 model,camera,_=scale(device,32);out={'colour':torch.empty((1,32,32,3),dtype=torch.float64),'mask':torch.empty((1,32,32),dtype=torch.float64)}
 original=torch.Tensor.clone;active=[False];copies=[]
 def observed(x,*a,**kw):
  if active[0] and tuple(x.shape) in ((4,3),(4,)):copies.append(tuple(x.shape))
  return original(x,*a,**kw)
 def checkpoint(progress):
  if 'plan' in progress and progress['completed_windows']==progress['plan']['window_count']:active[0]=True
 monkeypatch.setattr(torch.Tensor,'clone',observed)
 report,_=stream_into(model,camera,out,backward=True,tile_rows=32,group_point_limit=128,checkpoint=checkpoint)
 assert report['complete'] and report['material_groups']>32
 assert len(copies)<=3*32, 'only the final bounded queue may need owned copies'

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_remainders_own_storage_before_source_window_is_released(device):
 from tools.deferred_stream import _own_pending_groups
 source={k:torch.arange(128*c,dtype=torch.float64,device=device).reshape(128,c) if c>1 else torch.arange(128,dtype=torch.float64,device=device) for k,c in [('normal',3),('world_position',3),('mask',1)]}
 queue=[{k:v[j:j+4] for k,v in source.items()} for j in range(0,124,4)];pending={k:v[124:127] for k,v in source.items()}
 expected=[{k:v.clone() for k,v in record.items()} for record in queue+[pending]]
 owned=_own_pending_groups(queue,pending)
 for value in source.values():value.fill_(-999)
 for record,want in zip(queue+[owned],expected):
  for key in want:assert torch.equal(record[key],want[key])
 assert sum(v.numel()*v.element_size() for record in queue+[owned] for v in record.values())<=7168
 assert _own_pending_groups([],None) is None
