"""An intermediate complete frame must obey the existing conservative forecast."""
import copy
from tools.deferred_progressive_admission import forecast

def test_intermediate_frame_uses_both_complete_anchors_and_unchanged_margin():
 base=dict(status='passed',render_executed=True,driver_sha256='driver',scene_sha256='scene',source_sha256={'helper':'sha'},ownership_profile='stream',wheel={'sha256':'wheel'},profile=dict(device='cpu',backward=True,width=256,height=256),estimate={'principled_point_block_limit':4,'principled_group_point_limit':128},supervisor={'exit_code':0,'failure':None})
 anchors=[]
 for size,seconds in ((256,1.5),(512,4.5)):
  r=copy.deepcopy(base);r['profile'].update(width=size,height=size);r['measured_seconds']=seconds
  anchors.append(dict(path=str(size),sha256=str(size),report=r))
 current=copy.deepcopy(base);current['profile'].update(width=768,height=768)
 actual=forecast(anchors,current)
 assert actual['seconds']==max(1.5/256**2,4.5/512**2)*768**2*2+30
 assert len(actual['selected'])==2 and actual['margin']==2 and actual['reserve_seconds']==30

def test_intermediate_frame_cannot_be_used_as_a_partial_or_mismatched_anchor():
 import pytest
 base=dict(status='passed',render_executed=True,driver_sha256='d',scene_sha256='s',source_sha256={},ownership_profile='x',wheel={'sha256':'w'},profile=dict(width=512,height=512),estimate={},supervisor={'exit_code':0,'failure':None},measured_seconds=5.)
 current=copy.deepcopy(base);current['profile'].update(width=1920,height=1080)
 for change in ({'status':'stopped'},{'driver_sha256':'other'},{'render_executed':False}):
  r=copy.deepcopy(base);r.update(change)
  with pytest.raises(ValueError):forecast([dict(path='partial',sha256='hash',report=r)],current)
