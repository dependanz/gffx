"""CPU-owned streamed images and original globally ordered albedo gradients."""
import pytest,torch
from test_deferred_native_storage import scale
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('soft',[0.,.2])
def test_streamed_output_and_global_albedo_gradient_match_full(device,soft):
 from tools.deferred_stream import stream_into
 from gffx.torch import render
 model,camera,albedo=scale(device,16)
 expected=render(model,camera,soft=soft);eg,=torch.autograd.grad(expected['colour'].mean(),albedo)
 out={k:torch.empty_like(v,device='cpu') for k,v in expected.items()}
 for rows in (3,5,16):
  report,gradient=stream_into(model,camera,out,soft=soft,backward=True,tile_rows=rows)
  assert report['status']=='passed' and report['complete']
  for k in out:assert torch.equal(raw(out[k]),raw(expected[k]))
  assert torch.equal(raw(gradient),raw(eg)),(gradient,eg)


def test_stream_cancellation_and_input_mutation_never_return_success():
 from tools.deferred_stream import stream_into,StreamFailure
 model,camera,albedo=scale('cpu',16)
 out={'colour':torch.empty((1,16,16,3),dtype=torch.float64),'mask':torch.empty((1,16,16),dtype=torch.float64)}
 def cancel(progress):
  if progress['completed_windows']==1:raise RuntimeError('cancel test')
 with pytest.raises(StreamFailure) as e:stream_into(model,camera,out,tile_rows=3,checkpoint=cancel)
 assert not e.value.report['complete'] and e.value.report['completed_windows']==1
 def mutate(progress):
  if progress['completed_windows']==1:
   with torch.no_grad():albedo.add_(.01)
 with pytest.raises(StreamFailure,match='changed') as e:stream_into(model,camera,out,tile_rows=3,checkpoint=mutate)
 assert not e.value.report['complete']


def test_stream_backward_batches_global_groups_without_reducing_them_together(monkeypatch):
 from tools.deferred_stream import stream_into
 model,camera,albedo=scale('cpu',32);out={'colour':torch.empty((1,32,32,3),dtype=torch.float64),'mask':torch.empty((1,32,32),dtype=torch.float64)}
 active=[False];calls=[]
 for name in ('shading_principled_pairs','shading_principled_groups','shading_principled_shared_groups'):
  original=getattr(torch.ops.gffx,name)
  def observe(*args,_original=original,_name=name):
   if active[0]:calls.append(_name)
   return _original(*args)
  monkeypatch.setattr(torch.ops.gffx,name,observe)
 def checkpoint(progress):
  if 'plan' in progress and progress['completed_windows']==progress['plan']['window_count']:active[0]=True
 report,_=stream_into(model,camera,out,backward=True,tile_rows=5,checkpoint=checkpoint)
 assert 0<len(calls)<=(report['covered_points']+127)//128,calls

@pytest.mark.parametrize('mutation',['camera_shape','camera_tensor','destination'])
def test_stream_changed_bindings_and_destination_are_invalid(mutation):
 from tools.deferred_stream import stream_into,StreamFailure
 model,camera,albedo=scale('cpu',16);out={'colour':torch.empty((1,16,16,3),dtype=torch.float64),'mask':torch.empty((1,16,16),dtype=torch.float64)}
 def mutate(progress):
  if progress['completed_windows']==1:
   if mutation=='camera_shape':camera.image_width+=1
   elif mutation=='camera_tensor':camera.pose=camera.pose.clone()
   else:out['colour'].zero_()
 with pytest.raises(StreamFailure,match='changed') as e:stream_into(model,camera,out,tile_rows=3,checkpoint=mutate)
 assert not e.value.report['complete'] and e.value.report['completed_windows']==1


def test_stream_backward_cohort_fits_the_selected_window_bound(monkeypatch):
 from tools.deferred_stream import stream_into
 model,camera,albedo=scale('cpu',16);out={'colour':torch.empty((1,16,16,3),dtype=torch.float64),'mask':torch.empty((1,16,16),dtype=torch.float64)}
 active=[False];calls=[];original=torch.ops.gffx.shading_principled_shared_groups
 def observe(*args):
  if active[0]:calls.append(len(args[6]))
  return original(*args)
 monkeypatch.setattr(torch.ops.gffx,'shading_principled_shared_groups',observe)
 def checkpoint(progress):
  if 'plan' in progress and progress['completed_windows']==progress['plan']['window_count']:active[0]=True
 stream_into(model,camera,out,backward=True,tile_rows=3,checkpoint=checkpoint)
 assert max(calls)<=3*16,calls


def test_stream_cpu64_auto_plan_and_independent_albedo_differences():
 from tools.deferred_stream import stream_into
 model,camera,albedo=scale('cpu',64);out={'colour':torch.empty((1,64,64,3),dtype=torch.float64),'mask':torch.empty((1,64,64),dtype=torch.float64)}
 report,gradient=stream_into(model,camera,out,backward=True)
 assert report['plan']['estimate']['estimated_bytes']+report['plan']['carry_reserve_bytes']<=67108864
 assert bool((gradient!=0).all())
 model,camera,albedo=scale('cpu',16);out={'colour':torch.empty((1,16,16,3),dtype=torch.float64),'mask':torch.empty((1,16,16),dtype=torch.float64)}
 _,gradient=stream_into(model,camera,out,soft=.2,backward=True,tile_rows=3)
 direction=torch.tensor([.3,-.2,.1],dtype=torch.float64);analytic=(gradient*direction).sum();original=albedo.detach().clone()
 for step in (1e-5,5e-6):
  with torch.no_grad():albedo.copy_(original+step*direction)
  stream_into(model,camera,out,soft=.2,tile_rows=3);plus=out['colour'].clone()
  with torch.no_grad():albedo.copy_(original-step*direction)
  stream_into(model,camera,out,soft=.2,tile_rows=3);minus=out['colour'].clone()
  numerical=(plus-minus).mean()/(2*step)
  torch.testing.assert_close(analytic,numerical,atol=1e-10,rtol=1e-9)


def test_stream_cuda_repeated_storage_is_bounded(record_property):
 import json,gc
 from tools.deferred_stream import stream_into
 model,camera,albedo=scale('cuda',16);out={'colour':torch.empty((1,16,16,3),dtype=torch.float64),'mask':torch.empty((1,16,16),dtype=torch.float64)}
 report,g=stream_into(model,camera,out,soft=.2,backward=True,tile_rows=3);del g;gc.collect();torch.cuda.synchronize()
 baseline=torch.cuda.memory_allocated();live=[];torch.cuda.reset_peak_memory_stats()
 for _ in range(6):
  report,g=stream_into(model,camera,out,soft=.2,backward=True,tile_rows=3)
  assert torch.isfinite(g).all();del g;gc.collect();torch.cuda.synchronize();live.append(torch.cuda.memory_allocated())
 extra=torch.cuda.max_memory_allocated()-baseline;bound=report['plan']['estimate']['estimated_bytes']+report['plan']['carry_reserve_bytes']
 assert max(live)-min(live)<=32768 and extra<=bound
 record_property('stream_storage',json.dumps(dict(postwarm_live_bytes=live,peak_extra_bytes=extra,window_bound_bytes=bound,caller_output_bytes=report['plan']['caller_output_bytes'])))


def test_stream_largest_window_preserves_calibrated_four_point_partition():
 from tools.deferred_stream import plan_windows
 model,camera,_=scale('cpu',256)
 plan=plan_windows(model,camera)
 assert plan['estimate']['principled_point_block_limit']==4
