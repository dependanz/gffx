"""Native storage admission backed by actual complete-render measurements."""
import json
import torch
from gffx.torch import material,scene,render
from gffx.torch.camera import Camera
from tools.synthetic_flame_fixture import generate
from tools.deferred_flame import create_scene


def scale(device,size):
 model,cam=create_scene(generate(),'principled',device=device)
 camera=Camera.from_intrinsics(cam.fx*size/128,cam.fy*size/128,cam.cx*size/128,cam.cy*size/128,size,size,.1,10.,pose=cam.pose)
 albedo=next(p for p in model.prims if p.name=='skin').parameters['albedo'];albedo.requires_grad_()
 return model,camera,albedo


def test_native_storage_cpu_scale_backward_fits_existing_allowance():
 from gffx.torch._deferred import _plan_workload
 model,camera,albedo=scale('cpu',32)
 plan=_plan_workload(model,camera,1,64*1024**2,('colour','mask'))
 assert plan['backward'] and plan['estimated_bytes']<=64*1024**2
 y=render(model,camera)['colour'];g,=torch.autograd.grad(y.mean(),albedo)
 assert torch.isfinite(y).all() and torch.isfinite(g).all() and bool((g!=0).all())


def test_native_storage_gpu_scale_bound_and_device_values(record_property):
 from gffx.torch._deferred import _plan_workload
 rows=[]
 for device in ('cpu','cuda'):
  model,camera,albedo=scale(device,32);plan=_plan_workload(model,camera,1,64*1024**2,('colour','mask'))
  if device=='cuda':
   torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
  y=render(model,camera);g,=torch.autograd.grad(y['colour'].mean(),albedo)
  rows.append((y['colour'].detach().cpu(),y['mask'].detach().cpu(),g.detach().cpu()))
  if device=='cuda':
   torch.cuda.synchronize();extra=torch.cuda.max_memory_allocated()-baseline
   assert extra<=plan['estimated_bytes']
   record_property('native_storage',json.dumps(dict(baseline_bytes=baseline,peak_extra_bytes=extra,estimated_bytes=plan['estimated_bytes'])))
 from test_principled_native_operation import raw
 for a,b in zip(*rows):assert torch.equal(raw(a),raw(b))
