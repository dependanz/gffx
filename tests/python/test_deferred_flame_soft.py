"""Exact soft scale-head buffers and matched-input attribution."""
import json,torch
from tools.synthetic_flame_fixture import generate
from tools.deferred_flame import create_scene
from gffx.torch import render
from gffx.torch.camera import Camera
from test_deferred_cuda import exact

def capture(device,monkeypatch):
 from gffx.torch import _deferred as d
 model,cam=create_scene(generate(),'principled',device=device)
 camera=Camera.from_intrinsics(cam.fx/4,cam.fy/4,cam.cx/4,cam.cy/4,32,32,.1,10.,pose=cam.pose)
 rows=[];scores=[];original=d.primitives.rasterize
 logistic=getattr(d,'sigmoid',torch.sigmoid)
 def score(value):scores.append(value.detach().cpu());return logistic(value)
 def observed(*a,**kw):
  value=original(*a,**kw);rows.append(tuple(x.detach().cpu() for x in value));return value
 with monkeypatch.context() as m:
  m.setattr(d.primitives,'rasterize',observed)
  if hasattr(d,'sigmoid'):m.setattr(d,'sigmoid',score)
  with torch.no_grad():out=render(model,camera,soft=.2,outputs=('mask','normal','barycentric','face_index','world_position'))
 # Geometric outputs intentionally trigger a second hard-buffer pass.
 assert len(rows)==2
 if scores:out['coverage_score']=scores[0]
 return {k:v.detach().cpu() for k,v in out.items()},rows[0]

def test_flame_soft_common_stage_attribution(monkeypatch,record_property):
 cpu,a=capture('cpu',monkeypatch);gpu,b=capture('cuda',monkeypatch);rows=[]
 for name,x,y in zip(('index','affine','depth','distance'),a,b):
  rows.append(dict(stage=name,equal=bool(torch.equal(x.view(torch.uint8),y.view(torch.uint8))),max_abs=float((x.to(torch.float64)-y.to(torch.float64)).abs().nan_to_num().max())))
 for name in cpu:
  x,y=cpu[name],gpu[name];rows.append(dict(stage=name,equal=bool(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))),max_abs=float((x.to(torch.float64)-y.to(torch.float64)).abs().max())))
 score=-a[3].flatten()/.04;values=[]
 for device in ('cpu','cuda'):
  x=score.to(device).requires_grad_();y=torch.sigmoid(x);g,=torch.autograd.grad(y,x,torch.ones_like(y));values.append((y.detach().cpu(),g.detach().cpu()))
 rows.append(dict(stage='matched_sigmoid',value_equal=bool(torch.equal(values[0][0].view(torch.uint8),values[1][0].view(torch.uint8))),gradient_equal=bool(torch.equal(values[0][1].view(torch.uint8),values[1][1].view(torch.uint8)))))
 record_property('stages',json.dumps(rows))
 assert all(x['equal'] for x in rows[:4]),rows
 exact(cpu['mask'],gpu['mask'],'scale soft mask')
