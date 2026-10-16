"""Curve-only windows retain each pixel's canonical rule and replay order."""
import os,pytest,torch
from test_opaque_curved_pixels import fixture

def bits(x):return x.detach().cpu().contiguous().numpy().tobytes()

def test_curve_only_forward_window_uses_native_batch_and_exact_pixels(monkeypatch):
 from gffx.torch import render
 s,c,n=fixture(3,os.environ.get('GFFX_KNOT_DEVICE','cpu'))
 with torch.no_grad():expected=render.opaque_window(s,c,(0,0,3,3),cull_mode=1,tile_pixels=1)
 operation=render.opaque_material_quadrature;calls=[]
 def observe(*args,**kw):
  if kw.get('curves') is not None:
   assert args[2].shape[-1]==0;calls.append(args[3]*args[4])
  return operation(*args,**kw)
 monkeypatch.setattr(render,'opaque_material_quadrature',observe)
 with torch.no_grad():actual=render.opaque_window(s,c,(0,0,3,3),cull_mode=1,tile_pixels=3)
 assert calls and max(calls)>1,'curve-only windows unnecessarily fell back to individual pixels'
 for key in expected:assert bits(actual[key])==bits(expected[key])

def test_curve_only_window_retains_canonical_normal_gradient():
 from gffx.torch import render
 s,c,n=fixture(2,os.environ.get('GFFX_KNOT_DEVICE','cpu'));rows=[]
 for cap in (1,2):
  out=render.opaque_window(s,c,(0,0,2,2),cull_mode=1,tile_pixels=cap)
  g,=torch.autograd.grad(out['colour'],n,torch.full_like(out['colour'],.2))
  rows.append([bits(out['colour']),bits(out['mask']),bits(g)])
 assert rows[0]==rows[1]

def test_zero_curve_rows_do_not_change_native_material_sample_bytes():
 from gffx.torch import render
 dev=os.environ.get('GFFX_KNOT_DEVICE','cpu');v=torch.tensor([[.1,.1,0.],[.9,.1,0.],[.1,.9,0.]],dtype=torch.float64,device=dev)
 f=torch.tensor([[0,1,2]],dtype=torch.int32,device=dev);cuts=v.new_empty((1,3,0))
 plain=render.opaque_material_quadrature(v,f,cuts,1,2)
 zero=render.opaque_material_quadrature(v,f,cuts,1,2,curves=v.new_zeros((1,6)))
 for a,b in zip(plain,zero):assert bits(a)==bits(b)
