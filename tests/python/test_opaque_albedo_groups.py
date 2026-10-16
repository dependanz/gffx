"""Native pixel groups retain independent arithmetic and canonical carries."""
import os,torch,pytest
from test_opaque_full_original import original
from test_opaque_albedo_replay import reference,bits
from gffx.torch import render


def scene():return original(os.environ.get('GFFX_KNOT_DEVICE','cpu'))[:2]


def test_eight_pixel_native_batches_and_tail(monkeypatch):
 s,c=scene();a=s.prims[1].parameters['albedo'];calls=[];op=torch.ops.gffx.shading_opaque_albedo
 def observed(*args):calls.append(args[0].shape[0]);return op(*args)
 monkeypatch.setattr(torch.ops.gffx,'shading_opaque_albedo',observed)
 out=render.opaque_albedo_replay(s,c,(64,56,1,17),a.new_ones((1,1,17,3)),a.new_zeros((1,1,17)))
 assert calls==[8,8,1]
 assert all(not v.requires_grad for v in out.values())


@pytest.mark.parametrize('region',[(64,60,2,5),(64,56,1,17),(10,57,1,8),(0,0,1,9)])
def test_grouped_replay_exact_canonical_carries(region):
 s,c=scene();a=s.prims[1].parameters['albedo'];y,x,h,w=region
 g=a.new_tensor([-.0,-5e-324,.7]).expand(1,h,w,3).contiguous();gm=a.new_full((1,h,w),-.0);initial=a.new_tensor([-.0,5e-324,-.2])
 expected=reference(s,c,region,g,gm,initial)
 actual=render.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial)
 assert all(bits(actual[k])==bits(v) for k,v in expected.items())


def test_group_capacity_refusal_retries_only_canonical_pixels(monkeypatch):
 from gffx.torch import _opaque_scene as m
 s,c=scene();a=s.prims[1].parameters['albedo'];old=m.shade;calls=[]
 def refuse(*args,**kw):
  region=kw.get('_region');cot=kw.get('_albedo_cotangent')
  if cot is not None:
   calls.append(region[2]*region[3])
   if region[2]*region[3]>1:raise OverflowError('opaque quadrature injected capacity overflow')
  return old(*args,**kw)
 monkeypatch.setattr(m,'shade',refuse)
 out=render.opaque_albedo_replay(s,c,(64,60,1,8),a.new_ones((1,1,8,3)),a.new_zeros((1,1,8)))
 assert calls==[8]+[1]*8
 assert torch.isfinite(out['albedo_gradient']).all()


def test_wide_unsupported_native_never_allocates_eager_group(monkeypatch):
 from gffx.torch import _opaque_samples as samples,_opaque_scene as m
 s,c=scene();a=s.prims[1].parameters['albedo'];old=samples.compose;eager=m._sample_integral_eager;calls=[]
 def unsupported(*args,**kw):
  n=args[5].numel()//3200
  if kw.get('_albedo_cotangent') is not None:
   calls.append(n)
   if n>1:return None
  return old(*args,**kw)
 def bounded(*args,**kw):
  assert args[5].numel()<=3200
  return eager(*args,**kw)
 monkeypatch.setattr(samples,'compose',unsupported);monkeypatch.setattr(m,'_sample_integral_eager',bounded)
 render.opaque_albedo_replay(s,c,(64,60,1,8),a.new_ones((1,1,8,3)),a.new_zeros((1,1,8)))
 assert calls==[8]+[1]*8
