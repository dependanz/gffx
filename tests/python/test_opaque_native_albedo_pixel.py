"""Fixed-albedo continuation reuses native samples without losing pixel work."""
import os,torch,pytest
from test_opaque_full_original import original
from test_opaque_albedo_replay import reference,bits
from gffx.torch import render


def inputs(region):
 s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'));a=s.prims[1].parameters['albedo'];y,x,h,w=region
 return s,c,a.new_tensor([-.0,-5e-324,.7]).expand(1,h,w,3).contiguous(),a.new_full((1,h,w),.1),a.new_tensor([-.0,5e-324,-.2])


@pytest.mark.parametrize('region',[(10,59,1,1),(15,48,1,1),(117,67,1,1)])
def test_fixed_overflow_native_matches_exact_without_engine_replay(region,monkeypatch):
 s,c,g,gm,initial=inputs(region);expected=reference(s,c,region,g,gm,initial)
 def forbidden(*a,**kw):raise AssertionError('general autograd engine entered fixed overflow replay')
 monkeypatch.setattr(torch.autograd,'grad',forbidden)
 actual=render.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial)
 assert all(bits(actual[k])==bits(v) for k,v in expected.items())
 assert all(not t.requires_grad for t in actual.values())


def test_native_continuation_failure_propagates(monkeypatch):
 region=(10,59,1,1);s,c,g,gm,initial=inputs(region)
 def fail(*args):raise RuntimeError('injected native continuation failure')
 monkeypatch.setattr(torch.ops.gffx,'shading_opaque_albedo',fail)
 with pytest.raises(RuntimeError,match='injected native continuation'):
  render.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial)


def test_unsupported_continuation_keeps_canonical_fallback_and_shared_work(monkeypatch):
 from gffx.torch import _opaque_samples as samples,_opaque_pixel as pixel
 region=(10,59,1,1);s,c,g,gm,initial=inputs(region);expected=reference(s,c,region,g,gm,initial)
 old=samples.compose;work=pixel._Work;seen=[];attempts=[]
 def unavailable(*args,**kw):
  if kw.get('contributions') and kw.get('_albedo_cotangent') is not None:attempts.append(1);return None
  return old(*args,**kw)
 class Count(work):
  def __init__(self):super().__init__();seen.append(self)
 monkeypatch.setattr(samples,'compose',unavailable);monkeypatch.setattr(pixel,'_Work',Count)
 actual=render.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial)
 assert attempts and len(seen)==1 and seen[0].value>0
 assert all(bits(actual[k])==bits(v) for k,v in expected.items())


def test_native_continuation_rejects_receipt_corruption(monkeypatch):
 from gffx.torch import _opaque_scene as m
 region=(10,59,1,1);s,c,g,gm,initial=inputs(region);old=m.shade
 def corrupt(*args,**kw):
  out=old(*args,**kw)
  if kw.get('_albedo_cotangent') is not None and kw.get('_batch_cursors') is not None:
   rows=[list(r) for r in out['_receipts']];rows[0][4]+=1;out['_receipts']=tuple(tuple(r) for r in rows)
  return out
 monkeypatch.setattr(m,'shade',corrupt)
 with pytest.raises(RuntimeError,match='receipt'):
  render.opaque_albedo_replay(s,c,region,g,gm,initial_gradient=initial)
