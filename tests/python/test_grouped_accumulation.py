"""Bounded diagnosis of autograd input-edge accumulation order."""
import itertools,json,pytest,torch,os
pytestmark=pytest.mark.skipif(os.environ.get("GFFX_GROUPS33_DIAGNOSTIC")!="1",reason="Historical groups33 attribution requires its frozen wheel/harness")
from test_deferred_groups import case,evaluate
from test_principled_native_operation import raw


def test_grouped_shared_accumulation_attribution(monkeypatch,record_property):
 from gffx.torch import _principled_groups as groups
 budget,base=case(9,5,'cpu');cot=(torch.arange(135,dtype=torch.float64).reshape(9,5,3)%17-8)/19
 rows=[]
 original=groups.recompute
 for mode in ('reference','grouped','reversed_edges'):
  if mode=='reversed_edges':monkeypatch.setattr(groups,'recompute',lambda rows:tuple(reversed(original(list(reversed(rows))))))
  args=tuple(x.detach().clone().requires_grad_() for x in base);y=evaluate(budget,args,mode!='reference');g=torch.autograd.grad(y,args,cot);rows.append((y,*g))
 record_property('gradient_order',json.dumps({mode:[float((a-b).abs().max()) for a,b in zip(rows[0],row)] for mode,row in zip(('reference','grouped','reversed_edges'),rows)}))
 for a,b in zip(rows[0],rows[2]):assert torch.equal(raw(a),raw(b))


@pytest.mark.parametrize('points,lights',[(5,3),(129,3)])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_reversed_group_edges_preserve_shared_reference(monkeypatch,points,lights,device):
 from gffx.torch import _principled_groups as groups
 from test_deferred_groups import test_grouped_renderer_preserves_values_and_shared_first_gradients
 original=groups.recompute
 monkeypatch.setattr(groups,'recompute',lambda rows:tuple(reversed(original(list(reversed(rows))))))
 test_grouped_renderer_preserves_values_and_shared_first_gradients(points,lights,device)



def test_original_caller_light_broadcast_device_attribution(record_property):
 rows=[];per_point=[]
 for device in ('cpu','cuda'):
  budget,args=case(129,3,device);cot=(torch.arange(129*9,dtype=torch.float64,device=device).reshape(129,3,3)%17-8)/19
  y=evaluate(budget,args,False);g=torch.autograd.grad(y,args,cot);rows.append((y,*g))
 record_property('original_ungrouped_device_maxima',json.dumps([float((a.detach().cpu()-b.detach().cpu()).abs().max()) for a,b in zip(*rows)]))
 assert not torch.equal(raw(rows[0][3]),raw(rows[1][3]))
 # This is the caller's expand over129 pixels, outside the helper's input
 # boundary. Confirm that the helper's actual per-point direction VJP agrees.
 from gffx.torch import _deferred as d
 for device in ('cpu','cuda'):
  budget,args=case(129,3,device);n,v,l,a,r,f=args;directions=l.detach()[None].expand(129,-1,-1).clone().requires_grad_()
  params=dict(albedo=a,roughness=r,f0=f,budget=budget,allow_experimental_budget=device=='cuda')
  y=torch.cat([d._principled_recompute(n[i:i+4],v[i:i+4],directions[i:i+4],params) for i in range(0,129,4)])
  cot=(torch.arange(129*9,dtype=torch.float64,device=device).reshape(129,3,3)%17-8)/19
  per_point.append(torch.autograd.grad(y,(n,v,directions,a,r,f),cot))
 for a,b in zip(*per_point):assert torch.equal(raw(a),raw(b))
