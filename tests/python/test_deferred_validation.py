"""Validate complete original inputs before any grouped native dispatch."""
import pytest,torch
from test_deferred_groups import case,evaluate

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_original_inputs_checked_once_per_forward_and_backward(monkeypatch,device):
 from gffx.torch import shade
 budget,args=case(129,3,device);calls=[];original=shade._inputs
 def observed(**fields):calls.append(tuple(fields));return original(**fields)
 monkeypatch.setattr(shade,'_inputs',observed)
 y=evaluate(budget,args,True);torch.autograd.grad(y.sum(),args)
 assert len(calls)==2,calls

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('field,bad',[(0,float('nan')),(1,float('inf')),(2,float('nan')),(3,1.1),(4,0.),(5,-.1)])
def test_invalid_original_tail_refuses_before_native_dispatch(monkeypatch,device,field,bad):
 budget,args=case(129,3,device);calls=[]
 with torch.no_grad():args[field].reshape(-1)[-1]=bad
 def forbidden(*args):calls.append(True);raise AssertionError('invalid data reached native dispatch')
 monkeypatch.setattr(torch.ops.gffx,'shading_principled_groups',forbidden)
 with pytest.raises(ValueError):evaluate(budget,args,True)
 assert not calls
