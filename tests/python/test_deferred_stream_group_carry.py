"""Independent group trees may batch; the global gradient still adds each group in order."""
import pytest,torch
from test_deferred_stream_explicit_vjp import fixture
from test_principled_native_operation import raw

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('tail',[1,2,3,4])
def test_stream_group_reductions_batch_independent_rows_without_padding(device,tail,monkeypatch):
 from tools.deferred_stream import _cohort_partials
 from gffx.torch import _reference_channels as reference
 args=fixture(device,tail,'lit');previous=reference._sum_last;calls=[]
 def observed(value):calls.append(tuple(value.shape));return previous(value)
 monkeypatch.setattr(reference,'_sum_last',observed)
 _cohort_partials(*args,{})
 assert len(calls)<=2,'independent four-point groups should share one tensor dispatch'
 assert any(len(s)==3 and s[0]>=31 for s in calls)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('count',[0,1,3,32])
def test_stream_carry_preserves_every_addition_and_signed_tiny_terms(device,count):
 from tools.deferred_stream import _accumulate_partials
 t=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
 carry=t([-0.,1.,-1e-280]);parts=[t(x) for x in [[1e100,-1.,1e-280],[-1e100,1e-280,-1e-280],[1.,-1e-280,2e-280],[1e-280,-0.,-1e-280]]*(count//4+1)][:count]
 expected=carry.clone()
 for part in parts:expected.add_(part)
 _accumulate_partials(carry,parts)
 assert torch.equal(raw(carry),raw(expected))

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_carry_refuses_nonfinite_parts_and_excess_groups(device):
 from tools.deferred_stream import _accumulate_partials
 carry=torch.zeros(3,dtype=torch.float64,device=device)
 with pytest.raises((ValueError,RuntimeError)):_accumulate_partials(carry,[torch.full_like(carry,float('nan'))])
 with pytest.raises((ValueError,RuntimeError)):_accumulate_partials(carry,[torch.ones_like(carry)]*33)
