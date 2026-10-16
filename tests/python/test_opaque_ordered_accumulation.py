"""Native RGBA continuation preserves every separately rounded ascending addition."""
import os
import pytest
import torch
import gffx.torch


def call(carry, values):
    return torch.ops.gffx.shading_ordered_rgba(carry, values)


def reference(carry, values):
    result=carry.clone()
    for row in values:result.add_(row)
    return result


def bits(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


@pytest.mark.parametrize('count',[0,1,31,32,799,800])
def test_ordered_rgba_retains_each_rounded_add_and_partition(count):
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu')
    carry=torch.tensor([-0.,2.**53,1.,-1.],dtype=torch.float64,device=dev)
    pattern=torch.tensor([[0.,1.,-1.,1.],[float.fromhex('0x0.0000000000001p-1022'),-2.**53,1e-200,-1e-200],[-0.,3.,-1e-200,1e-200]],dtype=torch.float64,device=dev)
    values=pattern.repeat((count+2)//3,1)[:count].contiguous()
    expected=reference(carry,values);actual=call(carry,values)
    assert bits(actual)==bits(expected)
    assert bits(call(call(carry,values[:count//2]),values[count//2:]))==bits(expected)
    assert bits(carry)==bits(torch.tensor([-0.,2.**53,1.,-1.],dtype=torch.float64,device=dev))


def test_ordered_rgba_has_a_nonassociative_negative_control():
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu')
    carry=torch.full((4,),2.**53,dtype=torch.float64,device=dev)
    values=torch.tensor([[1.]*4,[-2.**53]*4],dtype=torch.float64,device=dev)
    actual=call(carry,values)
    assert torch.equal(actual,torch.zeros_like(carry))
    assert not torch.equal(actual,carry+(values[0]+values[1]))


def test_ordered_rgba_refuses_invalid_extents_types_and_nonfinite_values():
    dev=os.environ.get('GFFX_KNOT_DEVICE','cpu')
    carry=torch.zeros(4,dtype=torch.float64,device=dev);v=torch.zeros((2,4),dtype=torch.float64,device=dev)
    for a,b in [(carry[:3],v),(carry,v[:,:3].contiguous()),(carry,v.new_zeros(801,4)),(carry.float(),v.float()),(carry,v.reshape(8))]:
        with pytest.raises(RuntimeError):call(a,b)
    for special in [float('nan'),float('inf'),-float('inf')]:
        bad=v.clone();bad[-1,-1]=special
        with pytest.raises(RuntimeError):call(carry,bad)
        bad=carry.clone();bad[0]=special
        with pytest.raises(RuntimeError):call(bad,v[:0])
    big=torch.full_like(carry,torch.finfo(torch.float64).max)
    with pytest.raises(RuntimeError):call(big,big[None,:])


def test_ordered_rgba_old_provider_refuses_additive_operation():
    if os.environ.get('GFFX_OLD_PROVIDER_CONTROL')!='1':pytest.skip('Explicit preserved provider in fresh process only')
    x=torch.zeros(4,dtype=torch.float64,device='cuda')
    with pytest.raises(RuntimeError,match='provider lacks ordered RGBA'):call(x,x[None,:])
