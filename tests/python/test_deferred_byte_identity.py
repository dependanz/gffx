"""CPU-only controls for the byte comparator used by the CUDA evidence gate."""
import pytest
import torch
from test_deferred_cuda import exact


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_deferred_byte_identity_rejects_signed_zero(dtype):
    positive=torch.tensor(0.,dtype=dtype)
    negative=-positive
    assert torch.equal(positive,negative)  # Numerical equality is insufficient.
    assert not torch.equal(positive.reshape(-1).view(torch.uint8),
                           negative.reshape(-1).view(torch.uint8))
    with pytest.raises(AssertionError):
        exact(positive,negative,'signed zero')


def test_deferred_byte_identity_accepts_equal_scalar_empty_and_strided_values():
    samples=(torch.tensor(2.,dtype=torch.float64),
             torch.empty((0,),dtype=torch.float64),
             torch.empty((2,0),dtype=torch.int32),
             torch.tensor(True),
             torch.arange(6,dtype=torch.float32).reshape(2,3).T)
    for value in samples:
        exact(value,value.contiguous().clone(),'equal logical bytes')


def test_deferred_byte_identity_keeps_shape_and_dtype_guards():
    value=torch.tensor([1.],dtype=torch.float32)
    with pytest.raises(AssertionError,match='layout'):
        exact(value,value.to(torch.float64),'dtype')
    with pytest.raises(AssertionError,match='layout'):
        exact(value,value.reshape(()),'shape')


def test_deferred_byte_identity_handles_singleton_nonunit_stride():
    value=torch.tensor([.613,.1,.2],dtype=torch.float64)[::3]
    assert value.shape==(1,) and value.stride()==(3,)
    exact(value,torch.tensor([.613],dtype=torch.float64),'singleton stride')


def test_deferred_stage_byte_formatter_handles_singleton_nonunit_stride():
    from test_deferred_principled_stages import raw
    value=torch.tensor([.613,.1,.2],dtype=torch.float64)[::3]
    assert value.shape==(1,) and value.stride()==(3,)
    assert raw(value)==raw(torch.tensor([.613],dtype=torch.float64))
