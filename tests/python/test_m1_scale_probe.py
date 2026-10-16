"""Prospective independent and partition-equivalence controls for bounded scale measurements."""
import pytest
import torch


@pytest.mark.parametrize('mode',['composite','reconstruct'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_image_partition_preserves_values_and_input_vjps(mode,device):
    from tools.m1_scale_probe import image_inputs,image_operation,image_tiles
    if device=='cuda' and not torch.cuda.is_available():pytest.fail('CUDA required')
    source=image_inputs(mode,8,14,0,8)
    inputs=tuple(v.to(device).requires_grad_() for v in source)
    whole=image_operation(mode,inputs)
    cot=torch.arange(whole.numel(),dtype=whole.dtype,device=device).reshape(whole.shape).remainder(13)*.125-.75
    gradients=torch.autograd.grad(whole,inputs,cot)
    tiled,tg=image_tiles(mode,8,14,3,device,cot.cpu())
    assert torch.equal(whole.cpu(),tiled)
    for a,b in zip(gradients,tg):assert torch.equal(a.cpu(),b)
    if mode=='composite':
        a,b,c=source
        expected=a+b*(1-a[...,3:4]);expected=expected+c*(1-expected[...,3:4])
    else:expected=source[0].reshape(8,2,14,2,4).sum((1,3))/4
    torch.testing.assert_close(whole.cpu(),expected,atol=2e-7,rtol=2e-7)


def test_scale_admission_rejects_unbounded_inputs_before_tensor_allocation():
    from tools.m1_scale_probe import admission
    for mode,size in [('camera',1000001),('image',1081)]:
        with pytest.raises(ValueError):admission(mode,size)
    assert admission('image',1080)['tile_rows']==16
    assert admission('camera',1000000)['chunk_points']==65536
