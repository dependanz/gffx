"""Prospective crop-camera equivalence before any larger head partition profile."""
import pytest,torch
from tests.python.test_deferred import fixture

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('soft',[0.,.2])
def test_deferred_crop_partition_preserves_values_and_geometry(device,soft):
    from tools.deferred_image_partition import render_partitioned
    from gffx.torch import render
    model,camera,_,_,_=fixture(device=device)
    # Four square pixels and two two-row crops use the same centre samples.
    names=('colour','mask','depth','world_position','face_index','normal','uv')
    with torch.no_grad():
        expected=render(model,camera,outputs=names,soft=soft)
        actual,profiles=render_partitioned(model,camera,tile_rows=2,outputs=names,soft=soft)
    assert len(profiles)==2
    for name in names:
        assert torch.equal(actual[name],expected[name]),name

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_deferred_crop_partition_shared_first_vjps(device):
    from tools.deferred_image_partition import render_partitioned
    from gffx.torch import render
    model,camera,v,_,albedo=fixture(device=device);v.requires_grad_();albedo.requires_grad_()
    cot=torch.arange(48,dtype=torch.float64,device=device).reshape(1,4,4,3).remainder(7)*.125-.25
    expected=render(model,camera,outputs=('colour',))['colour']
    eg=torch.autograd.grad(expected,(v,albedo),cot)
    actual,_=render_partitioned(model,camera,tile_rows=2,outputs=('colour',))
    ag=torch.autograd.grad(actual['colour'],(v,albedo),cot)
    assert torch.equal(actual['colour'],expected)
    # Different caller reduction grouping is compared to the existing independent
    # algebra tolerance, not represented as a new byte-identity promise.
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
