"""Independent CPU derivatives for every differentiable stable geometry primitive."""
import pytest
import torch
from test_gpu_alpha import OPERATIONS, case


@pytest.mark.parametrize('name', OPERATIONS)
def test_stable_cpu_independent_geometry_gradient(name):
    function, inputs = case(name, 'cpu', torch.float64)
    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)


def test_stable_cpu_sampler_requires_rng_on_geometry_device():
    import gffx.torch as gt
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    key = torch.tensor([1, 2], dtype=torch.uint32, device='meta')
    counter = torch.tensor([0, 0], dtype=torch.uint32)
    with pytest.raises(ValueError, match='device'):
        gt.mesh.sample_surface(vertices, faces, 2, key, counter)
