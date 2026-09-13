"""Empty reduction axes still initialize every requested shared gradient."""
import gffx
import pytest
import torch

CASES = ('blend_batch', 'blend_vertices', 'blend_basis', 'rotation', 'fk_batch', 'fk_joints',
         'skin_batch', 'skin_vertices', 'bary_batch', 'bary_landmarks')


def check_empty_reduction(name, dtype, device='cpu'):
    adapter = gffx.torch
    def zeros(shape):
        return torch.zeros(shape, dtype=dtype, device=device, requires_grad=True)
    expected_nonempty = 0.
    if name.startswith('blend'):
        batch, vertices, basis = (0, 3, 2) if name == 'blend_batch' else (1, 0, 2) if name == 'blend_vertices' else (2, 3, 0)
        inputs = (zeros((vertices, 3)), zeros((basis, vertices, 3)), zeros((batch, basis)))
        output = adapter.mesh.blend_shapes(*inputs)
        expected_nonempty = 2. if name == 'blend_basis' else 0.
    elif name == 'rotation':
        inputs = (zeros((0, 3)),)
        output = adapter.transforms.rodrigues(*inputs)
    elif name.startswith('fk'):
        batch, joints = (0, 2) if name == 'fk_batch' else (1, 0)
        inputs = (zeros((batch, joints, 4, 4)),)
        parents = torch.arange(-1, joints - 1, dtype=torch.int32, device=device)
        output = adapter.skeleton.forward_kinematics(*inputs, parents)
    elif name.startswith('skin'):
        batch, vertices = (0, 3) if name == 'skin_batch' else (1, 0)
        inputs = (zeros((batch, vertices, 3)), zeros((batch, 2, 4, 4)), zeros((vertices, 2)))
        output = adapter.mesh.skin_lbs(*inputs)
    else:
        batch, landmarks = (0, 2) if name == 'bary_batch' else (1, 0)
        inputs = (zeros((batch, 3, 3)), zeros((landmarks, 3)))
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
        indices = torch.zeros(landmarks, dtype=torch.int32, device=device)
        output = adapter.mesh.evaluate_barycentric(inputs[0], faces, indices, inputs[1])
    gradients = torch.autograd.grad(output.sum(), inputs)
    for value, gradient in zip(inputs, gradients):
        assert gradient.shape == value.shape and gradient.dtype == dtype
        assert gradient.device == value.device
        torch.testing.assert_close(gradient, torch.full_like(value, expected_nonempty), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('name', CASES)
def test_animation_empty_reductions_initialize_shared_gradients(name, dtype):
    check_empty_reduction(name, dtype)
