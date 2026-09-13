"""Explicit boundary examples complement randomized animation references."""
import math

import gffx
import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rodrigues_rotation_invariants_and_signed_axis_turns(dtype):
    vectors = []
    for axis in range(3):
        for sign in (-1, 1):
            value = [0., 0., 0.]
            value[axis] = sign * math.pi / 2
            vectors.append(value)
    vectors += [[math.pi - 1e-5, .01, -.02], [.2, -.3, .4], [0., 0., 2 * math.pi]]
    axis = torch.tensor(vectors, dtype=dtype)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    actual = gffx.torch.transforms.rodrigues(axis)
    tol = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(actual, torch.matrix_exp(skew), **tol)
    torch.testing.assert_close(actual.transpose(-1, -2) @ actual, torch.eye(3, dtype=dtype).expand_as(actual), **tol)
    torch.testing.assert_close(torch.linalg.det(actual), torch.ones(len(axis), dtype=dtype), **tol)


def test_rodrigues_zero_jacobian_is_exact_skew_generators():
    axis = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(gffx.torch.transforms.rodrigues, axis)
    expected = torch.zeros((3, 3, 3), dtype=axis.dtype)
    expected[1, 2, 0], expected[2, 1, 0] = -1, 1
    expected[0, 2, 1], expected[2, 0, 1] = 1, -1
    expected[0, 1, 2], expected[1, 0, 2] = -1, 1
    torch.testing.assert_close(jacobian, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rodrigues_threshold_gradients_match_independent_matrix_exponential(dtype):
    threshold = torch.tensor(.0625, dtype=dtype)
    angles = torch.stack((torch.nextafter(threshold, threshold.new_tensor(0)), threshold,
                          torch.nextafter(threshold, threshold.new_tensor(1))))
    axis = torch.zeros((3, 3), dtype=dtype)
    axis[:, 0] = angles
    axis.requires_grad_()
    independent = axis.detach().clone().requires_grad_()
    x, y, z = independent.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    cotangent = torch.arange(27, dtype=dtype).reshape(3, 3, 3) / 8
    actual = torch.autograd.grad(gffx.torch.transforms.rodrigues(axis), axis, cotangent)[0]
    expected = torch.autograd.grad(torch.matrix_exp(skew), independent, cotangent)[0]
    tol = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(actual, expected, **tol)


@pytest.mark.parametrize("name", ["blend_shapes", "rodrigues", "forward_kinematics", "skin_lbs", "evaluate_barycentric"])
def test_animation_empty_outputs_preserve_gradient_shapes(name):
    adapter = gffx.torch
    def zeros(shape):
        return torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    if name == "blend_shapes":
        inputs = (zeros((0, 3)), zeros((2, 0, 3)), zeros((1, 2)))
        output = adapter.mesh.blend_shapes(*inputs)
    elif name == "rodrigues":
        inputs = (zeros((0, 3)),)
        output = adapter.transforms.rodrigues(*inputs)
    elif name == "forward_kinematics":
        inputs = (zeros((1, 0, 4, 4)),)
        output = adapter.skeleton.forward_kinematics(*inputs, torch.empty(0, dtype=torch.int32))
    elif name == "skin_lbs":
        inputs = (zeros((1, 0, 3)), zeros((1, 2, 4, 4)), zeros((0, 2)))
        output = adapter.mesh.skin_lbs(*inputs)
    else:
        inputs = (zeros((1, 3, 3)), zeros((0, 3)))
        output = adapter.mesh.evaluate_barycentric(inputs[0], torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.empty(0, dtype=torch.int32), inputs[1])
    assert output.numel() == 0
    gradients = torch.autograd.grad(output.sum(), inputs)
    for value, gradient in zip(inputs, gradients):
        assert gradient.shape == value.shape and gradient.dtype == value.dtype
        assert torch.count_nonzero(gradient) == 0


def test_skin_lbs_signed_and_zero_influences_have_literal_meaning():
    vertices = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.float64)
    transforms = torch.eye(4, dtype=vertices.dtype).repeat(1, 2, 1, 1)
    transforms[0, 1, 0, 3] = 2
    weights = vertices.new_tensor([[2., -1.], [0., 0.]])
    output = gffx.torch.mesh.skin_lbs(vertices, transforms, weights)
    torch.testing.assert_close(output, vertices.new_tensor([[[-1., 2., 3.], [0., 0., 0.]]]), atol=0, rtol=0)


def test_barycentric_repeated_vertices_and_edge_extrapolation():
    vertices = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]], dtype=torch.float64, requires_grad=True)
    faces = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.int32)
    indices = torch.tensor([0, 1, 1], dtype=torch.int32)
    coefficients = vertices.new_tensor([[.5, .25, .25], [.5, .5, 0.], [-1., 2., 0.]])
    actual = gffx.torch.mesh.evaluate_barycentric(vertices, faces, indices, coefficients)
    expected = vertices.new_tensor([[[1., 0., 0.], [1., 0., 0.], [4., 0., 0.]]])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    torch.testing.assert_close(vertices.grad, vertices.new_tensor([[[0., 0., 0.], [3., 3., 3.], [0., 0., 0.]]]), atol=0, rtol=0)
