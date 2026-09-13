"""Independent tensor VJPs for shared skeleton, skin and landmark inputs.

These small CPU oracles use no gffx operation to construct the expected result.
They complement exact CPU/CUDA parity, which alone could share the same bug.
"""
import gffx
import pytest
import torch


def _check_vjp(actual, expected, inputs, independent_inputs):
    tolerance = (dict(atol=2e-6, rtol=2e-5) if actual.dtype == torch.float32
                 else dict(atol=2e-12, rtol=2e-11))
    # Dyadic values avoid device-generated cotangent rounding differences.
    cotangent = (torch.arange(actual.numel(), dtype=actual.dtype)
                 .reshape(actual.shape).remainder(13) - 6) / 8
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    expected_gradients = torch.autograd.grad(expected, independent_inputs, cotangent)
    torch.testing.assert_close(actual, expected, **tolerance)
    for observed, reference in zip(actual_gradients, expected_gradients):
        assert bool(torch.isfinite(observed).all())
        assert bool(reference.abs().sum() > 0)
        torch.testing.assert_close(observed, reference, **tolerance)
    return actual_gradients, expected_gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fk_branching_shared_ancestor_vjp_matches_independent_matmul(dtype):
    # Joint1 is the root of two children, deliberately not index-topological.
    local = ((torch.arange(96, dtype=dtype).reshape(2, 3, 4, 4)
              .remainder(17) - 8) / 32 + torch.eye(4, dtype=dtype))
    local.requires_grad_()
    independent = local.detach().clone().requires_grad_()
    parents = torch.tensor([1, -1, 1], dtype=torch.int32)
    actual = gffx.torch.skeleton.forward_kinematics(local, parents)
    expected = torch.stack((independent[:, 1] @ independent[:, 0],
                            independent[:, 1],
                            independent[:, 1] @ independent[:, 2]), dim=1)
    observed, reference = _check_vjp(actual, expected, (local,), (independent,))
    # Check the shared root explicitly, including the general fourth row.
    assert bool(reference[0][:, 1, 3].abs().sum() > 0)
    assert bool(observed[0][:, 1].abs().sum() > 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lbs_multibatch_shared_weight_vjp_matches_independent_einsum(dtype):
    vertices = ((torch.arange(24, dtype=dtype).reshape(2, 4, 3) - 9) / 8).requires_grad_()
    transforms = ((torch.arange(96, dtype=dtype).reshape(2, 3, 4, 4)
                   .remainder(19) - 9) / 16).requires_grad_()
    weights = torch.tensor([[1., 0., 0.], [.25, .5, .25],
                            [-.5, 1., .25], [0., 0., 0.]], dtype=dtype, requires_grad=True)
    inputs = (vertices, transforms, weights)
    v, t, w = independent = tuple(x.detach().clone().requires_grad_() for x in inputs)
    homogeneous = torch.cat((v, torch.ones((2, 4, 1), dtype=dtype)), dim=-1)
    per_joint = torch.einsum("bjrc,bvc->bvjr", t[:, :, :3, :], homogeneous)
    expected = torch.einsum("vj,bvjr->bvr", w, per_joint)
    actual = gffx.torch.mesh.skin_lbs(*inputs)
    observed, reference = _check_vjp(actual, expected, inputs, independent)
    assert torch.count_nonzero(observed[1][:, :, 3, :]) == 0
    # A zero influence row still has a learnable weight derivative.
    assert bool(reference[2][3].abs().sum() > 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_barycentric_multibatch_shared_coefficients_vjp_matches_independent_indexing(dtype):
    vertices = ((torch.arange(24, dtype=dtype).reshape(2, 4, 3) - 7) / 8).requires_grad_()
    faces = torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.int32)
    indices = torch.tensor([1, 0, 1], dtype=torch.int32)
    coefficients = torch.tensor([[.5, .25, .25], [1.5, -.5, 0.],
                                 [-.25, .5, 1.]], dtype=dtype, requires_grad=True)
    inputs = (vertices, coefficients)
    v, c = independent = tuple(x.detach().clone().requires_grad_() for x in inputs)
    selected = faces.to(torch.int64)[indices.to(torch.int64)]
    expected = (v[:, selected, :] * c[None, :, :, None]).sum(dim=2)
    actual = gffx.torch.mesh.evaluate_barycentric(vertices, faces, indices, coefficients)
    _check_vjp(actual, expected, inputs, independent)
