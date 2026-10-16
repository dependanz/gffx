"""Portable camera inverse VJP oracle: exact rational arithmetic and fixed bytes."""
from fractions import Fraction as Q
import pytest
import torch


def ordered(left, right):
    terms = [left[..., :, k, None]*right[..., None, k, :] for k in range(4)]
    return (terms[0]+terms[1])+(terms[2]+terms[3])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('strided', [False, True])
def test_camera_inverse_vjp_exact_rational_bound_and_order(dtype, strided):
    from gffx.torch.camera import _Backend
    # Well-conditioned, non-diagonal, asymmetric synthetic input; no private
    # photograph/model/campaign data or optional NumPy dependency is involved.
    pose = (torch.eye(4, dtype=dtype)*2 + torch.arange(16, dtype=dtype).reshape(4, 4)/100).unsqueeze(0).requires_grad_()
    for channel in range(4):
        cotangent = ((torch.arange(16, dtype=dtype).reshape(1, 4, 4)+channel*3)/32)-.2
        if strided:
            packed = torch.zeros((1, 4, 8), dtype=dtype)
            packed[..., ::2] = cotangent
            cotangent = packed[..., ::2]
            assert not cotangent.is_contiguous()
        inverse = _Backend.inverse(pose)
        actual = torch.autograd.grad(inverse, pose, cotangent)[0]
        transpose = inverse.detach().transpose(-1, -2)
        expected = -ordered(ordered(transpose, cotangent), transpose)
        a, b, got = transpose.tolist()[0], cotangent.tolist()[0], actual.tolist()[0]
        for i in range(4):
            for j in range(4):
                exact = -sum((Q(a[i][k])*Q(b[k][l])*Q(a[l][j]) for k in range(4) for l in range(4)), Q(0))
                error = abs(Q(got[i][j])-exact)
                limit = Q(1, 10**10)+Q(1, 10**9)*abs(exact) if dtype == torch.float64 else Q(8)*Q(float(torch.finfo(dtype).eps))*max(Q(1), abs(exact))
                assert error <= limit
        left = actual.contiguous().reshape(-1).view(torch.uint8)
        right = expected.contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(left, right), 'inverse backward accumulation differs'


def test_camera_inverse_independent_derivative():
    from gffx.torch.camera import _Backend
    pose = (torch.eye(4, dtype=torch.float64)*2 + torch.arange(16, dtype=torch.float64).reshape(4, 4)/100).unsqueeze(0).requires_grad_()
    assert torch.autograd.gradcheck(_Backend.inverse, (pose,))
