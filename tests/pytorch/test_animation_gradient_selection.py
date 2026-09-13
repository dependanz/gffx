"""Selective gradients prevent unused corrective/skin tensors from becoming allocations."""
import torch
import gffx
import gffx.torch  # Explicit adapter activation before direct torch.ops schema probes.


def test_native_blend_backward_allocates_only_requested_gradient():
    """A coefficient-only fit must not allocate template or corrective gradients."""
    t = torch.zeros((2, 3), dtype=torch.float64)
    d = torch.ones((1, 2, 3), dtype=t.dtype)
    w = torch.ones((1, 1), dtype=t.dtype)
    g = torch.ones((1, 2, 3), dtype=t.dtype)
    gt, gd, gw = torch.ops.gffx.blend_shapes_backward(t, d, w, g, False, False, True)
    assert gt.numel() == 0 and gd.numel() == 0
    torch.testing.assert_close(gw, w.new_tensor([[6.]]))


def test_native_skin_backward_allocates_only_requested_gradient():
    """A pose-only fit needs matrix gradients, not input vertices or skinning weights."""
    v = torch.ones((1, 2, 3), dtype=torch.float64)
    t = torch.eye(4, dtype=v.dtype).reshape(1, 1, 4, 4)
    w = torch.ones((2, 1), dtype=v.dtype)
    gv, gt, gw = torch.ops.gffx.skin_lbs_backward(v, t, w, torch.ones_like(v), False, True, False)
    assert gv.numel() == 0 and gw.numel() == 0
    torch.testing.assert_close(gt[0, 0, :3], torch.full((3, 4), 2., dtype=v.dtype))
    assert torch.count_nonzero(gt[..., 3, :]) == 0


def test_native_barycentric_backward_omits_unrequested_embedding_gradient():
    """Fixed landmark embeddings should remain fixed storage during geometry fitting."""
    v = torch.ones((1, 3, 3), dtype=torch.float64)
    f = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    index = torch.tensor([0], dtype=torch.int32)
    w = torch.tensor([[.2, .3, .5]], dtype=v.dtype)
    gv, gw = torch.ops.gffx.evaluate_barycentric_backward(v, f, index, w,
        torch.ones((1, 1, 3), dtype=v.dtype), True, False)
    assert gw.numel() == 0
    torch.testing.assert_close(gv[0, :, 0], w[0])


def test_coefficient_only_autograd_preserves_other_inputs():
    """The regular public call propagates only the requested input derivative."""
    t = torch.zeros((2, 3), dtype=torch.float64)
    d = torch.ones((1, 2, 3), dtype=t.dtype)
    w = torch.ones((1, 1), dtype=t.dtype, requires_grad=True)
    gffx.torch.mesh.blend_shapes(t, d, w).sum().backward()
    torch.testing.assert_close(w.grad, torch.full_like(w, 6.))
    assert t.grad is None and d.grad is None
