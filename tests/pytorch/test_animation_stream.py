"""Caller-owned inference storage agrees with functional animation without hidden gradients."""
import pytest
import torch
import gffx


def case(name):
    v = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], dtype=torch.float64)
    if name == "blend_shapes":
        return gffx.torch.mesh.blend_shapes, (v[0], torch.ones((1, 3, 3), dtype=v.dtype), torch.ones((1, 1), dtype=v.dtype))
    if name == "rodrigues":
        return gffx.torch.transforms.rodrigues, (torch.zeros((1, 3), dtype=v.dtype),)
    if name == "forward_kinematics":
        return gffx.torch.skeleton.forward_kinematics, (torch.eye(4, dtype=v.dtype).repeat(1, 2, 1, 1), torch.tensor([-1, 0], dtype=torch.int32))
    if name == "skin_lbs":
        return gffx.torch.mesh.skin_lbs, (v, torch.eye(4, dtype=v.dtype).reshape(1, 1, 4, 4), torch.ones((3, 1), dtype=v.dtype))
    return gffx.torch.mesh.evaluate_barycentric, (v, torch.tensor([[0, 1, 2]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32), torch.tensor([[.2, .3, .5]], dtype=v.dtype))


@pytest.mark.parametrize("name", ["blend_shapes", "rodrigues", "forward_kinematics", "skin_lbs", "evaluate_barycentric"])
def test_animation_out_reuses_output_and_matches_functional(name):
    """Every preallocated variant returns the same values into the same caller storage."""
    function, inputs = case(name)
    expected = function(*inputs)
    stream = gffx.torch.stream
    size = stream.animation_workspace_size(name, *inputs)
    workspace = torch.empty(size, dtype=torch.uint8)
    out = torch.empty_like(expected)
    pointer = out.data_ptr()
    for _ in range(3):
        result = getattr(stream, name + "_out")(*inputs, out=out, workspace=workspace)
        assert result is out and out.data_ptr() == pointer
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_animation_out_rejects_gradient_tracked_inputs():
    """Inference storage must never silently discard a gradient requested by a caller."""
    _, inputs = case("blend_shapes")
    inputs[0].requires_grad_()
    with pytest.raises(ValueError, match="inference-only"):
        gffx.torch.stream.blend_shapes_out(*inputs, out=torch.empty((1, 3, 3), dtype=torch.float64),
                                          workspace=torch.empty(0, dtype=torch.uint8))


def test_animation_out_rejects_alias_and_small_workspace():
    """Inputs remain read-only and skeleton scratch must be sized by the query."""
    _, inputs = case("blend_shapes")
    with pytest.raises(ValueError, match="alias"):
        gffx.torch.stream.blend_shapes_out(*inputs, out=inputs[0].reshape(1, 3, 3),
                                          workspace=torch.empty(0, dtype=torch.uint8))
    _, inputs = case("forward_kinematics")
    with pytest.raises(RuntimeError, match="workspace"):
        gffx.torch.stream.forward_kinematics_out(*inputs, out=torch.empty_like(inputs[0]),
                                               workspace=torch.empty(0, dtype=torch.uint8))
