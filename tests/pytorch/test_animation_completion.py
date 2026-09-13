"""Focused acceptance gaps, separate from preserved strict-gradcheck evidence."""
import gffx
import pytest
import torch

NAMES = ("blend_shapes", "rodrigues", "forward_kinematics", "skin_lbs", "evaluate_barycentric")


def _same_bytes(a, b):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def check_repeated_backward(name, dtype, device):
    def values(shape):
        count = 1
        for n in shape:
            count *= n
        return ((torch.arange(count, dtype=dtype).reshape(shape).remainder(17) - 8) / 32)
    if name == "blend_shapes":
        function, inputs = gffx.torch.mesh.blend_shapes, (values((4, 3)), values((2, 4, 3)), values((2, 2)))
    elif name == "rodrigues":
        function, inputs = gffx.torch.transforms.rodrigues, (values((2, 3)),)
    elif name == "forward_kinematics":
        function, inputs = gffx.torch.skeleton.forward_kinematics, (values((2, 3, 4, 4)), torch.tensor([1, -1, 1], dtype=torch.int32))
    elif name == "skin_lbs":
        function, inputs = gffx.torch.mesh.skin_lbs, (values((2, 4, 3)), values((2, 3, 4, 4)), values((4, 3)))
    else:
        function, inputs = gffx.torch.mesh.evaluate_barycentric, (values((2, 4, 3)), torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.int32),
            torch.tensor([1, 0, 1], dtype=torch.int32), values((3, 3)))
    inputs = tuple(x.to(device).requires_grad_(x.is_floating_point()) for x in inputs)
    saved = tuple(x.detach().clone() for x in inputs)
    parameters = tuple(x for x in inputs if x.requires_grad)
    baseline = None
    for _ in range(3):
        output = function(*inputs)
        cotangent = (torch.arange(output.numel(), dtype=dtype).reshape(output.shape) / 8 - .5).to(device)
        gradients = torch.autograd.grad(output, parameters, cotangent)
        if baseline is None:
            baseline = tuple(x.detach().clone() for x in gradients)
        else:
            for actual, expected in zip(gradients, baseline):
                _same_bytes(actual, expected)
        for actual, expected in zip(inputs, saved):
            _same_bytes(actual, expected)


def check_inverse_bind_chain(device):
    template = torch.tensor([[0., 0., .1], [1., 0., .2], [0., 1., .3], [.2, .3, 1.]], dtype=torch.float64)
    basis = (torch.arange(24, dtype=template.dtype).reshape(2, 4, 3) - 10) / 128
    coefficients = template.new_tensor([[.125, -.25]])
    pose = template.new_tensor([[[.1, -.05, .02], [-.03, .08, .15]]])
    offsets = template.new_tensor([[[[.125], [0.], [0.]], [[0.], [.5], [0.]]]])
    inverse_bind = torch.eye(4, dtype=template.dtype).repeat(1, 2, 1, 1)
    inverse_bind[:, 0, 0, 3] = -.125
    inverse_bind[:, 1, 0, 3] = -.125
    inverse_bind[:, 1, 1, 3] = -.5
    weights = template.new_tensor([[1., 0.], [.75, .25], [.25, .75], [0., 1.]])
    bary = template.new_tensor([[.2, .3, .5], [.5, .25, .25]])
    source = (template, basis, coefficients, pose, offsets, inverse_bind, weights, bary)
    actual_inputs = tuple(x.to(device).detach().requires_grad_() for x in source)
    reference_inputs = tuple(x.clone().requires_grad_() for x in source)

    def chain(inputs, native):
        t, delta, coeff, r, offset, bind, w, bc = inputs
        parents = torch.tensor([-1, 0], dtype=torch.int32, device=t.device)
        faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32, device=t.device)
        indices = torch.tensor([0, 1], dtype=torch.int32, device=t.device)
        if native:
            vertices = gffx.torch.mesh.blend_shapes(t, delta, coeff)
            rotation = gffx.torch.transforms.rodrigues(r)
        else:
            vertices = t + torch.einsum("bk,kvc->bvc", coeff, delta)
            x, y, z = r.unbind(-1)
            zero = torch.zeros_like(x)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(1, 2, 3, 3)
            rotation = torch.matrix_exp(skew)
        bottom = t.new_tensor([0., 0., 0., 1.]).reshape(1, 1, 1, 4).repeat(1, 2, 1, 1)
        local = torch.cat((torch.cat((rotation, offset), -1), bottom), -2)
        world = (gffx.torch.skeleton.forward_kinematics(local, parents) if native else
                 torch.stack((local[:, 0], local[:, 0] @ local[:, 1]), 1))
        matrices = world @ bind
        if native:
            skinned = gffx.torch.mesh.skin_lbs(vertices, matrices, w)
            return gffx.torch.mesh.evaluate_barycentric(skinned, faces, indices, bc)
        homogeneous = torch.cat((vertices, torch.ones((1, 4, 1), dtype=t.dtype)), -1)
        skinned = torch.einsum("vj,bjca,bva->bvc", w, matrices[:, :, :3], homogeneous)
        return (skinned[:, faces[indices].long()] * bc[None, :, :, None]).sum(2)

    actual, expected = chain(actual_inputs, True), chain(reference_inputs, False)
    cotangent = template.new_tensor([[[.5, -.25, .75], [.125, .5, -.25]]])
    actual.backward(cotangent.to(device))
    expected.backward(cotangent)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach(), atol=2e-12, rtol=2e-11)
    for observed, reference in zip(actual_inputs, reference_inputs):
        assert observed.grad is not None and bool(torch.isfinite(observed.grad).all())
        assert bool(reference.grad.abs().sum() > 0)
        torch.testing.assert_close(observed.grad.cpu(), reference.grad, atol=2e-12, rtol=2e-11)


def check_overflow(name, dtype, device):
    from test_animation_stream import case
    function, original = case(name)
    inputs = [x.to(device=device, dtype=dtype if x.is_floating_point() else x.dtype) for x in original]
    maximum = torch.finfo(dtype).max
    if name == "blend_shapes":
        inputs[1].fill_(maximum); inputs[2].fill_(2)
    elif name == "rodrigues":
        inputs[0].fill_(maximum)
    elif name == "forward_kinematics":
        inputs[0].fill_(maximum)
    elif name == "skin_lbs":
        inputs[0].fill_(maximum); inputs[1].fill_(2)
    else:
        inputs[0].fill_(maximum); inputs[3].fill_(2)
    saved = tuple(x.clone() for x in inputs)
    with pytest.raises(OverflowError):
        function(*inputs)
    # Arithmetic failure promises an explicit error, not an unchanged output.
    for actual, expected in zip(inputs, saved):
        _same_bytes(actual, expected)


def check_storage_errors(name, device):
    from test_animation_stream import case
    function, original = case(name)
    inputs = [x.to(device) for x in original]
    expected = function(*inputs)
    size = gffx.torch.stream.animation_workspace_size(name, *inputs)
    workspace = torch.empty(size, dtype=torch.uint8, device=device)
    out = torch.full_like(expected, 73)
    caller = getattr(gffx.torch.stream, name + "_out")
    # A larger allocation lets even Rodrigues' larger output overlap its input.
    backing = torch.zeros(max(inputs[0].numel(), out.numel()), dtype=out.dtype, device=device)
    backing[:inputs[0].numel()].copy_(inputs[0].reshape(-1))
    inputs[0] = backing[:inputs[0].numel()].reshape(inputs[0].shape)
    saved = tuple(x.clone() for x in inputs)
    alias = backing[:out.numel()].reshape(out.shape)
    with pytest.raises(ValueError, match="alias"):
        caller(*inputs, out=alias, workspace=workspace)
    for actual, before in zip(inputs, saved):
        _same_bytes(actual, before)
    if size:
        with pytest.raises(RuntimeError, match="workspace|scratch"):
            caller(*inputs, out=out, workspace=workspace[:-1])
        _same_bytes(out, torch.full_like(out, 73))
    inputs[0].requires_grad_()
    with pytest.raises(ValueError, match="inference-only"):
        caller(*inputs, out=out, workspace=workspace)
    _same_bytes(out, torch.full_like(out, 73))
    inputs[0] = inputs[0].detach()
    out.requires_grad_()
    with pytest.raises(ValueError, match="inference-only"):
        caller(*inputs, out=out, workspace=workspace)
    _same_bytes(out, torch.full_like(out, 73))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_animation_repeated_backward_is_byte_stable(name, dtype):
    check_repeated_backward(name, dtype, "cpu")


def test_animation_nonidentity_inverse_bind_chain_all_inputs():
    check_inverse_bind_chain("cpu")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_animation_finite_arithmetic_overflow_is_explicit(name, dtype):
    check_overflow(name, dtype, "cpu")


@pytest.mark.parametrize("name", NAMES)
def test_animation_each_out_rejects_alias_scratch_and_gradients(name):
    check_storage_errors(name, "cpu")
