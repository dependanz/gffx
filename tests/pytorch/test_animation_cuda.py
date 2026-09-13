"""Exact native animation backend parity; run only in a supervised CUDA worker."""
import pytest
import torch
import gffx.torch
import math

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA adapter")


def _case(name, dtype):
    generator = torch.Generator().manual_seed(940)
    def random(shape):
        return (torch.rand(shape, generator=generator, dtype=dtype) * .4 - .2).requires_grad_()
    if name == "blend_shapes":
        return gffx.torch.mesh.blend_shapes, (random((4, 3)), random((2, 4, 3)), random((2, 2)))
    if name == "rodrigues":
        return gffx.torch.transforms.rodrigues, (random((2, 3)),)
    if name == "forward_kinematics":
        return gffx.torch.skeleton.forward_kinematics, (random((2, 3, 4, 4)), torch.tensor([1, -1, 1], dtype=torch.int32))
    if name == "skin_lbs":
        return gffx.torch.mesh.skin_lbs, (random((2, 4, 3)), random((2, 3, 4, 4)), random((4, 3)))
    return gffx.torch.mesh.evaluate_barycentric, (random((2, 4, 3)), torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32), torch.tensor([0, 1, 0], dtype=torch.int32), random((3, 3)))


def _identical(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    left, right = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8)), "reference-mode bytes differ"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["blend_shapes", "rodrigues", "forward_kinematics", "skin_lbs", "evaluate_barycentric"])
def test_animation_cuda_forward_and_gradients_are_byte_identical(name, dtype):
    assert torch.cuda.is_available(), "CUDA qualification requires a real adapter"
    function, inputs = _case(name, dtype)
    expected = function(*inputs)
    device_inputs = tuple(value.detach().cuda().requires_grad_(value.requires_grad) for value in inputs)
    actual = function(*device_inputs)
    _identical(expected, actual)
    cotangent = torch.arange(expected.numel(), dtype=dtype).reshape(expected.shape) / 7 + .1
    expected_gradients = torch.autograd.grad(expected, tuple(v for v in inputs if v.requires_grad), cotangent)
    actual_gradients = torch.autograd.grad(actual, tuple(v for v in device_inputs if v.requires_grad), cotangent.cuda())
    for left, right in zip(expected_gradients, actual_gradients):
        _identical(left, right)
    _identical(actual, function(*device_inputs))


def test_animation_cuda_uses_the_callers_current_stream():
    assert torch.cuda.is_available(), "CUDA qualification requires a real adapter"
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        template = torch.ones((4, 3), device="cuda")
        deltas = torch.full((1, 4, 3), 2., device="cuda")
        weights = torch.full((1, 1), 3., device="cuda")
        actual = torch.ops.gffx.blend_shapes(template, deltas, weights)
        assert torch.cuda.current_stream() == stream
        event = torch.cuda.Event()
        event.record()
    event.synchronize()
    _identical(actual, torch.full((1, 4, 3), 7.))


def test_animation_cuda_blend_empty_basis_and_preallocated_output():
    template = torch.arange(12, dtype=torch.float64).reshape(4, 3).cuda()
    deltas = torch.empty((0, 4, 3), dtype=torch.float64, device="cuda")
    weights = torch.empty((2, 0), dtype=torch.float64, device="cuda")
    size = gffx.torch.stream.animation_workspace_size("blend_shapes", template, deltas, weights)
    workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
    out = torch.empty((2, 4, 3), dtype=torch.float64, device="cuda")
    pointer = out.data_ptr()
    for _ in range(3):
        gffx.torch.stream.blend_shapes_out(template, deltas, weights, out=out, workspace=workspace)
        assert out.data_ptr() == pointer
        _identical(out, template.cpu().unsqueeze(0).repeat(2, 1, 1))


@pytest.mark.parametrize("name", ["blend_shapes", "rodrigues", "forward_kinematics", "skin_lbs", "evaluate_barycentric"])
def test_animation_cuda_every_out_reuses_storage_on_caller_stream(name):
    from test_animation_stream import case
    function, cpu_inputs = case(name)
    expected = function(*cpu_inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = tuple(value.cuda() for value in cpu_inputs)
        size = gffx.torch.stream.animation_workspace_size(name, *inputs)
        workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
        out = torch.empty(expected.shape, dtype=expected.dtype, device="cuda")
        pointer = out.data_ptr()
        for _ in range(3):
            result = getattr(gffx.torch.stream, name + "_out")(*inputs, out=out, workspace=workspace)
            assert result is out and out.data_ptr() == pointer
            assert torch.cuda.current_stream() == stream
        event = torch.cuda.Event()
        event.record()
    event.synchronize()
    _identical(out, expected)


def test_animation_cuda_native_invalid_values_leave_output_unchanged():
    template = torch.ones((1, 3), device="cuda")
    deltas = torch.ones((1, 1, 3), device="cuda")
    weights = torch.full((1, 1), float("nan"), device="cuda")
    out = torch.full((1, 1, 3), 73., device="cuda")
    workspace = torch.empty(8, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="finite"):
        torch.ops.gffx.blend_shapes_out(template, deltas, weights, out, workspace)
    _identical(out, torch.full((1, 1, 3), 73.))


def test_animation_cuda_native_rejects_alias_and_small_workspace():
    template = torch.ones((1, 3), device="cuda")
    deltas = torch.ones((1, 1, 3), device="cuda")
    weights = torch.ones((1, 1), device="cuda")
    workspace = torch.empty(8, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="alias"):
        torch.ops.gffx.blend_shapes_out(template, deltas, weights, template.reshape(1, 1, 3), workspace)
    with pytest.raises(RuntimeError, match="scratch"):
        torch.ops.gffx.blend_shapes_out(template, deltas, weights, torch.empty((1, 1, 3), device="cuda"), workspace[:0])


@pytest.mark.parametrize("invalid", ["vertex", "face"])
def test_animation_cuda_barycentric_invalid_indices_leave_output_unchanged(invalid):
    vertices = torch.ones((1, 3, 3), device="cuda")
    faces = torch.tensor([[0, 1, 3 if invalid == "vertex" else 2]], dtype=torch.int32, device="cuda")
    indices = torch.tensor([1 if invalid == "face" else 0], dtype=torch.int32, device="cuda")
    coefficients = torch.ones((1, 3), device="cuda")
    out = torch.full((1, 1, 3), 73., device="cuda")
    workspace = torch.empty(8, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="range"):
        torch.ops.gffx.evaluate_barycentric_out(vertices, faces, indices, coefficients, out, workspace)
    _identical(out, torch.full((1, 1, 3), 73.))


def test_animation_cuda_skin_fourth_row_is_ignored_and_gradient_is_zero():
    vertices = torch.tensor([[[1., 2., 3.]]], dtype=torch.float64, device="cuda", requires_grad=True)
    transforms = torch.eye(4, dtype=torch.float64, device="cuda").reshape(1, 1, 4, 4)
    transforms[:, :, 3] = torch.tensor([7., -4., 2., 9.], device="cuda")
    transforms.requires_grad_()
    weights = torch.ones((1, 1), dtype=torch.float64, device="cuda", requires_grad=True)
    output = gffx.torch.mesh.skin_lbs(vertices, transforms, weights)
    gradients = torch.autograd.grad(output.sum(), transforms)[0]
    _identical(output, vertices)
    _identical(gradients[:, :, 3], torch.zeros((1, 1, 4), dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [0., .0625 * (1 - 2**-18), .0625, .0625 * (1 + 2**-18), math.pi, 2 * math.pi, 10000.])
def test_animation_cuda_rodrigues_special_angles_are_byte_identical(dtype, angle):
    axis = torch.tensor([[angle, 0., 0.], [0., -angle, 0.], [0., 0., angle]], dtype=dtype, requires_grad=True)
    device_axis = axis.detach().cuda().requires_grad_()
    expected = gffx.torch.transforms.rodrigues(axis)
    actual = gffx.torch.transforms.rodrigues(device_axis)
    _identical(expected, actual)
    cotangent = torch.arange(27, dtype=dtype).reshape(3, 3, 3) / 8
    expected_gradient = torch.autograd.grad(expected, axis, cotangent)[0]
    actual_gradient = torch.autograd.grad(actual, device_axis, cotangent.cuda())[0]
    _identical(expected_gradient, actual_gradient)


@pytest.mark.parametrize("parents", [[0], [-2], [1, 0], [2, -1]])
def test_animation_cuda_fk_invalid_hierarchy_leaves_output_unchanged(parents):
    local = torch.eye(4, device="cuda").reshape(1, 1, 4, 4).repeat(1, len(parents), 1, 1)
    topology = torch.tensor(parents, dtype=torch.int32, device="cuda")
    out = torch.full_like(local, 73.)
    size = gffx.torch.stream.animation_workspace_size("forward_kinematics", local, topology)
    workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="parent|cycle"):
        torch.ops.gffx.forward_kinematics_out(local, topology, out, workspace)
    _identical(out, torch.full_like(local, 73.))


def test_animation_cuda_full_rig_backward_reaches_every_parameter():
    from test_animation import _synthetic_chain
    generator = torch.Generator().manual_seed(930)
    values = (torch.randn((4, 3), generator=generator, dtype=torch.float64),
              torch.randn((2, 4, 3), generator=generator, dtype=torch.float64) * .1,
              torch.tensor([[.2, -.1]], dtype=torch.float64),
              torch.tensor([[[.05, .1, 0.], [0., .02, -.1]]], dtype=torch.float64))
    parameters = tuple(value.cuda().requires_grad_() for value in values)
    output = _synthetic_chain(*parameters)
    output.square().sum().backward()
    for parameter in parameters:
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool(parameter.grad.abs().sum() > 0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('name', ['blend_batch', 'blend_vertices', 'blend_basis', 'rotation', 'fk_batch', 'fk_joints',
                                'skin_batch', 'skin_vertices', 'bary_batch', 'bary_landmarks'])
def test_animation_cuda_empty_reductions_initialize_shared_gradients(name, dtype):
    from test_animation_empty_reductions import check_empty_reduction
    check_empty_reduction(name, dtype, 'cuda')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('case', ['finite', 'first', 'middle', 'last', 'empty'])
def test_animation_cuda_finite_validation_stays_bounded_on_device(monkeypatch, dtype, case):
    from test_animation_finite_chunks import check_chunked_finite, CHUNK
    index = {'first': 0, 'middle': CHUNK, 'last': CHUNK * 2}.get(case)
    check_chunked_finite(monkeypatch, dtype, 'cuda', bad_index=index, empty=case == 'empty')
