"""AN-01..10: native deformation equations, domains and independent gradients."""
import pytest
import torch

import gffx


def test_animation_public_symbols():
    """Every required operation is reachable on both lazy and explicit surfaces."""
    for module, names in (("mesh", ("blend_shapes", "skin_lbs", "evaluate_barycentric")),
                          ("transforms", ("rodrigues",)),
                          ("skeleton", ("forward_kinematics",))):
        for name in names:
            assert callable(getattr(getattr(gffx.torch, module), name))
            assert callable(getattr(getattr(gffx, module), name))


def test_blend_shapes_closed_form():
    """A displacement is added to the template; coefficients need not be positive."""
    template = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    deltas = torch.tensor([[[2., 0., -1.]]], dtype=torch.float64)
    weights = torch.tensor([[1.], [-2.], [0.]], dtype=torch.float64)
    actual = gffx.torch.mesh.blend_shapes(template, deltas, weights)
    torch.testing.assert_close(actual, template.unsqueeze(0) + weights[:, :, None] * deltas)


def test_blend_shapes_all_gradients():
    """Template, basis displacements and each frame's coefficients all learn."""
    torch.manual_seed(910)
    inputs = tuple(torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                   for shape in ((3, 3), (2, 3, 3), (2, 2)))
    assert torch.autograd.gradcheck(gffx.torch.mesh.blend_shapes, inputs,
                                   eps=1e-6, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(gffx.torch.mesh.blend_shapes(*inputs),
                               inputs[0] + torch.einsum("bk,kvc->bvc", inputs[2], inputs[1]))


def test_rodrigues_zero_and_quarter_turn():
    """The zero rotation is identity; positive z turns x toward y."""
    r = torch.tensor([[0., 0., 0.], [0., 0., torch.pi / 2]], dtype=torch.float64,
                     requires_grad=True)
    actual = gffx.torch.transforms.rodrigues(r)
    torch.testing.assert_close(actual[0], torch.eye(3, dtype=r.dtype), atol=0, rtol=0)
    torch.testing.assert_close(actual[1, :, 0], r.new_tensor([0., 1., 0.]), atol=2e-12, rtol=0)
    assert torch.autograd.gradcheck(gffx.torch.transforms.rodrigues, (r,),
                                   eps=1e-6, atol=2e-5, rtol=2e-4)


def test_forward_kinematics_scrambled_forest():
    """Joint order is not hierarchy order; separate roots remain independent."""
    local = torch.eye(4, dtype=torch.float64).repeat(1, 3, 1, 1)
    local[0, 0, 0, 3] = 2
    local[0, 1, 1, 3] = 3
    local[0, 2, 2, 3] = 4
    parents = torch.tensor([1, -1, -1], dtype=torch.int32)
    actual = gffx.torch.skeleton.forward_kinematics(local, parents)
    torch.testing.assert_close(actual[0, 0], local[0, 1] @ local[0, 0])
    torch.testing.assert_close(actual[0, 1:], local[0, 1:])
    assert torch.autograd.gradcheck(lambda x: gffx.torch.skeleton.forward_kinematics(x, parents),
                                   (local.requires_grad_(),), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_skin_lbs_one_joint_and_all_gradients():
    """One-hot attachment follows its joint, including learned transform and weight."""
    vertices = torch.tensor([[[1., 2., 3.]]], dtype=torch.float64, requires_grad=True)
    transforms = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
    transforms[0, 0, 0, 3] = 5
    transforms.requires_grad_()
    weights = torch.ones((1, 1), dtype=torch.float64, requires_grad=True)
    actual = gffx.torch.mesh.skin_lbs(vertices, transforms, weights)
    torch.testing.assert_close(actual, vertices + vertices.new_tensor([5., 0., 0.]))
    assert torch.autograd.gradcheck(gffx.torch.mesh.skin_lbs, (vertices, transforms, weights),
                                   eps=1e-6, atol=2e-5, rtol=2e-4)


def test_barycentric_centroid_and_gradients():
    """A landmark's three coefficients and repeated vertex contributions retain gradients."""
    vertices = torch.tensor([[[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]]], dtype=torch.float64,
                            requires_grad=True)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    indices = torch.tensor([0, 0], dtype=torch.int32)
    bary = torch.tensor([[1/3, 1/3, 1/3], [1., -1., 1.]], dtype=torch.float64, requires_grad=True)
    function = lambda v, w: gffx.torch.mesh.evaluate_barycentric(v, faces, indices, w)
    actual = function(vertices, bary)
    torch.testing.assert_close(actual[0, 0], vertices.new_tensor([1., 1., 0.]))
    assert torch.autograd.gradcheck(function, (vertices, bary), eps=1e-6, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("parents", [[0], [1, 0], [2, -1], [-2]])
def test_forward_kinematics_invalid_hierarchy(parents):
    """Cycles and invalid references fail instead of hanging or reading outside storage."""
    local = torch.eye(4, dtype=torch.float64).repeat(1, len(parents), 1, 1)
    with pytest.raises(ValueError):
        gffx.torch.skeleton.forward_kinematics(local, torch.tensor(parents, dtype=torch.int32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rodrigues_threshold_matrix_exponential(dtype):
    """Both sides of the small-angle switch approximate an independent matrix exponential."""
    threshold = torch.tensor(0.0625, dtype=dtype)
    angles = torch.stack([torch.nextafter(threshold, threshold.new_tensor(0)), threshold,
                          torch.nextafter(threshold, threshold.new_tensor(1))])
    r = torch.zeros((3, 3), dtype=dtype)
    r[:, 0] = angles
    skew = torch.zeros((3, 3, 3), dtype=dtype)
    skew[:, 1, 2] = -angles
    skew[:, 2, 1] = angles
    tolerance = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(gffx.torch.transforms.rodrigues(r), torch.matrix_exp(skew), **tolerance)


def test_blend_shapes_empty_basis_and_empty_batch():
    """An empty corrective basis is identity; an empty batch remains empty."""
    template = torch.ones((2, 3), dtype=torch.float64)
    basis = torch.empty((0, 2, 3), dtype=template.dtype)
    torch.testing.assert_close(gffx.torch.mesh.blend_shapes(template, basis, torch.empty((2, 0), dtype=template.dtype)),
                               template.unsqueeze(0).repeat(2, 1, 1))
    actual = gffx.torch.mesh.blend_shapes(template, basis, torch.empty((0, 0), dtype=template.dtype))
    assert actual.shape == (0, 2, 3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_blend_shapes_nonfinite_rejected(bad):
    """Invalid coefficients cannot silently poison later geometry or rendering."""
    with pytest.raises(ValueError):
        gffx.torch.mesh.blend_shapes(torch.zeros((1, 3)), torch.zeros((1, 1, 3)), torch.tensor([[bad]]))


def test_skin_lbs_bottom_row_is_not_perspective_division():
    """Skinning transforms points affinely; its last matrix row has zero influence."""
    vertices = torch.ones((1, 1, 3), dtype=torch.float64)
    matrices = torch.eye(4, dtype=vertices.dtype).reshape(1, 1, 4, 4)
    matrices[0, 0, 3] = matrices.new_tensor([1., 2., 3., 4.])
    matrices.requires_grad_()
    actual = gffx.torch.mesh.skin_lbs(vertices, matrices, torch.ones((1, 1), dtype=vertices.dtype))
    torch.testing.assert_close(actual, vertices)
    actual.sum().backward()
    assert torch.count_nonzero(matrices.grad[..., 3, :]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_animation_random_independent_references(dtype):
    """Bounded random cases compare different expressions, not the same kernel twice."""
    generator = torch.Generator().manual_seed(920)
    random = lambda shape: torch.rand(shape, generator=generator, dtype=dtype) * 2 - 1
    vertices, matrices, weights = random((2, 4, 3)), random((2, 3, 4, 4)), random((4, 3))
    homogeneous = torch.cat((vertices, torch.ones((2, 4, 1), dtype=dtype)), -1)
    expected = torch.einsum("vj,bjca,bva->bvc", weights, matrices[:, :, :3], homogeneous)
    tolerance = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(gffx.torch.mesh.skin_lbs(vertices, matrices, weights), expected, **tolerance)
    rotation = random((5, 3))
    x, y, z = rotation.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    torch.testing.assert_close(gffx.torch.transforms.rodrigues(rotation), torch.matrix_exp(skew), **tolerance)
    template, delta, coefficients = random((4, 3)), random((2, 4, 3)), random((2, 2))
    torch.testing.assert_close(gffx.torch.mesh.blend_shapes(template, delta, coefficients),
        template + torch.einsum("bk,kvc->bvc", coefficients, delta), **tolerance)
    parents = torch.tensor([-1, 0, 0], dtype=torch.int32)
    world = gffx.torch.skeleton.forward_kinematics(matrices, parents)
    expected_world = torch.stack((matrices[:, 0], matrices[:, 0] @ matrices[:, 1],
                                  matrices[:, 0] @ matrices[:, 2]), 1)
    torch.testing.assert_close(world, expected_world, **tolerance)
    faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32)
    indices = torch.tensor([1, 0, 1], dtype=torch.int32)
    bary = random((3, 3))
    expected_landmarks = (vertices[:, faces[indices].long()] * bary[None, :, :, None]).sum(2)
    torch.testing.assert_close(gffx.torch.mesh.evaluate_barycentric(vertices, faces, indices, bary),
                               expected_landmarks, **tolerance)


def test_forward_kinematics_deep_nonrecursive_hierarchy():
    """A long skeleton does not consume a Python or native recursion stack."""
    count = 2048
    local = torch.eye(4, dtype=torch.float64).repeat(1, count, 1, 1)
    local[:, :, 0, 3] = 0.125
    parents = torch.arange(-1, count-1, dtype=torch.int32)
    world = gffx.torch.skeleton.forward_kinematics(local, parents)
    assert world[0, -1, 0, 3] == count * 0.125


def test_rodrigues_arbitrary_leading_dimensions():
    """The adapter preserves leading tensor dimensions beyond its original rank-six arena."""
    axis_angle = torch.zeros((1, 1, 1, 1, 1, 1, 3), dtype=torch.float64)
    actual = gffx.torch.transforms.rodrigues(axis_angle)
    assert actual.shape == (1, 1, 1, 1, 1, 1, 3, 3)
    torch.testing.assert_close(actual.reshape(3, 3), torch.eye(3, dtype=actual.dtype))


def _synthetic_chain(template, deltas, coefficients, pose):
    posed = gffx.torch.mesh.blend_shapes(template, deltas, coefficients)
    rotation = gffx.torch.transforms.rodrigues(pose)
    local = torch.cat((torch.cat((rotation, rotation.new_zeros((1, 2, 3, 1))), -1),
                       rotation.new_tensor([0., 0., 0., 1.]).reshape(1, 1, 1, 4).repeat(1, 2, 1, 1)), -2)
    world = gffx.torch.skeleton.forward_kinematics(local, torch.tensor([-1, 0], dtype=torch.int32, device=template.device))
    weights = template.new_tensor([[1., 0.], [.75, .25], [.25, .75], [0., 1.]])
    skin = gffx.torch.mesh.skin_lbs(posed, world, weights)
    return gffx.torch.mesh.evaluate_barycentric(skin, torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32, device=template.device),
                                               torch.tensor([0, 1], dtype=torch.int32, device=template.device),
                                               template.new_tensor([[.2, .3, .5], [.5, .3, .2]]))


def test_animation_complete_chain_gradients():
    """One landmark loss reaches pose, template, corrective shapes and coefficients."""
    torch.manual_seed(930)
    template = torch.randn((4, 3), dtype=torch.float64, requires_grad=True)
    deltas = (torch.randn((2, 4, 3), dtype=torch.float64) * .1).requires_grad_()
    coefficients = torch.tensor([[.2, -.1]], dtype=torch.float64, requires_grad=True)
    pose = torch.tensor([[[.05, .1, 0.], [0., .02, -.1]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_synthetic_chain, (template, deltas, coefficients, pose),
                                   eps=1e-6, atol=2e-5, rtol=2e-4)
    _synthetic_chain(template, deltas, coefficients, pose).square().sum().backward()
    for parameter in (template, deltas, coefficients, pose):
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool(parameter.grad.abs().sum() > 0)


@pytest.mark.parametrize("error", ["dtype", "layout", "shape"])
def test_animation_rejects_implicit_conversion(error):
    """Incorrect inputs require an explicit caller repair, not silent copying or casting."""
    template = torch.ones((2, 3), dtype=torch.float64)
    deltas = torch.zeros((1, 2, 3), dtype=template.dtype)
    weights = torch.ones((1, 1), dtype=template.dtype)
    if error == "dtype":
        weights = weights.float()
    elif error == "layout":
        template = torch.ones((3, 2), dtype=template.dtype).T
    else:
        deltas = torch.zeros((1, 3, 3), dtype=template.dtype)
    with pytest.raises(TypeError if error == "dtype" else ValueError):
        gffx.torch.mesh.blend_shapes(template, deltas, weights)
