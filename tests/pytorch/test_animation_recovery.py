"""Small synthetic inverse problem with an independently generated target."""
import math

import gffx
import torch


def fixture():
    template = torch.tensor([[0., 0., 0.], [1., 0., .1], [0., 1., .2], [.2, .3, 1.]], dtype=torch.float64)
    basis = torch.zeros((2, 4, 3), dtype=template.dtype)
    basis[0, 1, 0], basis[1, 3, 2] = .5, .5
    skin_weights = template.new_tensor([[1., 0.], [.75, .25], [.25, .75], [0., 1.]])
    translations = template.new_tensor([[[[0.], [0.], [0.]], [[0.], [.5], [0.]]]])
    inverse_bind = torch.eye(4, dtype=template.dtype).repeat(1, 2, 1, 1)
    inverse_bind[0, 1, 1, 3] = -.5
    return template, basis, skin_weights, translations, inverse_bind


def public_landmarks(parameters):
    template, basis, skin_weights, translations, inverse_bind = fixture()
    adapter = gffx.torch
    vertices = adapter.mesh.blend_shapes(template, basis, parameters[:2].reshape(1, 2))
    zero = parameters[2] * 0
    pose = torch.stack((zero, zero, zero, zero, zero, parameters[2])).reshape(1, 2, 3)
    rotations = adapter.transforms.rodrigues(pose)
    last_row = template.new_tensor([0., 0., 0., 1.]).reshape(1, 1, 1, 4).repeat(1, 2, 1, 1)
    local = torch.cat((torch.cat((rotations, translations), -1), last_row), -2)
    world = adapter.skeleton.forward_kinematics(local, torch.tensor([-1, 0], dtype=torch.int32))
    skinned = adapter.mesh.skin_lbs(vertices, world @ inverse_bind, skin_weights)
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int32)
    indices = torch.tensor([0, 0, 0, 1], dtype=torch.int32)
    bary = template.new_tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
    return adapter.mesh.evaluate_barycentric(skinned, faces, indices, bary).reshape(-1)


def independent_target(parameters):
    """No production deformation operation is used to construct the target."""
    template, basis, weights, _, _ = fixture()
    vertices = template + parameters[0] * basis[0] + parameters[1] * basis[1]
    c, s = math.cos(float(parameters[2])), math.sin(float(parameters[2]))
    rotation = template.new_tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    pivot = template.new_tensor([0., .5, 0.])
    child = (vertices - pivot) @ rotation.T + pivot
    return (vertices * weights[:, :1] + child * weights[:, 1:]).reshape(-1)


def test_animation_recovers_independent_synthetic_target(record_property):
    """Recover two shape coefficients and one joint angle from a different initial state."""
    truth = torch.tensor([.12, -.08, .30], dtype=torch.float64)
    target = independent_target(truth)
    estimate = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    initial_loss = float((public_landmarks(estimate).detach() - target).square().sum())
    losses = []
    for _ in range(8):
        residual = public_landmarks(estimate) - target
        jacobian = torch.autograd.functional.jacobian(public_landmarks, estimate)
        step = torch.linalg.solve(jacobian.T @ jacobian + torch.eye(3, dtype=estimate.dtype) * 1e-9,
                                  jacobian.T @ residual.detach())
        estimate = (estimate.detach() - step).requires_grad_()
        losses.append(float((public_landmarks(estimate).detach() - target).square().sum()))
    error = float((estimate.detach() - truth).abs().max())
    record_property("initial_squared_landmark_error", initial_loss)
    record_property("final_squared_landmark_error", losses[-1])
    record_property("max_parameter_error", error)
    record_property("iterations", len(losses))
    assert initial_loss > 1e-3
    assert losses[-1] < 1e-16
    assert error < 1e-7
    assert all(after <= before + 1e-18 for before, after in zip([initial_loss] + losses[:-1], losses))
    torch.testing.assert_close(public_landmarks(truth), target, atol=2e-12, rtol=2e-11)
