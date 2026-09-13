"""Final named Phase-0 edge groups; public APIs remain deliberately required."""
import importlib
import math

import pytest
import torch

from test_material_public_contract import digest, envelope


def t(x, grad=False, dtype=torch.float64):
    return torch.tensor(x, dtype=dtype, requires_grad=grad)


def data(turn=False):
    e = envelope()
    cs = [0., .1, .3, .6, 1.]
    pairs = [[.9, 0.]] + [[.7 + .02*math.log(c), .2 + .01*math.log(c)] for c in cs[1:]]
    if turn:
        pairs = [[.9, 0.], [.6, .2], [.8, .3], [.8, .3], [.7, .2]]
    e['payload'] = dict(roughness_nodes=[.001, .1, 1.], cosine_nodes=cs, losses=[pairs]*3)
    e['payload_sha256'] = digest(e['payload'])
    return e


def resource(turn=False):
    m = importlib.import_module('gffx.torch.material')
    return m.PrincipledBudget.from_envelope(data(turn), device='cpu', dtype=torch.float64, allow_unverified=True)


@pytest.mark.parametrize('cosine', [.1, .3, .6])
def test_public_resource_knot_uses_declared_right_derivative(cosine):
    r = resource()
    c = t(cosine, True)
    value = r.directional_loss(c, t(.2), t(.4))
    expected = .6*(.7+.02*math.log(cosine)) + .4*(.2+.01*math.log(cosine))
    assert value.item() == pytest.approx(expected, abs=1e-12)
    derivative, = torch.autograd.grad(value, (c,))
    # Includes the C0 horizon/log junction: its right derivative is required.
    assert derivative.item() == pytest.approx(.016/cosine, abs=1e-11)


@pytest.mark.parametrize('cosine', [.3, .45, .6])
def test_public_resource_plateau_limiter_keeps_zero_slope(cosine):
    r = resource(True)
    c = t(cosine, True)
    value = r.directional_loss(c, t(.2), t(.4))
    assert value.item() == pytest.approx(.6*.8+.4*.3, abs=1e-12)
    assert torch.autograd.grad(value, c)[0].item() == pytest.approx(0., abs=1e-12)


def test_workspace_rejection_precedes_query_tensor_allocation(monkeypatch):
    r = resource()
    c, rough, f = t([.2, .4], True), t(.2, True), t(.4, True)
    limit = r.workspace(2, backward=True) - 1
    assert limit >= 0
    def forbidden(*args, **kwargs):
        pytest.fail('allocation before workspace rejection')
    with monkeypatch.context() as patch:
        for name in ('empty', 'zeros', 'ones', 'tensor', 'as_tensor', 'stack', 'cat'):
            patch.setattr(torch, name, forbidden)
        with pytest.raises(ValueError):
            r.directional_loss(c, rough, f, workspace_limit_bytes=limit)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_normal_incidence_roundoff_keeps_value_and_tangent_gradients(dtype):
    m = importlib.import_module('gffx.torch.material')
    n, v, l = (t([[1., 1., 1.]], True, dtype) for _ in range(3))
    a, s, e = t([.2, .4, .6], True, dtype), t(.1, True, dtype), t(2., True, dtype)
    out = m.classic(n, v, l, albedo=a, specular=s, exponent=e)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(out[0, 0], a/math.pi+s*10/(8*math.pi), atol=tolerance, rtol=tolerance)
    for g in torch.autograd.grad(out.sum(), (n, v, l)):
        torch.testing.assert_close(g, torch.zeros_like(g), atol=tolerance, rtol=0)


@pytest.mark.parametrize('cosine', [-1e-8, 1.+1e-8])
def test_roundoff_policy_does_not_clamp_supplied_lobe_cosines(cosine):
    m = importlib.import_module('gffx.torch.shade')
    with pytest.raises(ValueError):
        m.fresnel_schlick(t(cosine), t(.04))


@pytest.mark.parametrize('measure', ['delta', 'solid_angle'])
def test_custom_light_batched_strided_integration_gradcheck(measure):
    m = importlib.import_module('gffx.torch.light')
    points = t([[[.1,.2,.3,9.],[.2,.3,.4,9.]],[[.3,.4,.5,9.],[.4,.5,.6,9.]]])[..., :3].requires_grad_()
    normals = t([[[.1,.2,1.],[.2,.1,1.]],[[.2,.3,1.],[.3,.2,1.]]], True)
    power = t(.3, True)
    def callback(p, *, parameters):
        weight = parameters['power'] * torch.ones_like(p[..., None, :])
        packet = dict(direction=p[..., None, :] + t([0.,0.,2.]), valid=torch.ones((*p.shape[:-1],1), dtype=torch.bool))
        if measure == 'delta':
            packet['irradiance'] = weight
        else:
            packet.update(radiance=weight, solid_angle=torch.ones((*p.shape[:-1],1), dtype=p.dtype)*.4)
        return packet
    def evaluate(p, n, w):
        packet = m.custom(callback, p, parameters={'power':w}, measure=measure)
        return m.integrate(n, torch.ones((*p.shape[:-1],1,3), dtype=p.dtype)*.2, packet)
    assert not points.is_contiguous()
    assert torch.autograd.gradcheck(evaluate, (points, normals, power), eps=1e-6, atol=1e-5, rtol=1e-3)
    actual = evaluate(points, normals, power)
    expected = torch.stack([evaluate(points[i], normals[i], power) for i in range(2)])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
