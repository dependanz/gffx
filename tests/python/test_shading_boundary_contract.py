"""Consolidated boundary red tests; synthetic resource algebra is not physics."""
import importlib
import math

import pytest
import torch

from test_material_public_contract import envelope


def api(kind):
    return importlib.import_module('gffx.torch.' + kind)


def t(value, grad=False, dtype=torch.float64):
    return torch.tensor(value, dtype=dtype, requires_grad=grad)


def budget(module):
    return module.PrincipledBudget.from_envelope(
        envelope(), device='cpu', dtype=torch.float64, allow_unverified=True)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('bad', ['number', 'nan', 'point_shape'])
def test_custom_parameters_validate_before_inactive_callback(bad):
    """An inactive surface must not conceal an invalid shared parameter."""
    module = api('material')
    n = t([[0., 0., 1.], [0., 0., 1.]])
    value = 1. if bad == 'number' else t(float('nan')) if bad == 'nan' else t([[1.], [2.], [3.]])
    def forbidden(*a, **k): pytest.fail('inactive callback invoked')
    with pytest.raises(TypeError if bad == 'number' else ValueError):
        module.custom(forbidden, n, n, -n[:1], parameters={'value': value})


def test_two_sided_batched_callback_order_and_parameter_gradients():
    """Pack active B/P/Q pairs in order and flip normals without losing field gradients."""
    module = api('material')
    n = t([0., 0., 1.]).expand(2, 2, 3)
    v = t([[0., 0., 1.], [0., 0., -1.]])
    fields = t([[[1.], [2.]], [[3.], [4.]]], True)
    seen = []
    def callback(n, v, l, *, parameters):
        seen.append(n.shape)
        torch.testing.assert_close(n[:, 2], t([1., -1., 1., -1.]))
        torch.testing.assert_close(parameters['value'][:, 0], t([1., 2., 3., 4.]))
        return parameters['value'].expand(-1, 3)
    out = module.custom(callback, n, v, v, parameters={'value': fields}, two_sided=True)
    expected = torch.zeros((2, 2, 2, 3), dtype=torch.float64)
    expected[:, 0, 0] = fields[:, 0]
    expected[:, 1, 1] = fields[:, 1]
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
    assert seen == [(4, 3)]
    gradient, = torch.autograd.grad(out.sum(), (fields,))
    torch.testing.assert_close(gradient, torch.full_like(fields, 3.))


def test_custom_interior_direction_and_parameter_gradcheck():
    """A smooth custom formula must retain wrapper normalization and parameter derivatives."""
    module = api('material')
    n, v, l = t([[.1, .2, 1.]], True), t([[.2, .1, 1.]], True), t([[.3, -.1, 1.]], True)
    scale = t(.3, True)
    def callback(n, v, l, *, parameters):
        value = ((n * v).sum(-1) * (n * l).sum(-1))[:, None]
        return (value * parameters['scale']).expand(-1, 3)
    def evaluate(n, v, l, scale):
        return module.custom(callback, n, v, l, parameters={'scale': scale})
    assert torch.autograd.gradcheck(evaluate, (n, v, l, scale), eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize('model', ['classic', 'principled'])
def test_empty_material_still_rejects_shared_domain_error(model):
    """An empty render must not hide a negative albedo or out-of-domain roughness."""
    module = api('material')
    empty = torch.empty((0, 3), dtype=torch.float64)
    args = dict(normals=empty, view=empty, directions=t([[0., 0., 1.]]), albedo=t([.2, .3, .4]))
    if model == 'classic': args.update(albedo=t([-.1, .3, .4]), specular=t(0.), exponent=t(2.))
    else: args.update(roughness=t(0.), f0=t(.04), budget=budget(module), allow_experimental_budget=True)
    with pytest.raises(ValueError): getattr(module, model)(**args)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_classic_horizon_zero_gradients_in_both_dtypes(dtype):
    """The exact horizon has the declared zero branch in both material precisions."""
    module = api('material')
    n, v, l = [t([value], True, dtype) for value in ([0., 0., 1.], [0., 0., 1.], [1., 0., 0.])]
    colour, strength, exponent = t([.2, .3, .4], True, dtype), t(.1, True, dtype), t(2., True, dtype)
    out = module.classic(n, v, l, albedo=colour, specular=strength, exponent=exponent)
    assert out.dtype == dtype and torch.equal(out, torch.zeros_like(out))
    gradients = torch.autograd.grad(out.sum(), (n, v, l, colour, strength, exponent))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in gradients)


def test_two_node_resource_mean_and_query_gradients_have_independent_closed_form():
    """The public resource's mean must integrate its own curve, not another lookup."""
    module = api('material')
    resource = budget(module)
    c, f = t(.4, True), t(.25, True)
    r = t(1.)
    # At the last roughness node: L0(c)=.9-.2c, L1(c)=.4c.
    # Their cosine-weighted means are .9-.2*2/3 and .4*2/3.
    expected = (1 - f) * (.9 - .2 * c) + f * (.4 * c)
    expected_mean = (1 - f) * (.9 - .2 * 2 / 3) + f * (.4 * 2 / 3)
    loss, mean = resource.directional_loss(c, r, f), resource.mean_loss(r, f)
    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(mean, expected_mean, atol=1e-12, rtol=1e-11)
    actual_gradients = torch.autograd.grad(loss + mean, (c, f))
    expected_gradients = torch.autograd.grad(expected + expected_mean, (c, f))
    for a, b in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'roughness', 'zero_workspace'])
def test_resource_query_shape_precision_domain_and_work_errors(bad):
    """Resource queries must not broadcast, cast, extrapolate or ignore work limits."""
    module = api('material')
    resource = budget(module)
    c, r, f = t([.2, .4]), t(.2), t(.04)
    kwargs = {}
    if bad == 'shape': r = t([.2])
    elif bad == 'dtype': f = f.float()
    elif bad == 'roughness': r = t(.0001)
    elif bad == 'zero_workspace': kwargs['workspace_limit_bytes'] = 0
    with pytest.raises(TypeError if bad == 'dtype' else ValueError):
        resource.directional_loss(c, r, f, **kwargs)


def test_resource_same_lane_transfer_has_independent_storage():
    """Explicit transfer returns a new resource and does not change an existing graph."""
    module = api('material')
    resource = budget(module)
    r = t(.2, True)
    original = resource.mean_loss(r, t(.04))
    moved = resource.to(device='cpu', dtype=torch.float64)
    assert moved is not resource and moved.metadata['admission'] == 'experimental'
    torch.testing.assert_close(moved.mean_loss(r, t(.04)), original, atol=1e-12, rtol=1e-11)
    for tensor in moved.inspect()['tensors'].values(): tensor.zero_()
    gradient, = torch.autograd.grad(original, (r,))
    assert torch.isfinite(gradient)
    torch.testing.assert_close(resource.mean_loss(r, t(.04)), original, atol=1e-12, rtol=1e-11)


def test_principled_rgb_budget_matches_independent_normal_incidence_formula():
    """RGB Fresnel fields keep their channels and use directional and mean losses together."""
    module = api('material')
    n, colour, f0 = t([[0., 0., 1.]]), t([.2, .4, .6]), t([.1, .4, .8], True)
    # r=1, N=V=L: GGX D=1/pi, Smith visibility=1/4, Schlick=F0.
    # Synthetic loss endpoints at r=1: horizon=.9*(1-F0), normal=.7-.3*F0.
    normal_loss = .7 - .3 * f0
    mean_loss = .9 * (1 - f0) / 3 + 2 * normal_loss / 3
    expected = f0 / (4 * math.pi) + colour * normal_loss.square() / (math.pi * mean_loss)
    out = module.principled(n, n, n, albedo=colour, roughness=t(1.), f0=f0,
                            budget=budget(module), allow_experimental_budget=True)
    torch.testing.assert_close(out[0, 0], expected, atol=1e-12, rtol=1e-11)
    a, = torch.autograd.grad(out.sum(), (f0,))
    b, = torch.autograd.grad(expected.sum(), (f0,))
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize('axis', ['points', 'samples'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_empty_light_integration_shapes_and_shared_gradients(axis, dtype):
    """Empty points or samples preserve output layout and connected zero derivatives."""
    module = api('light')
    p, q = (0, 1) if axis == 'points' else (2, 0)
    n = t([[0., 0., 1.]] * p, dtype=dtype).reshape(p, 3).requires_grad_()
    weight = t([.2, .3, .4], True, dtype)
    brdf = torch.ones((p, q, 3), dtype=dtype, requires_grad=True)
    samples = dict(measure='delta', direction=t([0., 0., 1.], dtype=dtype).expand(p, q, 3),
                   irradiance=weight.expand(p, q, 3), valid=torch.ones((p, q), dtype=torch.bool))
    out = module.integrate(n, brdf, samples)
    assert out.shape == (p, 3) and out.dtype == dtype
    assert torch.equal(out, torch.zeros_like(out))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(out.sum(), (n, brdf, weight)))


@pytest.mark.parametrize('bad', ['missing', 'invalid_placeholder', 'nonfinite', 'negative_angle'])
def test_custom_light_callback_packet_is_validated(bad):
    """A custom-light callback must return a complete, consistent packet even for invalid entries."""
    module = api('light')
    result = dict(direction=t([[[0., 0., 1.]]]), radiance=t([[[1., 1., 1.]]]),
                  solid_angle=t([[.5]]), valid=torch.tensor([[True]]))
    if bad == 'missing': result.pop('solid_angle')
    elif bad == 'invalid_placeholder': result['valid'] = torch.tensor([[False]])
    elif bad == 'nonfinite': result['radiance'][0, 0, 0] = float('nan')
    elif bad == 'negative_angle': result['solid_angle'] = t([[-.5]])
    with pytest.raises(ValueError):
        module.custom(lambda *a, **k: result, t([[0., 0., 0.]]), parameters={}, measure='solid_angle')
