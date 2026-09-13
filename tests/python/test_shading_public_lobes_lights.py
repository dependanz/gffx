"""Public lobe/light fixtures: independent formulas, no private recipe fallback."""
import importlib
import math

import pytest
import torch


def api(name):
    return importlib.import_module('gffx.torch.' + name)


def t(value, grad=False, dtype=torch.float64):
    return torch.tensor(value, dtype=dtype, requires_grad=grad)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(previous)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('name', ['lambert', 'blinn_phong', 'ggx_distribution', 'smith_visibility', 'fresnel_schlick'])
def test_each_public_lobe_matches_closed_form(name, dtype):
    """Each public building block must match an independent fixed-angle formula."""
    module = api('shade')
    c, one = t([.25, .5, 1.], dtype=dtype), t(1., dtype=dtype)
    colour = t([[0., .5, 1.]], dtype=dtype)
    if name == 'lambert': args, expected = (colour,), colour / math.pi
    elif name == 'blinn_phong': args, expected = (c, one * 2), c.square() * 10 / (8 * math.pi)
    elif name == 'ggx_distribution': args, expected = (c, one), torch.full_like(c, 1 / math.pi)
    elif name == 'smith_visibility': args, expected = (c, c, one), 1 / (4 * c)
    else:
        args = (c, colour[0])
        expected = colour + (1 - colour) * (1 - c[:, None]).pow(5)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(getattr(module, name)(*args), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('name,values', [
    ('lambert', [[[.2, .4, .6]]]),
    ('blinn_phong', [[.3, .7], [2.3, 5.1]]),
    ('ggx_distribution', [[.2, .8], [.4, .7]]),
    ('smith_visibility', [[.3, .8], [.7, .4], [.4, .7]]),
    ('fresnel_schlick', [[.3, .8], [.04, .2, .7]]),
])
def test_public_lobe_interior_gradients(name, values):
    """Fitting must differentiate every public lobe input in its smooth interior."""
    module = api('shade')
    assert torch.autograd.gradcheck(getattr(module, name), tuple(t(v, True) for v in values),
                                    eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize('exponent', [0., .5, 2.])
def test_blinn_horizon_corner_has_declared_value_and_zero_gradients(exponent):
    """The zero-cosine/exponent corner is explicit rather than a hidden log-zero NaN."""
    module = api('shade')
    c, n = t([0.], True), t(exponent, True)
    result = module.blinn_phong(c, n)
    assert result.item() == pytest.approx(1 / math.pi if exponent == 0 else 0., abs=1e-12)
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(result.sum(), (c, n)))


def test_smith_horizon_masks_before_division():
    """Zero view or light cosine returns finite zero values and first derivatives."""
    module = api('shade')
    v, l, r = t([0., .5, 0.], True), t([.5, 0., 0.], True), t(.3, True)
    result = module.smith_visibility(v, l, r)
    assert torch.equal(result, torch.zeros_like(result))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(result.sum(), (v, l, r)))


@pytest.mark.parametrize('roughness', [0., -1., 1.01, 1e-90])
def test_ggx_rejects_domain_and_lost_positive_divisor(roughness):
    """Reject invalid roughness and float64 underflow instead of introducing an epsilon floor."""
    module = api('shade')
    with pytest.raises(ValueError): module.ggx_distribution(t([1.]), t(roughness))


def test_classic_blinn_negative_control_retains_overreflection():
    """Classic Blinn's conventional scaling must not be advertised as energy-conserving."""
    module = api('shade')
    z = (torch.arange(4096, dtype=torch.float64) + .5) / 4096
    value = 2 * math.pi * (module.blinn_phong(((1 + z) / 2).sqrt(), t(2.)) * z).mean()
    assert value.item() == pytest.approx(25 / 24, abs=1e-7)
    assert value > 1.001


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_public_sh_constant_and_signed_environment(dtype):
    """Unit environment radiance gives pi irradiance, and signed SH is not clamped."""
    module = api('light')
    n = t([[1., 0., 0.], [0., 2., 0.], [0., 0., -3.]], dtype=dtype)
    coeff = torch.zeros((9, 3), dtype=dtype)
    coeff[0] = math.sqrt(4 * math.pi)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    for sign in (1, -1):
        torch.testing.assert_close(module.spherical_harmonics(n, sign * coeff),
                                   torch.full_like(n, sign * math.pi), atol=tolerance, rtol=tolerance)


def test_public_sh_directional_projection_matches_truncated_kernel():
    """Band-two lighting has a known ringing kernel, not an exact clamped cosine."""
    module = api('light')
    c = t([1., -1., -8 / 15])
    normals = torch.stack(((1 - c.square()).sqrt(), torch.zeros_like(c), c), dim=-1)
    coefficients = torch.zeros((9, 3), dtype=torch.float64)
    coefficients[0] = math.sqrt(1 / (4 * math.pi))
    coefficients[2] = math.sqrt(3 / (4 * math.pi))
    coefficients[6] = 2 * math.sqrt(5 / (16 * math.pi))
    expected = t([17 / 16, 1 / 16, -19 / 480])[:, None].expand(3, 3)
    torch.testing.assert_close(module.spherical_harmonics(normals, coefficients), expected, atol=1e-12, rtol=1e-12)


def test_public_point_inverse_square_translation_and_coincidence():
    """Point lights preserve inverse-square units and distinguish coincidence from darkness."""
    module = api('light')
    p, position, intensity = t([[0., 0., 0.]]), t([[0., 0., 0.], [0., 0., 2.]]), t([[4., 8., 12.]]).expand(2, 3)
    result = module.point(p, position, intensity)
    assert result['valid'].tolist() == [[False, True]]
    torch.testing.assert_close(result['irradiance'], t([[[0., 0., 0.], [1., 2., 3.]]]))
    torch.testing.assert_close(result['distance_squared'], t([[0., 4.]]))
    moved = module.point(p + 7, position + 7, intensity)
    assert all(torch.equal(result[key], moved[key]) for key in result)
    inputs = tuple(x.clone().requires_grad_() for x in (p, position, intensity))
    sample = module.point(*inputs)
    loss = sample['direction'][0, 0].sum() + sample['irradiance'][0, 0].sum() + sample['distance_squared'][0, 0]
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(loss, inputs))


def test_public_directional_has_no_hidden_distance_or_cosine_factor():
    """A directional light normalizes directions but preserves the incident weights."""
    module = api('light')
    weight = t([[1., 2., 3.], [4., 5., 6.]])
    result = module.directional(t([[0., 0., 7.], [2., 0., 0.]]), weight)
    torch.testing.assert_close(result['direction'], t([[0., 0., 1.], [1., 0., 0.]]))
    assert torch.equal(result['irradiance'], weight) and 'pdf' not in result


@pytest.mark.parametrize('name', ['sh', 'point', 'directional'])
def test_public_light_interior_gradients(name):
    """Learnable light coefficients, locations, directions and intensity keep their derivatives."""
    module = api('light')
    vector = t([[.3, .5, .8]], True)
    if name == 'sh':
        function = module.spherical_harmonics
        inputs = vector, torch.arange(27, dtype=torch.float64).reshape(9, 3).div(27).requires_grad_()
    elif name == 'point':
        inputs = vector, t([[1., 2., 3.]], True), t([[.4, .7, .9]], True)
        def function(*args):
            result = module.point(*args)
            return result['direction'], result['irradiance'], result['distance_squared']
    else:
        inputs = vector, t([[.4, .7, .9]], True)
        def function(*args):
            result = module.directional(*args)
            return result['direction'], result['irradiance']
    assert torch.autograd.gradcheck(function, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize('name', ['sh', 'point', 'directional'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_public_lights_shared_strided_batches_keep_values_and_gradients(name, dtype):
    m = api('light')
    raw = (torch.arange(24, dtype=dtype).reshape(2, 2, 6)/10+1).requires_grad_()
    q = raw[..., ::2]
    shared = torch.full((1, 9, 3) if name == 'sh' else (1, 2, 3), .6, dtype=dtype, requires_grad=True)
    positions = t([[[4., 3., 2.], [2., 4., 3.]]], True, dtype)
    def evaluate(query):
        if name == 'sh': return (m.spherical_harmonics(query, shared),)
        packet = m.point(query, positions, shared) if name == 'point' else m.directional(query, shared)
        return tuple(value for key, value in packet.items() if key != 'valid')
    together = evaluate(q)
    separate = tuple(torch.stack(parts) for parts in zip(*(evaluate(row) for row in q)))
    for a, b in zip(together, separate): torch.testing.assert_close(a, b)
    inputs = (raw, shared, positions) if name == 'point' else (raw, shared)
    ga = torch.autograd.grad(sum(x.sum() for x in together), inputs, retain_graph=True)
    gb = torch.autograd.grad(sum(x.sum() for x in separate), inputs)
    for a, b in zip(ga, gb): torch.testing.assert_close(a, b)
    assert all(torch.equal(a, b) for a, b in zip(together, evaluate(q)))


@pytest.mark.parametrize('name', ['sh', 'point-empty-points', 'point-empty-lights', 'directional'])
def test_public_light_empty_layouts_preserve_connected_gradients(name):
    m = api('light')
    q = torch.empty(2, 0, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.ones(1, 9 if name == 'sh' else 1, 3, dtype=torch.float64, requires_grad=True)
    if name == 'sh':
        out = m.spherical_harmonics(q, weights)
        assert out.shape == (2, 0, 3)
        inputs, loss = (q, weights), out.sum()
    elif name == 'directional':
        weights = weights[:, :0]
        packet = m.directional(q, weights)
        inputs, loss = (q, weights), packet['direction'].sum()+packet['irradiance'].sum()
    else:
        positions = t([[[1., 2., 3.]]], True)
        if name == 'point-empty-lights':
            q, positions, weights = torch.ones(2, 3, 3, dtype=torch.float64, requires_grad=True), positions[:, :0], weights[:, :0]
        packet = m.point(q, positions, weights)
        assert packet['valid'].shape == ((2, 0, 1) if name.endswith('points') else (2, 3, 0))
        inputs, loss = (q, positions, weights), sum(v.sum() for k, v in packet.items() if k != 'valid')
    for g in torch.autograd.grad(loss, inputs): assert torch.equal(g, torch.zeros_like(g))


@pytest.mark.parametrize('case', ['sh-shape', 'sh-batch', 'sh-zero', 'sh-nan', 'sh-mixed-dtype',
    'direction-zero', 'direction-negative', 'direction-number', 'direction-device',
    'point-count', 'point-batch', 'point-underflow', 'point-overflow', 'point-weight-overflow',
    'point-empty-negative', 'sh-overflow'])
def test_public_lights_reject_invalid_layout_domain_and_arithmetic(case):
    m = api('light'); p = t([[1., 2., 3.], [3., 2., 1.]]); c = torch.ones(9, 3, dtype=torch.float64)
    error, match = ValueError, 'shape|batch'
    if case == 'sh-shape': fn, args = m.spherical_harmonics, (p, c[:8])
    elif case == 'sh-batch': fn, args = m.spherical_harmonics, (p, c.expand(2, 9, 3))
    elif case == 'sh-zero': fn, args, match = m.spherical_harmonics, (p*0, c), 'zero'
    elif case == 'sh-nan': fn, args, match = m.spherical_harmonics, (p, c*float('nan')), 'finite'
    elif case == 'sh-mixed-dtype': fn, args, error, match = m.spherical_harmonics, (p, c.float()), TypeError, 'dtype'
    elif case == 'direction-zero': fn, args, match = m.directional, (p*0, p), 'zero'
    elif case == 'direction-negative': fn, args, match = m.directional, (p, -p), 'nonnegative'
    elif case == 'direction-number': fn, args, error, match = m.directional, (p.tolist(), p), TypeError, 'tensor'
    elif case == 'direction-device': fn, args, match = m.directional, (p, torch.empty_like(p, device='meta')), 'device'
    elif case == 'point-count': fn, args = m.point, (p, p, p[:1])
    elif case == 'point-batch': fn, args = m.point, (p, p.expand(2, 2, 3), p)
    elif case in ('point-underflow', 'point-overflow'):
        distance = 1e-200 if case.endswith('underflow') else 1e200
        fn, args, match = m.point, (p*0, t([[distance, 0., 0.]]), p[:1]), 'represent'
    elif case == 'point-weight-overflow': fn, args, match = m.point, (p*0, p*.001, torch.full_like(p, 1e308)), 'represent'
    elif case == 'point-empty-negative': fn, args, match = m.point, (p[:0], p, -p), 'nonnegative'
    elif case == 'sh-overflow': fn, args, match = m.spherical_harmonics, (p, c*1e308), 'represent'
    with pytest.raises(error, match=match): fn(*args)


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_public_lights_normalize_finite_extreme_nonzero_vectors(scale):
    m = api('light'); v = t([[1., 2., 3.]])
    expected = v/v.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(m.directional(v*scale, torch.ones_like(v))['direction'], expected)
    c = torch.arange(27, dtype=torch.float64).reshape(9, 3)/27
    torch.testing.assert_close(m.spherical_harmonics(v*scale, c), m.spherical_harmonics(v, c))


def test_public_light_demo_calls_all_three_installed_operations(monkeypatch):
    from tools.testing_dashboard import lighting_demo
    m, called = api('light'), set()
    with monkeypatch.context() as patch:
        for name in ('spherical_harmonics', 'point', 'directional'):
            original = getattr(m, name)
            def observe(*args, name=name, original=original):
                called.add(name)
                return original(*args)
            patch.setattr(m, name, observe)
        data = lighting_demo.examples(public=True)
    assert called == {'spherical_harmonics', 'point', 'directional'}
    assert data['image'].shape == (128, 256, 3)
    assert data['distance'][0] == pytest.approx((.5, 16., 4.))
    assert min(row[2] for row in data['angle']) < 0
