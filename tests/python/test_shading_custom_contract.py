"""Step-4 public extension red tests; no private implementation substitutes.

Registry and light packets are explicit local tensor interfaces. These small
fixtures test plumbing and analytical Lambert lighting, not a new physical model.
"""
import importlib
import math

import pytest
import torch


def api(kind):
    return importlib.import_module('gffx.torch.' + kind)


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def lambert(normals, view, directions, *, parameters):
    return parameters['colour'] / math.pi


def packet(points, *, parameters):
    shape = (*points.shape[:-1], 1)
    return dict(direction=t([0., 0., 1.]).expand(*shape, 3),
                irradiance=parameters['power'].expand(*shape, 3),
                valid=torch.ones(shape, dtype=torch.bool))


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('kind', ['material', 'light'])
def test_registries_are_local_and_return_stable_bindings(kind):
    """Changing a registry cannot silently change a renderer's selected function."""
    module = api(kind)
    first, second = module.Registry(), module.Registry()
    def old(*args, **kwargs): return 'old'
    def new(*args, **kwargs): return 'new'
    assert first.register('lab/model', old) is None
    assert first.names() == ('lab/model',) and second.names() == ()
    bound = first.resolve('lab/model')
    assert bound is old
    assert first.unregister('lab/model') is None
    first.register('lab/model', new)
    assert bound() == 'old' and first.resolve('lab/model')() == 'new'


@pytest.mark.parametrize('kind', ['material', 'light'])
@pytest.mark.parametrize('bad', ['duplicate', 'reserved', 'malformed', 'noncallable', 'wrong_type'])
def test_failed_registration_keeps_existing_state(kind, bad):
    """Reject ambiguous names and accidental replacement without damaging the registry."""
    module = api(kind)
    registry = module.Registry()
    registry.register('lab/first', lambert)
    before = registry.names()
    name, function = 'lab/second', lambert
    if bad == 'duplicate': name = 'lab/first'
    elif bad == 'reserved': name = 'gffx/classic'
    elif bad == 'malformed': name = '../classic'
    elif bad == 'noncallable': function = 5
    elif bad == 'wrong_type': name = 5
    with pytest.raises(TypeError if bad in ('noncallable', 'wrong_type') else ValueError):
        registry.register(name, function)
    assert registry.names() == before and registry.resolve('lab/first') is lambert


@pytest.mark.parametrize('kind', ['material', 'light'])
def test_registry_snapshots_missing_names_and_no_callback_execution(kind):
    """Registration stores code without running it; enumeration is a sorted snapshot."""
    module = api(kind)
    def forbidden(*args, **kwargs): pytest.fail('registration invoked user code')
    registry = module.Registry()
    registry.register('z/last', forbidden)
    snapshot = registry.names()
    registry.register('a/first', forbidden)
    assert snapshot == ('z/last',)
    assert registry.names() == ('a/first', 'z/last')
    for method in (registry.resolve, registry.unregister):
        with pytest.raises(KeyError): method('lab/missing')
    assert registry.names() == ('a/first', 'z/last')


def test_direct_custom_lambert_matches_builtin_and_shared_gradients():
    """A custom Lambert formula works without registration and fits a shared colour."""
    module = api('material')
    n = t([[0., 0., 2.], [0., 0., 3.]])
    v, directions = n.clone(), t([[0., 0., 4.], [0., 0., -1.]])
    colour = t([.2, .4, .6], True)
    seen = []
    def callback(n, v, l, *, parameters):
        seen.append(n.shape)
        assert n.shape == v.shape == l.shape == (2, 3)
        torch.testing.assert_close(n.norm(dim=-1), t([1., 1.]))
        return lambert(n, v, l, parameters=parameters)
    out = module.custom(callback, n, v, directions, parameters={'colour': colour})
    expected = module.classic(n, v, directions, albedo=colour, specular=t(0.), exponent=t(2.))
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
    assert seen == [(2, 3)]
    gradient, = torch.autograd.grad(out.sum(), (colour,))
    torch.testing.assert_close(gradient, torch.full_like(colour, 2 / math.pi), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('empty', [False, True])
def test_inactive_pairs_never_invoke_custom_material(empty):
    """No callback sees a horizon/backface half-vector or an empty active set."""
    module = api('material')
    n = torch.empty((0, 3), dtype=torch.float64) if empty else t([[0., 0., 1.]])
    n.requires_grad_()
    v = n.detach().clone().requires_grad_()
    l, colour = t([[0., 0., -1.]], True), t([.2, .3, .4], True)
    def forbidden(*args, **kwargs): pytest.fail('inactive callback invoked')
    out = module.custom(forbidden, n, v, l, parameters={'colour': colour})
    assert out.shape == (0 if empty else 1, 1, 3)
    assert torch.equal(out, torch.zeros_like(out))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(out.sum(), (n, v, l, colour)))


@pytest.mark.parametrize('bad', ['shape', 'negative', 'nan', 'dtype', 'object'])
def test_custom_material_rejects_bad_callback_output(bad):
    """A callback must return the promised finite BRDF tensor, not arbitrary data."""
    module = api('material')
    def callback(n, v, l, *, parameters):
        if bad == 'shape': return torch.ones((len(n), 1), dtype=n.dtype)
        if bad == 'object': return {'brdf': n}
        if bad == 'dtype': return torch.ones_like(n, dtype=torch.float32)
        return torch.full_like(n, -1. if bad == 'negative' else float('nan'))
    n = t([[0., 0., 1.]])
    with pytest.raises(TypeError if bad in ('dtype', 'object') else ValueError):
        module.custom(callback, n, n, n, parameters={})


def test_callback_exception_propagates_without_fallback():
    """A broken custom material must not quietly render with a default material."""
    module = api('material')
    def broken(*args, **kwargs): raise RuntimeError('custom failure sentinel')
    n = t([[0., 0., 1.]])
    with pytest.raises(RuntimeError, match='custom failure sentinel'):
        module.custom(broken, n, n, n, parameters={})


@pytest.mark.parametrize('measure', ['delta', 'solid_angle'])
def test_light_tag_applies_one_cosine_and_one_measure_weight(measure):
    """Half-angle illumination and environment weights must each be applied once."""
    module = api('light')
    points = t([[0., 0., 0.]])
    original = dict(direction=t([[[math.sqrt(3), 0., 1.]]]),
                    valid=torch.tensor([[True]]))
    if measure == 'delta': original['irradiance'] = t([[[2., 2., 2.]]])
    else:
        original['radiance'] = t([[[2., 2., 2.]]])
        original['solid_angle'] = t([[.25]])
    samples = module.custom(lambda *a, **k: original, points, parameters={}, measure=measure)
    assert samples is not original and 'measure' not in original
    assert samples['measure'] == measure
    out = module.integrate(t([[0., 0., 1.]]), t([[[1., 1., 1.]]]), samples)
    torch.testing.assert_close(out, t([[1., 1., 1.]]) * (.25 if measure == 'solid_angle' else 1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(original['direction'], t([[[math.sqrt(3), 0., 1.]]]))


def test_custom_delta_matches_builtin_point_light_and_parameter_gradients():
    """Custom light weights agree with inverse-square point lighting and remain learnable."""
    module = api('light')
    points, power = t([[0., 0., 0.], [0., 0., 0.]]), t([.2, .4, .6], True)
    samples = module.custom(packet, points, parameters={'power': power}, measure='delta')
    builtin = module.point(points, t([[0., 0., 2.]]), (power * 4)[None, :])
    torch.testing.assert_close(samples['direction'], builtin['direction'], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(samples['irradiance'], builtin['irradiance'], atol=1e-12, rtol=1e-12)
    normals = t([[0., 0., 1.], [0., 0., 1.]])
    brdf = torch.full((2, 1, 3), 1 / math.pi, dtype=torch.float64)
    output = module.integrate(normals, brdf, samples)
    gradient, = torch.autograd.grad(output.sum(), (power,))
    torch.testing.assert_close(gradient, torch.full_like(power, 2 / math.pi), atol=1e-12, rtol=1e-12)


def test_environment_sample_splitting_preserves_integral_and_signed_radiance():
    """Dividing a sky patch into two samples must not double its light or erase signed SH."""
    module = api('light')
    def samples(count):
        return dict(measure='solid_angle', direction=t([0., 0., 1.]).expand(1, count, 3),
                    radiance=t([-.2, .4, .6]).expand(1, count, 3),
                    solid_angle=torch.full((1, count), .75 / count, dtype=torch.float64),
                    valid=torch.ones((1, count), dtype=torch.bool))
    n = t([[0., 0., 1.]])
    results = [module.integrate(n, torch.ones((1, count, 3), dtype=torch.float64), samples(count)) for count in (1, 2)]
    for result in results:
        torch.testing.assert_close(result, t([[-.15, .3, .45]]), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('bad', ['pdf', 'unknown', 'mixed', 'negative', 'valid_dtype', 'zero_direction', 'shape'])
def test_bad_light_packets_are_not_silently_reinterpreted(bad):
    """Reject wrong measures, units and layouts rather than making plausible-looking light."""
    module = api('light')
    n = t([[0., 0., 1.]])
    samples = packet(n, parameters={'power': t([1., 1., 1.])})
    samples['measure'] = 'delta'
    if bad == 'pdf': samples['pdf'] = t([[1.]])
    elif bad == 'unknown': samples['measure'] = 'monte_carlo'
    elif bad == 'mixed': samples['radiance'] = samples.pop('irradiance')
    elif bad == 'negative': samples['irradiance'] = -samples['irradiance']
    elif bad == 'valid_dtype': samples['valid'] = t([[1.]])
    elif bad == 'zero_direction': samples['direction'] = torch.zeros_like(samples['direction'])
    elif bad == 'shape': samples['direction'] = n  # missing point/light axis
    with pytest.raises(TypeError if bad == 'valid_dtype' else ValueError):
        module.integrate(n, t([[[1., 1., 1.]]]), samples)


def test_invalid_light_is_distinct_from_visibility_and_has_zero_gradients():
    """Invalid samples are safely zero, while occlusion is a separate explicit factor."""
    module = api('light')
    direction = t([[[0., 0., 0.], [0., 0., 1.]]], True)
    weight = t([[[0., 0., 0.], [1., 1., 1.]]], True)
    samples = dict(measure='delta', direction=direction, irradiance=weight,
                   valid=torch.tensor([[False, True]]))
    n, brdf = t([[0., 0., 1.]]), torch.ones((1, 2, 3), dtype=torch.float64)
    out = module.integrate(n, brdf, samples)
    torch.testing.assert_close(out, t([[1., 1., 1.]]))
    gradients = torch.autograd.grad(out.sum(), (direction, weight))
    assert all(torch.equal(g[:, 0], torch.zeros_like(g[:, 0])) for g in gradients)
    visibility = t([[1., 0.]])
    hidden = module.integrate(n, brdf * visibility[..., None], samples)
    assert torch.equal(hidden, torch.zeros_like(hidden))
    assert torch.equal(samples['valid'], torch.tensor([[False, True]]))


def test_custom_light_rejects_unsupported_measure_before_callback():
    """An unsupported PDF-based measure must not even invoke user light code."""
    module = api('light')
    def forbidden(*args, **kwargs): pytest.fail('unsupported callback invoked')
    with pytest.raises(ValueError):
        module.custom(forbidden, t([[0., 0., 0.]]), parameters={}, measure='pdf')
