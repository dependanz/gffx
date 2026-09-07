"""Independent Phase-0 lobe recipes: not public gffx shader acceptance."""
import importlib
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
sys.path.append(str(Path(__file__).resolve().parents[2]))


def recipe():
    return importlib.import_module('tools.shade_contract_reference')


def t(value, **kwargs):
    return torch.tensor(value, dtype=torch.float64, **kwargs)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_lobe_closed_form_values_and_dtype(dtype):
    r = recipe(); c = torch.tensor([0., .5, 1.], dtype=dtype)
    one = torch.tensor(1., dtype=dtype)
    albedo = torch.tensor([[0., .5, 1.]], dtype=dtype)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(r.lambert(albedo), albedo/math.pi, atol=tol, rtol=tol)
    torch.testing.assert_close(r.ggx_distribution(c, one), torch.full_like(c, 1/math.pi), atol=tol, rtol=tol)
    torch.testing.assert_close(r.blinn_phong(c, one*2), c.square()*10/(8*math.pi), atol=tol, rtol=tol)
    expected = torch.tensor([[1., 1., 1.], [.03125, .515625, 1.], [0., .5, 1.]], dtype=dtype)
    torch.testing.assert_close(r.fresnel_schlick(c, albedo[0]), expected, atol=tol, rtol=tol)
    torch.testing.assert_close(r.smith_visibility(c[1:], c[1:], one), 1/(4*c[1:]), atol=tol, rtol=tol)


def test_blinn_half_vector_is_not_reflected_vector_phong():
    r = recipe()
    # Normal view and a 60-degree light: half-vector cosine sqrt(3)/2, not 1/2.
    value = r.blinn_phong(t([math.sqrt(3)/2]), t(2.))
    assert value.item() == pytest.approx(15/(16*math.pi), abs=1e-12)
    assert not torch.allclose(value, r.blinn_phong(t([.5]), t(2.)))


def test_blinn_approximate_scaling_is_not_energy_conservation(request):
    r = recipe(); z = (torch.arange(4096, dtype=torch.float64)+.5)/4096
    result = 2*math.pi*(r.blinn_phong(((1+z)/2).sqrt(), t(2.))*z).mean()
    # Analytic: nH^2=(1+z)/2, integral of z+z^2 is 5/6.
    assert result.item() == pytest.approx(25/24, abs=1e-7)
    assert result > 1.001
    plugin = request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin:
        plugin.Evidence(request.node.nodeid).record('classic-negative-control',
            {'outgoing_over_unit_input': result.item(), 'analytic': 25/24,
             'scope': 'approximate Blinn highlight, not the principled material'})


@pytest.mark.parametrize('roughness', [.2, .5, 1.])
def test_ggx_projected_distribution_integrates_to_one(roughness):
    r = recipe(); z = (torch.arange(65536, dtype=torch.float64)+.5)/65536
    value = 2*math.pi*(r.ggx_distribution(z, t(roughness))*z).mean()
    assert abs(value.item()-1) < 5e-5


def test_microfacet_product_matches_independent_furnace():
    r = recipe(); z = (torch.arange(16384, dtype=torch.float64)+.5)/16384
    d = r.ggx_distribution(((1+z)/2).sqrt(), t(1.))
    v = r.smith_visibility(torch.ones_like(z), z, t(1.))
    f = r.fresnel_schlick(((1+z)/2).sqrt(), t([1., 1., 1.]))[:, 0]
    integral = 2*math.pi*(d*v*f*z).mean()
    assert integral.item() == pytest.approx(1-math.log(2), abs=1e-9)


@pytest.mark.parametrize('name,args', [
    ('lambert', [[[.2, .5, .7]]]),
    ('blinn_phong', [[.3, .8], [2.3, 16.]]),
    ('ggx_distribution', [[.2, .8], [.3, .7]]),
    ('smith_visibility', [[.3, .8], [.7, .4], [.3, .7]]),
    ('fresnel_schlick', [[.3, .8], [.04, .2, .7]]),
])
def test_lobe_interior_gradients(name, args):
    values = tuple(t(value, requires_grad=True) for value in args)
    assert torch.autograd.gradcheck(getattr(recipe(), name), values, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_masked_boundaries_have_finite_zero_gradients():
    r = recipe()
    for exponent in [0., .5, 2.]:
        c, n = t([0.], requires_grad=True), t(exponent, requires_grad=True)
        value = r.blinn_phong(c, n)
        assert value.item() == pytest.approx(1/math.pi if exponent == 0 else 0.)
        assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(value.sum(), (c, n)))
    v, l, rough = t([0., .5, 0.], requires_grad=True), t([.5, 0., 0.], requires_grad=True), t(.2, requires_grad=True)
    out = r.smith_visibility(v, l, rough)
    assert torch.equal(out, torch.zeros_like(out))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(out.sum(), (v, l, rough)))


def test_schlick_endpoint_derivatives_are_not_clamped():
    c, f0 = t([0., 1.], requires_grad=True), t([.04, .2, .6], requires_grad=True)
    out = recipe().fresnel_schlick(c, f0)
    gc, gf = torch.autograd.grad(out.sum(), (c, f0))
    torch.testing.assert_close(gc, t([-5*(3-f0.detach().sum().item()), 0.]))
    torch.testing.assert_close(gf, torch.ones_like(f0))


def test_lobe_batches_and_shared_gradients_do_not_mix():
    r = recipe(); c = t([[.25, .5], [.75, 1.]])
    n = t(2., requires_grad=True)
    together = r.blinn_phong(c, n)
    separate = torch.stack([r.blinn_phong(row, n) for row in c])
    assert torch.equal(together, separate)
    a, = torch.autograd.grad(together.sum(), (n,), retain_graph=True)
    b, = torch.autograd.grad(separate.sum(), (n,))
    torch.testing.assert_close(a, b)
    f0 = t([[[.1, .2, .3], [.4, .5, .6]]], requires_grad=True)
    out = r.fresnel_schlick(c, f0)
    assert out.shape == (2, 2, 3)
    out.sum().backward()
    expected = (1-(1-c)**5).sum(0)[None, :, None].expand_as(f0)
    torch.testing.assert_close(f0.grad, expected)


def test_empty_strided_inputs_are_deterministic_and_unchanged():
    r = recipe(); source = t([[.2, .3, .4, .5], [.6, .7, .8, .9]])
    c = source[:, ::2]; before = source.clone(); rough = t(.5)
    assert not c.is_contiguous()
    a = r.ggx_distribution(c, rough)
    assert torch.equal(a, r.ggx_distribution(c, rough)) and torch.equal(source, before)
    empty = torch.empty(2, 0, dtype=torch.float64)
    assert r.blinn_phong(empty, t(4.)).shape == (2, 0)
    assert r.smith_visibility(empty, empty, rough).shape == (2, 0)
    assert r.fresnel_schlick(empty, t([.1, .2, .3])).shape == (2, 0, 3)
    assert r.lambert(torch.empty(2, 0, 3, dtype=torch.float64)).shape == (2, 0, 3)
    with pytest.raises(ValueError): r.ggx_distribution(empty, t(0.))


@pytest.mark.parametrize('field,bad', [('cosine', [-.1]), ('cosine', [1.1]), ('cosine', [float('nan')]),
                                     ('roughness', 0.), ('roughness', 1.1), ('roughness', float('inf'))])
def test_lobe_invalid_domains_are_explicit(field, bad):
    cosine = t(bad) if field == 'cosine' else t([.5])
    roughness = t(bad) if field == 'roughness' else t(.5)
    with pytest.raises(ValueError): recipe().ggx_distribution(cosine, roughness)


def test_lobe_rejects_ambiguous_shapes_types_and_devices():
    r = recipe()
    for cosine, parameter, error in [(t(.5), t(2.), ValueError), (t([.5]), 2., TypeError),
        (t([.5]), torch.tensor(2.), TypeError), (torch.ones(1, dtype=torch.int64), t(2.), TypeError),
        (t([.5]), t([[2.], [3.]]), ValueError), (t([[.5, .6]]), t([2.]), ValueError),
        (torch.empty(0, 2, dtype=torch.float64), t(2.), ValueError)]:
        with pytest.raises(error): r.blinn_phong(cosine, parameter)
    with pytest.raises(ValueError): r.fresnel_schlick(t([.5]), t(.04))
    with pytest.raises(ValueError): r.lambert(t([[1.1, .2, .3]]))
    with pytest.raises(ValueError): r.blinn_phong(t([.5]), t(-1.))
    with pytest.raises(ValueError, match='device'):
        r.blinn_phong(t([.5]), torch.empty((), device='meta', dtype=torch.float64))


def test_small_roughness_is_not_silently_floored():
    r = recipe(); result = r.ggx_distribution(t([1.]), t(1e-4))
    assert result.item() == pytest.approx(1/(math.pi*1e-16), rel=1e-12)
    with pytest.raises(ValueError, match='represent'):
        r.ggx_distribution(torch.tensor([1.]), torch.tensor(1e-20))


def test_lobe_demo_and_textbook_explain_model_boundaries():
    demo = importlib.import_module('tools.testing_dashboard.shade_demo')
    data = demo.examples()
    assert data['image'].shape == (360, 1280, 3)
    assert data['image'].dtype == torch.uint8
    assert len(data['angles']) == 181
    assert data['scope'].startswith('Phase-0')
    assert data['image'].numel() < 2_000_000
    from tools.testing_dashboard.build_lessons import articles
    article = articles()['learn/shading/energy-conservation']['html']
    for phrase in ('Blinn', 'half-vector', '25/24', 'Smith', '1-log(2)'):
        assert phrase in article
