"""Complete public material tests; synthetic budgets are algebra, not physics."""
import builtins
import importlib
import io
import math
from pathlib import Path

import pytest
import torch

from test_material_public_contract import envelope


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def setup():
    module = importlib.import_module('gffx.torch.material')
    resource = module.PrincipledBudget.from_envelope(
        envelope(), device='cpu', dtype=torch.float64, allow_unverified=True)
    return module, resource


def evaluate(module, resource, n, v, l, a, r, f, **kwargs):
    return module.principled(n, v, l, albedo=a, roughness=r, f0=f,
                            budget=resource, allow_experimental_budget=True, **kwargs)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_principled_oblique_value_matches_independent_scalar_formula():
    """Check the assembled BRDF off-axis without using its public lobes as oracle."""
    module, resource = setup()
    nv, nl = .8, .6
    # V=(.6,0,.8), L=(0,.8,.6); at r=1, D=1/pi and Vsmith=1/(2*(nv+nl)).
    vh = math.sqrt((1 + nv * nl) / 2)
    expected = []
    for a, f in zip((.2, .4, .6), (.1, .3, .7)):
        fresnel = f + (1 - f) * (1 - vh)**5
        loss = lambda c: (1 - f) * (.9 - .2 * c) + f * .4 * c
        mean = (1 - f) * (.9 - .4 / 3) + f * .8 / 3
        expected.append(fresnel / (2 * math.pi * (nv + nl)) + a * loss(nv) * loss(nl) / (math.pi * mean))
    out = evaluate(module, resource, t([[0., 0., 1.]]), t([[.6, 0., .8]]),
                   t([[0., .8, .6]]), t([.2, .4, .6]), t(1.), t([.1, .3, .7]))
    torch.testing.assert_close(out[0, 0], t(expected), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('roughness', [.03, .4, 1.])
def test_principled_reciprocity_preserves_values_and_input_gradients(roughness):
    """Exchanging view and light must preserve the BRDF and its mapped derivatives."""
    module, resource = setup()
    inputs = (t([[.1, .2, 1.]], True), t([[.4, .2, 1.]], True),
              t([[-.2, .3, 1.]], True), t([.2, .4, .6], True),
              t(roughness, True), t([.1, .3, .7], True))
    n, v, l, a, r, f = inputs
    first = evaluate(module, resource, n, v, l, a, r, f)
    second = evaluate(module, resource, n, l, v, a, r, f)
    torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)
    for x, y in zip(torch.autograd.grad(first.sum(), inputs), torch.autograd.grad(second.sum(), inputs)):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize('model', ['classic', 'principled'])
def test_complete_material_all_input_gradcheck(model):
    """Fitting reaches normals, both directions and all material parameters together."""
    # Classic has no budget dependency: do not block its acceptance on an
    # unrelated, not-yet-implemented principled resource factory.
    if model == 'principled':
        module, resource = setup()
    else:
        module = importlib.import_module('gffx.torch.material')
    vectors = (t([[.1, .2, 1.]], True), t([[.4, .2, 1.]], True), t([[-.2, .3, 1.]], True))
    if model == 'principled':
        inputs = (*vectors, t([.2, .4, .6], True), t(.4, True), t([.1, .3, .7], True))
        function = lambda *args: evaluate(module, resource, *args)
    else:
        inputs = (*vectors, t([.2, .4, .6], True), t([.1, .3, .7], True), t(2.3, True))
        function = lambda n, v, l, a, s, e: module.classic(n, v, l, albedo=a, specular=s, exponent=e)
    assert torch.autograd.gradcheck(function, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_principled_batched_strided_fields_match_separate_calls_and_gradients():
    """Explicit batch expansion must not mix channels or drop shared gradients."""
    module, resource = setup()
    base = t([[[.2, .4, .6, .9], [.3, .5, .7, .8]],
              [[.4, .2, .3, .7], [.5, .3, .2, .6]]], True)
    a = base[..., :3]  # non-contiguous [B,P,3]
    assert not a.is_contiguous()
    n = t([0., 0., 1.]).expand(2, 2, 3)
    v, l = t([[.1, .2, 1.], [.2, .1, 1.]]), t([[.3, .2, 1.], [.1, -.2, 1.]])
    r, f = t(.4, True), t([.1, .3, .7], True)
    before = base.detach().clone()
    batched = evaluate(module, resource, n, v, l, a, r, f)
    separate = torch.stack([evaluate(module, resource, n[i], v, l, a[i], r, f) for i in range(2)])
    torch.testing.assert_close(batched, separate, atol=1e-12, rtol=1e-12)
    for x, y in zip(torch.autograd.grad(batched.sum(), (base, r, f)),
                    torch.autograd.grad(separate.sum(), (base, r, f))):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-11)
    assert torch.equal(base.detach(), before)


@pytest.mark.parametrize('view,light', [([0., 0., 1.], [1., 0., 0.]),
                                      ([1., 0., 0.], [0., 0., 1.]),
                                      ([0., 0., 1.], [0., 0., -1.])])
def test_principled_inactive_pairs_have_zero_all_input_gradients(view, light):
    """Horizon and opposite vectors must not create invalid intermediate divisions."""
    module, resource = setup()
    inputs = (t([[0., 0., 1.]], True), t([view], True), t([light], True),
              t([.2, .4, .6], True), t(.4, True), t([.1, .3, .7], True))
    value = evaluate(module, resource, *inputs)
    assert torch.equal(value, torch.zeros_like(value))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(value.sum(), inputs))


def test_principled_two_sided_equals_explicit_normal_flip():
    """Two-sided orientation changes the normal, not the material or incident light."""
    module, resource = setup()
    n, v, l = t([[0., 0., 1.]]), t([[.2, .1, -1.]]), t([[-.1, .3, -1.]])
    a, r, f = t([.2, .4, .6]), t(.4), t([.1, .3, .7])
    actual = evaluate(module, resource, n, v, l, a, r, f, two_sided=True)
    expected = evaluate(module, resource, -n, v, l, a, r, f)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert (actual > 0).all()


def test_resource_queries_do_not_read_files_or_parse_json(monkeypatch):
    """Shading reuses loaded constants without hidden disk reads or JSON work."""
    _, resource = setup()
    c, r, f = t([.2, .4], True), t(.4, True), t(.2, True)
    def forbidden(*args, **kwargs):
        pytest.fail('resource query attempted I/O or parsing')
    import json
    with monkeypatch.context() as patch:
        for owner, name in ((builtins, 'open'), (io, 'open'), (Path, 'open'), (json, 'load'), (json, 'loads')):
            patch.setattr(owner, name, forbidden)
        value = resource.directional_loss(c, r, f).sum() + resource.mean_loss(r, f)
        gradients = torch.autograd.grad(value, (c, r, f))
    assert torch.isfinite(value) and all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize('backward', [False, True])
def test_resource_query_enforces_its_reported_workspace_requirement(backward):
    """A caller-provided memory estimate limit must be enforced rather than ignored."""
    _, resource = setup()
    c, r, f = t([.2, .4], backward), t(.4, backward), t(.2, backward)
    required = resource.workspace(2, backward=backward)
    assert isinstance(required, int) and not isinstance(required, bool) and required > 0
    with pytest.raises(ValueError):
        resource.directional_loss(c, r, f, workspace_limit_bytes=required - 1)
    result = resource.directional_loss(c, r, f, workspace_limit_bytes=required)
    assert torch.isfinite(result).all()
    if backward:
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(result.sum(), (c, r, f)))


def test_resource_empty_strided_queries_keep_shapes_and_gradients():
    """Empty work and strided cosine views retain the declared resource query layout."""
    _, resource = setup()
    base, r, f = t([.2, .9, .4, .8], True), t(.4, True), t(.2, True)
    c = base[::2]
    actual = resource.directional_loss(c, r, f)
    expected = torch.stack([resource.directional_loss(x, r, f) for x in c])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for x, y in zip(torch.autograd.grad(actual.sum(), (base, r, f)), torch.autograd.grad(expected.sum(), (base, r, f))):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-11)
    empty = resource.directional_loss(base[:0], r, f)
    assert empty.shape == (0,)
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(empty.sum(), (base, r, f)))
