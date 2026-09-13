"""Public resource safety and complete-material regression; no physical admission."""
import importlib
import json
import math
import pytest
import torch
from test_material_public_contract import envelope, digest


def api():
    return importlib.import_module('gffx.torch.material')


def t(x, grad=False):
    return torch.tensor(x, dtype=torch.float64, requires_grad=grad)


def budget():
    return api().PrincipledBudget.from_envelope(envelope(), device='cpu',
                                              dtype=torch.float64, allow_unverified=True)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('device,dtype', [('cpu', torch.float32), ('cuda:0', torch.float64), ('meta', torch.float64)])
def test_budget_unsupported_lane_rejected_before_allocation(monkeypatch, device, dtype):
    module = api()
    value = budget()
    def forbidden(*a, **k):
        pytest.fail('unqualified target allocated storage')
    monkeypatch.setattr(torch, 'tensor', forbidden)
    with pytest.raises(ValueError):
        # CUDA-f64 now has an explicit research path; default requests remain
        # unsupported. CPU-f32 and meta remain unsupported even with research opt-in.
        module.PrincipledBudget.from_envelope(envelope(), device=device, dtype=dtype,
                                             allow_unverified=device!='cuda:0')
    with pytest.raises(ValueError):
        value.to(device=device, dtype=dtype)


def test_budget_metadata_storage_and_constructor_contract():
    module = api()
    with pytest.raises(TypeError):
        module.PrincipledBudget()
    value = budget()
    info = value.inspect()
    assert info['persistent_tensor_bytes'] == sum(x.numel()*x.element_size() for x in info['tensors'].values())
    assert value.metadata['supported_query_entries'] == 32
    assert value.metadata['execution_device'] == 'cpu'
    assert value.metadata['execution_dtype'] == 'float64'
    assert value.metadata['acceptance_profile_id'] is None
    assert value.workspace(query_entries=1, backward=False) == value.workspace(1, backward=False)
    assert not hasattr(value, 'evaluate')
    with pytest.raises(AttributeError):
        value.metadata = {}
    with pytest.raises(ValueError):
        value.workspace(33, backward=False)


@pytest.mark.parametrize('bad', ['boolean_domain', 'tensor_payload', 'huge_integer', 'nan', 'overflow', 'deep_json'])
def test_budget_parser_rejects_invalid_numbers_and_nested_data(tmp_path, bad):
    value = envelope()
    if bad == 'boolean_domain':
        value['domain']['cosine'][0] = False
    elif bad == 'tensor_payload':
        value['payload']['losses'][0][0][0] = t(.8, True)
    elif bad == 'huge_integer':
        value['payload']['losses'][0][0][0] = 10**1000
    else:
        path = tmp_path/'bad.json'
        if bad == 'deep_json':
            text = '['*2000+'0'+']'*2000
        else:
            text = json.dumps(value).replace('0.8', 'NaN' if bad == 'nan' else '1e999')
        path.write_text(text, encoding='utf-8')
        with pytest.raises(ValueError):
            api().PrincipledBudget.load(path, device='cpu', dtype=torch.float64, allow_unverified=True)
        return
    with pytest.raises(ValueError):
        api().PrincipledBudget.from_envelope(value, device='cpu', dtype=torch.float64, allow_unverified=True)


@pytest.mark.parametrize('option', ['count_bool', 'backward_number', 'query_limit_bool'])
def test_budget_static_option_types_are_explicit(option):
    value = budget()
    with pytest.raises(TypeError):
        if option == 'count_bool':
            value.workspace(True, backward=False)
        elif option == 'backward_number':
            value.workspace(1, backward=1)
        else:
            value.mean_loss(t(.4), t(.1), workspace_limit_bytes=True)


@pytest.mark.parametrize('axis', ['points', 'lights', 'inactive'])
def test_principled_empty_and_tiny_inactive_keep_all_gradients(axis):
    n = t([[0., 0., 1e-310]] if axis == 'inactive' else [[0., 0., 1.]], True)
    v, l = n.detach().clone().requires_grad_(), n.detach().clone().requires_grad_()
    if axis == 'points':
        n, v = n[:0], v[:0]
    elif axis == 'lights':
        l = l[:0]
    else:
        l = -l
    a, r, f = t([.2, .3, .4], True), t(.4, True), t(.1, True)
    out = api().principled(n, v, l, albedo=a, roughness=r, f0=f,
                           budget=budget(), allow_experimental_budget=True)
    assert out.shape == (len(n), len(l), 3)
    assert torch.equal(out, torch.zeros_like(out))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(out.sum(), (n, v, l, a, r, f)))


def test_principled_preflight_counts_expanded_channels_before_allocation(monkeypatch):
    resource = budget()
    n, a, r, f = t([[0., 0., 1.]]), t([.2, .3, .4]), t(.4), t(.1)
    def forbidden(*a, **k):
        pytest.fail('expanded allocation before material workspace rejection')
    monkeypatch.setattr(torch, 'ones_like', forbidden)
    # One scalar query fits, but three channel queries and material state do not.
    with pytest.raises(ValueError, match='workspace'):
        api().principled(n, n, n, albedo=a, roughness=r, f0=f, budget=resource,
                         allow_experimental_budget=True,
                         workspace_limit_bytes=resource.workspace(1, backward=False))


def test_principled_reuses_view_mean_and_rebuilds_fitting_graph(monkeypatch):
    module = api()
    resource = budget()
    losses = module.PrincipledBudget._material_losses
    calls = []
    def tracked(self, view, light, roughness, f0, indices):
        calls.append(roughness.shape)
        return losses(self, view, light, roughness, f0, indices)
    monkeypatch.setattr(module.PrincipledBudget, '_material_losses', tracked)
    n, l, a = t([[0., 0., 1.]]), t([[.2, .1, 1.], [-.1, .3, 1.]]), t([.2, .3, .4], True)
    values = []
    for _ in range(2):
        out = module.principled(n, n, l, albedo=a, roughness=t(.4), f0=t(.1),
                                budget=resource, allow_experimental_budget=True)
        g, = torch.autograd.grad(out.sum(), (a,))
        values.append(out.detach().sum().item())
        with torch.no_grad():
            a.sub_(.01*g)
    assert calls == [torch.Size([1])]*2
    assert values[1] < values[0]


def test_principled_and_custom_facades_match_backend():
    import gffx
    module = api()
    n = t([[0., 0., 1.]])
    args = dict(albedo=t([.2, .4, .6]), roughness=t(.4), f0=t(.1),
                budget=budget(), allow_experimental_budget=True)
    torch.testing.assert_close(gffx.material.principled(n, n, n, **args), module.principled(n, n, n, **args))
    def lambert(n, v, l, *, parameters):
        return parameters['colour']/math.pi
    fields = dict(parameters={'colour': args['albedo']})
    torch.testing.assert_close(gffx.material.custom(lambert, n, n, n, **fields), module.custom(lambert, n, n, n, **fields))


@pytest.mark.parametrize('roughness,f0', [(.07, .04), (.4, .7), (1., 1.)])
def test_public_budget_mean_matches_independent_segment_quadrature(roughness, f0):
    from test_shading_public_edges import data
    from tools.shading_energy_probe import _gauss_unit
    value = api().PrincipledBudget.from_envelope(data(True), device='cpu', dtype=torch.float64, allow_unverified=True)
    r, f = t(roughness, True), t(f0, True)
    u, weights = _gauss_unit(32)
    total = t(0.)
    # Different integration algorithm: Gauss samples each interval of the public
    # curve, versus the implementation's constant analytic Bernstein moments.
    for lo, hi in zip(data(True)['payload']['cosine_nodes'][:-1], data(True)['payload']['cosine_nodes'][1:]):
        if lo == 0:
            c, jac = hi*u, torch.full_like(u, hi)
        else:
            width = math.log(hi/lo)
            c = lo*torch.exp(width*u)
            jac = c*width
        total = total+(2*c*value.directional_loss(c, r, f)*jac*weights).sum()
    exact = value.mean_loss(r, f)
    torch.testing.assert_close(exact, total, atol=1e-12, rtol=1e-11)
    for a, b in zip(torch.autograd.grad(exact, (r, f)), torch.autograd.grad(total, (r, f))):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)


def test_native_texture_roughness_fresnel_to_principled_gradcheck():
    from test_shading_public_sources import sample
    from gffx.torch import render
    value = budget()
    n = t([[0., 0., 1.]])
    texture = t([[[.3, .1, .2], [.4, .2, .3]], [[.5, .3, .4], [.6, .4, .5]]], True)
    uv = t([[.35, .45]], True)
    def evaluate(tex, coords):
        fields = sample(render, tex, coords)
        return api().principled(n, n, t([[.2, .3, 1.]]), albedo=t([.2, .4, .6]),
            roughness=fields[:, 0], f0=fields[:, 1:2], budget=value, allow_experimental_budget=True)
    assert torch.autograd.gradcheck(evaluate, (texture, uv), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_complete_material_demo_is_finite_and_labels_experimental_budget():
    from tools.testing_dashboard import complete_material_demo as demo
    result = demo.evaluate(size=16)
    assert result['scope'].startswith('Experimental')
    assert len(result['images']) == 5
    assert result['custom_max_abs_error'] <= 1e-12
    assert all(image.shape == (16, 16, 3) and torch.isfinite(image).all() for image in result['images'].values())
