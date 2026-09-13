"""Public material contract, section 8.12: deliberately red before implementation.

No private recipe is substituted for the public API. Synthetic envelopes test
plumbing, never physical accuracy or resource admission. Default installed-wheel
mode needs a functioning adapter; source mode requires its own native build.
"""
import copy
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def api():
    return importlib.import_module('gffx.torch.material')


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def fields(points=2):
    return dict(normals=t([[0., 0., 1.]] * points).reshape(points, 3),
                view=t([[0., 0., 1.]] * points).reshape(points, 3),
                directions=t([[0., 0., 1.], [0., 0., -1.]]),
                albedo=t([.2, .4, .6]), specular=t(.1), exponent=t(2.))


def digest(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def envelope():
    payload = dict(roughness_nodes=[.001, 1.], cosine_nodes=[0., 1.],
                   losses=[[[.8, 0.], [.6, .3]], [[.9, 0.], [.7, .4]]])
    return dict(format_version=1,
                model='ggx-correlated-smith-schlick-reciprocal-matte-v1',
                representation='bounded-cubic-loss-v1', dtype='float64',
                domain=dict(roughness=[.001, 1.], cosine=[0, 1], f0=[0, 1]),
                generator=dict(source_sha256=None, policy='synthetic test only'),
                evidence_id='synthetic-not-physical-acceptance',
                payload_sha256=digest(payload), payload=payload)


def resource(module):
    return module.PrincipledBudget.from_envelope(
        envelope(), device='cpu', dtype=torch.float64, allow_unverified=True)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_import_gffx_stays_lazy():
    """Plain import must not require torch or eagerly load material data."""
    import gffx
    code = (f'import sys; sys.path.insert(0, {str(Path(gffx.__file__).parent.parent)!r}); '
            'import gffx; assert "torch" not in sys.modules; '
            'assert "gffx.torch.material" not in sys.modules; '
            'assert "gffx._core" not in sys.modules')
    result = subprocess.run([sys.executable, '-I', '-c', code],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', ['classic', 'principled'])
def test_public_functions_have_explicit_keyword_contracts(name):
    """Users see the exact fields and explicit resource/experimental choices."""
    module = api()
    expected = {
        'classic': ('normals', 'view', 'directions', 'albedo', 'specular',
                    'exponent', 'two_sided'),
        'principled': ('normals', 'view', 'directions', 'albedo', 'roughness',
                       'f0', 'budget', 'two_sided', 'workspace_limit_bytes',
                       'allow_experimental_budget'),
    }
    names = expected[name]
    params = inspect.signature(getattr(module, name)).parameters
    assert tuple(params) == names
    assert all(params[key].kind == inspect.Parameter.KEYWORD_ONLY for key in names[3:])
    assert params['two_sided'].default is False
    if name == 'principled':
        assert params['budget'].default is inspect.Parameter.empty
        assert params['workspace_limit_bytes'].default == 64 * 1024**2
        assert params['allow_experimental_budget'].default is False


def test_facade_dispatch_matches_backend():
    """The friendly facade performs the same computation as the torch API."""
    module = api()
    import gffx
    torch.testing.assert_close(gffx.material.classic(**fields()), module.classic(**fields()))


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('layout', ['shared', 'point', 'explicit'])
def test_classic_shapes_and_closed_form(batched, layout):
    """Scene, surface-point and light axes must not be confused or inferred."""
    module = api()
    args = fields()
    if batched:
        args['normals'] = args['normals'].expand(3, 2, 3)
    if layout == 'point':
        args['directions'] = args['directions'].expand(2, 2, 3)
    elif layout == 'explicit':
        args['directions'] = args['directions'].expand(1, 2, 2, 3)
    out = module.classic(**args)
    expected = torch.zeros((3, 2, 2, 3) if batched else (2, 2, 3), dtype=torch.float64)
    expected[..., 0, :] = args['albedo'] / math.pi + .1 * 10 / (8 * math.pi)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)


def test_shared_fields_and_strided_inputs_preserve_gradients():
    """One shared colour receives all consumers' derivatives, even for strided views."""
    module = api()
    args = fields(4)
    args['normals'] = args['normals'][::2]
    args['view'] = args['view'][::2]
    args['albedo'].requires_grad_()
    out = module.classic(**args)
    gradient, = torch.autograd.grad(out.sum(), (args['albedo'],))
    torch.testing.assert_close(gradient, torch.full_like(gradient, 2 / math.pi),
                               atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('axis', ['points', 'lights'])
def test_empty_calls_keep_zero_gradient_connections(axis):
    """An empty image/light list has the right shape and a usable zero gradient."""
    module = api()
    args = fields(0 if axis == 'points' else 2)
    if axis == 'lights':
        args['directions'] = torch.empty((0, 3), dtype=torch.float64)
    args = {key: value.requires_grad_() for key, value in args.items()}
    out = module.classic(**args)
    assert out.shape == ((0, 2, 3) if axis == 'points' else (2, 0, 3))
    gradients = torch.autograd.grad(out.sum(), tuple(args.values()))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in gradients)


@pytest.mark.parametrize('bad,exception', [
    ('python_scalar', TypeError), ('dtype', TypeError), ('bool', TypeError),
    ('normal_zero', ValueError), ('nan', ValueError), ('rgb', ValueError),
    ('scene_lights', ValueError), ('exponent', ValueError), ('unknown', TypeError),
])
def test_invalid_material_fields_fail_explicitly(bad, exception):
    """Bad values and ambiguous layouts are rejected, not guessed or clamped."""
    module = api()
    args = fields()
    if bad == 'python_scalar': args['specular'] = .1
    elif bad == 'dtype': args['albedo'] = args['albedo'].float()
    elif bad == 'bool': args['two_sided'] = 1
    elif bad == 'normal_zero': args['normals'].zero_()
    elif bad == 'nan': args['view'][0, 0] = float('nan')
    elif bad == 'rgb': args['albedo'] = t([.1, .2])
    elif bad == 'scene_lights':
        args['normals'] = args['normals'].expand(3, 2, 3)
        args['directions'] = args['directions'].expand(3, 2, 3)  # B,Q,3 is not P,Q,3
    elif bad == 'exponent': args['exponent'] = t(-1.)
    elif bad == 'unknown': args['roughness'] = t(.5)
    with pytest.raises(exception):
        module.classic(**args)


@pytest.mark.parametrize('view,light,two_sided,active', [
    ([0, 0, 1], [0, 0, 1], False, True),
    ([0, 0, -1], [0, 0, -1], False, False),
    ([0, 0, -1], [0, 0, -1], True, True),
    ([1, 0, 0], [0, 0, 1], True, False),
    ([0, 0, 1], [1, 0, 0], False, False),
    ([0, 0, 1], [0, 0, -1], False, False),
])
def test_view_light_and_two_sided_boundaries(view, light, two_sided, active):
    """Backfaces/horizons produce finite zero results, including opposite directions."""
    module = api()
    args = fields(1)
    args.update(view=t([view]), directions=t([light]))
    args = {key: value.requires_grad_() for key, value in args.items()}
    out = module.classic(**args, two_sided=two_sided)
    assert torch.isfinite(out).all()
    if active:
        expected = args['albedo'] / math.pi + .1 * 10 / (8 * math.pi)
        torch.testing.assert_close(out[0, 0], expected, atol=1e-12, rtol=1e-12)
    else:
        assert torch.equal(out, torch.zeros_like(out))
        gradients = torch.autograd.grad(out.sum(), tuple(args.values()))
        assert all(torch.equal(g, torch.zeros_like(g)) for g in gradients)


def test_direction_normalization_and_interior_gradcheck():
    """Vector lengths do not change shading; fitting differentiates their directions."""
    module = api()
    n, v, l = t([[.1, .2, 1]], True), t([[.2, .1, 1]], True), t([[.3, -.1, 1]], True)
    def evaluate(n, v, l):
        return module.classic(n, v, l, albedo=t([.2, .3, .4]), specular=t(.1), exponent=t(2.3))
    torch.testing.assert_close(evaluate(n, v, l), evaluate(n * 3, v * 2, l * 4), atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(evaluate, (n, v, l), eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize('bad', ['extra', 'version', 'model', 'hash', 'domain',
                               'axis', 'bool_number', 'shape', 'horizon', 'oversized_axis'])
def test_malformed_envelopes_fail_before_tensor_allocation(monkeypatch, bad):
    """Untrusted resource metadata is checked before allocating constant tensors."""
    module = api()
    value = envelope()
    if bad == 'extra': value['accepted'] = True
    elif bad == 'version': value['format_version'] = 999
    elif bad == 'model': value['model'] = 'other-brdf'
    elif bad == 'hash': value['payload_sha256'] = '0' * 64
    elif bad == 'domain': value['domain']['roughness'] = [0, 1]
    elif bad == 'axis': value['payload']['roughness_nodes'] = [1., .001]
    elif bad == 'bool_number': value['payload']['cosine_nodes'][0] = False
    elif bad == 'shape': value['payload']['losses'][0].pop()
    elif bad == 'horizon': value['payload']['losses'][0][0][1] = .1
    elif bad == 'oversized_axis': value['payload']['roughness_nodes'] = [.001] * 130
    if bad not in ('hash',): value['payload_sha256'] = digest(value['payload'])
    def forbidden(*args, **kwargs):
        pytest.fail('tensor allocation before invalid envelope rejection')
    for name in ('tensor', 'as_tensor', 'empty', 'zeros', 'ones'):
        monkeypatch.setattr(torch, name, forbidden)
    with pytest.raises(ValueError):
        module.PrincipledBudget.from_envelope(value, device='cpu', dtype=torch.float64, allow_unverified=True)


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'oversized', 'malformed'])
def test_loader_preserves_file_and_format_error_categories(tmp_path, bad):
    """Missing files differ from malformed/oversized files; none triggers a download."""
    module = api()
    path = tmp_path / 'budget.json'
    if bad == 'duplicate': path.write_text('{"format_version":1,"format_version":1}', encoding='utf-8')
    elif bad == 'oversized': path.write_bytes(b' ' * (8 * 1024**2 + 1))
    elif bad == 'malformed': path.write_text('{not json}', encoding='utf-8')
    with pytest.raises(FileNotFoundError if bad == 'missing' else ValueError):
        module.PrincipledBudget.load(path, device='cpu', dtype=torch.float64, allow_unverified=True)


@pytest.mark.parametrize('label', ['synthetic-not-physical-acceptance', 'accepted', 'production-v1'])
def test_evidence_labels_cannot_grant_admission(label):
    """Calling a file accepted does not prove it has passed physical tests."""
    module = api()
    value = envelope()
    value['evidence_id'] = label
    with pytest.raises(ValueError):
        module.PrincipledBudget.from_envelope(value, device='cpu', dtype=torch.float64)
    candidate = module.PrincipledBudget.from_envelope(value, device='cpu', dtype=torch.float64, allow_unverified=True)
    assert candidate.metadata['admission'] == 'experimental'
    with pytest.raises(ValueError):
        module.PrincipledBudget.bundled('synthetic-test-never-shipped', device='cpu', dtype=torch.float64)


def test_principled_requires_separate_experimental_use_opt_in():
    """Loading research data does not silently enable it in a production material call."""
    module = api()
    candidate = resource(module)
    args = fields(1)
    args.pop('specular'); args.pop('exponent')
    args.update(roughness=t(.2), f0=t(.04), budget=candidate)
    with pytest.raises(ValueError): module.principled(**args)
    out = module.principled(**args, allow_experimental_budget=True)
    assert out.shape == (1, 2, 3) and torch.isfinite(out).all()
    assert torch.equal(out[:, 1], torch.zeros_like(out[:, 1]))


def test_resource_roundtrip_and_inspection_are_independent_copies(tmp_path):
    """Inspecting or changing caller-owned data must not alter a live material's constants."""
    module = api()
    value = envelope()
    candidate = module.PrincipledBudget.from_envelope(value, device='cpu', dtype=torch.float64, allow_unverified=True)
    expected = candidate.directional_loss(t(.4), t(.2), t(.04))
    original = copy.deepcopy(candidate.to_envelope())
    value['payload']['losses'][0][0][0] = .1
    candidate.metadata['admission'] = 'accepted'
    for tensor in candidate.inspect()['tensors'].values(): tensor.zero_()
    assert candidate.to_envelope() == original
    assert candidate.metadata['admission'] == 'experimental'
    path = tmp_path / 'synthetic.json'
    path.write_text(json.dumps(original), encoding='utf-8')
    loaded = module.PrincipledBudget.load(path, device='cpu', dtype=torch.float64, allow_unverified=True)
    torch.testing.assert_close(loaded.directional_loss(t(.4), t(.2), t(.04)), expected, atol=1e-12, rtol=1e-11)


def test_synthetic_fixture_has_no_physical_admission_claim():
    """Check the test data itself without depending on the missing public implementation."""
    value = envelope()
    assert value['payload_sha256'] == digest(value['payload'])
    assert value['generator']['source_sha256'] is None
    assert value['evidence_id'] == 'synthetic-not-physical-acceptance'
    assert set(value) == {'format_version', 'model', 'representation', 'dtype', 'domain',
                          'generator', 'evidence_id', 'payload_sha256', 'payload'}
