"""Public installed-wheel shading: layouts, boundaries and independent integrals."""
import importlib
import math
import subprocess
import sys

import pytest
import torch

NAMES = ('lambert', 'blinn_phong', 'ggx_distribution', 'smith_visibility', 'fresnel_schlick')


def shade():
    return importlib.import_module('gffx.torch.shade')


def t(value, grad=False, dtype=torch.float64):
    return torch.tensor(value, dtype=dtype, requires_grad=grad)


def call(name, q, parameter):
    fn = getattr(shade(), name)
    if name == 'lambert':
        return fn(q[..., None].expand(*q.shape, 3))
    if name == 'smith_visibility':
        return fn(q, q, parameter)
    return fn(q, parameter)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_public_lobe_strided_batches_match_values_and_shared_gradients(name, dtype):
    source = t([[.2, .3, .4, .5], [.6, .7, .8, .9]], True, dtype)
    q = source[:, ::2]
    parameter = t([.1, .2, .3] if name == 'fresnel_schlick' else .6, True, dtype)
    before = source.detach().clone()
    together = call(name, q, parameter)
    separate = torch.stack([call(name, row, parameter) for row in q])
    torch.testing.assert_close(together, separate)
    assert torch.equal(together, call(name, q, parameter))
    inputs = (source,) if name == 'lambert' else (source, parameter)
    actual = torch.autograd.grad(together.sum(), inputs, retain_graph=True)
    expected = torch.autograd.grad(separate.sum(), inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert torch.equal(source, before)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('batched', [False, True])
def test_public_lobe_empty_calls_keep_connected_gradients(name, batched):
    q = torch.empty((2, 0) if batched else (0,), dtype=torch.float64, requires_grad=True)
    p = t([.1, .2, .3] if name == 'fresnel_schlick' else .6, True)
    value = call(name, q, p)
    expected = q.shape + ((3,) if name in ('lambert', 'fresnel_schlick') else ())
    assert value.shape == expected
    for g in torch.autograd.grad(value.sum(), (q,) if name == 'lambert' else (q, p)):
        assert torch.equal(g, torch.zeros_like(g))


@pytest.mark.parametrize('shape', [(), (3,), (1, 3), (2, 3)])
def test_public_scalar_sharing_does_not_expand_query(shape):
    q = t([[.2, .4, .6], [.3, .5, .7]])
    n = torch.full(shape, 2., dtype=torch.float64, requires_grad=True)
    out = shade().blinn_phong(q, n)
    torch.testing.assert_close(out, q.square() * 10 / (8 * math.pi))
    assert out.shape == q.shape
    out.sum().backward()
    assert n.grad.shape == n.shape and torch.isfinite(n.grad).all()


@pytest.mark.parametrize('shape', [(3,), (3, 3), (1, 3, 3), (2, 3, 3)])
def test_public_rgb_sharing_keeps_explicit_channels(shape):
    q = t([[.2, .4, .6], [.3, .5, .7]])
    f0 = torch.full(shape, .2, dtype=torch.float64, requires_grad=True)
    out = shade().fresnel_schlick(q, f0)
    expected = (.2 + .8 * (1-q[..., None]).pow(5)).expand(2, 3, 3)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    assert f0.grad.shape == f0.shape and torch.isfinite(f0.grad).all()


@pytest.mark.parametrize('case', ['scalar-query', 'empty-batch', 'point-broadcast',
    'batch-expansion', 'per-scene-scalar', 'rgb-scalar', 'rgb-point-broadcast',
    'lambert-channel', 'lambert-empty-batch', 'non-tensor', 'integer', 'half',
    'mixed-dtype', 'mixed-device', 'cosine-low', 'cosine-high', 'nan', 'infinity',
    'negative-exponent', 'empty-invalid-roughness', 'empty-invalid-rgb'])
def test_public_lobe_rejections_name_the_input_category(case):
    m = shade()
    q, n = t([[.2, .4, .6], [.3, .5, .7]]), t(2.)
    fn, args, error, match = m.blinn_phong, (q, n), ValueError, 'shape|batch'
    if case == 'scalar-query': args = (t(.5), n)
    elif case == 'empty-batch': args = (torch.empty(0, 3, dtype=torch.float64), n)
    elif case == 'point-broadcast': args = (q, t([2.]))
    elif case == 'batch-expansion': args = (q[0], t([[2., 2., 2.], [2., 2., 2.]]))
    elif case == 'per-scene-scalar': args = (q, t([2., 3.]))
    elif case.startswith('rgb-'):
        fn, args = m.fresnel_schlick, (q, t(.2) if case == 'rgb-scalar' else t([[.1, .2, .3]]))
    elif case.startswith('lambert-'):
        fn, args = m.lambert, (torch.empty(0, 3, 3, dtype=torch.float64) if case.endswith('empty-batch') else t([[.1, .2]]),)
    elif case == 'non-tensor': args, error, match = (q, 2.), TypeError, 'tensor'
    elif case in ('integer', 'half', 'mixed-dtype'):
        dtype = {'integer': torch.int64, 'half': torch.float16, 'mixed-dtype': torch.float32}[case]
        args, error, match = (q, torch.tensor(2, dtype=dtype)), TypeError, 'dtype'
    elif case == 'mixed-device': args, match = (q, torch.empty((), device='meta', dtype=torch.float64)), 'device'
    elif case in ('cosine-low', 'cosine-high', 'nan', 'infinity'):
        bad = {'cosine-low': -1e-8, 'cosine-high': 1+1e-8, 'nan': float('nan'), 'infinity': float('inf')}[case]
        fn, args, match = m.fresnel_schlick, (t([bad]), t([.1, .2, .3])), 'v_dot_h'
    elif case == 'negative-exponent': args, match = (q, t(-1.)), 'exponent'
    elif case == 'empty-invalid-roughness': fn, args, match = m.ggx_distribution, (t([]), t(0.)), 'roughness'
    elif case == 'empty-invalid-rgb': fn, args, match = m.fresnel_schlick, (t([]), t([.1, .2, float('nan')])), 'f0'
    with pytest.raises(error, match=match):
        fn(*args)


@pytest.mark.parametrize('dtype,tiny', [(torch.float32, 1e-8), (torch.float64, 1e-60)])
def test_public_ggx_lost_squared_divisor_and_smith_mask_are_explicit(dtype, tiny):
    # r^4 survives, but its square at c=1 is not representable.
    with pytest.raises(ValueError, match='represent'):
        shade().ggx_distribution(t([1.], dtype=dtype), t(tiny, dtype=dtype))
    v, l, r = t([0., .5, 0.], True, dtype), t([.5, 0., 0.], True, dtype), t(tiny, True, dtype)
    result = shade().smith_visibility(v, l, r)
    assert torch.equal(result, torch.zeros_like(result))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in torch.autograd.grad(result.sum(), (v, l, r)))


def test_public_schlick_endpoints_have_polynomial_derivatives():
    c, f0 = t([0., 1.], True), t([0., .2, 1.], True)
    value = shade().fresnel_schlick(c, f0)
    gc, gf = torch.autograd.grad(value.sum(), (c, f0))
    torch.testing.assert_close(gc, t([-5*(3-1.2), 0.]))
    torch.testing.assert_close(gf, torch.ones_like(f0))


@pytest.mark.parametrize('roughness', [.2, .5, 1.])
def test_public_ggx_projected_area_is_normalized(roughness):
    c = (torch.arange(65536, dtype=torch.float64)+.5)/65536
    integral = 2*math.pi*(shade().ggx_distribution(c, t(roughness))*c).mean()
    assert integral.item() == pytest.approx(1., abs=5e-5)


def test_public_microfacet_product_has_independent_roughness_one_furnace():
    m = shade()
    c = (torch.arange(16384, dtype=torch.float64)+.5)/16384
    h = ((1+c)/2).sqrt()
    brdf = m.ggx_distribution(h, t(1.))*m.smith_visibility(torch.ones_like(c), c, t(1.))
    brdf = brdf*m.fresnel_schlick(h, t([1., 1., 1.]))[:, 0]
    assert (2*math.pi*(brdf*c).mean()).item() == pytest.approx(1-math.log(2), abs=1e-9)


def test_public_shade_facade_is_lazy_and_matches_adapter():
    child = subprocess.run([sys.executable, '-c',
        "import sys; import gffx; import gffx.shade; "
        "assert 'torch' not in sys.modules; assert 'gffx._core' not in sys.modules; "
        "assert 'gffx._torch' not in sys.modules"], capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    import gffx
    assert torch.equal(gffx.shade.lambert(t([[.1, .2, .3]])), shade().lambert(t([[.1, .2, .3]])))


def test_public_lobe_demo_calls_installed_functions_and_labels_scope(monkeypatch):
    from tools.testing_dashboard import shade_demo
    m = shade()
    called = set()
    with monkeypatch.context() as patch:
        for name in NAMES:
            original = getattr(m, name)
            def observe(*args, name=name, original=original):
                called.add(name)
                return original(*args)
            patch.setattr(m, name, observe)
        data = shade_demo.examples(public=True)
    assert called == set(NAMES)
    assert data['scope'].startswith('Public elementary')
    assert data['image'].shape == (360, 1280, 3) and data['image'].dtype == torch.uint8
    assert len(data['angles']) == 181
