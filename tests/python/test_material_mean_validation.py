"""Test-only streaming gradients preserve the frozen mean oracle's quadrature."""
import gc
import pytest

torch = pytest.importorskip('torch')


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def inputs(r=.173, f=.41):
    return tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (r, f))


@pytest.mark.parametrize('roughness,fresnel', [(.00473, 0.), (.173, .41), (.733, 1.)])
@pytest.mark.parametrize('method', ['sinh', 'log'])
def test_streamed_mean_preserves_value_and_moving_view_derivatives(roughness, fresnel, method):
    from tools.material_loss_reference import average
    from tools.material_mean_validation import average_value_and_gradients
    r, f = inputs(roughness, fresnel)
    expected = average(r, f, resolution=16, view_resolution=16, method=method)
    gradients = torch.autograd.grad(expected, (r, f))
    actual, observed = average_value_and_gradients(r, f, resolution=16, view_resolution=16, method=method)
    assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-12)
    assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-11) for a, b in zip(observed, gradients))
    assert not actual.requires_grad and all(not g.requires_grad for g in observed)
    assert r.grad is None and f.grad is None


def saved_peak(call):
    state = dict(live=0, peak=0)
    class Saved:
        def __init__(self, tensor):
            self.tensor = tensor
            self.size = tensor.numel()*tensor.element_size()
            state['live'] += self.size
            state['peak'] = max(state['peak'], state['live'])
        def __del__(self):
            state['live'] -= self.size
    with torch.autograd.graph.saved_tensors_hooks(Saved, lambda item: item.tensor):
        call()
    gc.collect()
    assert state['live'] == 0
    return state['peak']


def test_reference_graph_pressure_is_bounded_per_view_not_all_views():
    from tools.material_loss_reference import average
    from tools.material_mean_validation import average_value_and_gradients
    def legacy():
        r, f = inputs()
        torch.autograd.grad(average(r, f, resolution=16, view_resolution=16), (r, f))
    def streamed(views):
        return lambda: average_value_and_gradients(*inputs(), resolution=16, view_resolution=views)
    old = saved_peak(legacy)
    small = saved_peak(streamed(16))
    large = saved_peak(streamed(32))
    assert large <= small*1.1
    assert old > large*4
    print(dict(legacy_saved_reference_bytes=old, streamed_16=small, streamed_32=large,
               qualification='logical saved references, not unique storage or RSS'))


def test_streaming_preserves_sample_order_count_and_finishes_each_backward(monkeypatch):
    from tools import material_loss_reference as reference
    from tools.material_mean_validation import average_value_and_gradients
    from tools.shading_energy_probe import _gauss_unit
    seen = []
    state = dict(pending=0)
    class Term(torch.autograd.Function):
        @staticmethod
        def forward(ctx, c, r, f):
            assert state['pending'] == 0
            state['pending'] += 1
            return c+r+f
        @staticmethod
        def backward(ctx, grad):
            state['pending'] -= 1
            return grad, grad, grad
    def directional(c, r, f, **kwargs):
        seen.append((c.item(), kwargs))
        return Term.apply(c, r, f)
    monkeypatch.setattr(reference, 'directional', directional)
    r, f = inputs()
    value, gradients = average_value_and_gradients(r, f, resolution=16, view_resolution=16)
    u, w = _gauss_unit(16)
    a = r.square(); bound = torch.asinh(1/a)
    cosines = a*torch.sinh(u*bound)
    jacobian = a*torch.cosh(u*bound)*bound
    expected = (2*(cosines+r+f)*cosines*jacobian*w).sum()
    assert [x for x, _ in seen] == cosines.tolist()
    assert all(k == dict(resolution=16, method='sinh') for _, k in seen)
    assert state['pending'] == 0
    assert torch.allclose(value, expected, atol=1e-13, rtol=1e-12)
    assert all(torch.allclose(x,y,atol=1e-12,rtol=1e-11) for x,y in zip(gradients, torch.autograd.grad(expected,(r,f))))


@pytest.mark.parametrize('kwargs', [dict(resolution=256,view_resolution=64), dict(view_resolution=128), dict(method='unknown')])
def test_streamed_oracle_rejects_invalid_work_before_any_directional_call(monkeypatch, kwargs):
    from tools import material_loss_reference as reference
    from tools.material_mean_validation import average_value_and_gradients
    monkeypatch.setattr(reference,'directional',lambda *a,**k:pytest.fail('allocated before validation'))
    with pytest.raises(ValueError): average_value_and_gradients(*inputs(), **kwargs)
