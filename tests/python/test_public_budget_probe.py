"""Test the prospective physical gate without generating a lookup table."""
import math
import pytest
from tools import public_budget_probe as probe


def test_guard_cohort_counts_and_conditional_boundary():
    points = list(probe.guard_points())
    assert len(points) == len(set(points)) == 40
    assert len(probe.directional_points()) == 80
    for r in probe.ROUGHNESSES:
        assert all((r, r*r/16*k) in points for k in (.999, 1., 1.001))


def test_guard_runs_both_orders_and_preserves_every_endpoint():
    calls, snapshots = [], []
    def pair(c, r, *, resolution, azimuth_resolution):
        calls.append((c, r, resolution, azimuth_resolution))
        return (.5, 0.)
    rows = probe.run_guard(pair, lambda: None, lambda rows: snapshots.append(len(rows)))
    assert len(rows) == 80 and snapshots[-1] == 80
    assert len(calls) == 80
    assert {x[2:] for x in calls} == {(280, 32), (512, 64)}
    assert probe.guard_passed(rows)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -.1, 1.1])
def test_guard_rejects_nonphysical_reference_values(bad):
    rows = probe.run_guard(lambda *a, **k: (bad, 0.), lambda: None, lambda rows: None)
    assert not probe.guard_passed(rows)


@pytest.mark.parametrize('count', [0, 79, 81])
def test_guard_cannot_pass_incomplete_evidence(count):
    rows = [dict(valid=True, error=0., limit=1e-12)] * count
    assert not probe.guard_passed(rows)


def test_guard_failure_is_preserved_without_relaxing_tolerance():
    def pair(c, r, *, resolution, **kwargs):
        return (.5 + (1e-4 if resolution == 280 else 0.), 0.)
    rows = probe.run_guard(pair, lambda: None, lambda rows: None)
    assert len(rows) == 80 and not probe.guard_passed(rows)
    assert rows[0]['limit'] == .00005


def test_guard_checks_deadline_before_each_reference_call():
    calls = []
    def stop():
        raise TimeoutError('bounded worker')
    with pytest.raises(TimeoutError):
        probe.run_guard(lambda *a, **k: calls.append(1), stop, lambda rows: None)
    assert not calls


def test_preflight_rejects_changed_reference_before_fixture_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'FROZEN', {'reference.py': '0'*64})
    (tmp_path/'reference.py').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='reference SHA'):
        probe.verify_references(tmp_path)


@pytest.mark.parametrize('mode', ['directional', 'mean', 'furnace'])
def test_physical_stage_requires_successful_supervised_predecessor(mode):
    with pytest.raises(ValueError, match='prerequisite'):
        probe.verify_predecessor({}, mode)


def test_public_adapter_never_substitutes_private_evaluation():
    import torch
    calls = []
    class Public:
        def directional_loss(self, c, r, f):
            calls.append('directional')
            return c*r*f
        def mean_loss(self, r, f):
            calls.append('mean')
            return r+f
    class Oracle:
        def integrate_for_test(self, r, f, *, order):
            calls.append('quadrature')
            return r+f
    adapter = probe.PublicComparison(Public(), Oracle(), lambda: None)
    c, r, f = [torch.tensor(v, requires_grad=True) for v in (.3, .2, .4)]
    result = adapter.evaluate(c, r, f)
    assert len(torch.autograd.grad(result['loss'], (c, r, f))) == 3
    assert calls == ['directional', 'mean']
    assert adapter.integrate_for_test(r, f, order=16) == r+f
    assert calls[-1] == 'quadrature'
