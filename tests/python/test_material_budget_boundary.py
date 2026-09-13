"""Predeclared boundary screens: measure reference failures, never relax gates."""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from tools import material_budget_sharp as sharp
from tools import material_loss_reference as loss
from tools import shading_energy_probe as oracle
from tools.testing_dashboard.pytest_plugin import Evidence


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def test_residual_probe_has_explicit_methods_and_work_limits():
    from tools import material_residual_probe as probe
    from tools.testing_dashboard.explanations import explain
    for name in ('Reference value refinement', 'Reference partial derivative refinement',
                 'Independent oracle refinement and agreement', 'Reflected energy range'):
        assert explain('tests/python/test_material_budget_boundary.py::test_coupled_grazing_reference_converges', name)['authored']
    for name in ('Positive and relatively resolved remaining budget',
                 'Residual derivative refinement', 'White furnace under both normalizations'):
        assert explain('tests/python/test_material_budget_boundary.py::test_tiny_mean_residual_is_resolved', name)['authored']
    expected = 1-4*(1-math.log(2))/3
    for method in ('legacy', 'direct', 'transformed', 'independent'):
        value = probe.residual(t(1.), t(1.), resolution=128,
                               view_resolution=64, method=method)
        assert abs(value.item()-expected) < 5e-5
    for kwargs in (dict(method='automatic'), dict(resolution=256, view_resolution=64)):
        with pytest.raises(ValueError):
            probe.residual(t(.01), t(1.), **kwargs)
    with pytest.raises(ValueError, match='detached'):
        probe.residual(t(.01, True), t(1.), method='independent')


def test_residual_demo_publishes_unresolved_evidence(tmp_path, monkeypatch):
    from tools.testing_dashboard import residual_demo as demo
    from tools.testing_dashboard.server import TestingServer
    data = dict(rows=[dict(roughness=.01, f0=1., independent_disagreement=.2,
        methods={key:dict(fine=1e-7,relative_change=.01) for key in
                 ('legacy','direct','transformed','independent')})], scope='fixture')
    monkeypatch.setattr(demo.probe, 'study', lambda:data)
    with TestingServer(tmp_path, port=0, allow_loopback_for_tests=True) as server:
        demo.publish('http://'+server.address, 'residual-demo-protocol')
        page = server.index.page('shading/residual-demo')
        assert page['evidence']['exit_code'] == 0
        assert len(page['artifacts']) == 5


def measurement(c, r, f, resolution):
    args = tuple(t(x, True) for x in (c, r, f))
    value = sharp.directional(*args, resolution=resolution)
    return value.item(), [g.item() for g in torch.autograd.grad(value, args)]


@pytest.mark.parametrize("roughness", [.001, .01, .05, .199])
@pytest.mark.parametrize("scale", [.01, .1, 1., 10.])
def test_coupled_grazing_reference_converges(roughness, scale, request):
    cosine = scale * roughness**2
    rows = []
    for f0 in (0., .04, 1.):
        a, ga = measurement(cosine, roughness, f0, 256)
        b, gb = measurement(cosine, roughness, f0, 512)
        va, vb = [oracle.directional_albedo(cosine, roughness, f0=f0,
                    resolution=n, radial_rule="tail-dense") for n in (256, 512)]
        rows.append(dict(roughness=roughness, cosine=cosine, scale=scale, f0=f0,
            value=b, value_change=abs(b-a), gradients=gb,
            gradient_change=[abs(x-y) for x, y in zip(ga, gb)],
            independent_value=vb, independent_change=abs(vb-va),
            independent_error=abs(b-vb)))
    evidence = Evidence(request.node.nodeid)
    evidence.record("coupled-boundary", dict(rows=rows,
        scope="48 finite triples; derivatives hold cosine, roughness and F0 independently. Not fast-table or all-domain admission."))
    # All measurements are retained even when the first admission check fails.
    with evidence.check("Reference value refinement"):
        assert all(x["value_change"] <= 5e-5 for x in rows), rows
    with evidence.check("Reference partial derivative refinement"):
        assert all(e <= .001+.002*abs(g) for x in rows
                   for e, g in zip(x["gradient_change"], x["gradients"])), rows
    with evidence.check("Independent oracle refinement and agreement"):
        assert all(x["independent_change"] <= 5e-5 and
                   x["independent_error"] <= 5e-5 for x in rows), rows
    with evidence.check("Reflected energy range"):
        assert all(0 <= x["value"] <= 1.001 for x in rows), rows


@pytest.mark.parametrize("roughness", [.001, .003, .01])
@pytest.mark.parametrize("f0", [1-1e-8, 1.])
def test_tiny_mean_residual_is_resolved(roughness, f0, request):
    r, f = t(roughness, True), t(f0, True)
    # The contract requires a residual, not subtraction from a rounded mean.
    # Historical failing subtraction is retained as a separate negative control.
    coarse = loss.average(r, f, resolution=64, view_resolution=32)
    fine = loss.average(r, f, resolution=128, view_resolution=64)
    gc = torch.autograd.grad(coarse, (r, f))
    gf = torch.autograd.grad(fine, (r, f))
    cv, fv = coarse.item(), fine.item()
    view_residual = loss.directional(t(.43), t(roughness), t(f0), resolution=512).item()
    specular = oracle.directional_albedo(.43, roughness, f0=f0,
                                         resolution=512, radial_rule="tail-dense")
    row = dict(roughness=roughness, f0=f0, coarse_residual=cv, fine_residual=fv,
        residual_change=abs(fv-cv), gradients=[x.item() for x in gf],
        gradient_change=[abs(x.item()-y.item()) for x, y in zip(gc, gf)],
        relative_change=abs(fv-cv)/abs(fv) if fv else None,
        forward_energy=specular+view_residual*fv/cv if cv else None,
        reverse_energy=specular+view_residual*cv/fv if fv else None)
    evidence = Evidence(request.node.nodeid)
    evidence.record("tiny-residual", row)
    with evidence.check("Positive and relatively resolved remaining budget"):
        assert cv > 0 and fv > 0, row
        assert row["relative_change"] <= .001, row
    with evidence.check("Residual derivative refinement"):
        assert all(e <= .001+.002*abs(g) for e, g in
                   zip(row["gradient_change"], row["gradients"])), row
    with evidence.check("White furnace under both normalizations"):
        assert all(0 <= row[key] <= 1.001 and abs(row[key]-1) < .001
                   for key in ("forward_energy", "reverse_energy")), row


@pytest.mark.parametrize("exponent", [10, 30, 45, 50])
def test_tiny_synthetic_budget_retains_signal_and_gradients(exponent):
    # Binary-exact residuals isolate arithmetic from quadrature error.
    d = t(2.**-exponent, True)
    v, l, mean = 1-2*d, 1-3*d, 1-d
    value = oracle.coupled_diffuse(t(.4), v, l, mean)
    expected = .4*6*d/math.pi
    assert torch.allclose(value, expected, rtol=1e-12, atol=0)
    derivative = torch.autograd.grad(value, d)[0]
    assert abs(derivative.item()-.4*6/math.pi) < 1e-12
    with pytest.raises(ValueError, match="zero average residual"):
        oracle.coupled_diffuse(t(.4), v.detach(), l.detach(), t(1.))
