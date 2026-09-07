"""Phase-0 math gates, independent of gffx implementation.

Preserve the rejected model as a negative control alongside the new energy gate.
No clamp, xfail or relaxed energy threshold is used to accept the revision.
"""
import sys
import math
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.append(str(ROOT))
from tools.shading_energy_probe import furnace
from tools import shading_energy_probe as probe


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(previous)


def test_white_furnace_quadrature_integrates_lambert():
    for resolution in (32,64,128):
        assert abs(furnace(.5,.5,resolution=resolution,model='lambert')-1) < 1e-12


def test_white_furnace_checks_single_scattering_ggx_separately():
    # A constant F=1 microfacet reflector loses energy at high roughness; no diffuse.
    energy=furnace(1.,1.,resolution=512,model='unit_fresnel_ggx')
    assert abs(energy-(1-math.log(2))) < 1e-6


def test_rejected_local_fresnel_mixture_still_overreflects(request):
    coarse=furnace(.1,.2,resolution=1024)
    fine=furnace(.1,.2,resolution=2048)
    measurement={
        'scope':'proposed formula, not implemented gffx shading',
        'incoming_radiance':1.,'albedo':1.,'f0':.04,'roughness':.2,'cos_view':.1,
        '1024x2048':coarse,'2048x4096':fine,'allowed_upper_bound':1.001}
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin:plugin.Evidence(request.node.nodeid).record('white-furnace-candidate',measurement)
    assert abs(fine-coarse) < .001, 'Refine the integration before interpreting an energy failure'
    assert fine > 1.001, 'The old formula is a negative control, not an accepted material'
    assert fine == pytest.approx(1.4278734917345142, abs=2e-6)


def test_revised_coupling_retains_energy_and_reciprocity():
    # Constant directional albedo has a closed-form coupled diffuse value.
    a = torch.tensor([0., .5, 1.], dtype=torch.float64)
    e = torch.tensor(.25, dtype=torch.float64)
    assert torch.allclose(probe.coupled_diffuse(a, e, e, e), a*.75/math.pi)
    left = probe.coupled_diffuse(a, e, e*2, e)
    right = probe.coupled_diffuse(a, e*2, e, e)
    assert torch.equal(left, right)


def test_revised_coupling_has_interior_gradients_and_safe_zero_budget():
    args = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True)
                 for x in (.6, .2, .4, .3))
    assert torch.autograd.gradcheck(probe.coupled_diffuse, args)
    one = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    result = probe.coupled_diffuse(one, one, one, one)
    assert result == 0
    result.backward()
    assert one.grad == 0


def test_vndf_energy_matches_independent_hemisphere_integral():
    for view, roughness in ((1., 1.), (.5, .5), (.1, .2)):
        direct = probe.furnace(view, roughness, resolution=2048, model='specular')
        coarse = probe.directional_albedo(view, roughness, resolution=128)
        fine = probe.directional_albedo(view, roughness, resolution=256)
        assert abs(fine-coarse) < .001
        assert abs(fine-direct) < .001


def test_revised_principled_satisfies_white_furnace(request):
    report = probe.revision_report()
    rows = report['rows']
    assert len(rows) == 108
    plugin = request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin:
        plugin.Evidence(request.node.nodeid).record('revised-white-furnace', report)
    assert max(row['delta'] for row in rows) < .001
    assert all(0 <= row['fine'] <= 1.001 for row in rows)
    assert all(abs(row['fine']-1) < .001 for row in rows if row['albedo'] == 1)


def test_revised_coupling_rejects_invalid_budgets():
    for values in ((1., -.1, .3, .2), (1.1, .2, .3, .2), (1., .2, .3, 1.),
                   (1., float('nan'), .3, .2)):
        with pytest.raises(ValueError):
            probe.coupled_diffuse(*(torch.tensor(x) for x in values))


def test_gauss_quadrature_and_nonconstant_budget_have_known_integrals():
    mu, weights = probe._gauss_unit(32)
    assert bool(((mu > 0) & (mu < 1) & (weights > 0)).all())
    for power in range(8):
        assert float((mu**power*weights).sum()) == pytest.approx(1/(power+1), abs=1e-14)
    # E(mu)=0.2+0.3*mu gives mean E=0.4; use that independent analytic mean.
    e_view = torch.tensor(.27, dtype=torch.float64)
    e_average = torch.tensor(.4, dtype=torch.float64)
    albedo = torch.tensor(.6, dtype=torch.float64)
    diffuse = probe.coupled_diffuse(albedo, e_view, .2+.3*mu, e_average)
    assert float((diffuse*mu*weights).sum()*2*math.pi) == pytest.approx(.6*(1-.27), abs=1e-14)
    zero = torch.tensor(0., dtype=torch.float64)
    assert probe.coupled_diffuse(albedo, zero, zero, zero) == pytest.approx(.6/math.pi)
