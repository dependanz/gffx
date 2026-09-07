"""Differentiable quadrature controls, not a shipping material or speed claim."""
import importlib
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from tools import shading_energy_probe as oracle


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def recipe():
    return importlib.import_module('tools.material_budget_reference')


def scalar(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize('cosine', [.13, .47, .86])
def test_budget_has_independent_analytic_value_and_view_derivative(cosine):
    c = scalar(cosine, True)
    value = recipe().directional(c, scalar(1.), scalar(1.), resolution=64)
    expected = 1 - cosine * math.log((1 + cosine) / cosine)
    derivative = -math.log((1 + cosine) / cosine) + 1 / (1 + cosine)
    assert value.item() == pytest.approx(expected, abs=1e-12)
    assert torch.autograd.grad(value, c)[0].item() == pytest.approx(derivative, abs=1e-11)


def test_average_budget_has_independent_analytic_integral():
    value = recipe().average(scalar(1.), scalar(1.), resolution=64, view_resolution=32)
    assert value.item() == pytest.approx(4 * (1 - math.log(2)) / 3, abs=2e-9)


@pytest.mark.parametrize('cosine,roughness,f0', [(.17, .37, .11), (.43, .63, .29), (.79, .87, .73)])
def test_budget_held_out_values_and_gradients_against_independent_oracles(cosine, roughness, f0, request):
    args = tuple(scalar(x, True) for x in (cosine, roughness, f0))
    result = recipe().directional(*args, resolution=256)
    fine = recipe().directional(*args, resolution=512)
    gradients = torch.autograd.grad(result, args)
    reference = oracle.directional_albedo(cosine, roughness, f0=f0, resolution=512)
    # Uniform direct-hemisphere integration differs from the recipe's Gauss grid.
    # Refinement also checks the finite-difference reference, not just the recipe.
    delta = 1e-4
    derivatives = []
    refinements = []
    for axis in range(3):
        estimates = []
        for resolution in (1024, 2048):
            values = []
            for sign in (-1, 1):
                point = [cosine, roughness, f0]
                point[axis] += sign * delta
                values.append(oracle.furnace(point[0], point[1], f0=point[2],
                                             model='specular', resolution=resolution))
            estimates.append((values[1] - values[0]) / (2 * delta))
        refinements.append(abs(estimates[1] - estimates[0]))
        derivatives.append(estimates[1])
    report = dict(cosine=cosine, roughness=roughness, f0=f0, value=result.item(),
                  independent_vndf=reference, refinement=abs(fine.item()-result.item()),
                  gradients=[x.item() for x in gradients], independent_gradients=derivatives,
                  derivative_refinements=refinements)
    plugin = request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin:
        plugin.Evidence(request.node.nodeid).record('independent-budget-gradients', report)
    assert report['refinement'] < 1e-5
    assert abs(result.item() - reference) < 2e-5
    assert max(refinements) < 1e-4
    assert max(abs(g.item() - d) for g, d in zip(gradients, derivatives)) < 5e-4


def test_schlick_budget_is_affine_in_f0_without_detaching():
    mod = recipe()
    r, f = scalar(.67, True), scalar(.31, True)
    for evaluate in (lambda x: mod.directional(scalar(.41), r, x, resolution=64),
                     lambda x: mod.average(r, x, resolution=64, view_resolution=16)):
        value, zero, one = evaluate(f), evaluate(scalar(0.)), evaluate(scalar(1.))
        assert torch.allclose(value, zero + f * (one-zero), atol=1e-13, rtol=1e-13)
        dr, df = torch.autograd.grad(value, (r, f))
        assert torch.isfinite(dr) and abs(dr.item()) > 1e-4
        assert df.item() == pytest.approx((one-zero).item(), abs=1e-12)


def test_coupled_matte_gradients_include_all_three_computed_budgets():
    mod = recipe()
    def evaluate(a, cv, cl, r, f):
        ev = mod.directional(cv, r, f, resolution=48)
        el = mod.directional(cl, r, f, resolution=48)
        mean = mod.average(r, f, resolution=48, view_resolution=16)
        return oracle.coupled_diffuse(a, ev, el, mean)
    args = tuple(scalar(x, True) for x in (.6, .4, .8, .7, .25))
    assert torch.autograd.gradcheck(evaluate, args, eps=1e-6, atol=1e-5, rtol=1e-3)
    gradients = torch.autograd.grad(evaluate(*args), args)
    assert all(torch.isfinite(g) and abs(g.item()) > 1e-5 for g in gradients)
    assert torch.allclose(evaluate(*args), evaluate(args[0], args[2], args[1], *args[3:]), atol=1e-14)


def test_budget_quadrature_refines_average_values_and_gradients():
    mod = recipe()
    estimates = []
    for light, view in ((128, 32), (128, 64)):
        r, f = scalar(.67, True), scalar(.31, True)
        value = mod.average(r, f, resolution=light, view_resolution=view)
        estimates.append([value.item(), *(g.item() for g in torch.autograd.grad(value, (r, f)))])
    assert max(abs(a-b) for a, b in zip(*estimates)) < 1e-5


def test_budget_rejects_invalid_domain_type_device_and_work_size():
    mod = recipe()
    for values in ((0., .5, .2), (1., .5, .2), (.5, 0., .2), (.5, 1.1, .2),
                   (.5, .5, -.1), (.5, .5, float('nan'))):
        with pytest.raises(ValueError):
            mod.directional(*(scalar(x) for x in values))
    with pytest.raises(TypeError):
        mod.directional(.5, scalar(.5), scalar(.2))
    with pytest.raises(TypeError):
        mod.directional(scalar(.5).float(), scalar(.5), scalar(.2))
    with pytest.raises(ValueError):
        mod.directional(scalar(.5).reshape(1), scalar(.5), scalar(.2))
    with pytest.raises(ValueError):
        mod.directional(torch.empty((), device='meta', dtype=torch.float64), scalar(.5), scalar(.2))
    for size in (True, 8, 513, 32.5):
        with pytest.raises(ValueError):
            mod.directional(scalar(.5), scalar(.5), scalar(.2), resolution=size)
    with pytest.raises(ValueError, match='work'):
        mod.average(scalar(.5), scalar(.2), resolution=512, view_resolution=64)
    with pytest.raises(ValueError, match='represent'):
        mod.directional(scalar(.5), scalar(1e-100), scalar(.2))


def test_budget_repeats_without_mutation_or_retained_parameter_cache():
    mod = recipe()
    args = tuple(scalar(x, True) for x in (.4, .7, .2))
    before = tuple(x.clone() for x in args)
    first = mod.directional(*args, resolution=32)
    first.backward()
    for x in args:
        x.grad = None
    second = mod.directional(*args, resolution=32)
    second.backward()
    assert torch.equal(first, second)
    assert all(torch.equal(a, b) for a, b in zip(args, before))
    assert all(x.grad is not None and torch.isfinite(x.grad) for x in args)


def test_budget_demo_is_finite_and_explains_energy_and_gradient_limits():
    demo = importlib.import_module('tools.testing_dashboard.material_budget_demo')
    data = demo.examples()
    assert len(data['rows']) == 33
    assert all(0 < row['specular'] < 1 and math.isfinite(row['dE_dr']) for row in data['rows'])
    assert all(abs(row['specular']+row['matte_white']-1) < 1e-14 for row in data['rows'])
    assert 'not an independent furnace' in data['scope']
    from tools.testing_dashboard.build_lessons import budget_section
    lesson = budget_section()
    assert 'detaching' in lesson and 'fixed' in lesson and 'held-out' in lesson
    assert 'shading/budget-contract' in lesson


@pytest.mark.parametrize('c,r',[(.025,.213),(.11,.213),(.025,.337),(.11,.337)])
def test_budget_grazing_reference_is_resolved(c,r,request):
    """Refine the measuring stick independently of the candidate lookup table."""
    rows=[]
    for f in (0.,.04,1.):
        args=tuple(scalar(x,True) for x in (c,r,f))
        results=[]
        for grid in ('midpoint','gauss'):
            estimates=[]
            for resolution in (256,512):
                value=recipe().directional(*args,resolution=resolution,azimuth=grid)
                gradients=torch.autograd.grad(value,args)
                estimates.append((value.item(),torch.stack(gradients).detach()))
            low,high=estimates
            results.append(dict(grid=grid,value=high[0],gradient=high[1].tolist(),
                value_change=abs(high[0]-low[0]),gradient_change=(high[1]-low[1]).abs().tolist(),
                resolved=abs(high[0]-low[0])<=5e-5 and bool(
                    ((high[1]-low[1]).abs()<=.001+.002*high[1].abs()).all())))
        independent=[oracle.directional_albedo(c,r,f0=f,resolution=n) for n in (256,512)]
        rows.append(dict(cosine=c,roughness=r,f0=f,estimates=results,
            independent_vndf=independent[1],independent_refinement=abs(independent[1]-independent[0]),
            independent_value_error=abs(results[1]['value']-independent[1])))
    plugin=request.config.pluginmanager.get_plugin('tools.testing_dashboard.pytest_plugin')
    if plugin:plugin.Evidence(request.node.nodeid).record('grazing-reference-refinement',dict(rows=rows))
    assert all(not row['estimates'][0]['resolved'] for row in rows), 'Legacy negative control unexpectedly resolved'
    assert all(row['estimates'][1]['resolved'] for row in rows), str(rows)
    assert all(row['independent_refinement']<=5e-5 and row['independent_value_error']<=5e-5 for row in rows), str(rows)


def test_budget_azimuth_choice_is_explicit_and_bounded():
    mod=recipe();args=tuple(scalar(x,True) for x in (.43,.67,.31))
    for invalid in ('adaptive',None,12):
        with pytest.raises(ValueError,match='azimuth'):
            mod.directional(*args,azimuth=invalid)
        with pytest.raises(ValueError,match='azimuth'):
            mod.average(*args[1:],azimuth=invalid)
    for grid in ('gauss','midpoint'):
        with pytest.raises(ValueError,match='work'):
            mod.average(*args[1:],resolution=512,view_resolution=64,azimuth=grid)
    assert torch.autograd.gradcheck(lambda *xs:mod.directional(*xs,resolution=64,azimuth='gauss'),
                                    args,eps=1e-6,atol=1e-5,rtol=1e-3)


def test_budget_angular_demo_explains_reference_convergence():
    demo=importlib.import_module('tools.testing_dashboard.reference_angular_demo')
    data=demo.examples()
    assert [row['resolution'] for row in data['rows']]==[32,64,128,256,512]
    assert len(data['rows'])==5 and 'not table acceptance' in data['scope']
    low,high=data['rows'][-2:]
    assert abs(high['gauss']-data['independent_vndf'])<5e-5
    assert abs(high['gauss']-low['gauss'])<5e-5
    assert abs(high['midpoint']-low['midpoint'])>5e-5
    assert abs(high['gauss_dc']-low['gauss_dc'])<.001+.002*abs(high['gauss_dc'])
    assert all(math.isfinite(value) for row in data['rows'] for value in row.values())
