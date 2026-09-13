"""Full constant-environment image harness: independent pixels and bounded work."""
import importlib
import math
import time

import pytest
import torch


def tool():
    return importlib.import_module('tools.environment_full_images')


@pytest.fixture
def curve():
    from gffx.torch.material import PrincipledBudget
    from tools.principled_direct_images import PreparedCurve, RESOURCE, scalar
    budget = PrincipledBudget.bundled(RESOURCE, device='cpu', dtype=torch.float64)
    return PreparedCurve(scalar.LossCurve(budget.to_envelope()['payload']), roughness=1.)


@pytest.mark.parametrize('model', ['classic', 'principled'])
@pytest.mark.parametrize('cosine', [.014622699987302479, .4, 1.])
def test_fixed_profile_reference_samples_match_frozen_scalar(model, cosine, curve):
    from tools.principled_direct_images import scalar
    n, v = [0.,0.,1.], [math.sqrt(1-cosine*cosine),0.,cosine]
    directions = torch.tensor([[.6,0.,.8], [-.8,0.,.6], [0.,0.,1.]], dtype=torch.float64)
    actual = tool().reference_brdf(cosine, directions, model=model)
    if model == 'classic':
        expected = [scalar.classic(n,v,l,albedo=[.2,.4,.6],specular=.1,exponent=2.) for l in directions.tolist()]
    else:
        expected = [scalar.principled(n,v,l,albedo=0.,roughness=1.,f0=.04,budget=curve) for l in directions.tolist()]
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize('model', ['classic', 'principled'])
def test_fast_reference_matches_scalar_convergence_history(model, curve):
    from tools.constant_environment_reference import reference_point
    c = .014622699987302479
    expected = reference_point(c, model=model, curve=curve, radial_rule='square', separated=True)
    actual = tool().reference(c, model=model, curve=curve, deadline=time.perf_counter()+30)
    assert actual['evaluations'] == expected['evaluations']
    assert [x['streak'] for x in actual['history']] == [x['streak'] for x in expected['history']]
    assert actual['radiance'] == pytest.approx(expected['radiance'], abs=1e-13, rel=1e-12)


def test_reference_work_and_time_fail_closed(curve):
    from tools.shading_environment_reference import ReferenceConvergenceError
    with pytest.raises(ReferenceConvergenceError):
        tool().reference(.4, model='principled', curve=curve, deadline=time.perf_counter()+30, max_evaluations=256)
    with pytest.raises(TimeoutError):
        tool().reference(.4, model='classic', curve=None, deadline=time.perf_counter()-1)


@pytest.mark.parametrize('cosine', [0., -1., 1.01, math.nan])
def test_reference_rejects_invalid_pixels(cosine):
    with pytest.raises(ValueError):
        tool().reference(cosine, model='classic', curve=None, deadline=time.perf_counter()+30)


def test_tile_layout_covers_every_pixel_once_and_rejects_bad_indices():
    ranges = [tool().tile_range(i) for i in range(52)]
    assert [j for start,end in ranges for j in range(start,end)] == list(range(6536))
    for i in (-1,52,True):
        with pytest.raises(ValueError): tool().tile_range(i)


@pytest.mark.parametrize('model', ['classic','principled'])
def test_public_pixel_uses_two_estimates_and_rejects_bad_reference(model, curve):
    from gffx.torch.material import PrincipledBudget
    from tools.principled_direct_images import RESOURCE
    budget = PrincipledBudget.bundled(RESOURCE, device='cpu', dtype=torch.float64)
    ref = tool().reference(.4, model=model, curve=curve, deadline=time.perf_counter()+30)
    row = tool().candidate(.4, model=model, budget=budget, ref=ref, deadline=time.perf_counter()+30)
    assert len(row['estimates']) == 2 and row['directions'] <= 65536
    ref['radiance'] = [10.,10.,10.]
    with pytest.raises(ValueError, match='tolerance'):
        tool().candidate(.4, model=model, budget=budget, ref=ref, deadline=time.perf_counter()+30)


def test_local_frame_matches_world_frame_public_integral():
    from gffx.torch import material, light
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    directions, weights = tool().quadrature(16,.5)
    c=.4; n=t([[0.,0.,1.]]); v=t([[math.sqrt(1-c*c),0.,c]])
    rotation=t([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    values=[]
    for r in (torch.eye(3,dtype=torch.float64),rotation):
        nn,vv,ll=n@r,v@r,directions@r
        brdf=material.classic(nn,vv,ll,albedo=t([.2,.4,.6]),specular=t(.1),exponent=t(2.))
        values.append(light.integrate(nn,brdf,dict(measure='solid_angle',direction=ll[None],
            radiance=torch.ones((1,len(ll),3),dtype=torch.float64),valid=torch.ones((1,len(ll)),dtype=torch.bool),solid_angle=weights[None])))
    torch.testing.assert_close(*values,atol=1e-13,rtol=1e-12)


def test_assembly_refuses_missing_duplicate_or_failed_tiles():
    with pytest.raises(ValueError, match='complete'):
        tool().assemble([], torch.zeros((128,128),dtype=torch.bool))
    rows=[dict(index=0,status='stopped')]*52
    with pytest.raises(ValueError): tool().assemble(rows,torch.zeros((128,128),dtype=torch.bool))
