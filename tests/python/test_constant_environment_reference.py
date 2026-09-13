"""Restricted angular refinement, with unchanged accuracy and work limits."""
import importlib
import math
import pytest


def tool():return importlib.import_module('tools.constant_environment_reference')


@pytest.mark.parametrize('split', [.25,.5])
def test_split_azimuth_weights_cover_half_hemisphere_and_polynomials(split):
    nodes,weights=tool().azimuth(8,split)
    assert len(nodes)==len(weights)==16
    assert all(0<x<math.pi for x in nodes) and min(weights)>0
    for power in range(8):
        assert math.fsum(w*x**power for x,w in zip(nodes,weights))==pytest.approx(math.pi**(power+1)/(power+1),rel=1e-13)


@pytest.mark.parametrize('cosine',[.014622699987302479,.04,.4,1.])
def test_split_complete_classic_matches_exact_environment(cosine):
    result=tool().reference_point(cosine,model='classic')
    expected=[a+.1*(5*cosine/8+5/12) for a in (.2,.4,.6)]
    assert result['status']=='converged'
    assert result['radiance']==pytest.approx(expected,rel=1e-9,abs=1e-9)
    assert result['evaluations']<=87296
    assert result['history'][-1]['streak']==2


def test_split_reference_keeps_work_limit_and_failure_history():
    from tools.shading_environment_reference import ReferenceConvergenceError
    with pytest.raises(ReferenceConvergenceError) as caught:
        tool().reference_point(.04,model='classic',max_evaluations=256)
    assert caught.value.report['reason']=='work_limit'
    assert caught.value.report['evaluations']==256
    assert len(caught.value.report['history'])==1


def test_split_reference_preserves_cooperative_timeout():
    from tools.shading_environment_reference import ReferenceConvergenceError
    with pytest.raises(ReferenceConvergenceError) as caught:
        tool().reference_point(.04,model='classic',deadline_seconds=1e-12)
    assert caught.value.report['reason']=='time_limit'


@pytest.mark.parametrize('cosine',[0.,-1.,1.01,math.nan])
def test_split_reference_rejects_invalid_view(cosine):
    with pytest.raises(ValueError):tool().reference_point(cosine,model='classic')


def test_square_radial_mapping_preserves_measure_and_moments():
    nodes,weights=tool().radial(8,'square')
    assert min(nodes)>0 and max(nodes)<1 and min(weights)>0
    for power in range(8):
        assert math.fsum(w*x**power for x,w in zip(nodes,weights))==pytest.approx(1/(power+1),abs=1e-14)


@pytest.mark.parametrize('cosine',[.014622699987302479,.04,.4,1.])
def test_square_radial_complete_classic_matches_known_answer(cosine):
    row=tool().reference_point(cosine,model='classic',radial_rule='square')
    assert row['profile']=='constant-env-split-square-v1'
    assert row['radiance']==pytest.approx([a+.1*(5*cosine/8+5/12) for a in (.2,.4,.6)],abs=1e-10)


def test_separated_matte_uses_residual_budget_not_plain_lambert():
    from tools.principled_direct_images import PreparedCurve
    from tools.shading_independent_evaluator import LossCurve
    from tools.testing_dashboard.complete_material_demo import synthetic_envelope
    original=LossCurve(synthetic_envelope()['payload'])
    curve=PreparedCurve(original,roughness=1.)
    result=tool().matte_integral(curve,.4)
    expected=[a*original.loss(.4,1.,.04)*original.mean(1.,.04)/curve.mean(1.,.04) for a in (.2,.4,.6)]
    assert result==pytest.approx(expected,abs=1e-13)
    assert result!=pytest.approx([.2,.4,.6])
