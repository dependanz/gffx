"""Reference preflight is not a full environment-image acceptance result."""
import importlib
import math
import os
import pytest
import torch


def tool():
    return importlib.import_module('tools.environment_image_preflight')


def test_reference_pixel_selection_spans_view_angles_deterministically():
    c = torch.linspace(.01, 1., 16, dtype=torch.float64).flip(0)
    n = torch.stack((torch.sqrt(1-c*c), torch.zeros_like(c), c), -1).reshape(4, 4, 3)
    v = torch.zeros_like(n); v[..., 2] = 1
    data = dict(normal=n, view=v, covered=torch.ones((4,4), dtype=torch.bool))
    result = tool().select_pixels(data)
    assert [row['flat_index'] for row in result] == [15,13,11,9,7,5,3,0]
    assert result[0]['cosine'] == pytest.approx(.01)
    assert result[-1]['cosine'] == pytest.approx(1.)
    assert result == tool().select_pixels(data)


@pytest.mark.parametrize('cosine', [.4, 1.])
def test_classic_constant_environment_closed_form_matches_scalar_quadrature(cosine):
    n = [0.,0.,1.]; v = [math.sqrt(1-cosine*cosine),0.,cosine]
    result = tool().reference_point(n, v, model='classic')
    expected = [a+.1*(5*cosine/8+5/12) for a in (.2,.4,.6)]
    assert result['status'] == 'converged'
    assert result['radiance'] == pytest.approx(expected, abs=1e-9, rel=1e-9)


def test_legacy_uniform_grazing_reference_remains_explicitly_unresolved():
    from tools.shading_environment_reference import ReferenceConvergenceError
    with pytest.raises(ReferenceConvergenceError) as caught:
        tool().reference_point([0.,0.,1.],[math.sqrt(1-.04**2),0.,.04],model='classic')
    assert caught.value.report['reason']=='level_limit'
    assert caught.value.report['evaluations']==87296


def test_refined_reference_agrees_with_converged_original_and_exact_value():
    n=[0.,0.,1.];v=[math.sqrt(1-.4**2),0.,.4]
    old=tool().reference_point(n,v,model='classic')
    new=tool().reference_point(n,v,model='classic',rule='split')
    assert new['radiance']==pytest.approx(old['radiance'],abs=1e-10)
    assert new['radiance']==pytest.approx(new['closed_form'],abs=1e-10)


def test_environment_reference_unresolved_diagnostics_are_not_results(monkeypatch):
    from tools import shading_environment_reference as reference
    report = dict(profile='env-sh2-v1',status='unresolved',reason='level_limit',history=[],evaluations=256)
    def unresolved(*args, **kwargs):
        raise reference.ReferenceConvergenceError(report)
    monkeypatch.setattr(reference,'environment_reference',unresolved)
    with pytest.raises(reference.ReferenceConvergenceError) as caught:
        tool().reference_point([0.,0.,1.],[0.,0.,1.],model='classic')
    assert caught.value.report == report


@pytest.mark.parametrize('model', ['diffuse','unknown',''])
def test_environment_preflight_rejects_unknown_material(model):
    with pytest.raises(ValueError, match='material'):
        tool().reference_point([0.,0.,1.],[0.,0.,1.],model=model)


@pytest.mark.parametrize('model',['classic','principled'])
@pytest.mark.parametrize('index',range(8))
def test_saved_complete_environment_reference_has_two_valid_transitions(model,index,request):
    root=os.environ.get('GFFX_ENVIRONMENT_REPORTS')
    if not root:pytest.skip('explicit saved environment reports required')
    from tools.testing_dashboard.environment_image_progress import read
    from tools.testing_dashboard.pytest_plugin import Evidence
    row=read(root,f'separated-{model}-{index}')
    report=row['reference'];evidence=Evidence(request.node.nodeid)
    with evidence.check('Converged independent full-material reference'):
        assert report['status']=='converged' and report['profile']=='constant-env-separated-v1'
        assert row['supervisor']['exit_code']==0 and row['supervisor']['failure'] is None
        assert row['worker_pid']==row['supervisor']['monitored_pid']
        assert row['supervisor']['sampled_peak_working_set_bytes']<=1024**3
        assert report['evaluations']<=87296 and report['history'][-1]['streak']==2
        for level in report['history'][-2:]:
            assert all(a<=t and b<=t for a,b,t in zip(level['change'],level['offset_change'],level['threshold']))
    evidence.record('Reference values and costs',dict(reference=report,supervisor=row['supervisor'],point=row['point']))


def test_saved_public_environment_point_matches_reference_but_is_not_an_image():
    root=os.environ.get('GFFX_ENVIRONMENT_REPORTS')
    if not root:pytest.skip('explicit saved environment reports required')
    from tools.testing_dashboard.environment_image_progress import read
    row=read(root,'public-principled-0')
    assert row['status']=='passed public point comparison; no image acceptance'
    assert row['result']['directions']==65536 and row['result']['calls']==6554
    assert row['supervisor']['exit_code']==0 and row['supervisor']['failure'] is None
    for errors in row['absolute_errors']:assert all(x<=limit for x,limit in zip(errors,row['limits']))


def test_environment_progress_refuses_changed_report(tmp_path):
    from tools.testing_dashboard.environment_image_progress import read
    (tmp_path/'classic-0.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):read(tmp_path,'classic-0')


def test_environment_progress_uses_accepted_producer_metadata():
    from gffx.dashboard._server import check_provenance
    from tools.testing_dashboard.environment_image_progress import provenance
    value=provenance()
    assert check_provenance(value)==value
    assert value['artifact_sha256']=='6ab5ccb2335281b8c05e6d1466c791cc94dfb56fbd425b77af11546516c56285'
