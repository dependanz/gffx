"""A usable derivative must lower the measured objective, not just exist."""
import importlib
import pytest


def tool():
    return importlib.import_module('tools.shading_fit_closure')


def test_fit_closure_has_a_finite_complete_parameter_schedule():
    schedule = tool().schedule()
    assert len(schedule) == 17
    assert len(set(schedule)) == 17
    assert set(name for model, name in schedule if model == 'classic') == {
        'texture', 'uv', 'normal', 'view', 'directions', 'shape', 'specular', 'radiance'}
    assert schedule[-1] == ('sh', 'coefficients')


@pytest.mark.parametrize('bad', ['model', 'parameter', 'deadline'])
def test_fit_closure_rejects_invalid_work_before_evaluation(monkeypatch, bad):
    import time
    module = tool()
    def forbidden(*args, **kwargs):
        raise AssertionError('public chain reached')
    monkeypatch.setattr(module, 'chain', forbidden)
    model, parameter, deadline = 'classic', 'texture', time.perf_counter()+30
    if bad == 'model': model = 'unknown'
    if bad == 'parameter': parameter = 'topology'
    if bad == 'deadline': deadline = 0
    with pytest.raises((ValueError, TimeoutError)):
        module.evaluate(model, parameter, deadline=deadline)


def test_fit_closure_rejects_wrong_derivative_and_non_descent():
    module = tool()
    with pytest.raises(ValueError, match='derivative'):
        module.validate_result(dict(before=1., after=.9, analytic=-1., differences=[0., 0.], fresh_finite=True))
    with pytest.raises(ValueError, match='descent'):
        module.validate_result(dict(before=1., after=1., analytic=-1., differences=[-1., -1.], fresh_finite=True))


def test_saved_fit_closure_requires_exact_evidence():
    import os
    from pathlib import Path
    path = os.environ.get('GFFX_FIT_CLOSURE_REPORT')
    if not path:
        pytest.skip('explicit supervised fitting report required')
    result = tool().replay(Path(path), os.environ['GFFX_FIT_CLOSURE_SHA256'])
    assert len(result['cases']) == 17
    with pytest.raises(ValueError, match='identity'):
        tool().replay(Path(path), '0'*64)


@pytest.fixture(scope='module')
def fitting_report():
    import os
    from pathlib import Path
    path=os.environ.get('GFFX_FIT_CLOSURE_REPORT')
    if not path:pytest.skip('explicit supervised fitting report required')
    from tools.shading_cuda_closeout import identity
    identity(tool().WHEEL,tool().WHEEL_SHA)
    return tool().replay(Path(path),os.environ['GFFX_FIT_CLOSURE_SHA256'])


@pytest.mark.parametrize('model,parameter',[(m,p) for m in ('classic','principled') for p in
    ('texture','uv','normal','view','directions','shape','specular','radiance')]+[('sh','coefficients')])
def test_each_saved_shading_parameter_has_checked_fresh_descent(fitting_report,model,parameter,request):
    from tools.testing_dashboard.pytest_plugin import Evidence
    row=next(x for x in fitting_report['cases'] if (x['model'],x['parameter'])==(model,parameter))
    evidence=Evidence(request.node.nodeid)
    with evidence.check('Two-scale loss derivative') as check:
        for value in row['differences']:
            assert abs(value-row['analytic'])<=1e-10+1e-5*abs(row['analytic'])
        check.record('comparison',dict(analytic=row['analytic'],finite_differences=row['differences'],steps=[1e-5,5e-6]))
    with evidence.check('Fresh gradient lowers loss') as check:
        assert row['fresh_finite'] is True and row['after']<row['before']
        check.scalar('loss',row['before'],step=0)
        check.scalar('loss',row['after'],step=1)
        check.text('scope','One small unclamped gradient step for this parameter group; saved exact-wheel measurements, not a new optimization run or converged fit.')
