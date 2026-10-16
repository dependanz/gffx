"""Bounded diagnostic contract: failures are evidence, never piece certificates."""
import pytest
from tools.opaque_shading_intervals import I,Jet


def fixture():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    return MaterialCorrection(e,lambda x,y:(x*I('.001')+y*I('.001')+I('.4992'),x.lift('.75'),x.lift(1),x.lift(1)),curved=False,transverse=True)


@pytest.mark.parametrize('boxes',[[],[(I(0,1),I(0,1))]*4,[(I(0),I(0,1))],[(I(-1,0),I(0,1))],[(I(0,1),)],[(I(0,2),I(0,1))]])
def test_diagnostic_refuses_invalid_scope_before_evaluation(boxes):
    from tools.opaque_boundary_diagnostic import diagnose
    c=fixture();c.inputs=lambda *a:pytest.fail('invalid scope evaluated')
    with pytest.raises(ValueError):diagnose(c,boxes)


@pytest.mark.parametrize('limits',[dict(max_inputs=0),dict(max_inputs=2049),dict(max_seconds=0),dict(max_seconds=21),dict(max_seconds=float('nan')),dict(max_inputs=True)])
def test_diagnostic_refuses_expanded_limits(limits):
    from tools.opaque_boundary_diagnostic import diagnose
    with pytest.raises(ValueError):diagnose(fixture(),[(I(0,1),I(0,1))],**limits)


def test_diagnostic_counts_attempts_stops_before_extra_input_and_restores():
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    c=fixture();before=c.inputs;check=c.checkpoint
    with pytest.raises(DiagnosticStopped) as error:diagnose(c,[(I(0,1),I(0,1))],max_inputs=1)
    assert error.value.report['input_calls']==1
    assert error.value.report['status']=='stopped'
    assert c.inputs is before and c.checkpoint is check


def test_diagnostic_deadline_cancellation_restores_without_retries():
    from tools.opaque_boundary_diagnostic import diagnose,DiagnosticStopped
    c=fixture();before=c.inputs;times=iter([0,21])
    with pytest.raises(DiagnosticStopped):diagnose(c,[(I(0,1),I(0,1))],clock=lambda:next(times))
    assert c.inputs is before
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):diagnose(c,[(I(0,1),I(0,1))],checkpoint=cancel)
    assert c.inputs is before


@pytest.mark.parametrize('order',[1,2,3])
def test_kernel_parts_reconstruct_outward_enclosure(order):
    from tools.opaque_transverse_peano import weighted_kernel_integral,weighted_kernel_parts
    args=(I('1.2','1.4'),I('.1','.11'),I('.8','.81'),order)
    p=weighted_kernel_parts(*args);v=weighted_kernel_integral(*args)
    assert p['value'].lo==v.lo and p['value'].hi==v.hi
    assert p['variation_radius'].lo>=0 and p['anchor_mass'].width>0
    assert p['value'].width>=p['anchor_mass'].width


def test_diagnostic_reports_two_orientations_without_integrating_field(monkeypatch):
    from tools.opaque_boundary_diagnostic import diagnose
    import tools.opaque_square_reference as square
    monkeypatch.setattr(square,'integrate_square',lambda *a,**k:pytest.fail('adaptive integration forbidden'))
    c=fixture();before=c.inputs
    r=diagnose(c,[(I(0,1),I(0,1))]);assert r['status']=='diagnostic_only'
    assert len(r['boxes'])==1 and len(r['boxes'][0]['orientations'])==2
    assert r['input_calls']>0 and r['input_calls']<=2048
    assert r['witnesses']==2 and c.inputs is before and r['full_piece_certificate'] is False
    for v in r['boxes'][0]['orientations']:
        assert len(v['witnesses'])==1 and v['witnesses'][0]['admission']=='integrated'
        assert len(v['witnesses'][0]['terms'])==3
        assert 'uniform' in v and 'tensor_error' in v


def test_diagnostic_exposes_fallback_reason():
    from tools.opaque_boundary_diagnostic import diagnose
    c=fixture();c.inputs=lambda x,y:(y*I('.001')+I('.4995'),x.lift('.75'),x.lift(1),x.lift(1))
    r=diagnose(c,[(I(0,1),I(0,1))])
    w=r['boxes'][0]['orientations'][1]['witnesses'][0]
    assert w['admission']=='fallback' and w['reason']=='derivative_contains_zero'
