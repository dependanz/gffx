"""Domain guidance affects work order only; complete enclosures still decide."""
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('axis',[0,1])
def test_direction_hint_keeps_closed_integral_with_fixed_depth(axis):
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import I,Unresolved
    class Correction:
        domain_split_axis=axis
        def __call__(self,box):
            if box[axis].width>I(Q(1,8)).lo:raise ValueError('bounded angular segment count')
            return [I(0)]
    r=integrate_square(lambda x,y:[x*y],correction_integrator=Correction(),max_depth=3)
    assert Q(r['value'][0].lo)<=Q(1,4)<=Q(r['value'][0].hi);assert r['cells']==15 and r['max_depth']==3
    with pytest.raises(Unresolved):integrate_square(lambda x,y:[x*y],correction_integrator=Correction(),max_depth=2)

@pytest.mark.parametrize('hint',[None,True,-1,2,'0'])
def test_invalid_hints_keep_fallback_and_cannot_admit(hint):
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    class Correction:
        domain_split_axis=hint
        def __call__(self,box):raise ValueError('bounded angular segment count')
    with pytest.raises(Unresolved) as e:integrate_square(lambda x,y:[x*y],correction_integrator=Correction(),max_depth=1)
    assert e.value.details['last']['box']==['0','1/2','0','1']

def test_cosine_direction_evidence_reuses_existing_evaluations():
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_shading_intervals import I
    calls=[];evidence=[]
    def field(x,y):calls.append(1);return [2*x+y,x-3*y]
    r=bounded_cosines(field,I(0,1),I(0,1),directional=evidence.append)
    assert len(calls)==4 and len(evidence)==1
    assert [[v.hi for v in row] for row in evidence[0]]==[[2,1],[1,3]]
    assert r[0].lo==0 and r[0].hi==3

def test_range_fallback_retains_complete_residual_and_domain_refusals(monkeypatch):
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_shading_intervals import I,Jet
    from types import SimpleNamespace
    e=SimpleNamespace(mean=[I(1),I(1)])
    c=MaterialCorrection(e,lambda *a:None,curved=True,range_fallback=True)
    monkeypatch.setattr(c,'_certified_integral',lambda box:(_ for _ in ()).throw(ValueError('curved angular crossings need certified monotonicity')))
    monkeypatch.setattr(c,'ordinary',lambda u,s:[u.lift(0),u+2*s])
    r=c((I(0,I(Q(1,2)).hi),I(0,I(Q(1,4)).hi)))
    assert r[1].lo<=I(Q(1,16)).lo<=r[1].hi
    assert r[1].lo==0 and r[1].hi==I(Q(1,8)).hi
    assert c.error_breakdown is None
    monkeypatch.setattr(c,'_certified_integral',lambda box:(_ for _ in ()).throw(ValueError('bounded angular segment count')))
    with pytest.raises(ValueError,match='bounded angular segment count'):c((I(0,1),I(0,1)))
    with pytest.raises(ValueError):MaterialCorrection(e,lambda *a:None,curved=False,range_fallback=True)

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_material_range_fallback_encloses_independent_closed_integral(monkeypatch,sx,sy):
    from tools.opaque_shading_intervals import I
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True,range_fallback=True)
    monkeypatch.setattr(correction,'_certified_integral',lambda box:(_ for _ in ()).throw(ValueError('curved angular crossings need certified monotonicity')))
    got=correction((I(0,1),I(0,1)))[1];exact=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert got.lo<=exact.lo<=exact.hi<=got.hi
