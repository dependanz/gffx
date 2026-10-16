"""Fresh complete envelope/aggregate scheduling; acceptance is unchanged."""
import pytest
from fractions import Fraction as Q

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
@pytest.mark.parametrize('aggregate',[False,True])
def test_direct_bound_encloses_independent_closed_integral(monkeypatch,sx,sy,aggregate):
    from tools.opaque_shading_intervals import I
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True,envelope_fallback=True,aggregate_fallback=aggregate)
    got=correction.envelope_integral((I(0,1),I(0,1)))[1];exact=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert got.lo<=exact.lo<=exact.hi<=got.hi

@pytest.mark.parametrize('sign',[-1,1])
def test_direct_bound_matches_existing_fallback_exactly_without_preliminary_work(monkeypatch,sign):
    from tools.opaque_shading_intervals import I
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    def inputs(x,y):return x*I('.001')*sign+y*I('.001')+I('.5'),x.lift('.75'),x.lift(1),1+2*x
    c=MaterialCorrection(e,inputs,curved=True,envelope_fallback=True,aggregate_fallback=True);box=(I(0,1),I(0,1));calls=[]
    def refuse(box):calls.append(1);raise ValueError('bounded angular segment count')
    monkeypatch.setattr(c,'_certified_integral',refuse)
    old=c(box);new=c.envelope_integral(box)
    assert calls==[1] and [(v.lo,v.hi) for v in old]==[(v.lo,v.hi) for v in new]

@pytest.mark.parametrize('kind',['cancel','monotonicity','disjoint'])
def test_direct_bound_retains_stop_and_domain_boundaries(monkeypatch,kind):
    from tools.opaque_shading_intervals import I,Unresolved
    from tools.opaque_peano_material import MaterialCorrection
    from tools import opaque_aggregate_peano as module
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    def inputs(x,y):return x*I('.001')+y*I('.001')+I('.5'),x.lift('.75'),x.lift(1),1+2*x
    c=MaterialCorrection(e,inputs,curved=True,envelope_fallback=True,aggregate_fallback=True);box=(I(0,1),I(0,1))
    class Proof:
        def __init__(self,*a):pass
        def __call__(self,box):
            if kind=='cancel':raise Unresolved('stop direct')
            if kind=='monotonicity':raise ValueError('aggregate angular crossings need certified monotonicity')
            return [I(0),I(100)]
    monkeypatch.setattr(module,'AggregateMaterial',Proof)
    if kind=='cancel':
        with pytest.raises(Unresolved,match='stop direct'):c.envelope_integral(box)
    elif kind=='disjoint':
        with pytest.raises(ValueError):c.envelope_integral(box)
    else:
        got=c.envelope_integral(box);assert len(got)==2 and got[1].width>0
