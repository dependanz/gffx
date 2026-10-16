"""All branch derivatives and all knot jumps, with independent integral controls."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest

@pytest.mark.parametrize('n',[3,16])
def test_aggregate_jump_sums_cover_independent_log_derivatives(n):
    from tests.packaging.test_opaque_residual_range import curve
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I
    c=curve(n);index=ResidualRangeIndex(c);lo=c.cn[1].lo;hi=c.cn[n-1].hi;got=index.jump_sums(I(lo,hi))
    with localcontext() as ctx:
        ctx.prec=90;expected=[[D(0)]*3 for _ in range(2)]
        for k in range(1,n):
            q=(D(k+1)/D(n+1));x=q.ln()
            for ch in range(2):
                a=[c.global_power(i,ch) for i in (k-1,k)];p=[D((b-a).numerator)/D((b-a).denominator) for a,b in zip(*a)]
                d=[sum(p[j]*x**j for j in range(4)),(p[1]+2*p[2]*x+3*p[3]*x*x)/q,((2*p[2]-p[1])+(6*p[3]-2*p[2])*x-3*p[3]*x*x)/(q*q)]
                for j in range(3):expected[ch][j]+=abs(d[j])
        for ch in range(2):
            for j in range(3):assert got[ch][j].lo>=0 and got[ch][j].hi>=expected[ch][j]

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_aggregate_material_bound_contains_closed_log_integral(sx,sy):
    from tools.opaque_shading_intervals import I
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy;a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True);proof=AggregateMaterial(correction)
    result=proof((I(0,1),I(0,1)))[1];expected=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert result.lo<=expected.lo<=expected.hi<=result.hi

@pytest.mark.parametrize('planar',[False,True])
def test_more_than_eight_discontinuous_steps_keep_every_jump(planar):
    from tools.opaque_shading_intervals import I
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from types import SimpleNamespace
    n=16;cn=[I(Q(1,4)+Q(3*i,4*n)) for i in range(n+1)];x=[v.ln() for v in cn]
    values=[I(Q((i*7)%13,10**8)) for i in range(n)];controls=[[[v]*4]*2 for v in values];e=HorizonExtension(SimpleNamespace(cn=cn,x=x,controls=controls,mean=[I(1)]*2,pi=I(1),horizon=[values[0]]*2,values=[[v]*2 for v in values]+[[values[-1]]*2]));e.indices=[0,n-1]
    lo=cn[0];span=1-lo
    def inputs(u,v):return u.lift(lo)+u*span,u.lift(1),u.lift(1),u.lift(1)
    correction=MaterialCorrection(e,inputs,curved=not planar,envelope_fallback=planar,aggregate_fallback=planar,planar_envelope=planar)
    proof=AggregateMaterial(correction);result=(correction if planar else proof)((I(0,1),I(0,1)))[1]
    expected=(sum((v for v in values),I(0))/n-values[0])*values[-1]
    assert result.lo<=expected.lo<=expected.hi<=result.hi
    rows=proof.axis_bounds((I(0,1),I(0,1)),0);assert rows[1]['jumps'][0].hi>0

def test_unknown_monotonicity_and_cancellation_refuse():
    from tools.opaque_shading_intervals import I,Unresolved
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    def inputs(x,y):return (x-I('.5'))**2*I('.01')+I('.499'),x.lift('.75'),x.lift(1),x.lift(1)
    proof=AggregateMaterial(MaterialCorrection(e,inputs,curved=True))
    with pytest.raises(ValueError,match='monotonicity'):proof.axis_bounds((I(0,1),I(0,1)),0)
    def stop():raise Unresolved('stop aggregate work')
    with pytest.raises(Unresolved):AggregateMaterial(MaterialCorrection(e,inputs,curved=True,checkpoint=stop))


@pytest.mark.parametrize('n',[7,16])
def test_all_ordinary_jet_orders_enclose_independent_formulas(n):
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_residual_range import ResidualRangeIndex
    from tests.packaging.test_opaque_residual_range import curve
    c=curve(n);index=ResidualRangeIndex(c);base=1;q=Jet.variable(I(c.cn[1].lo,c.cn[-1].hi),order=3);got=index.delta_jet(q,base)
    with localcontext() as ctx:
        ctx.prec=90
        for branch in range(1,n):
            v=D(2*branch+3)/D(2*(n+1));x=v.ln()
            for channel in range(2):
                a,b=c.global_power(base,channel),c.global_power(branch,channel);p=[D((v-u).numerator)/D((v-u).denominator) for u,v in zip(a,b)]
                expected=[sum(p[j]*x**j for j in range(4)),(p[1]+2*p[2]*x+3*p[3]*x*x)/v,((2*p[2]-p[1])+(6*p[3]-2*p[2])*x-3*p[3]*x*x)/(2*v*v),((6*p[3]-6*p[2]+2*p[1])+(4*p[2]-18*p[3])*x+6*p[3]*x*x)/(6*v**3)]
                for j,want in enumerate(expected):assert got[channel][j,0].lo<=want<=got[channel][j,0].hi

def test_simultaneous_view_and_light_steps_preserve_product_jumps():
    from tools.opaque_shading_intervals import I
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from types import SimpleNamespace
    n=16;cn=[I(Q(1,4)+Q(3*i,4*n)) for i in range(n+1)];x=[v.ln() for v in cn];values=[I(Q((i*7)%13,10**4)) for i in range(n)];controls=[[[v]*4]*2 for v in values]
    e=HorizonExtension(SimpleNamespace(cn=cn,x=x,controls=controls,mean=[I(1)]*2,pi=I(1),horizon=[values[0]]*2,values=[[v]*2 for v in values]+[[values[-1]]*2]));e.indices=[0,0];span=1-cn[0]
    def inputs(u,v):
        q=u.lift(cn[0])+u*span;return q,q,u.lift(1),u.lift(1)
    proof=AggregateMaterial(MaterialCorrection(e,inputs,curved=True));result=proof((I(0,1),I(0,1)))[1]
    expected=sum(((values[i]**2-values[0]**2)*(cn[i+1]**2-cn[i]**2)/(2*span) for i in range(n)),I(0))
    assert result.lo<=expected.lo<=expected.hi<=result.hi


def test_empty_knot_range_has_exact_zero_jump_sum():
    from tools.opaque_shading_intervals import I
    from tools.opaque_residual_range import ResidualRangeIndex
    from tests.packaging.test_opaque_residual_range import curve
    c=curve(16);index=ResidualRangeIndex(c);index.jump_sums(I(0,1))
    q=I((Q(c.cn[7].hi)+Q(c.cn[8].lo))/2)
    assert all(v.zero for row in index.jump_sums(q) for v in row)


def test_aggregate_adoption_intersects_bounds_and_preserves_failures(monkeypatch):
    from tools.opaque_shading_intervals import I,Unresolved
    from tools.opaque_peano_material import MaterialCorrection
    from tools import opaque_aggregate_peano as module
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture()
    def inputs(x,y):return x*I('.001')+I('.4995'),x.lift('.75'),x.lift(1),x.lift(1)
    def checker():
        c=MaterialCorrection(e,inputs,curved=True,envelope_fallback=True,aggregate_fallback=True)
        c._certified_integral=lambda box:(_ for _ in ()).throw(ValueError('bounded angular segment count'))
        return c
    box=(I(0,1),I(0,1));old=MaterialCorrection(e,inputs,curved=True,envelope_fallback=True);old._certified_integral=lambda box:(_ for _ in ()).throw(ValueError('bounded angular segment count'));range_bound=old(box)
    class Refuse:
        def __init__(self,c):pass
        def __call__(self,box):raise ValueError('aggregate angular crossings need certified monotonicity')
    monkeypatch.setattr(module,'AggregateMaterial',Refuse);got=checker()(box);assert [(v.lo,v.hi) for v in got]==[(v.lo,v.hi) for v in range_bound]
    class Narrow(Refuse):
        def __call__(self,box):return [I(0),I((Q(range_bound[1].lo)+Q(range_bound[1].hi))/2)]
    monkeypatch.setattr(module,'AggregateMaterial',Narrow);got=checker()(box);assert got[1].width<range_bound[1].width
    class Cancel(Refuse):
        def __call__(self,box):raise Unresolved('stop aggregate proof')
    monkeypatch.setattr(module,'AggregateMaterial',Cancel)
    with pytest.raises(Unresolved):checker()(box)
    class Wrong(Refuse):
        def __call__(self,box):return [I(0),I(1)]
    monkeypatch.setattr(module,'AggregateMaterial',Wrong)
    with pytest.raises(ValueError):checker()(box)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_conditioned_aggregate_encloses_independent_material_integral(sx,sy,axis):
    from tools.opaque_shading_intervals import I
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy;a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True);proof=AggregateMaterial(correction)
    result=proof((I(0,1),I(0,1)),integrated_axis=axis)[1];expected=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert result.lo<=expected.lo<=expected.hi<=result.hi
