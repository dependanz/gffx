"""Independent discontinuous polynomial and material integral range controls."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import math,pytest

def curve(n):
    from tools.opaque_shading_intervals import I
    class Curve:
        cn=[I(Q(i+1,n+1)) for i in range(n+1)]
        x=[v.ln() for v in cn]
        horizon=[I(Q(1,3)),I(Q(1,5))]
        def global_power(self,i,c):return [Q(i%3,7)+Q(c,5),Q((-1)**i,11),Q(i+1,100),Q((-1)**c,1000)]
        def power(self,i,c,anchor):
            p=list(map(I,self.global_power(i,c)))
            return [sum((math.comb(j,k)*p[j]*anchor**(j-k) for j in range(k,4)),I(0)) for k in range(4)]
    c=Curve();c.values=[[c.power(i if i<n else n-1,k,c.x[i])[0] for k in range(2)] for i in range(n+1)];return c

@pytest.mark.parametrize('n',[2,7,16])
@pytest.mark.parametrize('base',[-1,0,1])
def test_range_index_contains_independent_discontinuous_polynomials(n,base):
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I
    c=curve(n);index=ResidualRangeIndex(c)
    for first,last in [(0,n-1),(0,0),(n//2,n-1)]:
        lo=c.cn[first].lo;hi=c.cn[last+1].hi;got=index.delta(I(lo,hi),base)
        assert index.last_query_blocks<=3
        with localcontext() as ctx:
            ctx.prec=90
            for branch in range(first,last+1):
                for t in (Q(1,7),Q(1,2),Q(6,7)):
                    v=Q(c.cn[branch].lo)*(1-t)+Q(c.cn[branch+1].hi)*t;vv=D(v.numerator)/D(v.denominator);z=vv.ln()
                    for channel in range(2):
                        p=c.global_power(branch,channel);actual=sum((D(a.numerator)/D(a.denominator))*z**k for k,a in enumerate(p))
                        if base<0:
                            # The stored first value is uncertain; check its midpoint
                            h=(c.horizon[channel].lo+c.horizon[channel].hi)/2;start=(c.values[0][channel].lo+c.values[0][channel].hi)/2;expected=actual-(h+vv*(start-h)/c.cn[0].lo)
                        else:expected=actual-sum((D(a.numerator)/D(a.denominator))*z**k for k,a in enumerate(c.global_power(base,channel)))
                        assert got[channel].lo<=expected<=got[channel].hi

def test_horizon_zero_and_snapshot_are_retained():
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I
    c=curve(8);index=ResidualRangeIndex(c)
    assert all(v.zero for v in index.delta(I(0,(c.cn[0]/2).lo),-1))
    assert all(v.lo<=0<=v.hi for v in index.delta(I(c.cn[0].lo,c.cn[0].hi),-1))
    a=index.delta(I('.01','.9'),2);c.horizon[0]=I(100);c.x[0]=I(-100)
    b=index.delta(I('.01','.9'),2);assert [(v.lo,v.hi) for v in a]==[(v.lo,v.hi) for v in b]

def test_invalid_resource_and_cancellation_are_refused():
    from tools.opaque_residual_range import ResidualRangeIndex
    c=curve(8);c.cn[1]=c.cn[0]
    with pytest.raises(ValueError):ResidualRangeIndex(c)
    def stop():raise RuntimeError('stop envelope work')
    with pytest.raises(RuntimeError,match='stop envelope work'):ResidualRangeIndex(curve(8),checkpoint=stop)

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_whole_residual_range_encloses_closed_material_integral(monkeypatch,sx,sy):
    from tools.opaque_shading_intervals import I
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I('.4988')+(I('.002') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True,envelope_fallback=True)
    monkeypatch.setattr(correction,'_certified_integral',lambda box:(_ for _ in ()).throw(ValueError('bounded angular segment count')))
    correction.domain_split_axis=1
    got=correction((I(0,1),I(0,1)))[1];assert correction.preferred_split_axis==1;exact=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert got.lo<=exact.lo<=exact.hi<=got.hi


def test_missing_top_domain_is_refused_instead_of_omitted():
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I
    c=curve(8);c.cn[-1]=I('.99')
    with pytest.raises(ValueError):ResidualRangeIndex(c)
