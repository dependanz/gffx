"""Independent formulas for light-weighted ordinary residual derivatives."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest

@pytest.mark.parametrize('base',[-1,0,1])
@pytest.mark.parametrize('order',[0,1,2,3])
def test_weighted_residual_contains_independent_log_cubic_derivatives(base,order):
    from tests.packaging.test_opaque_residual_range import curve
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    c=curve(7);index=ResidualRangeIndex(c);q=Jet.variable(I(c.cn[1].lo,1),order=order);got=index.weighted_delta_jet(q,base)
    with localcontext() as ctx:
        ctx.prec=90
        for branch in range(1,7):
            v=D(2*branch+3)/16;x=v.ln()
            for ch in range(2):
                p=[D(a.numerator)/D(a.denominator) for a in c.global_power(branch,ch)]
                if base>=0:p=[a-D(b.numerator)/D(b.denominator) for a,b in zip(p,c.global_power(base,ch))]
                f=sum(p[j]*x**j for j in range(4));g=p[1]+2*p[2]*x+3*p[3]*x*x;h=2*p[2]+6*p[3]*x
                want=[v*f,f+g,(g+h)/(2*v),(6*p[3]-g)/(6*v*v)]
                if base==-1:
                    z=(c.horizon[ch].lo+c.horizon[ch].hi)/2;k=((c.values[0][ch].lo+c.values[0][ch].hi)/2-z)/c.cn[0].lo
                    want=[a-b for a,b in zip(want,[z*v+k*v*v,z+2*k*v,k,D(0)])]
                for j in range(order+1):assert got[ch][j,0].lo<=want[j]<=got[ch][j,0].hi


def test_weighted_horizon_zero_and_invalid_profile():
    from tests.packaging.test_opaque_residual_range import curve
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    c=curve(7);index=ResidualRangeIndex(c)
    assert all(q[j,0].zero for q in index.weighted_delta_jet(Jet.variable(I(0,c.cn[0].lo/2),order=3),-1) for j in range(4))
    with pytest.raises(ValueError):index.weighted_delta_jet(Jet.variable(I('.2','.5'),order=4),-1)
    with pytest.raises(ValueError):index.weighted_delta_jet(Jet.variable(I('-.1','.5'),order=3),-1)


@pytest.mark.parametrize('weighted',[False,True])
def test_weighted_material_keeps_complete_independent_step_product_integral(weighted):
    from tools.opaque_shading_intervals import I
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_aggregate_peano import AggregateMaterial
    from types import SimpleNamespace
    n=16;cn=[I(Q(1,4)+Q(3*i,4*n)) for i in range(n+1)];values=[I(Q((i*7)%13,10**4)) for i in range(n)]
    e=HorizonExtension(SimpleNamespace(cn=cn,x=[v.ln() for v in cn],controls=[[[v]*4]*2 for v in values],mean=[I(1)]*2,pi=I(1),horizon=[values[0]]*2,values=[[v]*2 for v in values]+[[values[-1]]*2]));e.indices=[0,0];span=1-cn[0]
    def inputs(u,v):
        q=u.lift(cn[0])+u*span;return q,q,u.lift(1),u.lift(1)
    correction=MaterialCorrection(e,inputs,curved=True);correction.weighted_light_residual=weighted
    got=AggregateMaterial(correction)((I(0,1),I(0,1)))[1]
    expected=sum(((values[i]**2-values[0]**2)*(cn[i+1]**2-cn[i]**2)/(2*span) for i in range(n)),I(0))
    assert got.lo<=expected.lo<=expected.hi<=got.hi
