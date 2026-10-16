"""Independent truncated-polynomial moments for all ten Peano kernels."""
from fractions import Fraction as Q
import math
import pytest

@pytest.mark.parametrize('order',range(1,11))
def test_all_kernel_orders_enclose_independent_truncated_power_error(order):
    from decimal import Decimal as D,localcontext
    from tools.opaque_high_peano import kernel_range
    from tools.opaque_shading_intervals import I
    with localcontext() as ctx:
        ctx.prec=90;a=((D(5)+2*(D(10)/7).sqrt()).sqrt())/6;b=((D(5)-2*(D(10)/7).sqrt()).sqrt())/6
        nodes=[D('.5')-a,D('.5')-b,D('.5'),D('.5')+b,D('.5')+a];u=(D(322)-13*D(70).sqrt())/1800;v=(D(322)+13*D(70).sqrt())/1800;weights=[u,v,D(64)/225,v,u]
        for k in range(21):
            t=D(k)/20;actual=sum(w*(x-t)**(order-1)/math.factorial(order-1) for x,w in zip(nodes,weights) if x>t)-(1-t)**order/math.factorial(order);bound=kernel_range(I(str(t)),order)
            assert bound.lo<=actual<=bound.hi

@pytest.mark.parametrize('base',[-1,1,7])
def test_high_branch_derivatives_enclose_independent_recurrence(base):
    from decimal import Decimal as D,localcontext
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_residual_range import curve
    c=curve(16);index=ResidualRangeIndex(c)
    with localcontext() as ctx:
        ctx.prec=90
        for branch in [1,6,14]:
            q=Q(2*branch+3,34);value=D(q.numerator)/D(q.denominator);x=value.ln();got=index.high_delta_jet(Jet.variable(I(q),order=10),base)
            for ch in [0,1]:
                p=list(c.global_power(branch,ch));ref=list(c.global_power(base,ch)) if base>=0 else None
                for k in range(11):
                    expected=sum(D(v.numerator)/D(v.denominator)*x**j for j,v in enumerate(p))/(value**k*math.factorial(k))
                    if ref is not None:expected-=sum(D(v.numerator)/D(v.denominator)*x**j for j,v in enumerate(ref))/(value**k*math.factorial(k))
                    elif k<=1:
                        start=(c.values[0][ch].lo+c.values[0][ch].hi)/2;h=(c.horizon[ch].lo+c.horizon[ch].hi)/2;slope=(start-h)/c.cn[0].lo;expected-=h+slope*value if k==0 else slope
                    assert got[ch][k,0].lo<=expected<=got[ch][k,0].hi
                    p=[(j+1)*p[j+1]-k*p[j] for j in range(3)]+[-k*p[3]]
                    if ref is not None:ref=[(j+1)*ref[j+1]-k*ref[j] for j in range(3)]+[-k*ref[3]]

def test_old_low_order_guard_remains_and_high_order_is_bounded():
    from tools.opaque_residual_range import ResidualRangeIndex
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_residual_range import curve
    index=ResidualRangeIndex(curve(16))
    with pytest.raises(ValueError):index.delta_jet(Jet.variable(I('.5'),order=4),1)
    with pytest.raises(ValueError):index.high_delta_jet(Jet.variable(I('.5'),order=11),1)


@pytest.mark.parametrize('order',range(2,11))
@pytest.mark.parametrize('degree',range(13))
def test_complete_high_identity_contains_exact_truncated_polynomial_integral(degree,order):
    from tools.opaque_high_peano import high_integral
    from tools.opaque_shading_intervals import I,Jet
    knot=I(Q(3,7))
    def field(u,v):
        q=u[0,0]-knot
        if q.hi<0:z=I(0)
        elif q.lo>=0:z=q**degree
        else:z=I(0,(I(0,q.hi)**degree).hi)
        return [u.lift(z)]
    def bounds(box,axis):
        tenth=I(Q(math.factorial(degree),math.factorial(degree-order)))*(1-knot)**(degree-order) if degree>=order and axis==0 else I(0)
        jumps=[]
        if degree<order and axis==0:jumps=[dict(position=knot,values=[I(math.factorial(degree)) if j==degree else I(0) for j in range(order)],required=True)]
        return [dict(order=order,derivative=tenth,jumps=jumps)]
    got=high_integral(field,(I(0,1),I(0,1)),bounds)[0];expected=(1-Q(3,7))**(degree+1)/Q(degree+1)
    assert got.lo<=I(expected).lo<=I(expected).hi<=got.hi
