"""Independent polynomial checks for angular-knot residual enclosures."""
import math
from decimal import Decimal as D,localcontext
import pytest
from tools.opaque_shading_intervals import I,Jet

class Curve:
    cn=list(map(I,[.2,.4,.6,.8,1.]))
    x=list(map(lambda x:I(math.log(x)),[.2,.4,.6,.8,1.]))
    controls=[[[I((i+1)*(k+1)/16) for k in range(4)], [I((i+2)*(4-k)/32) for k in range(4)]] for i in range(4)]

def test_loss_extension_polynomial_difference_encloses_independent_piecewise_values():
    from tools.opaque_loss_extension import LossExtension
    q=LossExtension(Curve());q.indices=[1,1]
    delta=q.delta(Jet(I(.3,.9)),1)
    with localcontext() as c:
        c.prec=90
        for value in [.3,.4,.5,.6,.7,.8,.9]:
            z=D.from_float(value).ln()
            def polynomial(i,ch):
                u=(z-Curve.x[i].lo)/(Curve.x[i+1].lo-Curve.x[i].lo);b=[v.lo for v in Curve.controls[i][ch]]
                return (1-u)**3*b[0]+3*(1-u)**2*u*b[1]+3*(1-u)*u*u*b[2]+u**3*b[3]
            index=min(sum(v.lo<=D.from_float(value) for v in Curve.cn)-1,3)
            for ch in range(2):assert delta[ch].lo<=polynomial(index,ch)-polynomial(1,ch)<=delta[ch].hi

def test_loss_extension_same_branch_cancels_exactly_and_refuses_moving_angles():
    from tools.opaque_loss_extension import LossExtension
    q=LossExtension(Curve())
    assert all(v.zero for v in q.delta(Jet(I(.45,.55)),1))
    with pytest.raises(ValueError,match='angle tangent'):q.delta(Jet(.5,tangent=1),1)
    with pytest.raises(ValueError,match='horizon'):q.delta(Jet(I(.1,.3)),1)


def test_cosine_mean_value_bounds_preserve_geometry_dependency():
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_shading_intervals import Jet,I
    from fractions import Fraction as Q
    # An exact affine cancellation must be recovered while a varying nonlinear
    # component keeps its nonzero range. Neither is a renderer error tolerance.
    def field(x,y):
        d=x*x+y*y+1;return [x-x+1,d.sqrt()]
    box=(I('.2','.21'),I('.3','.31'));bounds=bounded_cosines(field,*box)
    assert bounds[0].lo<=1<=bounds[0].hi and bounds[0].width < I('1e-4').lo
    for i in range(5):
        for j in range(5):
            q=field(Jet(Q(1,5)+Q(i,400)),Jet(Q(3,10)+Q(j,400)))
            for b,v in zip(bounds,q):assert b.lo<=v[0,0].lo and b.hi>=v[0,0].hi
    # The independent centre alone would exclude the nonconstant endpoint.
    assert bounds[1].width>I('1e-3').lo


@pytest.mark.parametrize('compiled',[False,True])
def test_centered_loss_preserves_constant_under_wide_high_mixed_jets(compiled):
    from contextlib import nullcontext
    from tools.opaque_arb_jets import backend
    from tools.opaque_loss_extension import LossExtension
    class Constant:
        x=[I(0),I(1)]
        controls=[[[I(1)]*4,[I(2)]*4]]
    with backend(profile='ball') if compiled else nullcontext():
        q=Jet(order=10,coeff={(0,0):I('.1','.9'),(1,0):I(1),(0,1):I('.01'),(1,1):I('.02')})
        result=LossExtension(Constant()).at_centered(q,0)
    for result,value in zip(result,[1,2]):
        assert result[0,0].lo<=value<=result[0,0].hi
        for i in range(11):
            for j in range(2):
                if i+j:assert result[i,j].lo<=0<=result[i,j].hi and result[i,j].width<I('1e-40').lo


@pytest.mark.parametrize('compiled',[False,True])
def test_centered_loss_matches_independent_cubic_and_first_motion(compiled):
    from contextlib import nullcontext
    from fractions import Fraction as Q
    from tools.opaque_arb_jets import backend
    from tools.opaque_loss_extension import LossExtension
    # Bernstein [1,2,4,8] equals 1+3*z+3*z^2+z^3. Log(q)=z.
    class Cubic:
        x=[I(0),I(1)]
        controls=[[[I(k) for k in [1,2,4,8]]]*2]
    with backend(profile='ball') if compiled else nullcontext():
        for z in [Q(-1,2),Q(0),Q(1,2),Q(1),Q(3,2)]:
            q=Jet(order=10,coeff={(i,j):I(z).exp()*I(Q(1,math.factorial(i)))*(I(Q(1,8)) if j else I(1)) for i in range(11) for j in range(2)})
            results=LossExtension(Cubic()).at_centered(q,0)
            for result in results:
                for i in range(11):
                    value=I(math.comb(3,i)*(1+z)**(3-i)) if i<=3 else I(0)
                    tangent=I(Q(1,8)*math.comb(3,i)*(3-i)*(1+z)**(2-i)) if i<=2 else I(0)
                    for actual,exact in [(result[i,0],value),(result[i,1],tangent)]:
                        assert actual.lo<=exact.lo<=exact.hi<=actual.hi
                        assert actual.width<I('1e-35').lo



def test_centered_horizon_extension_retains_linear_join_and_exact_branch_residual():
    from tools.opaque_full_material_reference import CenteredHorizonExtension
    class Horizon(Curve):
        horizon=[I(0),I(1)]
        values=[[I(1),I(2)]]
    e=CenteredHorizonExtension(Horizon())
    q=Jet(order=2,coeff={(0,0):I('.1'),(1,0):I(1),(0,1):I('.01')})
    for actual,a,b in zip(e.at(q,-1),e.horizon,e.values[0]):
        expected=q.lift(a)+q/q.lift(e.cn[0])*(b-a)
        assert actual.c==expected.c or all(actual[i,j].lo==expected[i,j].lo and actual[i,j].hi==expected[i,j].hi for i in range(3) for j in range(2))
    assert all(v.zero for v in e.delta(Jet(I('.45','.55')),1))
