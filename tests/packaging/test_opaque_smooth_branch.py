"""Independent controls for an equivalent centered cubic smooth branch."""
import math,itertools
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from tools.opaque_shading_intervals import I,Jet


def fixture(coefficients):
    from tools.opaque_loss_extension import LossExtension
    a,b,c,d=map(I,coefficients);x0=I(-2);h=I(2)
    value=a+b*x0+c*x0*x0+d*x0*x0*x0
    slope=b+2*c*x0+3*d*x0*x0;quad=c+3*d*x0
    controls=[value,value+h*slope/3,value+2*h*slope/3+h*h*quad/3,value+h*slope+h*h*quad+h*h*h*d]
    curve=SimpleNamespace(x=[x0,I(0)],controls=[[controls,controls]])
    return LossExtension(curve)


@pytest.mark.parametrize('coefficients',[(1,0,0,0),(1,-2,0,0),(1,-2,3,0),(1,-2,3,-4)])
@pytest.mark.parametrize('point',['.25','.5','1'])
@pytest.mark.parametrize('rate',[-1,1])
def test_centered_branch_encloses_independent_log_polynomial_derivatives(coefficients,point,rate):
    from tools.opaque_smooth_branch import SmoothBranches
    e=fixture(coefficients)
    # Construct q0=point with exactly the selected spatial derivative.
    q=Jet(order=3,coeff={(0,0):I(point),(1,0):I(rate)})
    got=SmoothBranches(e).at(q,0)
    with localcontext() as ctx:
        ctx.prec=100;a,b,c,d=map(D,coefficients);v=D(point);z=v.ln()
        p=a+b*z+c*z*z+d*z*z*z;dp=b+2*c*z+3*d*z*z;ddp=2*c+6*d*z;dddp=6*d
        known=[p,dp/v*rate,(ddp-dp)/(2*v*v),(dddp-3*ddp+2*dp)/(6*v*v*v)*rate]
        for row in got:
            for k,value in enumerate(known):assert row[k,0].lo<=value<=row[k,0].hi


def test_centered_branch_keeps_uncertain_controls_over_interval():
    from tools.opaque_smooth_branch import SmoothBranches
    from tools.opaque_loss_extension import LossExtension
    bs=[I(1,2),I(-2,-1),I(3,4),I(0,1)];e=LossExtension(SimpleNamespace(x=[I(-2),I(0)],controls=[[bs,bs]]));got=SmoothBranches(e).at(Jet.variable(I('.3','.8'),order=3),0)
    with localcontext() as ctx:
        ctx.prec=100
        for controls in itertools.product(*[(v.lo,v.hi) for v in bs]):
            a,b,c,d=controls
            for point in map(D,['.3','.47','.8']):
                u=(point.ln()+2)/2;t=1-u
                p=t**3*a+3*t*t*u*b+3*t*u*u*c+u**3*d
                du=3*((b-a)*t*t+2*(c-b)*t*u+(d-c)*u*u)
                ddu=6*((c-2*b+a)*t+(d-2*c+b)*u);dddu=6*(d-3*c+3*b-a)
                known=[p,du/(2*point),(ddu/4-du/2)/(2*point**2),(dddu/8-3*ddu/4+du)/(6*point**3)]
                for row in got:
                    for k,value in enumerate(known):assert row[k,0].lo<=value<=row[k,0].hi


def test_centered_branch_cache_is_bounded_invalidates_and_returns_fresh_jets():
    from tools.opaque_smooth_branch import SmoothBranches
    e=fixture((1,-2,3,-4));s=SmoothBranches(e);q=Jet.variable(I('.5'),order=3)
    first=s.at(q,0);before=[[(v[k,0].lo,v[k,0].hi) for k in range(4)] for v in first];first[0].c[0,0].lo=I(-10000).lo
    assert [[(v[k,0].lo,v[k,0].hi) for k in range(4)] for v in s.at(q,0)]==before
    for n in range(5):
        e.curve.controls[0][0][0]=I(n);s.at(q,0);assert len(s._cache)<=2
    assert [[(v[k,0].lo,v[k,0].hi) for k in range(4)] for v in s.at(q,0)]!=before


@pytest.mark.parametrize('index',[True,1,-2])
def test_centered_branch_refuses_invalid_indices(index):
    from tools.opaque_smooth_branch import SmoothBranches
    with pytest.raises(ValueError):SmoothBranches(fixture((1,0,0,0))).at(Jet('.5'),index)


@pytest.mark.parametrize('order',[4,6])
def test_centered_branch_refuses_unqualified_order(order):
    from tools.opaque_smooth_branch import SmoothBranches
    with pytest.raises(ValueError):SmoothBranches(fixture((1,0,0,0))).at(Jet('.5',order=order),0)


def test_centered_branch_keeps_horizon_formula_and_rejects_moving_angle():
    from tools.opaque_smooth_branch import SmoothBranches
    from tools.opaque_full_material_reference import HorizonExtension
    e=HorizonExtension(SimpleNamespace(cn=[I('.25')],values=[[I(2),I(3)]],horizon=[I(1),I(0)]));s=SmoothBranches(e);q=Jet.variable(I('.1'),order=3)
    assert [(v[0,0].lo,v[1,0].hi) for v in s.at(q,-1)]==[(I('1.4').lo,I(4).hi),(I('1.2').lo,I(12).hi)]
    with pytest.raises(ValueError):s.at(Jet('.1',tangent=1),-1)
