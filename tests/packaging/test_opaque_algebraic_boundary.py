"""Independent positive normalized-cosine branch and implicit derivative controls."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
from types import SimpleNamespace
import pytest

def fixture(role):
    from tools.opaque_moving_reference import P
    from tools.opaque_peano_material import radial_cosines
    fixed=[[1,0,0],[0,0,0],[0,0,0]];moving=[[1,0,0],[0,1,1],[0,0,0]]
    n,v=(fixed,moving) if role==0 else (moving,fixed)
    fields=SimpleNamespace(coefficients={k:[[P(x) for x in row] for row in values] for k,values in [('n',n),('v',v)]})
    return radial_cosines(fields,[1,0,0])

@pytest.mark.parametrize('role',[0,1])
@pytest.mark.parametrize('order',[1,2,3,4])
def test_algebraic_implicit_curve_matches_independent_reciprocal(role,order):
    from tools.opaque_shading_intervals import I
    from tools.opaque_boundary_taylor import curve_jet
    fn=fixture(role);x=I(Q(3,4));y=I(0);k=I(Q(4,5))
    got=curve_jet(lambda a,b:fn.implicit(role,k,a,b),x,y,order=order)
    for n in range(1,order+1):
        want=Q((-1)**n)*Q(3,4)/Q(3,4)**(n+1)
        assert Q(got[n,0].lo)<=want<=Q(got[n,0].hi)

@pytest.mark.parametrize('role',[0,1])
@pytest.mark.parametrize('order',[0,1,3,10])
def test_role_only_cosine_retains_every_original_coefficient(role,order):
    from tools.opaque_shading_intervals import I,Jet
    fn=fixture(role);x=Jet.variable(I('.7','.8'),order=order) if order else Jet(I('.7','.8'));y=Jet(I('.1','.2'),order=order)
    old=fn(x,y)[role];new=fn.role(role,x,y)
    assert old.order==new.order
    for key in set(old.c)|set(new.c):assert (old[key].lo,old[key].hi)==(new[key].lo,new[key].hi)

@pytest.mark.parametrize('role',[0,1])
def test_algebraic_curve_range_encloses_independent_derivatives(role):
    from tools.opaque_shading_intervals import I
    from tools.opaque_boundary_taylor import curve_jet
    fn=fixture(role);x=I('.7','.8');y=I('-.1','.1');k=I(Q(4,5))
    got=curve_jet(lambda a,b:fn.implicit(role,k,a,b),x,y,order=3)
    for n in range(1,4):
        wants=[Q((-1)**n)*Q(3,4)/v**(n+1) for v in (Q(7,10),Q(4,5))]
        assert Q(got[n,0].lo)<=min(wants)<=max(wants)<=Q(got[n,0].hi)

def test_algebraic_boundary_rejects_wrong_branch_and_invalid_role():
    from tools.opaque_shading_intervals import Jet,I
    fn=fixture(0);x,y=Jet('.75'),Jet(0)
    for role in (-1,2,True):
        with pytest.raises(ValueError):fn.role(role,x,y)
    for k in (I(0),I(-1),I(0,1),I(2)):
        with pytest.raises(ValueError):fn.implicit(0,k,x,y)
    # Negative numerator has the same squared equation but the wrong q=k branch.
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_moving_reference import P
    f=SimpleNamespace(coefficients={'n':[[P(-1),P(0),P(0)],[P(0)]*3,[P(0)]*3],'v':[[P(1),P(0),P(0)],[P(0)]*3,[P(0)]*3]})
    with pytest.raises(ValueError):radial_cosines(f,[1,0,0]).implicit(0,I('.8'),x,y)

@pytest.mark.parametrize('positive',[False,True])
@pytest.mark.parametrize('role',[0,1])
@pytest.mark.parametrize('order',[0,1,3,10])
def test_horizon_role_only_retains_exact_coefficients(positive,role,order):
    from tests.packaging.test_opaque_horizon_factor import setup
    from tools.opaque_curved_reference import flattened_radial_map
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_shading_intervals import I,Jet
    p,f=setup(True,positive);fn=radial_cosines(f,[1,1,2],horizon_patch=p)
    u=Jet.variable(I('.2','.21'),order=order) if order else Jet(I('.2','.21'));s=Jet(Q(1,3),order=order)
    t,_,distance,numerator=flattened_radial_map(p,u,s,with_numerator=True)
    kw=dict(horizon_distance=distance,horizon_numerator=numerator);a=fn(t,s,**kw)[role];b=fn.role(role,t,s,**kw)
    assert all((a[k].lo,a[k].hi)==(b[k].lo,b[k].hi) for k in set(a.c)|set(b.c))
