"""Exact conic-root identity, derivative preservation and mismatch refusals."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest

def setup(quadratic=True,positive=True):
    from tools.opaque_curved_reference import CurvedTriangle,perspective_numerator,radial_homogeneous_fields
    n=[[-Q(1,4),0,1],[Q(3,4),0,1],[Q(3,4),1,1]]
    v=[[1,0,0],[1,0,0],[1,-Q(1,10) if quadratic else 0,0]];tri=[(0,0),(1,0),(0,1)]
    q=perspective_numerator(tri,inverse_w=[1]*3,normals=n,views=v)
    p=CurvedTriangle(tri,q,positive=positive)
    f=radial_homogeneous_fields(p,tri,inverse_w=[1]*3,fields=dict(n=n,v=v))
    return p,f

@pytest.mark.parametrize('quadratic',[False,True])
@pytest.mark.parametrize('positive',[False,True])
@pytest.mark.parametrize('order',[0,1,3,10])
def test_root_factor_preserves_point_jets_and_exact_zero(quadratic,positive,order):
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_peano_material import radial_cosines
    p,f=setup(quadratic,positive);stable=radial_cosines(f,[1,1,2],horizon_patch=p);direct=radial_cosines(f,[1,1,2])
    s=Jet.variable(I(Q(1,5)),order=order) if order else Jet(Q(1,5))
    # Complete transverse interval has exactly zero numerator on the conic.
    z=p.horizon_numerator(Jet(0,order=order),Jet.variable(I(0,1),order=order) if order else Jet(I(0,1)))
    assert all(v.zero for v in z.c.values())
    for d in [Q(0),Q(1,10000),Q(1,4),Q(1)]:
        distance=Jet(d,order=order);u=1-distance if p.inside else distance;t,_=p.radial_map(u,s)
        a=stable(t,s,horizon_distance=distance);b=direct(t,s)
        for x,y in zip(a,b):
            for k in set(x.c)|set(y.c):assert max(x[k].lo,y[k].lo)<=min(x[k].hi,y[k].hi)
        # Independent scalar rational/sqrt evaluation at 90 decimal digits.
        with localcontext() as c:
            c.prec=90;ss=D(1)/5;aa=-ss*ss/10 if quadratic else D(0);r=D('.5')/(1+(1+aa).sqrt());dd=D(d.numerator)/D(d.denominator)
            tt=r*(1-dd) if p.inside else r+dd*(1-r);yy=tt*ss
            expected=(tt-D('.25')-(yy*yy/10 if quadratic else 0))/(((tt-D('.25'))**2+yy**2+1)*(1+(yy*yy/100 if quadratic else 0))).sqrt()
            assert a[0][0,0].lo<=expected<=a[0][0,0].hi or d==0 and abs(expected)<D('1e-85')

@pytest.mark.parametrize('positive',[False,True])
def test_radial_derivatives_overlap_direct_identity(positive):
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_peano_material import radial_cosines
    p,f=setup(True,positive);a=radial_cosines(f,[1,1,2],horizon_patch=p);b=radial_cosines(f,[1,1,2]);d=Jet.variable(I(Q(1,8)),order=10)**4;s=Jet(Q(1,3),order=10)
    t,_=p.radial_map(1-d if p.inside else d,s)
    for x,y in zip(a(t,s,horizon_distance=d),b(t,s)):
        for k in set(x.c)|set(y.c):assert max(x[k].lo,y[k].lo)<=min(x[k].hi,y[k].hi)

def test_mismatched_horizon_and_missing_distance_refuse():
    from tools.opaque_shading_intervals import Jet
    from tools.opaque_peano_material import radial_cosines
    p,f=setup(True);wrong,_=setup(False)
    with pytest.raises(ValueError,match='horizon numerator'):radial_cosines(f,[1,1,2],horizon_patch=wrong)
    a=radial_cosines(f,[1,1,2],horizon_patch=p)
    with pytest.raises(ValueError,match='horizon distance'):a(Jet(1),Jet(0))
    with pytest.raises(ValueError,match='horizon distance'):radial_cosines(f,[1,1,2])(Jet(1),Jet(0),horizon_distance=Jet(0))
