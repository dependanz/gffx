"""Independent implicit-curve and weighted Peano-moment controls."""
from fractions import Fraction as Q
import math
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved

@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('point',[Q(1,10),Q(1,4),Q(1,2)])
def test_circle_level_set_derivatives_match_closed_form(sign,point):
    from tools.opaque_boundary_taylor import curve_jet
    from decimal import Decimal as D,localcontext
    x=I(point);y=(1-x*x).sqrt()*sign
    got=curve_jet(lambda x,y:x*x+y*y,x,y,order=3)
    with localcontext() as c:
        c.prec=90;z=D(point.numerator)/D(point.denominator);v=(1-z*z).sqrt()*sign
        expected=[v,-z/v,-1/(2*v**3),-z/(2*v**5)]
        for i,v in enumerate(expected):assert got[i,0].lo<=v<=got[i,0].hi

@pytest.mark.parametrize('sign',[-1,1])
def test_curve_box_encloses_all_roots_and_derivatives(sign):
    from tools.opaque_boundary_taylor import curve_jet
    from decimal import Decimal as D,localcontext
    x=I('.1','.2');y=I((1-I('.2')**2).sqrt().lo,(1-I('.1')**2).sqrt().hi)*sign
    got=curve_jet(lambda x,y:x*x+y*y,x,y,order=3)
    with localcontext() as c:
        c.prec=90
        for n in range(11):
            z=D('.1')+D(n)/100;v=sign*(1-z*z).sqrt()
            for i,w in enumerate([v,-z/v,-1/(2*v**3),-z/(2*v**5)]):assert got[i,0].lo<=w<=got[i,0].hi

def test_implicit_curve_refuses_zero_denominator_and_cancellation():
    from tools.opaque_boundary_taylor import curve_jet
    with pytest.raises(ValueError):curve_jet(lambda x,y:x*x+y*y,I(0,1),I(-1,1),order=3)
    def stop():raise Unresolved('intentional stop')
    with pytest.raises(Unresolved):curve_jet(lambda x,y:x+y,I(0),I(1),order=3,checkpoint=stop)

@pytest.mark.parametrize('kernel_order',[1,2,3,4,6])
@pytest.mark.parametrize('power',[0,1,2,3])
def test_weighted_moment_encloses_independent_piecewise_polynomial(kernel_order,power):
    from tools.opaque_boundary_taylor import moment
    from decimal import Decimal as D,localcontext
    a,b,c=I(Q(1,7)),I(Q(4,5)),I('.5');got=moment(a,b,c,kernel_order,power)
    with localcontext() as ctx:
        ctx.prec=90
        a,b,c=D(1)/7,D(4)/5,D('.5');root=(D(10)/7).sqrt();x=(5+2*root).sqrt()/6;y=(5-2*root).sqrt()/6
        ns=[D('.5')-x,D('.5')-y,D('.5'),D('.5')+y,D('.5')+x];u=(322-13*D(70).sqrt())/1800;v=(322+13*D(70).sqrt())/1800;ws=[u,v,D(64)/225,v,u]
        expected=D(0);edges=sorted({a,b,*[x for x in ns if a<x<b]})
        for lo,hi in zip(edges,edges[1:]):
            co=[D(0)]*(kernel_order+1)
            for n,w in zip(ns,ws):
                if n>(lo+hi)/2:
                    for j in range(kernel_order):co[j]+=w*math.comb(kernel_order-1,j)*n**(kernel_order-1-j)*(-1)**j/D(math.factorial(kernel_order-1))
            for j in range(kernel_order+1):co[j]-=D(math.comb(kernel_order,j)*(-1)**j)/math.factorial(kernel_order)
            for j,z in enumerate(co):
                for k in range(power+1):expected+=z*math.comb(power,k)*(-c)**(power-k)*(hi**(j+k+1)-lo**(j+k+1))/(j+k+1)
        assert got.lo<=expected<=got.hi

def test_weighted_taylor_rejects_unproved_center_and_negative_remainder():
    from tools.opaque_boundary_taylor import weighted_taylor
    with pytest.raises(ValueError):weighted_taylor([I(1)],I(0),I('.5'),I('.2','.7'),I('.8'),3)
    with pytest.raises(ValueError):weighted_taylor([I(1)],I(-1),I('.5'),I(0),I(1),3)

@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('side',[0,1])
def test_certified_outside_crossing_is_exactly_clipped(sign,side):
    from tools.opaque_boundary_taylor import clipped_root
    def q(x):return sign*(x*x+x*I('.1')+x.lift(2))
    knot=I(sign*(1 if side==0 else 4));got=clipped_root(q,I(0,1),knot)
    assert got.lo==got.hi==side

@pytest.mark.parametrize('sign',[-1,1])
def test_bounded_root_encloses_independent_quadratic_formula(sign):
    from tools.opaque_boundary_taylor import clipped_root
    got=clipped_root(lambda x:sign*(x*x+x*2),I(0,1),I(sign))
    exact=I(2).sqrt()-1
    assert got.lo<=exact.lo<=exact.hi<=got.hi and got.width<I('.001').lo

def test_bounded_root_refuses_unknown_sign_and_preserves_cancellation():
    from tools.opaque_boundary_taylor import clipped_root
    with pytest.raises(ValueError):clipped_root(lambda x:x*x,I(-1,1),I('.5'))
    def stop():raise Unresolved('intentional stop')
    with pytest.raises(Unresolved):clipped_root(lambda x:x,I(0,1),I('.5'),checkpoint=stop)

@pytest.mark.parametrize('count',[2,4])
@pytest.mark.parametrize('a,b',[(I(0),I(1)),(I('.12','.13'),I('.72','.74'))])
def test_fixed_kernel_partition_covers_every_uncertain_endpoint(count,a,b):
    from tools.opaque_boundary_taylor import supports,moment
    parts=supports(a,b,count)
    assert len(parts)==count
    for actual_a in [a.lo,a.hi]:
        for actual_b in [b.lo,b.hi]:
            lengths=sum((max(Q(0),min(Q(actual_b),Q(j+1,count))-max(Q(actual_a),Q(j,count))) for j in range(count)),Q(0))
            assert lengths==Q(actual_b)-Q(actual_a)
    summed=sum((moment(x,y,I('.5'),3,2) for x,y in parts),I(0))
    direct=moment(a,b,I('.5'),3,2)
    assert summed.lo<=direct.hi and direct.lo<=summed.hi
