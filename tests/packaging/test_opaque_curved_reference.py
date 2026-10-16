"""Independent curved-horizon area,motion,Jacobian and refusal controls."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import pytest,json

def enclosed(interval,value):
    assert interval.lo<=value<=interval.hi
    assert interval.width<=D('2e-12')

def test_parabolic_coverage_and_boundary_motion_have_closed_integrals(record_property):
    from tools.opaque_curved_reference import CurvedTriangle,integrate_curved
    # q=x+y-a-b*y^2, q>0 front. d a/d theta=1.
    patch=CurvedTriangle([(0,0),(1,0),(0,1)],[0,0,-Q(1,10),1,1,[-Q(1,4),-1]])
    result=integrate_curved(patch,lambda x,y:[x.lift(1)])
    with localcontext() as c:
        c.prec=90;a=D(1)/4;b=D(1)/10;r=2*a/(1+(1-4*a*b).sqrt())
        area=D(1)/2-a*r+r*r/2-b*r**3/3
        enclosed(result['value'][0],area);enclosed(result['derivative'][0],-r)
        # A fixed-domain pointwise derivative of constant1 is zero: it misses
        # the entire moving-horizon jump,whose magnitude exceeds0.25 here.
        assert result['derivative'][0].hi<D('-.25')
    record_property('curved_reference',json.dumps(result,default=str))

def test_complementary_regions_and_linear_limit_preserve_partition_and_vjp():
    from tools.opaque_curved_reference import CurvedTriangle,integrate_curved
    tri=[(0,0),(1,0),(0,1)];coeff=[0,0,0,1,1,[-Q(1,4),-1]]
    a=integrate_curved(CurvedTriangle(tri,coeff),lambda x,y:[x.lift(1)])
    b=integrate_curved(CurvedTriangle(tri,coeff,positive=False),lambda x,y:[x.lift(1)])
    enclosed(a['value'][0],D(15)/32);enclosed(a['derivative'][0],-D(1)/4)
    enclosed(a['value'][0]+b['value'][0],D(1)/2);enclosed(a['derivative'][0]+b['derivative'][0],D(0))

def test_moving_triangle_jacobian_and_horizon_transport_both_survive():
    from tools.opaque_curved_reference import CurvedTriangle,integrate_curved
    # Triangle scales by1+theta; q=x+y-(1+theta)/4. Area scales byscale^2.
    p=CurvedTriangle([(0,0),(1,0),(0,1)],[0,0,0,1,1,[-Q(1,4),-Q(1,4)]],motion=[(0,0),(1,0),(0,1)])
    r=integrate_curved(p,lambda x,y:[x.lift(1)])
    enclosed(r['value'][0],D(15)/32);enclosed(r['derivative'][0],D(15)/16)

def test_uncertified_topology_and_work_limits_are_refused():
    from tools.opaque_curved_reference import CurvedTriangle,integrate_curved
    from tools.opaque_shading_intervals import Unresolved
    tri=[(0,0),(1,0),(0,1)]
    # Tangency q=(x+y-.5)^2 and a root through a vertex never get certified.
    for q in ([1,2,1,-1,-1,Q(1,4)],[0,0,0,1,1,0],[1,2,1,0,0,-Q(1,4)]):
        with pytest.raises(Unresolved):CurvedTriangle(tri,q)
    patch=CurvedTriangle(tri,[0,0,-Q(1,10),1,1,-Q(1,4)])
    for n in (0,257):
        with pytest.raises(ValueError):integrate_curved(patch,lambda x,y:[x.lift(1)],max_cells=n)
    with pytest.raises(Unresolved):integrate_curved(patch,lambda x,y:[x.lift(1)],max_cells=1)


def test_perspective_quadratic_matches_independent_rational_corner_algebra():
    from tools.opaque_curved_reference import perspective_numerator
    from tools.opaque_moving_reference import P
    tri=[(0,0),(1,0),(0,1)];iw=[Q(1),Q(1,2),Q(1,3)]
    normals=[[1,0,1],[0,1,1],[1,1,2]];views=[[1,2,3],[3,2,1],[2,4,1]];motion=[[0,0,1]]*3
    coeff=perspective_numerator(tri,inverse_w=iw,normals=normals,views=views,normal_motion=motion)
    for x,y,theta in ((Q(1,5),Q(1,7),Q(1,11)),(Q(1,3),Q(1,4),-Q(1,13))):
        beta=[1-x-y,x,y];raw=[b*w for b,w in zip(beta,iw)]
        n=[sum(raw[i]*(normals[i][k]+theta*motion[i][k]) for i in range(3)) for k in range(3)]
        v=[sum(raw[i]*views[i][k] for i in range(3)) for k in range(3)]
        q=sum(sum(z*theta**j for j,z in enumerate(p.c))*m for p,m in zip(coeff,(x*x,x*y,y*y,x,y,1)))
        assert q==sum(a*b for a,b in zip(n,v))


def test_radial_field_composition_keeps_integral_and_parameter_derivative():
    from tools.opaque_curved_reference import CurvedTriangle,integrate_curved,radial_homogeneous_fields
    p=CurvedTriangle([(0,0),(1,0),(0,1)],[0,0,0,1,1,[-Q(1,4),-1]])
    fields=radial_homogeneous_fields(p,[(0,0),(1,0),(0,1)],inverse_w=[1,1,1],fields={'f':[[0],[1],[2]]})
    r=integrate_curved(p,lambda t,s:fields(t,s)['f'],radial=True)
    enclosed(r['value'][0],D(63)/128);enclosed(r['derivative'][0],-D(3)/32)
