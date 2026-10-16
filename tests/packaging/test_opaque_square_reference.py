"""Closed integrals qualify direct-square outward quadrature and first tangents."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
import pytest

def test_square_polynomial_and_first_tangent():
    from tools.opaque_square_reference import integrate_square
    r=integrate_square(lambda u,v:[(u+u.lift(1,tangent=1))*u*v**3])
    assert Q(r['value'][0].lo)<=Q(5,24)<=Q(r['value'][0].hi)
    assert Q(r['derivative'][0].lo)<=Q(1,8)<=Q(r['derivative'][0].hi)
    assert r['cells']==1

def test_square_nonlinear_closed_integral_and_tangent():
    from tools.opaque_square_reference import integrate_square
    r=integrate_square(lambda u,v:[1/(u+v+u.lift(2,tangent=1))])
    with localcontext() as c:
        c.prec=90;a=D(2);value=(a+2)*(a+2).ln()-2*(a+1)*(a+1).ln()+a*a.ln();derivative=((a+2)*a/(a+1)**2).ln()
    for key,want in [('value',value),('derivative',derivative)]:
        q=r[key][0];assert q.lo<=want<=q.hi and q.width<=D('2e-12')

def test_square_curved_area_includes_boundary_flux():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_curved_reference import CurvedTriangle
    p=CurvedTriangle([(0,0),(1,0),(0,1)],[0,0,-Q(1,10),1,1,[-Q(1,4),-1]])
    r=integrate_square(lambda u,s:[p.radial_map(u,s)[1]])
    with localcontext() as c:
        c.prec=90;a=D(1)/4;b=D(1)/10;x=2*a/(1+(1-4*a*b).sqrt());value=D(1)/2-a*x+x*x/2-b*x**3/3
    assert r['value'][0].lo<=value<=r['value'][0].hi
    assert r['derivative'][0].lo<=-x<=r['derivative'][0].hi

def test_square_refuses_unresolved_and_resource_growth():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    for n in (0,257):
        with pytest.raises(ValueError):integrate_square(lambda u,v:[u],max_cells=n)
    with pytest.raises(Unresolved):integrate_square(lambda u,v:[1/(u-Q(1,2))],max_depth=0)


def test_horizon_extension_encloses_first_linear_and_log_segments():
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_shading_intervals import I,Jet
    import math
    class Curve:
        cn=list(map(I,[.2,.4,.6]))
        x=[I(math.log(v)) for v in (.2,.4,.6)]
        horizon=[I(.9),I(.7)]
        values=[[I(.8),I(.6)],[I(.7),I(.5)],[I(.6),I(.4)]]
        controls=[[[I(.8),I(.77),I(.73),I(.7)],[I(.6),I(.57),I(.53),I(.5)]],[[I(.7),I(.67),I(.63),I(.6)],[I(.5),I(.47),I(.43),I(.4)]]]
    e=HorizonExtension(Curve());q=Jet(I(.15,.3));bounds=e.delta(q,-1)
    for x in (.15,.2,.25,.3):
        actual=e.at(Jet(x),-1 if x<.2 else 0);smooth=e.at(Jet(x),-1)
        for b,a,c in zip(bounds,actual,smooth):
            d=a[0,0]-c[0,0];assert b.lo<=d.lo and b.hi>=d.hi
    assert all(v.zero for v in e.delta(Jet(I(0,.1)),-1))
    with pytest.raises(ValueError,match='angle tangent'):e.delta(Jet(.25,tangent=1),0)


def test_piece_error_budget_is_area_weighted_and_conserves_whole_pixel_radius():
    from tools.opaque_full_material_reference import piece_tolerances
    from tools.opaque_shading_intervals import I
    triangles=[[[0,0],[1,0],[0,1]],[[0,0],[2,0],[0,1]],[[0,0],[0,0],[0,0]]]
    got=piece_tolerances([{'triangle':t} for t in triangles])
    assert len(got)==3 and got[2]==0
    assert Q(got[1])==2*Q(got[0]) or abs(Q(got[1])-2*Q(got[0]))<Q(1,10**70)
    assert 0<sum(map(Q,got))<=Q(1,10**12)/2
    assert Q(got[1])>Q(1,6*10**12)
    # Both independent basis channels fit their allocated radius; max albedo<=1
    # keeps every RGB channel inside the unchanged total radius 1e-12.
    bounds=[I(-v,v) for v in got]
    total=sum(bounds,I(0))
    assert total.width<=I('1e-12').lo
    with pytest.raises(ValueError):piece_tolerances([{'triangle':triangles[2]}])


def test_outward_products_match_exact_endpoints_for_all_signs_and_ambient_precision():
    from decimal import localcontext
    from itertools import product
    from tools.opaque_shading_intervals import I
    points=[Q(-1234567890123456789,10**17),Q(-2,3),Q(0),Q(2,7),Q(9876543210987654321,10**18)]
    intervals=[I(I(a).lo,I(b).hi) for a,b in product(points,points) if a<=b]
    with localcontext() as c:
        c.prec=7
        for a,b in product(intervals,intervals):
            got=a*b;exact=[Q(x)*Q(y) for x,y in product((a.lo,a.hi),(b.lo,b.hi))]
            assert Q(got.lo)<=min(exact)<=max(exact)<=Q(got.hi)
            assert Q(got.hi)-Q(got.lo)>=max(exact)-min(exact)


def test_complete_cover_accounting_replaces_parent_and_retains_failure():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import I,Unresolved
    class Correction:
        preferred_split_axis=0
        def __call__(self,box):
            area=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo));radius=area**2*I('1e-12')
            self.error_breakdown=[[dict(ordinary=radius/2,global_jumps=I(0),localized_jumps=I(0),signed_correction=I(0))] for _ in (0,1)]
            return [I(radius.hi.copy_negate(),radius.hi)]
    got=integrate_square(lambda x,y:[x.lift(0)],correction_integrator=Correction(),tolerance='8e-13')
    audit=got['error_accounting'];assert audit['leaves']==2 and len(audit['largest_leaves'])==2
    assert list(map(D,audit['channels'][0]['ordinary_radius']))==[D('5e-13')]*2
    assert all(r['depth']==1 for r in audit['largest_leaves'])
    with pytest.raises(Unresolved) as e:integrate_square(lambda x,y:[x.lift(0)],correction_integrator=Correction(),tolerance='1e-13',max_cells=1)
    complete=e.value.details['last_complete_enclosure'];assert complete['error_accounting']['leaves']==1
    assert list(map(D,complete['error_accounting']['channels'][0]['ordinary_radius']))==[D('1e-12')]*2
    assert e.value.details['last']['box']==['0','1','0','1']


def test_complete_cover_accounting_caps_leaf_details():
    from tools.opaque_square_reference import cover_accounting
    rows=[dict(box=[str(i),str(i+1),'0','1'],depth=4,widths=[[str(i+1),'0']],base_widths=[['0','0']],correction_widths=[[str(i+1),'0']],correction_error_components=None) for i in range(12)]
    r=cover_accounting(rows);assert r['leaves']==12 and len(r['largest_leaves'])==5
    assert r['largest_leaves'][0]['box'][0]=='11'
    assert r['components_complete'] is False
    with pytest.raises(ValueError):cover_accounting(rows*22)
