"""Exact controls for certified local moving-region integration."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest

def api():
    from tools.opaque_moving_reference import moving_fragments,integrate_moving_pixel,Poly,BranchCertificate,moving_perspective_field
    return moving_fragments,integrate_moving_pixel,Poly,BranchCertificate,moving_perspective_field

def contains(a,q):
    q=Q(q)
    with localcontext() as c:
        c.prec=100;v=D(q.numerator)/D(q.denominator)
    assert a.lo<=v<=a.hi,(a,q)

def silhouette(t=Q(1,4)):
    return [(t,-1,0),(t,3,0),(4+t,-1,0)],[(0,1,2)],[(1,0,0)]*3

def intersection(t=Q(1,4)):
    xy=[(-2,-2),(6,-2),(-2,6)]
    return [(x,y,x-t) for x,y in xy]+[(x,y,0) for x,y in xy],[(0,1,2),(3,4,5)],[(0,0,-1)]*3+[(0,0,0)]*3

def test_moving_silhouette_integrates_area_field_and_measure_derivatives():
    _,integrate,*_=api();v,f,m=silhouette()
    row=integrate(v,f,m,[lambda x,y:[x*x,y,x.lift(2,tangent=3)]])
    assert row['mask']==Q(3,4) and row['mask_derivative']==-1
    for a,b in zip(row['value'],[Q(21,64),Q(3,8),Q(3,2)]):contains(a,b)
    for a,b in zip(row['derivative'],[-Q(1,16),-Q(1,2),Q(1,4)]):contains(a,b)
    assert row['derivative'][0].hi<0 # Freezing visibility would incorrectly give zero.

@pytest.mark.parametrize('t',[Q(1,4),Q(3,4)])
def test_moving_depth_intersection_matches_closed_form(t):
    _,integrate,*_=api();v,f,m=intersection(t)
    fields=[lambda x,y:[x*x,x.lift(0),x.lift(0)],lambda x,y:[x.lift(0),x.lift(0),x.lift(1)]]
    row=integrate(v,f,m,fields)
    assert row['mask']==1 and row['mask_derivative']==0
    for a,b in zip(row['value'],[t**3/3,0,1-t]):contains(a,b)
    for a,b in zip(row['derivative'],[t*t,0,-1]):contains(a,b)


def test_moving_hidden_surface_and_background_have_exact_derivatives():
    _,integrate,*_=api();v,f,m=silhouette();v=v+[(x,y,z+1) for x,y,z in v];f=[(0,1,2),(3,4,5)];m=m+m
    fields=[lambda x,y:[x.lift(1),x.lift(0),x.lift(0)],lambda x,y:[x.lift(9,tangent=9)]*3]
    row=integrate(v,f,m,fields,background=(0,0,1))
    for a,b in zip(row['value'],[Q(3,4),0,Q(1,4)]):contains(a,b)
    for a,b in zip(row['derivative'],[-1,0,1]):contains(a,b)


def test_moving_subpixel_triangle_retains_positive_area_and_gradient():
    _,integrate,*_=api();v=[(Q(1,4),Q(1,4),0),(Q(33,128),Q(1,4),0),(Q(1,4),Q(33,128),0)]
    row=integrate(v,[(0,1,2)],[(0,0,0),(1,0,0),(0,0,0)],[lambda x,y:[x.lift(1)]*3])
    assert row['mask']==Q(1,32768) and row['mask_derivative']==Q(1,256)
    assert all(q.lo>0 for q in row['value']+row['derivative'])

@pytest.mark.parametrize('reverse',[False,True])
def test_moving_geometry_agrees_with_independent_rational_differences(reverse):
    fragments,integrate,*_=api();from tools.opaque_pixel_oracle import integrate as exact
    v,f,m=intersection();f=[tuple(reversed(x)) for x in f] if reverse else f
    colours=[(Q(1,3),Q(1,4),Q(1,2)),(Q(2,3),Q(3,4),Q(1,4))]
    row=integrate(v,f,m,[lambda x,y,c=c:[x.lift(z) for z in c] for c in colours])
    nominal=exact(v,f,colours,height=1,width=1)
    for a,b in zip(row['value'],nominal['colour'][0][0]):contains(a,b)
    for eps in (Q(1,1024),Q(1,4096)):
        plus=exact([[Q(x)+eps*Q(z) for x,z in zip(p,d)] for p,d in zip(v,m)],f,colours,height=1,width=1)
        minus=exact([[Q(x)-eps*Q(z) for x,z in zip(p,d)] for p,d in zip(v,m)],f,colours,height=1,width=1)
        for a,b,c in zip(row['derivative'],plus['colour'][0][0],minus['colour'][0][0]):contains(a,(b-c)/(2*eps))


def test_moving_branch_certificate_rejects_first_order_invisible_transition():
    _,_,P,C,_=api();c=C()
    assert c.classify(P([1,2]))==1 and c.classify(P([0]))==0
    with pytest.raises(ValueError,match='topology'):c.classify(P([0,0,1]))
    for entry in c.report()['strict']:
        q=list(map(Q,entry['coefficients']));r=Q(c.report()['radius'])
        assert sum(abs(x)*r**i for i,x in enumerate(q) if i)<abs(q[0])


def test_moving_visibility_refuses_tie_changes_and_collapsing_geometry():
    _,integrate,*_=api();v,f,m=intersection();v=[(x,y,0) for x,y,z in v]
    with pytest.raises(ValueError,match='topology'):integrate(v,f,m,[lambda x,y:[x.lift(1)]*3]*2)
    with pytest.raises(ValueError):integrate([(0,0,0),(1,0,0),(2,0,0)],[(0,1,2)],[(0,0,0),(0,1,0),(0,0,0)],[lambda x,y:[x.lift(1)]*3])


def test_moving_reference_preserves_work_and_cancellation_guards():
    _,integrate,*_=api();v,f,m=silhouette()
    with pytest.raises(ValueError):integrate(v,f,m,[lambda x,y:[x.lift(1)]*3],max_cells=0)
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):integrate(v,f,m,[lambda x,y:[x.lift(1)]*3],checkpoint=stop)

@pytest.mark.parametrize('faces',[[(0,1,2),(0,2,3)],[(0,1,3),(1,2,3)],[(0,1,4),(1,2,4),(2,3,4),(3,0,4)]])
def test_moving_shared_edges_do_not_invent_internal_coverage_gradients(faces):
    _,integrate,*_=api()
    v=[(Q(-6,5),Q(-11,10),0),(Q(16,5),Q(-11,10),0),(Q(16,5),Q(17,5),0),(Q(-6,5),Q(17,5),0),(Q(3,10),Q(7,10),0)]
    row=integrate(v,faces,[(1,0,0)]*5,[lambda x,y:[x,y,x+y]]*len(faces))
    assert row['mask']==1 and row['mask_derivative']==0
    for a,b in zip(row['value'],[Q(1,2),Q(1,2),1]):contains(a,b)
    for a in row['derivative']:contains(a,0)


def test_general_moving_geometry_matches_independent_affine_oracle():
    _,integrate,*_=api();from tools.opaque_pixel_oracle import integrate_affine
    v=[[-Q(6,5),-Q(11,10),-Q(7,5)],[Q(16,5),-Q(9,10),Q(21,10)],[-Q(11,10),Q(17,5),-Q(6,5)],[-Q(13,10),-Q(13,10),Q(1,10)],[Q(17,5),-Q(6,5),Q(1,10)],[-Q(6,5),Q(7,2),Q(1,10)]]
    f=[(0,1,2),(3,4,5)];m=[(Q(1,10),Q(1,20),-Q(1,10)),(Q(-1,10),Q(1,30),Q(1,20)),(Q(1,30),Q(-1,20),Q(1,10)),(Q(1,20),Q(-1,10),0),(0,Q(1,20),Q(-1,20)),(Q(-1,10),0,Q(1,20))]
    funcs=[lambda x,y:[(x+2)/8,(y+2)/8,(x+y+4)/16],lambda x,y:[(y+3)/8,(x+3)/8,(x+2*y+6)/24]]
    row=integrate(v,f,m,funcs)
    def exact(eps):
        vv=[[q+eps*d for q,d in zip(p,vel)] for p,vel in zip(v,m)]
        c=[[funcs[j](*vv[i][:2]) for i in face] for j,face in enumerate(f)]
        return integrate_affine(vv,f,c,height=1,width=1)['colour'][0][0]
    for a,b in zip(row['value'],exact(0)):contains(a,b)
    for h in (Q(1,2**24),Q(1,2**25)):
        assert h<Q(row['certificate']['radius'])
        for a,b,c in zip(row['derivative'],exact(h),exact(-h)):
            with localcontext() as context:
                context.prec=90;q=(b-c)/(2*h);value=D(q.numerator)/D(q.denominator)
                assert max(a.lo-value,value-a.hi,D(0))<D('1e-12')
    radius=Q(row['certificate']['radius'])
    for item in row['certificate']['strict']:
        coeff=list(map(Q,item['coefficients']))
        assert sum(abs(z)*radius**k for k,z in enumerate(coeff) if k)<abs(coeff[0])


def test_empty_moving_reference_validates_tolerance_and_returns_exact_zero():
    _,integrate,*_=api()
    with pytest.raises(ValueError):integrate([],[],[],[],tolerance='0')
    row=integrate([],[],[],[])
    assert row['mask']==row['mask_derivative']==0 and all(q.zero for q in row['value']+row['derivative'])


def test_integrated_attached_field_translation_cancels_false_shading_gradient():
    _,integrate,_,_,transport=api()
    from tools.opaque_principled_reference import perspective_field
    class AffineMaterial:
        def radiance(self,n,v,l,albedo,f0):return albedo
    tri=[(Q(1,10),Q(1,10),0),(Q(7,10),Q(1,10),0),(Q(1,10),Q(7,10),0)];motion=[(1,0,0)]*3
    colours=[[.2,.4,.6],[.4,.3,.5],[.3,.5,.2]]
    attrs=dict(inverse_w=[1,1,1],normals=[[0,0,1]]*3,views=[[0,0,1]]*3,albedo=colours,f0=[0,0,0],direction=[0,0,1])
    curve=AffineMaterial();attached=transport(curve,tri,motion,**attrs);fixed=perspective_field(curve,tri,**attrs)
    row=integrate(tri,[(0,1,2)],motion,[attached]);wrong=integrate(tri,[(0,1,2)],motion,[fixed])
    assert row['mask']==Q(9,50) and row['mask_derivative']==0
    for k in range(3):contains(row['value'][k],Q(9,50)*sum(Q(c[k]) for c in colours)/3);contains(row['derivative'][k],0)
    assert any(not q.lo<=0<=q.hi for q in wrong['derivative'])
