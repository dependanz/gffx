"""Independent exact integrals for jump-aware quadrature error bounds."""
from fractions import Fraction as Q
import math
import pytest
from tools.opaque_shading_intervals import I,Jet

@pytest.mark.parametrize('degree',[0,1,2,3,5])
def test_peano_encloses_step_ramp_and_curvature_jumps(degree):
    from tools.opaque_peano import residual_integral
    knot=Q(37,100)
    def field(x,y):
        z=x[0,0]-I(knot)
        if z.hi<0:return [x.lift(0)]
        if z.lo>0:return [(x-knot)**degree*(1+y)]
        return [x.lift(I(0,max(I(0).hi,z.hi)**degree))*(1+y)]
    def axis(box,axis):
        h=box[axis].width
        third=I(0) if axis or degree<3 else I(math.factorial(degree)//math.factorial(degree-3))*(I(1)-I(knot))**(degree-3)*2
        jumps=[I(0)]*3
        if not axis and degree<=2:jumps[degree]=I(2*math.factorial(degree))
        return [dict(third=third,jumps=jumps)]
    result=residual_integral(field,(I(0,1),I(0,1)),axis)
    exact=Q(3,2)*(1-knot)**(degree+1)/(degree+1)
    assert Q(result[0].lo)<=exact<=Q(result[0].hi)


def test_peano_tensor_scaled_box_and_tiny_residual():
    from tools.opaque_peano import residual_integral
    scale=Q(1,10**9);knot=Q(3,8)
    def field(x,y):
        z=x[0,0]-y[0,0]-I(knot)
        lo=max(Q(0),Q(z.lo));hi=max(Q(0),Q(z.hi))
        return [x.lift(I(I(lo**2).lo,I(hi**2).hi))*scale]
    def axis(box,axis):return [dict(third=I(0),jumps=[I(0),I(0),I(2*scale)])]
    box=(I('.25','.75'),I(0,'.25'))
    result=residual_integral(field,box,axis)
    # Integrate (x-y-knot)_+^2 exactly,first x then y.
    exact=scale*((Q(3,4)-knot)**4-(Q(3,4)-Q(1,4)-knot)**4)/12
    assert Q(result[0].lo)<=exact<=Q(result[0].hi)
    assert result[0].width<I('1e-12').lo


def test_peano_negative_bounds_and_unqualified_tangent_refuse():
    from tools.opaque_peano import residual_integral
    with pytest.raises(ValueError,match='nonnegative'):
        residual_integral(lambda x,y:[x],(I(0,1),I(0,1)),lambda box,axis:[dict(third=I(-1),jumps=[I(0)]*3)])
    with pytest.raises(ValueError,match='fixed'):
        residual_integral(lambda x,y:[x.lift(1,tangent=1)],(I(0,1),I(0,1)),lambda box,axis:[dict(third=I(0),jumps=[I(0)]*3)])


def test_piecewise_log_residual_encloses_derivatives_and_binary_knot_jumps():
    from types import SimpleNamespace
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_peano_material import segmented,difference
    # Two log-coordinate quadratic polynomials,second has an explicit tiny jump.
    cn=[I('.25'),I('.5'),I(1)];x=[q.ln() for q in cn];eps=I('1e-17')
    controls=[]
    for i in range(2):
        h=x[i+1]-x[i];shift=x[i]-x[1];factor=I(i+1)
        a=factor*shift**2+(eps if i else I(0));b=2*factor*shift;c=factor
        controls.append([[a,a+h*b/3,a+2*h*b/3+h*h*c/3,a+h*b+h*h*c]]*2)
    e=HorizonExtension(SimpleNamespace(cn=cn,x=x,controls=controls))
    q=Jet.variable(I('.49','.51'),order=3);result=segmented(e,q,0)
    for value in ('.49','.499','.5','.501','.51'):
        v=I(value);z=v.ln()-x[1]
        expected=[I(0)]*4 if v.hi<cn[1].lo else [z*z+eps,2*z/v,(1-z)/(v*v),(2*z-3)/(3*v*v*v)]
        for row in result:
            for i,want in enumerate(expected):assert row[i,0].lo<=want.lo and row[i,0].hi>=want.hi
    jump=difference(e,Jet.variable(cn[1],order=3),1,0)
    for row in jump:assert row[0,0].lo<=eps.lo<=eps.hi<=row[0,0].hi
    with pytest.raises(ValueError,match='moving knot'):
        segmented(e,Jet('.49',tangent=1),0)


def test_square_integrator_uses_peano_correction_without_relaxing_tolerance():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_peano import residual_integral
    scale=Q(1,10**9);knot=Q(37,100)
    def field(x,y):
        v=x[0,0]-I(knot);lo=max(Q(0),Q(v.lo));hi=max(Q(0),Q(v.hi))
        return [x.lift(I(I(lo**2).lo,I(hi**2).hi))*scale]
    def axis(box,axis):
        crosses=box[0].lo<=I(knot).hi and box[0].hi>=I(knot).lo
        return [dict(third=I(0),jumps=[I(0),I(0),I(2*scale if crosses and not axis else 0)])]
    q=integrate_square(lambda x,y:[x.lift(0)],correction_integrator=lambda box:residual_integral(field,box,axis),tolerance='1e-12')
    exact=scale*(1-knot)**3/3
    assert Q(q['value'][0].lo)<=exact<=Q(q['value'][0].hi)
    assert q['value'][0].width<=I('2e-12').lo


def test_radial_cosines_exact_polynomials_preserve_value_and_spatial_derivatives():
    from types import SimpleNamespace
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_moving_reference import P
    prepared={'n':[[P(0),P(1),P(0)],[P(0),P(0),P(0)],[P(1),P(0),P(0)]], 'v':[[P(0)]*3,[P(0)]*3,[P(2),P(0),P(0)]]}
    cosine=radial_cosines(SimpleNamespace(coefficients=prepared),[0,1,1])
    x=Jet.variable(I('.2','.3'),order=3);y=Jet(I('.4','.5'),order=3)
    nv,nl,vh=cosine(x,y);known=(1+x*x).sqrt().reciprocal()
    for got,want in [(nv,known),(nl,known/I(2).sqrt()),(vh,x.lift(((I(1)+1/I(2).sqrt())/2).sqrt()))]:
        for i in range(4):assert got[i,0].lo<=want[i,0].hi and got[i,0].hi>=want[i,0].lo
    # Independent point formulas for Taylor coefficients of(1+x*x)^(-1/2).
    x=Jet.variable(I('.25'),order=3);nv,nl,vh=cosine(x,y);v=I('.25');d=1+v*v
    exact=[1/d.sqrt(),-v/(d*d.sqrt()),(2*v*v-1)/(2*d*d*d.sqrt()),v*(3-2*v*v)/(2*d*d*d*d.sqrt())]
    for i,want in enumerate(exact):assert nv[i,0].lo<=want.hi and nv[i,0].hi>=want.lo


def test_peano_kernel_hulls_are_certified_not_sampled_maxima():
    from tools.opaque_peano import kernel_bounds
    sup,l1=kernel_bounds()
    assert sup[2].hi<I('.000309').lo
    assert l1[2].hi<sup[2].lo
    assert all(0<z.hi<1 for z in sup)


def test_reference_stop_is_not_retried_as_a_subdivision_domain():
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_shading_intervals import Unresolved
    calls=[]
    def stop(box):calls.append(box);raise Unresolved('external deadline')
    with pytest.raises(Unresolved) as error:
        integrate_square(lambda x,y:[x.lift(0)],correction_integrator=stop)
    assert error.value.details['reason']=='external deadline'
    assert len(calls)==1


def test_radial_cosines_retains_exact_rational_complexity_limit():
    from types import SimpleNamespace
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_moving_reference import P
    huge=P(2**5000);zero=P(0)
    coefficients={'n':[[huge,zero,zero],[zero]*3,[zero]*3],'v':[[huge,zero,zero],[zero]*3,[zero]*3]}
    with pytest.raises(ValueError,match='rational complexity budget'):
        radial_cosines(SimpleNamespace(coefficients=coefficients),[1,0,0])


@pytest.mark.parametrize('axis',[0,1])
def test_residual_direction_guides_refinement_without_relaxing_error(axis):
    """A one-dimensional kink must not spend its cell budget splitting a constant axis."""
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_peano import residual_integral
    knot=Q(1,3)
    def field(x,y):
        q=(x,y)[axis];v=q[0,0]-I(knot);lo=max(Q(0),Q(v.lo));hi=max(Q(0),Q(v.hi))
        return [q.lift(I(I(lo**2).lo,I(hi**2).hi))]
    def bounds(box,j):
        crossing=box[axis].lo<=I(knot).hi and box[axis].hi>=I(knot).lo
        return [dict(third=I(0),jumps=[I(0),I(0),I(2 if crossing and j==axis else 0)])]
    class Correction:
        preferred_split_axis=axis
        def __call__(self,box):return residual_integral(field,box,bounds)
    result=integrate_square(lambda x,y:[x.lift(0)],correction_integrator=Correction(),tolerance='1e-12',max_cells=64,max_depth=10)
    exact=(1-knot)**3/3
    assert Q(result['value'][0].lo)<=exact<=Q(result['value'][0].hi)
    assert result['value'][0].width<=I('2e-12').lo
    assert result['cells']<=64
    # Successful leaves may vary along the active coordinate only.
    assert all((row[2:4] if axis==0 else row[:2])==['0','1'] for row in result['leaves'])


def test_peano_directional_components_retain_complete_error_enclosure():
    from tools.opaque_peano import residual_integral
    rows=[]
    def field(x,y):return [x**3+7*y**3]
    def bounds(box,axis):return [dict(third=I(6 if axis==0 else 42),jumps=[I(0)]*3)]
    result=residual_integral(field,(I(0,1),I(0,1)),bounds,error_components=rows.append)
    assert result[0].lo<=2<=result[0].hi and len(rows)==1
    assert rows[0][0][0].lo==0 and rows[0][0][0].hi>0
    assert abs(Q(rows[0][1][0].hi)/Q(rows[0][0][0].hi)-7)<Q(1,10**55)
    total=rows[0][0][0]+rows[0][1][0]
    assert (I(2)-total).lo>=result[0].lo and (I(2)+total).hi<=result[0].hi
