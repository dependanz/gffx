"""Closed-form controls for validated pixel quadrature, not grid-agreement claims."""
from fractions import Fraction as Q
import pytest

def api():
    from tools.opaque_shading_intervals import Interval,Jet,integrate_triangle,Unresolved
    return Interval,Jet,integrate_triangle,Unresolved

def contains(interval,q):
    from decimal import Decimal,localcontext
    with localcontext() as ctx:
        ctx.prec=100
        q=Q(q);a=Decimal(q.numerator)/Decimal(q.denominator)
    assert interval.lo<=a<=interval.hi,(interval,a)

@pytest.mark.parametrize('power',[0,1,2,3,4,5,6])
def test_validated_triangle_polynomials_enclose_exact_integrals(power):
    I,J,integrate,_=api();row=integrate(lambda x,y:[x**power],[(0,0),(1,0),(0,1)],tolerance='1e-12')
    contains(row['value'][0],Q(1,(power+1)*(power+2)))
    assert row['value'][0].width<=I('2e-12').hi

def test_validated_triangle_parameter_derivative_is_integrated_separately():
    I,J,integrate,_=api()
    row=integrate(lambda x,y:[x.lift(2,tangent=1)*x*x+y],[(0,0),(1,0),(0,1)])
    contains(row['value'][0],Q(1,3));contains(row['derivative'][0],Q(1,12))

def test_validated_interval_arithmetic_and_jet_transcendentals():
    I,J,_,_=api();one=I(1)
    contains((one/3)*3,1);contains(I(2).sqrt()**2,2);contains(I(2).ln().exp(),2)
    x=J.variable(I(2),axis=0,order=6);v=x.ln();w=x.sqrt()
    for k in range(1,7):contains(v[k,0],Q((-1)**(k+1),k*2**k))
    assert w[0,0].lo>0
    with pytest.raises(ValueError):one/I(-1,1)

def test_validated_quadrature_detects_sample_missed_polynomial():
    I,J,integrate,Unresolved=api()
    def hidden(x,y):return [((x-x.lift('.5'))*(x*x-x+x.lift('.1')))**2]
    with pytest.raises(Unresolved) as e:integrate(hidden,[(0,0),(1,0),(0,1)],max_cells=1,tolerance='1e-12')
    assert e.value.details['reason']=='cell budget'
    row=integrate(hidden,[(0,0),(1,0),(0,1)])
    contains(row['value'][0],Q(1,5600));assert row['value'][0].lo>0

def test_validated_quadrature_refuses_invalid_domains_limits_and_cancellation():
    I,J,integrate,Unresolved=api()
    with pytest.raises(ValueError):integrate(lambda x,y:[x],[(0,0),(1,0),(0,1)],tolerance='0')
    with pytest.raises(ValueError):integrate(lambda x,y:[x],[(0,0),(1,0),(0,1)],max_cells=0)
    with pytest.raises(Unresolved):integrate(lambda x,y:[(x-x.lift('.5')).ln()],[(0,0),(1,0),(0,1)],max_cells=4)
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):integrate(lambda x,y:[x],[(0,0),(1,0),(0,1)],checkpoint=stop)

def test_validated_triangle_orientation_and_tiny_positive_area():
    _,_,integrate,_=api()
    tri=[(Q(1,4),Q(1,4)),(Q(33,128),Q(1,4)),(Q(1,4),Q(33,128))]
    for points in (tri,tri[::-1]):
        row=integrate(lambda x,y:[x.lift(1)],points)
        contains(row['value'][0],Q(1,32768));assert row['value'][0].lo>0

def test_validated_jet_keeps_interval_constant_separate_from_formal_increment():
    I,J,_,_=api();x=J.variable(I(1,2),order=6);v=x.ln();expected=I(1,2).ln()
    assert v[0,0].lo==expected.lo and v[0,0].hi==expected.hi
    for k in range(1,7):
        for q in (Q((-1)**(k+1),k),Q((-1)**(k+1),k*2**k)):contains(v[k,0],q)

def test_validated_triangle_mixed_coordinates_and_moving_geometry():
    _,_,integrate,_=api()
    row=integrate(lambda x,y:[x*x*y],[(0,0),(1,0),(0,1)])
    contains(row['value'][0],Q(1,60))
    row=integrate(lambda x,y:[x.lift(1)],[(0,0),(1,0),(0,1)],motion=[(0,0),(1,0),(0,0)])
    contains(row['value'][0],Q(1,2));contains(row['derivative'][0],Q(1,2))

@pytest.mark.parametrize('power',[0,1,4,8,9,10])
def test_validated_gauss5_rule_encloses_exact_high_degree_controls(power):
    _,_,integrate,_=api();row=integrate(lambda x,y:[x**power],[(0,0),(1,0),(0,1)],gauss_points=5)
    contains(row['value'][0],Q(1,(power+1)*(power+2)))

def test_validated_gauss5_refuses_sample_missed_degree_ten():
    _,_,integrate,Unresolved=api()
    def hidden(x,y):return [((252*x**5-630*x**4+560*x**3-210*x*x+30*x-1)/252)**2]
    with pytest.raises(Unresolved):integrate(hidden,[(0,0),(1,0),(0,1)],gauss_points=5,max_cells=1)
    row=integrate(hidden,[(0,0),(1,0),(0,1)],gauss_points=5)
    contains(row['value'][0],Q(1,22*252**2));assert row['value'][0].lo>0

@pytest.mark.parametrize('operation',['reciprocal','ln','sqrt'])
def test_order_ten_mixed_jets_enclose_independent_series(operation):
    from decimal import Decimal,localcontext
    I,J,_,_=api();x=J.variable(2,order=10);x.c[0,1]=I(1)
    got=getattr(x,operation)()
    def rational_coefficient(k):
        if operation=='reciprocal':return Q((-1)**k,2**(k+1))
        if operation=='ln':return Q((-1)**(k+1),k*2**k)
        q=Q(1)
        for j in range(k):q*=Q(1,2)-j;q/=j+1
        return q/Q(2)**k
    with localcontext() as ctx:
        ctx.prec=100
        for k in range(11):
            for tangent in range(2):
                order=k+tangent
                if operation=='ln' and order==0:value=Decimal(2).ln()
                else:
                    q=rational_coefficient(order)*(k+1 if tangent else 1)
                    value=Decimal(q.numerator)/Decimal(q.denominator)
                    if operation=='sqrt':value*=Decimal(2).sqrt()
                assert got[k,tangent].lo<=value<=got[k,tangent].hi


def test_interval_quadrature_adds_independent_signed_correction_and_tangent():
    from tools.opaque_shading_intervals import integrate_triangle
    def base(x,y):return [x.lift(0)]
    def correction(x,y):return [x.lift('-0.003',tangent='.002')]
    row=integrate_triangle(base,[(0,0),(1,0),(0,1)],correction=correction,gauss_points=5)
    from decimal import Decimal as D
    assert row['value'][0].lo<=D('-.0015')<=row['value'][0].hi
    assert row['derivative'][0].lo<=D('.001')<=row['derivative'][0].hi

def test_interval_quadrature_bounds_a_nonsmooth_residual_without_differentiating_it():
    from tools.opaque_shading_intervals import integrate_triangle,I
    from decimal import Decimal as D
    def zero(x,y):return [x.lift(0)]
    def residual(x,y):
        q=x[0,0];lo=max(D(0),q.lo-D('.5'));hi=max(D(0),q.hi-D('.5'))
        return [x.lift(I(lo,hi)**2*I('1e-8'))]
    row=integrate_triangle(zero,[(0,0),(1,0),(0,1)],correction=residual,tolerance='1e-8')
    exact=D('1e-8')/D(192)
    assert row['value'][0].lo<=exact<=row['value'][0].hi
    assert integrate_triangle(zero,[(0,0),(1,0),(0,1)])['value'][0].hi==0


def test_interval_correction_includes_exact_moving_area_derivative():
    from tools.opaque_shading_intervals import integrate_triangle
    from decimal import Decimal as D
    def zero(x,y):return [x.lift(0)]
    def residual(x,y):return [x.lift('-.003',tangent='.002')]
    row=integrate_triangle(zero,[(0,0),(1,0),(0,1)],motion=[(0,0),(1,0),(0,0)],correction=residual)
    assert row['value'][0].lo<=D('-.0015')<=row['value'][0].hi
    assert row['derivative'][0].lo<=D('-.0005')<=row['derivative'][0].hi


def test_interval_residual_freezes_one_smooth_extension_per_cell():
    from tools.opaque_shading_intervals import integrate_triangle,I
    from decimal import Decimal as D
    state={};calls=[]
    def prepare(x,y):state['right']=x>=D('.5');calls.append(state['right'])
    def smooth(x,y):return [(x-D('.5'))**2 if state['right'] else x.lift(0)]
    def residual(x,y):
        q=x[0,0]
        if state['right']:
            lo=min(D(0),q.lo-D('.5'));hi=min(D(0),q.hi-D('.5'));v=-I(lo,hi)**2
        else:
            lo=max(D(0),q.lo-D('.5'));hi=max(D(0),q.hi-D('.5'));v=I(lo,hi)**2
        return [x.lift(v)]
    row=integrate_triangle(smooth,[(0,0),(1,0),(0,1)],correction=residual,prepare=prepare)
    from fractions import Fraction
    assert row['value'][0].lo<=Fraction(1,192)<=row['value'][0].hi
    assert len(calls)==row['cells'] and True in calls and False in calls
