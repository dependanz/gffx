"""Independent integration-rule controls; never imports the production renderer."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_sh_texture_reference import certify,rule,field,fixture,numeric

def test_gauss5_moments_and_positive_mass():
    nodes,weights=rule()
    assert all(w.lo>0 for w in weights)
    for degree in range(10):
        value=sum((w*x**degree for x,w in zip(nodes,weights)),I(0))
        exact=I(Q(1,degree+1))
        assert value.lo<=exact.hi and exact.lo<=value.hi

def test_tenth_degree_remainder_and_parameter_tangent():
    def f(u,v):return [u**10+2*v**10,(u**10+v**10)*Jet(3,order=u.order,tangent=7)]
    c=certify(f)
    for j,want in enumerate((Q(3,11),Q(6,11))):
        assert c['value'][j].lo<=I(want).lo<=c['value'][j].hi
    assert c['tangent'][1].lo<=I(Q(14,11)).lo<=c['tangent'][1].hi
    assert c['remainder'][0][0].hi>0

def test_polynomial_bilinear_texture_and_moving_area():
    a=fixture();a['tex']=[[0,0,0],[1,2,3],[2,4,6],[4,8,12]]
    a['det']=2.;a['d_det']=3.
    c=certify(lambda u,v:field(a,u,v))
    # Texture = s+2t+st on the unit triangle; integral=13/24.
    for k in range(3):
        exact=I(Q(13*(k+1),12));deriv=I(Q(13*(k+1),8))
        assert c['value'][k].lo<=exact.lo and c['value'][k].hi>=exact.hi
        assert c['tangent'][k].lo<=deriv.lo and c['tangent'][k].hi>=deriv.hi
        assert c['remainder'][0][k].hi< I('1e-45').hi

def test_zero_normal_and_texture_knot_refuse():
    a=fixture();a['normal']=[[0,0,0]]*3
    with pytest.raises(ValueError):certify(lambda u,v:field(a,u,v))
    a=fixture();a['uv'][2]=[0,1.001]
    with pytest.raises(ValueError):field(a,Jet(.3),Jet(.4))

@pytest.mark.parametrize('direction',['d_normal','d_uv','d_tex','d_sh','d_det'])
def test_nonconstant_normal_all_first_differentials(direction):
    a=fixture();a['normal']=[[.1,.2,1],[.12,.19,.98],[.11,.22,1.01]]
    a['sh']=[[.9,.8,.7],[.03,-.02,.01],[.02,.04,-.03],[.01,.02,.03],[.01,0,.01],[.005,.01,0],[.01,-.01,.02],[.01,.005,0],[.005,.003,-.005]]
    if direction=='d_det':a[direction]=.3
    else:
        key=direction[2:];a[direction]=[[((i*len(row)+j)%5-2)*.01 for j in range(len(row))] for i,row in enumerate(a[key])]
    c=certify(lambda u,v:field(a,u,v));value,tangent=numeric(a)
    for name,actual in [('value',value),('tangent',tangent)]:
        for x,b in zip(actual,c[name]):assert float(b.lo)-1e-14<=x<=float(b.hi)+1e-14
    assert max(float(x.hi) for row in c['remainder'] for x in row)<1e-10

def test_tensor_mixed_error_and_nonpolynomial_integral():
    c=certify(lambda u,v:[u**10*v**10,1/((1+u)*(1+v))])
    exact=[I(Q(1,121)),I(2).ln()**2]
    for x,y in zip(c['value'],exact):assert x.lo<=y.lo and x.hi>=y.hi

def test_tiny_positive_area_and_tangent_are_retained():
    a=fixture();a['det']=1e-200;a['d_det']=1e-210
    c=certify(lambda u,v:field(a,u,v))
    assert all(x.lo>0 for x in c['value']+c['tangent'])

def test_normal_crossing_zero_refuses_even_with_finite_corner_lengths():
    a=fixture();a['normal']=[[0,0,1],[0,0,-1],[0,0,1]]
    with pytest.raises(ValueError):certify(lambda u,v:field(a,u,v))

def test_universal_first_differential_remainder_encloses_specific_direction():
    from tools.opaque_sh_texture_reference import remainder
    a=fixture();a['normal']=[[.1,.2,1],[.12,.19,.98],[.11,.22,1.01]]
    specific=dict(a);general=dict(a)
    for key in ('normal','uv','tex','sh'):
        specific['d_'+key]=[[.13 for x in row] for row in a[key]]
        general['d_'+key]=[[I(-1,1) for x in row] for row in a[key]]
    specific['d_det']=.13;general['d_det']=I(-1,1)
    s=remainder(lambda u,v:field(specific,u,v));g=remainder(lambda u,v:field(general,u,v))
    assert all(x.hi<=y.hi for x,y in zip(s[1],g[1]))

@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_input_refuses(bad):
    a=fixture();a['tex'][0][0]=bad
    with pytest.raises(ValueError):certify(lambda u,v:field(a,u,v))
