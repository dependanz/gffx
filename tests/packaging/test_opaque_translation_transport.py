"""Exact transport controls: internal jumps, moving endpoints and invalid inputs."""
from fractions import Fraction as Q
from decimal import Decimal as D
import pytest
from tools.opaque_shading_intervals import I

def test_translation_contains_closed_polynomial_and_parameter_directions():
    from tools.opaque_translation_transport import translation_derivative
    # Integral x^3 shifted by 2 theta with endpoints 1-theta, 3+4theta.
    # Leibniz: 2*(27-1)+4*27+1=161, independently differentiated primitive.
    result=translation_derivative(lambda x:[I(x**3)],Q(1),Q(3),speed=2,lower_speed=-1,upper_speed=4)
    assert result[0].lo<=161<=result[0].hi and result[0].width<D('1e-40')

def test_internal_step_transport_is_not_the_zero_pointwise_derivative():
    from tools.opaque_translation_transport import translation_derivative
    # H(x-.5-theta): area=.5-theta, ordinary derivative zero almost everywhere.
    result=translation_derivative(lambda x:[I(int(x>Q(1,2)))],Q(0),Q(1),speed=-1)
    assert result[0].lo==result[0].hi==-1

def test_attached_visibility_edge_cancels_lower_transport_term():
    from tools.opaque_translation_transport import translation_derivative
    # Integral from .25+theta to1 of (x-theta)^2 = ((1-theta)^3-.25^3)/3.
    result=translation_derivative(lambda x:[I(x*x)],Q(1,4),Q(1),speed=-1,lower_speed=1)
    assert result[0].lo==result[0].hi==-1

def test_interval_endpoint_uncertainty_is_preserved_with_negative_coefficients():
    from tools.opaque_translation_transport import translation_derivative
    r=translation_derivative(lambda x:[I(1,2) if x==0 else I(3,4)],0,1,speed=-2)
    assert r[0].lo<=-6 and r[0].hi>=-2

@pytest.mark.parametrize('a,b,speed',[(1,0,1),(0,0,1),(0,1,float('nan')),(0,1,True)])
def test_invalid_transport_parameters_refuse(a,b,speed):
    from tools.opaque_translation_transport import translation_derivative
    with pytest.raises(ValueError):translation_derivative(lambda x:[I(1)],a,b,speed=speed)

def test_endpoint_shape_mismatch_refuses():
    from tools.opaque_translation_transport import translation_derivative
    with pytest.raises(ValueError):translation_derivative(lambda x:[I(1)]*(1+int(x)),0,1,speed=1)

@pytest.mark.parametrize('role',['view','normal'])
def test_original_endpoint_intervals_contain_independent_decimal_algebra(role):
    import os,json,hashlib
    from pathlib import Path
    from decimal import localcontext
    from tools.opaque_translation_transport import original_endpoint
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.principled_material_oracle import material_pair
    p=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(p.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    data=json.loads(p.read_bytes())['budget'];curve=PreparedPrincipled(data,1.)
    for x in (Q(0),Q(3,4),Q(1)):
        for z0,dz in ((0.,.02),(1.3,.0001)):
            result=original_endpoint(curve,x,role=role,z0=z0,dz=dz)
            with localcontext() as ctx:
                ctx.prec=90;z=D.from_float(z0)+D.from_float(dz)*(D(x.numerator)/D(x.denominator)-D('.5'));var=[D(1),D(0),z];fixed=[D(0),D(0),D(1)];n,v=(fixed,var) if role=='view' else (var,fixed);l=list(map(D,[1,1,2]));expected=material_pair(data,[*n,*v,*l,.2,.4,.6,1.,.04,.04,.04]);cos=sum(a*b for a,b in zip(n,l))/(sum(a*a for a in n)*D(6)).sqrt()
                for a,b in zip(result,expected):assert a.lo<=b*max(cos,D(0))<=a.hi and a.width<D('1e-40')
    with pytest.raises(ValueError,match='horizon'):original_endpoint(curve,Q(1,2),role=role,z0=0,dz=.02)

def test_endpoint_touching_uncertain_resource_knot_refuses():
    from types import SimpleNamespace
    from tools.opaque_translation_transport import original_endpoint
    # A declared knot enclosure overlaps the exact normalized endpoint cosine.
    q=I(1)/I(2).sqrt();curve=SimpleNamespace(roughness=I(1),cn=[q])
    with pytest.raises(ValueError,match='resource knot'):original_endpoint(curve,Q(1,2),role='view',z0=1,dz=1)
