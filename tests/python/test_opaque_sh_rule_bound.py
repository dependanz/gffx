import pytest
from tools.opaque_sh_rule_bound import bounds
from tools.opaque_sh_texture_reference import fixture
@pytest.mark.parametrize('order',[3,4,5])
def test_constant_normal_exactness_including_unit_field_directions(order):
 b=bounds(fixture(),order)
 assert all(x.zero for vals in (b['value'],b['unit_direction']) for x in vals)
@pytest.mark.parametrize('order',[2,6,True,3.0])
def test_invalid_order_refuses(order):
 with pytest.raises(ValueError):bounds(fixture(),order)
def test_zero_normal_refuses():
 a=fixture();a['normal']=[[0.,0.,0.]]*3
 with pytest.raises(ValueError):bounds(a)
def test_texture_cell_crossing_refuses():
 a=fixture();a['uv'][1][0]=1.1
 with pytest.raises(ValueError):bounds(a)
def test_area_direction_has_no_area_scaling_shortcut():
 a=fixture();a['normal']=[[.1,.2,1.],[.2,.25,.9],[.12,.19,1.02]];a['det']=1e-100;a['sh'][1]=[.3,.4,.5]
 b=bounds(a)
 assert min(x.hi for x in b['unit_direction'])>max(x.hi for x in b['value'])*(10**50)

@pytest.mark.parametrize('order',[3,4,5])
def test_majorant_covers_pointwise_taylor_coefficients_and_unit_directions(order):
 from tools.opaque_shading_intervals import I,Jet
 from tools.opaque_sh_production_reference import polynomial
 from fractions import Fraction as F
 import math
 a=fixture();a['normal']=[[.1,.2,1.],[.2,.25,.9],[.12,.19,1.02]]
 a['sh']=[[((i*3+j)%7-3)*.1 for j in range(3)] for i in range(9)]
 for key in ('normal','uv','tex','sh'):a['d_'+key]=[[1. if (i+j)%2 else -1. for j in range(len(row))] for i,row in enumerate(a[key])]
 a['d_det']=-1.;b=bounds(a,order);n=2*order
 factor=I(F(math.factorial(order)**4,(2*order+1)*math.factorial(n)**2))
 for axis in (0,1):
  for point in (0.,.5,1.):
   u=Jet.variable(I(point),order=n) if axis==0 else Jet(I(point),order=n)
   v=Jet.variable(I(point),order=n) if axis==1 else Jet(I(point),order=n)
   z=polynomial(a,u,v)
   for k in range(3):
    assert (factor*I(z[k][n,0].bound)).hi<=b['value'][k].hi
    assert (factor*I(z[k][n,1].bound)).hi<=b['unit_direction'][k].hi

def test_norm_bernstein_bound_retains_orthogonality_cancellation():
 from tools.opaque_sh_rule_bound import norm_axis_coefficients
 n=[[.6,0.,.8],[.8,0.,.6],[.6,0.,.8]]
 a,b=norm_axis_coefficients(n,0)
 assert float(a.hi)<=.080000000000001
 assert float(b.hi)<=.080000000000001
 a,b=norm_axis_coefficients(n,1)
 assert a.zero and b.zero
