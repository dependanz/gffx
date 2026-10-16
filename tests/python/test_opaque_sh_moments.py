from fractions import Fraction as F
import pytest
from tools.opaque_sh_moments import evaluate,tail_bound
from tools.opaque_sh_texture_reference import fixture,certify_partition,field
from tools.opaque_shading_intervals import I


def test_exact_bilinear_cell_moment_and_area_derivative():
 a=fixture();a['tex']=[[0,0,0],[1,2,3],[2,4,6],[4,8,12]];a['det']=2.;a['d_det']=3.
 z=evaluate(a,degree=0)
 for k in range(3):
  assert abs(z['value'][k]-float(F(13*(k+1),12)))<2e-15
  assert abs(z['tangent'][k]-float(F(13*(k+1),8)))<3e-15

@pytest.mark.parametrize('direction',['d_normal','d_uv','d_tex','d_sh','d_det'])
def test_values_and_first_differentials_against_independent_reference(direction):
 a=fixture();a['normal']=[[.1,.2,1],[.12,.19,.98],[.11,.22,1.01]]
 a['sh']=[[.9,.8,.7],[.03,-.02,.01],[.02,.04,-.03],[.01,.02,.03],[.01,0,.01],[.005,.01,0],[.01,-.01,.02],[.01,.005,0],[.005,.003,-.005]]
 if direction=='d_det':a[direction]=.3
 else:
  key=direction[2:];a[direction]=[[((i*len(row)+j)%5-2)*.01 for j in range(len(row))] for i,row in enumerate(a[key])]
 z=evaluate(a,degree=8);ref=certify_partition(lambda u,v:field(a,u,v),2);bounds=tail_bound(a,degree=8,q0=z['q0'])
 for name in ('value','tangent'):
  for x,b,e in zip(z[name],ref[name],bounds[name]):
   assert (I(x)-b).bound<I('1e-10').lo
   assert e.hi<I('1e-10').lo


def test_continuous_texture_knot_and_moving_boundary():
 # Two cells meeting at x=c, with slopes 1 and 2: integral on unit
 # triangle of x + max(0,x-c). Its moving-knot derivative is -(1-c)^2/2.
 # Clip analytically into a left quadrilateral (two triangles) and right triangle.
 c=.375;dc=.2
 vertices=[([(0,0),(c,0),(c,1-c)],[(0,0),(dc,0),(dc,-dc)],False), ([(0,0),(c,1-c),(0,1)],[(0,0),(dc,-dc),(0,0)],False), ([(c,0),(1,0),(c,1-c)],[(dc,0),(0,0),(dc,-dc)],True)]
 total=dt=0.
 for p,dp,right in vertices:
  a=fixture();a['uv']=[[x,y] for x,y in p];a['d_uv']=[[x,y] for x,y in dp]
  a['tex']=[[(-c if right else 0.)]*3,[(2-c if right else 1.)]*3,[(-c if right else 0.)]*3,[(2-c if right else 1.)]*3]
  a['d_tex']=[[-dc if right else 0.]*3 for _ in range(4)]
  u=[p[1][i]-p[0][i] for i in (0,1)];v=[p[2][i]-p[0][i] for i in (0,1)];du=[dp[1][i]-dp[0][i] for i in (0,1)];dv=[dp[2][i]-dp[0][i] for i in (0,1)]
  a['det']=u[0]*v[1]-u[1]*v[0];a['d_det']=du[0]*v[1]+u[0]*dv[1]-du[1]*v[0]-u[1]*dv[0]
  z=evaluate(a,degree=0);total+=z['value'][0];dt+=z['tangent'][0]
 assert abs(total-(1/6+(1-c)**3/6))<1e-14
 assert abs(dt+dc*(1-c)**2/2)<1e-14


def test_tiny_positive_cell_is_retained():
 a=fixture();a['det']=1e-200;a['d_det']=1e-210;z=evaluate(a,degree=0)
 assert min(z['value'])>0 and min(z['tangent'])>0

@pytest.mark.parametrize('degree',[-1,9,True,2.0])
def test_bounded_degree_is_enforced(degree):
 with pytest.raises(ValueError):evaluate(fixture(),degree=degree)


def test_zero_or_uncertifiable_norm_refuses():
 a=fixture();a['normal']=[[0,0,0]]*3
 with pytest.raises(ValueError):evaluate(a,degree=8)

def test_texture_extension_is_explicit_bounded_and_not_clamped():
 a=fixture();a['uv'][0]=[-1e-9,0.]
 with pytest.raises(ValueError):evaluate(a)
 z=evaluate(a,cell_extension=1e-8)
 assert z['value'][0]!=evaluate(fixture())['value'][0]
 with pytest.raises(ValueError):evaluate(a,cell_extension=1e-7)
