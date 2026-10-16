import ctypes,os
from fractions import Fraction as F
import pytest
from tools.opaque_sh_texture_reference import fixture

@pytest.fixture
def kernel():
 path=os.environ.get('GFFX_SH_RULE_DLL')
 if not path:pytest.skip('isolated development kernel required')
 lib=ctypes.CDLL(path);ptr=ctypes.POINTER(ctypes.c_double)
 lib.sh_gauss_rule.argtypes=[ptr,ctypes.c_int,ptr];lib.sh_gauss_rule.restype=ctypes.c_int
 return lib

def packed(a):
 out=[]
 for prefix in ('','d_'):
  for key in ('normal','uv','tex','sh'):
   for row in a.get(prefix+key,[[0.]*len(r) for r in a[key]]):out.extend(row)
  out.append(a['det'] if not prefix else a.get('d_det',0.))
 return (ctypes.c_double*110)(*out)

@pytest.mark.parametrize('order',[3,4,5])
def test_exact_texture_and_moving_area(kernel,order):
 a=fixture();a['tex']=[[0,0,0],[1,2,3],[2,4,6],[4,8,12]];a['det']=2.;a['d_det']=3.
 z=(ctypes.c_double*6)();assert kernel.sh_gauss_rule(packed(a),order,z)==0
 for k in range(3):
  assert abs(z[k]-float(F(13*(k+1),12)))<3e-15
  assert abs(z[k+3]-float(F(13*(k+1),8)))<4e-15

@pytest.mark.parametrize('order',[3,4,5])
def test_moving_continuous_texture_boundary(kernel,order):
 c=.375;dc=.2;total=dt=0.
 regions=[([(0,0),(c,0),(c,1-c)],[(0,0),(dc,0),(dc,-dc)],False), ([(0,0),(c,1-c),(0,1)],[(0,0),(dc,-dc),(0,0)],False), ([(c,0),(1,0),(c,1-c)],[(dc,0),(0,0),(dc,-dc)],True)]
 for p,dp,right in regions:
  a=fixture();a['uv']=p;a['d_uv']=dp
  a['tex']=[[(-c if right else 0.)]*3,[(2-c if right else 1.)]*3,[(-c if right else 0.)]*3,[(2-c if right else 1.)]*3]
  a['d_tex']=[[-dc if right else 0.]*3 for _ in range(4)]
  u=[p[1][i]-p[0][i] for i in (0,1)];v=[p[2][i]-p[0][i] for i in (0,1)];du=[dp[1][i]-dp[0][i] for i in (0,1)];dv=[dp[2][i]-dp[0][i] for i in (0,1)]
  a['det']=u[0]*v[1]-u[1]*v[0];a['d_det']=du[0]*v[1]+u[0]*dv[1]-du[1]*v[0]-u[1]*dv[0]
  z=(ctypes.c_double*6)();assert kernel.sh_gauss_rule(packed(a),order,z)==0;total+=z[0];dt+=z[3]
 assert abs(total-(1/6+(1-c)**3/6))<1e-14
 assert abs(dt+dc*(1-c)**2/2)<1e-14

@pytest.mark.parametrize('order',[3,4,5])
def test_tiny_positive_area(kernel,order):
 a=fixture();a['det']=1e-200;a['d_det']=1e-210;z=(ctypes.c_double*6)();assert kernel.sh_gauss_rule(packed(a),order,z)==0
 assert min(z)>0

@pytest.mark.parametrize('case',['low_order','high_order','zero_norm','nan','negative_area','overflow'])
def test_refusals(kernel,case):
 a=fixture();order=3
 if case=='low_order':order=2
 if case=='high_order':order=6
 if case=='zero_norm':a['normal']=[[0.,0.,0.]]*3
 if case=='nan':a['tex'][0][0]=float('nan')
 if case=='negative_area':a['det']=-1.
 if case=='overflow':a['tex']=[[1e308]*3]*4;a['sh']=[[1e308]*3]*9
 z=(ctypes.c_double*6)();assert kernel.sh_gauss_rule(packed(a),order,z)!=0
