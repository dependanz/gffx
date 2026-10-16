"""Independent convex normal-square value enclosures retain all Jet tangents."""
from fractions import Fraction as F
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_sh_production_reference import normal_square

def inside(box,x):assert box.lo<=I(x).lo<=I(x).hi<=box.hi

def test_wide_duffy_field_has_positive_proven_normal_length():
 rows=[[.22266739792694795,.5442601963331791,.8055327949961126],[.032724206094290290,.9082585316022711,.22920594969927727],[.047848001974119696,.8974939794206683,.2179250265892063]]
 u=Jet.variable(I(0,1),order=10);v=Jet(I(0,1),order=10)
 n=[Jet(rows[0][k],order=10)+u*(rows[1][k]-rows[0][k])+(1-u)*v*(rows[2][k]-rows[0][k]) for k in range(3)]
 naive=sum(x*x for x in n);assert naive[0,0].lo<0
 bounded=normal_square(n,rows,u,v);assert bounded[0,0].lo>I('.67').hi
 for key in naive.c:
  if key!=(0,0):assert (bounded[key].lo,bounded[key].hi)==(naive[key].lo,naive[key].hi)

@pytest.mark.parametrize('sign',[-1,1])
def test_constant_vector_preserves_exact_first_differential(sign):
 rows=[[1,2,2]]*3;u=Jet.variable(I(0,1),order=2);v=Jet(I(0,1),order=2)
 n=[Jet(x,order=2,tangent=sign*x) for x in rows[0]];q=normal_square(n,rows,u,v);length=q.sqrt()
 inside(length[0,0],3);inside(length[0,1],3*sign)

@pytest.mark.parametrize('u0,v0',[(-.1,.5),(.5,1.1)])
def test_no_convex_tightening_outside_duffy_square(u0,v0):
 rows=[[1,0,0],[2,0,0],[3,0,0]];u=Jet(u0);v=Jet(v0);n=[1+u+(1-u)*v*2,Jet(0),Jet(0)]
 plain=sum(x*x for x in n);q=normal_square(n,rows,u,v)
 assert (q[0,0].lo,q[0,0].hi)==(plain[0,0].lo,plain[0,0].hi)

@pytest.mark.parametrize('u0,v0',[(F(0),F(0)),(F(1,2),F(1,3)),(F(1),F(1)),(F(1,8),F(7,8))])
def test_encloses_exact_rational_squared_norm(u0,v0):
 rows=[[1,-2,3],[2,1,-1],[-1,3,2]];u=Jet(I(u0));v=Jet(I(v0));n=[Jet(rows[0][k])+u*(rows[1][k]-rows[0][k])+(1-u)*v*(rows[2][k]-rows[0][k]) for k in range(3)]
 q=normal_square(n,rows,u,v);exact=sum((rows[0][k]+u0*(rows[1][k]-rows[0][k])+(1-u0)*v0*(rows[2][k]-rows[0][k]))**2 for k in range(3));inside(q[0,0],exact)
