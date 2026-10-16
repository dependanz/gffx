"""Development-only a-priori quadrature truncation bound.

Bounds every supplied corner-field direction with component infinity norm <=1,
including determinant. Does not certify upstream geometry, floating quadrature
roundoff, clipping, texture-knot extension or runtime selection transitions.
Positive Taylor majorants over each Duffy axis; no sampled-grid inference.
"""
import math
from fractions import Fraction as F
from tools.opaque_shading_intervals import I
from tools.opaque_sh_texture_reference import validate

class M:
 def __init__(self,v,d=None):self.v=v;self.d=[I(0)]*len(v) if d is None else d
 def __add__(self,b):
  if not isinstance(b,M):b=constant(b,len(self.v)-1)
  return M([x+y for x,y in zip(self.v,b.v)],[x+y for x,y in zip(self.d,b.d)])
 __radd__=__add__
 def __mul__(self,b):
  if not isinstance(b,M):b=constant(b,len(self.v)-1)
  n=len(self.v);v=[I(0)]*n;d=[I(0)]*n
  for i in range(n):
   for j in range(n-i):
    if not self.v[i].zero and not b.v[j].zero:v[i+j]=v[i+j]+self.v[i]*b.v[j]
    if not self.d[i].zero and not b.v[j].zero:d[i+j]=d[i+j]+self.d[i]*b.v[j]
    if not self.v[i].zero and not b.d[j].zero:d[i+j]=d[i+j]+self.v[i]*b.d[j]
  return M(v,d)
 __rmul__=__mul__

def constant(v,n,d=0):return M([I(v)]+[I(0)]*n,[I(d)]+[I(0)]*n)

def inverse_majorant(qmin,q,p,n):
 if p==F(1,2):h0=1/qmin.sqrt()
 elif p==1:h0=1/qmin
 elif p==F(3,2):h0=1/(qmin*qmin.sqrt())
 elif p==2:h0=1/(qmin*qmin)
 else:raise ValueError('bounded reciprocal exponent')
 h=[h0]
 for k in range(1,n+1):
  x=I(F(k-1)+p)*q.v[1]*h[k-1]
  if k>=2:x=x+I(F(k-2)+2*p)*q.v[2]*h[k-2]
  h.append(x/(k*qmin))
 return M(h)

def norm_axis_coefficients(normal,axis):
 """Bernstein convex-hull bounds for q' and q''/2 along one Duffy axis."""
 rows=[[I(x) for x in row] for row in normal]
 sub=lambda a,b:[x-y for x,y in zip(a,b)]
 dot=lambda a,b:sum(x*y for x,y in zip(a,b))
 if axis==0:
  derivative=[sub(rows[1],rows[0]),sub(rows[1],rows[2])]
  endpoints=[(rows[0],rows[2]),(rows[1],rows[1])]
 elif axis==1:
  derivative=[sub(rows[2],rows[0]),[I(0)]*3]
  endpoints=[(rows[0],rows[1]),(rows[2],rows[1])]
 else:raise ValueError('Duffy axis')
 controls=[]
 for a,b in endpoints:
  controls.extend([2*dot(a,derivative[0]),dot(a,derivative[1])+dot(b,derivative[0]),2*dot(b,derivative[1])])
 return I(max(x.bound for x in controls)),I(max(dot(x,x).hi for x in derivative))

def bounds(a,order=3,*,cell_extension=0.):
 if type(order) is not int or order not in (3,4,5):raise ValueError('order must be3,4,5')
 if not math.isfinite(cell_extension) or not 0<=cell_extension<=1e-8:raise ValueError('bounded cell extension')
 validate(dict(a,uv=[[0.,0.]]*3))
 if len(a['uv'])!=3 or any(len(row)!=2 for row in a['uv']) or any(not math.isfinite(x) or x < -cell_extension or x>1+cell_extension for row in a['uv'] for x in row):raise ValueError('single texture cell required')
 n=2*order;normal=[[I(x) for x in row] for row in a['normal']]
 dots=[sum(x*y for x,y in zip(row,other)) for row in normal for other in normal];lo=min(x.lo for x in dots)
 if lo<=0:raise ValueError('positive Bernstein normal bound required')
 qmin=I(lo);factor=I(F(math.factorial(order)**4,(2*order+1)*math.factorial(2*order)**2))
 result=[[I(0)]*3 for _ in range(2)]
 for axis in (0,1):
  def affine(key):
   rows=[[I(x) for x in row] for row in a[key]];out=[]
   for k in range(len(rows[0])):
    differences=[rows[1][k]-rows[0][k],rows[1][k]-rows[2][k]] if axis==0 else [rows[2][k]-rows[0][k]]
    v=[I(max(row[k].bound for row in rows)),I(max(x.bound for x in differences))]+[I(0)]*(n-1)
    out.append(M(v,[I(1),I(2)]+[I(0)]*(n-1)))
   return out
  x,y,z=affine('normal');u,v=affine('uv');q=x*x+y*y+z*z
  q.v[1],q.v[2]=norm_axis_coefficients(a['normal'],axis)
  h=inverse_majorant(qmin,q,F(1,2),n);inv=inverse_majorant(qmin,q,F(1),n)
  dq=M(q.d)
  h.d=(inverse_majorant(qmin,q,F(3,2),n)*dq*I(F(1,2))).v
  inv.d=(inverse_majorant(qmin,q,F(2),n)*dq).v
  jac=constant(a['det'],n,1)*M([I(1),I(1 if axis==0 else 0)]+[I(0)]*(n-1))
  tex=[[I(x) for x in row] for row in a['tex']]
  for k in range(3):
   albedo=constant(tex[0][k].bound,n,1)+u*constant((tex[1][k]-tex[0][k]).bound,n,2)+v*constant((tex[2][k]-tex[0][k]).bound,n,2)+u*v*constant((tex[3][k]-tex[1][k]-tex[2][k]+tex[0][k]).bound,n,4)
   sh=[constant(I(row[k]).bound,n,1) for row in a['sh']]
   c=constant((I(a['sh'][0][k])-I(a['sh'][6][k])).bound,n,2)
   lin=x*sh[1]+y*sh[2]+z*sh[3]
   quad=x*y*sh[4]+y*z*sh[5]+3*z*z*sh[6]+x*z*sh[7]+(x*x+y*y)*sh[8]
   f=jac*albedo*(c+lin*h+quad*inv)
   result[0][k]=result[0][k]+factor*f.v[n]
   result[1][k]=result[1][k]+factor*f.d[n]
 return dict(value=result[0],unit_direction=result[1],scope='truncation only; all 55 corner-field direction components <=1; excludes floating arithmetic and upstream construction')
