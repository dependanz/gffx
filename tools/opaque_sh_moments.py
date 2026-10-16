"""Development-only triangle moments; no production admission changes.

Retains one bilinear texture cell per supplied triangle. Only reciprocal normal
length is approximated. Degree <=8. q0 is held fixed for first differentials.
Tail certificates bound exact polynomial truncation, NOT binary64 roundoff;
independent reference comparison is required for observed candidate accuracy.
"""
import math
from fractions import Fraction as F
from tools.opaque_shading_intervals import I
from tools.opaque_sh_texture_reference import validate

class D:
    __slots__=('v','d')
    def __init__(self,v=0.,d=0.):self.v,self.d=v,d
    def __add__(self,b):
        b=b if isinstance(b,D) else D(b);return D(self.v+b.v,self.d+b.d)
    __radd__=__add__
    def __neg__(self):return D(-self.v,-self.d)
    def __sub__(self,b):return self+-asdual(b)
    def __rsub__(self,b):return asdual(b)+-self
    def __mul__(self,b):
        b=asdual(b);return D(self.v*b.v,self.d*b.v+self.v*b.d)
    __rmul__=__mul__

def asdual(x):return x if isinstance(x,D) else D(x)

class P:
    def __init__(self,x=0.,terms=None):self.terms={k:v for k,v in ({(0,0):asdual(x)} if terms is None else terms).items() if v.v!=0 or v.d!=0}
    def __add__(self,b):
        b=b if isinstance(b,P) else P(b);r=dict(self.terms)
        for k,x in b.terms.items():r[k]=r.get(k,D())+x
        return P(terms=r)
    __radd__=__add__
    def __neg__(self):return P(terms={k:-x for k,x in self.terms.items()})
    def __sub__(self,b):return self+-(b if isinstance(b,P) else P(b))
    def __rsub__(self,b):return -self+b
    def __mul__(self,b):
        b=b if isinstance(b,P) else P(b);r={}
        for (i,j),x in self.terms.items():
            for (k,l),y in b.terms.items():
                key=i+k,j+l;r[key]=r.get(key,D())+x*y
        return P(terms=r)
    __rmul__=__mul__
    def integral(self):
        return sum((x*float(F(math.factorial(i)*math.factorial(j),math.factorial(i+j+2))) for (i,j),x in self.terms.items()),D())

def configuration(a,degree,q0=None,cell_extension=0.):
    if not math.isfinite(cell_extension) or not 0<=cell_extension<=1e-8:raise ValueError("bounded cell extension required")
    validate(dict(a,uv=[[0.,0.]]*3))
    if len(a["uv"])!=3 or any(len(r)!=2 for r in a["uv"]) or any(not math.isfinite(x) or x < -cell_extension or x > 1+cell_extension for r in a["uv"] for x in r):raise ValueError("texture cell extension exceeded")
    if type(degree) is not int or not 0<=degree<=8:raise ValueError('degree must be integer 0..8')
    n=[[I(x) for x in row] for row in a['normal']]
    dots=[sum(x*y for x,y in zip(r,s)) for r in n for s in n]
    q=I(min(x.lo for x in dots),max(x.hi for x in dots))
    if q.lo<=0:raise ValueError('positive normal Bernstein bound required')
    if q0 is None:q0=float((I(q.lo)/2+I(q.hi)/2).hi)
    if not math.isfinite(q0) or q0<=0:raise ValueError('positive finite q0 required')
    delta=I((q/I(q0)-1).bound)
    if delta.hi>=1:raise ValueError('normal series must converge uniformly')
    return q0,delta

def evaluate(a,degree=4,*,cell_extension=0.):
    q0,delta=configuration(a,degree,cell_extension=cell_extension)
    def data(key):
        ds=a.get('d_'+key,[[0.]*len(row) for row in a[key]])
        return [[D(x,dx) for x,dx in zip(row,drow)] for row,drow in zip(a[key],ds)]
    u=P(terms={(1,0):D(1)});v=P(terms={(0,1):D(1)})
    def affine(rows):return [P(rows[0][k])+u*(rows[1][k]-rows[0][k])+v*(rows[2][k]-rows[0][k]) for k in range(len(rows[0]))]
    nx,ny,nz=affine(data('normal'));s,t=affine(data('uv'));tex=data('tex');sh=data('sh')
    q=nx*nx+ny*ny+nz*nz;z=q*(1/q0)-1
    rh=P(0);ri=P(0);power=P(1);coef=F(1)
    for k in range(degree+1):
        rh=rh+power*float(coef);ri=ri+power*((-1)**k)
        if k<degree:power=power*z;coef*=F(-2*k-1,2*k+2)
    rh=rh*(1/math.sqrt(q0));ri=ri*(1/q0)
    out=[];counts=[]
    for k in range(3):
        albedo=P(tex[0][k])+s*(tex[1][k]-tex[0][k])+t*(tex[2][k]-tex[0][k])+s*t*(tex[3][k]-tex[1][k]-tex[2][k]+tex[0][k])
        lin=nx*sh[1][k]+ny*sh[2][k]+nz*sh[3][k]
        quad=nx*ny*sh[4][k]+ny*nz*sh[5][k]+3*nz*nz*sh[6][k]+nx*nz*sh[7][k]+(nx*nx-ny*ny)*sh[8][k]
        integrand=albedo*(P(sh[0][k]-sh[6][k])+lin*rh+quad*ri)
        out.append(integrand.integral()*D(a['det'],a.get('d_det',0.)));counts.append(len(integrand.terms))
    if not all(math.isfinite(x.v) and math.isfinite(x.d) for x in out):raise ValueError('nonfinite polynomial result')
    return dict(value=[x.v for x in out],tangent=[x.d for x in out],q0=q0,delta=float(delta.hi),degree=degree,coefficient_counts=counts)

def tail_bound(a,degree=4,q0=None,*,cell_extension=0.):
    q0,delta=configuration(a,degree,q0,cell_extension);center=I(q0)
    tails=[delta**(degree+1)/(1-delta)/center.sqrt(),delta**(degree+1)/(1-delta)/center]
    slope=delta**degree*((degree+1)-degree*delta)/(1-delta)**2
    slopes=[slope/(center*center.sqrt()),slope/(center*center)]
    def data(key,hull=False):
        ds=a.get('d_'+key,[[0.]*len(row) for row in a[key]])
        rows=[[D(I(x),I(dx)) for x,dx in zip(row,drow)] for row,drow in zip(a[key],ds)]
        if not hull:return rows
        return [D(I(min(row[k].v.lo for row in rows),max(row[k].v.hi for row in rows)),I(min(row[k].d.lo for row in rows),max(row[k].d.hi for row in rows))) for k in range(len(rows[0]))]
    nx,ny,nz=data('normal',True);s,t=data('uv',True);tex=data('tex');sh=data('sh');q=nx*nx+ny*ny+nz*nz
    jac=D(I(a['det']),I(a.get('d_det',0)))
    values=[];tangents=[]
    def mag(x):return I(x.bound)
    for k in range(3):
        albedo=tex[0][k]+s*(tex[1][k]-tex[0][k])+t*(tex[2][k]-tex[0][k])+s*t*(tex[3][k]-tex[1][k]-tex[2][k]+tex[0][k])
        lin=nx*sh[1][k]+ny*sh[2][k]+nz*sh[3][k]
        quad=nx*ny*sh[4][k]+ny*nz*sh[5][k]+3*nz*nz*sh[6][k]+nx*nz*sh[7][k]+(nx*nx-ny*ny)*sh[8][k]
        e=mag(lin.v)*tails[0]+mag(quad.v)*tails[1]
        de=mag(lin.d)*tails[0]+mag(quad.d)*tails[1]+mag(q.d)*(mag(lin.v)*slopes[0]+mag(quad.v)*slopes[1])
        values.append(mag(jac.v)*mag(albedo.v)*e/2)
        tangents.append((mag(jac.d)*mag(albedo.v)*e+mag(jac.v)*(mag(albedo.d)*e+mag(albedo.v)*de))/2)
    return dict(value=values,tangent=tangents,scope='exact polynomial truncation only; excludes binary64 arithmetic and upstream domain construction')
