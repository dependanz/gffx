"""Fixed-field first spatial cosine slopes with exact polynomial cancellation.

For q=A/sqrt(B), q_d=(B*A_d-A*B_d/2)/B^(3/2).
Expand each numerator over rationals before outward evaluation. Compose with
both physical coordinate derivatives. Only coefficient(1,0) is refined;
higher and mixed coefficients remain owned by the original full Jet.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_angle_light import _clean,_add,_mul,_diff,PreparedIntervalPolynomial

class FixedCosineSlopes:
    def __init__(self,coefficients,direction):
        def field(rows):
            if len(rows)!=3 or any(len(row)!=3 for row in rows):raise ValueError('three fixed affine components required')
            if any(any(v[k] for k in range(1,len(v.c))) for row in rows for v in row):raise ValueError('fixed fields only; intrinsic motion is not admitted')
            return [_clean({(i,j,0):v[0] for (i,j),v in zip(((0,0),(1,0),(1,1)),row)}) for row in rows]
        if len(direction)!=3:raise ValueError('three light coordinates required')
        n,v=field(coefficients['n']),field(coefficients['v']);light=list(map(Q,direction))
        ll=sum(x*x for x in light)
        if ll<=0:raise ValueError('positive light norm required')
        def dot(a,b):
            out={}
            for x,y in zip(a,b):out=_add(out,_mul(x,y))
            return out
        N,V,A=dot(n,n),dot(v,v),dot(n,v);C={}
        for row,w in zip(n,light):C=_add(C,row,w)
        self.roles=[]
        for A,B in ((A,_mul(N,V)),(C,_clean({k:ll*x for k,x in N.items()}))):
            numerators=[_add(_mul(B,_diff(A,axis)),_mul(A,_diff(B,axis)),Q(-1,2)) for axis in (0,1)]
            self.roles.append(tuple(PreparedIntervalPolynomial(p) for p in (B,*numerators)))
    def bound(self,role,t,s):
        if type(role)!=int or role not in (0,1) or not isinstance(t,Jet) or not isinstance(s,Jet) or t.order!=s.order or t.order<1:raise ValueError('role0or1 and matching spatial jets required')
        B,dt,ds=(p(t[0,0],s[0,0]) for p in self.roles[role])
        if B.lo<=0:raise ValueError('positive cosine norm product required')
        return (dt*t[1,0]+ds*s[1,0])/(B**3).sqrt()


def validate_partition(parts):
    if type(parts)!=tuple or len(parts)!=2 or any(type(n)!=int or n not in (1,2,4) for n in parts) or parts[0]*parts[1]>8:raise ValueError('at most eight explicit dyadic slope boxes required')


def subdivided_slopes(field,t,s,*,parts=(2,4),checkpoint=lambda:None):
    """Hull first spatial derivatives over an exhaustive at-most-eight-box cover.

    First composition coefficients depend only on input values and first
    coefficients. Keep BOTH input derivatives, including signed/tiny values.
    This refines neither mixed nor higher coefficients and performs no sampled
    sign inference. Every field call encloses its complete subrectangle.
    """
    validate_partition(parts)
    if not isinstance(t,Jet) or not isinstance(s,Jet) or t.order!=s.order or t.order<1:raise ValueError('matching spatial jets required')
    edges=[]
    for q,n in zip((t,s),parts):
        a,b=Q(q[0,0].lo),Q(q[0,0].hi)
        edges.append([I(a+(b-a)*Q(j,n)) for j in range(n+1)])
    hull=None
    for i in range(parts[0]):
        for j in range(parts[1]):
            checkpoint();args=[]
            for q,row,k in zip((t,s),edges,(i,j)):
                args.append(Jet(order=1,coeff={(0,0):I(row[k].lo,row[k+1].hi),(1,0):q[1,0]}))
            values=[q[1,0] for q in field(*args)];checkpoint()
            if hull is None:hull=values
            else:
                if len(values)!=len(hull):raise ValueError('matching slope channels required')
                hull=[I(min(a.lo,b.lo),max(a.hi,b.hi)) for a,b in zip(hull,values)]
    return hull
