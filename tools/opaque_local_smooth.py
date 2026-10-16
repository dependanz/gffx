"""Localized Gauss5 remainder with the original 25 tensor sample points.

For I-Q the order-ten Peano kernel is nonnegative: Hermite interpolation at
five Gauss nodes has remainder f^(10)(xi)*prod(x-x_i)^2/10!, and Gaussian
exactness integrates the degree-nine interpolant. Positivity of that error
functional on nonnegative tenth derivatives gives positivity of its kernel.
We integrate that kernel on a fixed dyadic partition, using outward algebra
of truncated powers. No sampled extrema or convergence estimate is used.

For [a,b] the mass multiplied by 10! equals
 ((1-a)^11-(1-b)^11)/11 - sum_i w_i*((x_i-a)_+^10-(x_i-b)_+^10).
Multiply these masses by enclosed Taylor coefficients f^(10)/10!.
Tensor identity: IxIy-QxQy=(Ix-Qx)Iy+Qx(Iy-Qy); positive Gauss weights sum to
one. Each derivative enclosure includes the entire transverse interval.
The caller certifies C^10 smoothness; material jumps remain separate.
"""
from fractions import Fraction as Q
from functools import lru_cache
import math
from tools.opaque_shading_intervals import I,Jet
import tools.opaque_shading_intervals as scalar
from tools.opaque_peano import gauss5

def validate(parts):
    if type(parts)!=tuple or len(parts)!=2 or any(type(n)!=int or n not in (1,2,4,8) for n in parts):raise ValueError('two bounded dyadic derivative partitions required')

@lru_cache(maxsize=16)
def _masses(parts,precision,factory):
    if parts==1:return ((I(Q(math.factorial(5)**4,11*math.factorial(10)**2)).lo,I(Q(math.factorial(5)**4,11*math.factorial(10)**2)).hi),)
    nodes,weights=gauss5();rows=[]
    def positive(v):return I(max(I(0).lo,v.lo),max(I(0).hi,v.hi))
    for j in range(parts):
        a,b=I(Q(j,parts)),I(Q(j+1,parts))
        mass=((1-a)**11-(1-b)**11)/11
        for x,w in zip(nodes,weights):mass=mass-w*(positive(x-a)**10-positive(x-b)**10)
        if mass.hi<0:raise ValueError('inconsistent positive Gauss remainder mass')
        mass=I(max(I(0).lo,mass.lo),mass.hi);rows.append((mass.lo,mass.hi))
    return tuple(rows)

def masses(parts):
    validate((parts,1))
    return tuple(I(*q) for q in _masses(parts,scalar.PRECISION,I.__init__))

def remainder(field,box,parts,*,checkpoint=lambda:None):
    validate(parts);box=tuple(map(I,box));width=[I(v.hi)-I(v.lo) for v in box]
    if len(box)!=2 or any(v.lo<=0 for v in width):raise ValueError('positive two-dimensional box required')
    result=[]
    for axis,n in enumerate(parts):
        accum=None
        for j,mass in enumerate(masses(n)):
            checkpoint();lo=I(box[axis].lo)+width[axis]*Q(j,n);hi=I(box[axis].lo)+width[axis]*Q(j+1,n)
            args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(I(lo.lo,hi.hi),order=10)
            rows=field(*args);checkpoint()
            if accum is None:accum=[[I(0),I(0)] for _ in rows]
            if len(rows)!=len(accum):raise ValueError('matching smooth channel count')
            for k,q in enumerate(rows):
                for tangent in (0,1):accum[k][tangent]=accum[k][tangent]+q[10,tangent]*mass
        factor=width[axis]**11*width[1-axis]
        result.append([[v*factor for v in row] for row in accum])
    return result
