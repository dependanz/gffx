"""Explicit 24-point directional tensor rule for smooth checker extensions.

Gauss3 in the first direction uses a sixth-derivative remainder. Gauss8 in the
second direction uses an outward L1 bound on its order-ten Peano kernel;
no sixteenth derivative, sampled error, or positive-kernel assumption is used.
The positive tensor weights and IxIy-QxQy decomposition add the two independent
axis error bounds. Physical Jacobians, ordinary residual and parameter jump
transport remain the caller's full separately included contributions.
The optional local2 proof encloses the transverse tenth derivative on two
halves. Symmetry and polynomial exactness give K10(1-t)=K10(t), so each
half has exactly half the true absolute kernel mass and at most half its
certified whole mass. Both derivative boxes and all mixed terms are retained.
The local4 variant uses separately certified absolute Peano kernel masses
for all four quarter-domains. No equal-quarter kernel mass is assumed.
One attempted region costs ceil(24/25)=1 sample-equivalent cell, including
failed attempts, plus unchanged wall/depth/memory limits. Correction work is
not folded into this sample count and remains timed in the same worker.
"""
import math
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I

def validate(name):
    if name not in (None,'gauss3x8','gauss3x8-local2','gauss3x8-local4','gauss3x8-local4x4','gauss3x12-local4','gauss4x12-local4'):raise ValueError('qualified directional tensor rule required')

def rule(*,points=8,radial=3,checkpoint=lambda:None):
    if type(points)!=int or points not in (8,12):raise ValueError("qualified transverse point count required")
    if points==12:from tools.opaque_gauss12 import rule as gauss8
    else:from tools.opaque_gauss8 import rule as gauss8
    from tools.opaque_high_peano import _mass
    if type(radial)!=int or radial not in (3,4):raise ValueError('qualified radial point count required')
    checkpoint();root=I(Q(3,5)).sqrt()/2
    nx=[I('.5')-root,I('.5'),I('.5')+root];wx=[I(Q(5,18)),I(Q(4,9)),I(Q(5,18))]
    if radial==4:
        small=((I(15)-2*I(30).sqrt())/35).sqrt();large=((I(15)+2*I(30).sqrt())/35).sqrt()
        roots=[-large,-small,small,large];nx=[(1+r)/2 for r in roots]
        wx=[1/((1-r*r)*((140*r**3-60*r)/8)**2) for r in roots]
    ny,wy=gauss8(checkpoint=checkpoint);data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(ny,wy))
    factors=(I(Q(math.factorial(radial)**4,(2*radial+1)*math.factorial(2*radial)**2)),I(math.factorial(10))*I(_mass(10,data)))
    checkpoint();return (nx,wx),(ny,wy),(2*radial,10),factors


from functools import lru_cache

@lru_cache(maxsize=16)
def _quarter_masses(data,context,order=10):
    from tools.opaque_high_peano import _power,_bernstein
    nodes=[I(a,b) for a,b,_,_ in data];weights=[I(c,d) for _,_,c,d in data];edges=[I(0),*nodes,I(1)];answer=[]
    for part in range(4):
        total=I(0)
        for segment,(left,right) in enumerate(zip(edges,edges[1:])):
            lo=max(I(Q(part,4)).lo,left.lo);hi=min(I(Q(part+1,4)).hi,right.hi)
            if hi<=lo:continue
            key=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes[segment:],weights[segment:]));coeff=[I(*v) for v in _power(order,key)]
            # Bernstein convexity and integral B_i^n=1/(n+1) bound the
            # absolute polynomial integral. All16disjoint subdomains remain.
            a,b=Q(lo),Q(hi)
            for j in range(16):
                l,h=I(a+(b-a)*Q(j,16)),I(a+(b-a)*Q(j+1,16))
                row=_bernstein(l,h,coeff)
                total=total+sum((I(v.bound) for v in row),I(0))*(h-l)/(order+1)
        answer.append((total*math.factorial(order)).hi)
    return tuple(answer)


def quarter_factors(*,points=8,checkpoint=lambda:None):
    if type(points)!=int or points not in (8,12):raise ValueError("qualified transverse point count required")
    if points==12:from tools.opaque_gauss12 import rule as gauss8
    else:from tools.opaque_gauss8 import rule as gauss8
    import tools.opaque_shading_intervals as scalar
    checkpoint();nodes,weights=gauss8(checkpoint=checkpoint);data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights))
    result=tuple(I(v) for v in _quarter_masses(data,(scalar.PRECISION,I.__init__)))
    checkpoint();return result


def radial_quarter_factors(*,checkpoint=lambda:None):
    """Four complete Gauss3 Peano6 masses; bounds use the existing Bernstein proof.

    Multiply by local sixth Taylor-coefficient bounds and the original h^7.
    The quadrature remains24nodes. All four disjoint proof intervals remain.
    """
    import tools.opaque_shading_intervals as scalar
    checkpoint();root=I(Q(3,5)).sqrt()/2
    nodes=[I('.5')-root,I('.5'),I('.5')+root];weights=[I(Q(5,18)),I(Q(4,9)),I(Q(5,18))]
    data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights))
    answer=tuple(I(v) for v in _quarter_masses(data,(scalar.PRECISION,I.__init__),6))
    checkpoint();return answer
