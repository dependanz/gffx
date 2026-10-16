"""Signed monotone transport of every finite-table jump through selected order.

No derivatives across knots: retain separate ordinary and all jump terms.
Clipped roots cover absent/partial/full sweeps. Unknown transverse monotonicity
keeps the conservative whole-box term. All arithmetic remains outward Decimal.
"""
import math
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano import gauss5
from tools.opaque_high_peano import kernel_range,high_integral
from tools.opaque_high_material import HighMaterial
from tools.opaque_transverse_peano import clamped_crossing

def kernel_integral(a,b,order):
    a,b=map(I,(a,b))
    if type(order)!=int or not 1<=order<=10 or min(a.lo,b.lo)<0 or max(a.hi,b.hi)>1 or a.lo>b.hi:raise ValueError('ordered normalized kernel support required')
    nodes,weights=gauss5()
    def primitive(t):return (1-t)**(order+1)/math.factorial(order+1)-sum((w*I(max(I(0).lo,(x-t).lo),max(I(0).hi,(x-t).hi))**order/math.factorial(order) for x,w in zip(nodes,weights)),I(0))
    def enclosed(t):
        natural=primitive(t)
        if not t.width:return natural
        mid=I(I((Q(t.lo)+Q(t.hi))/2).lo);center=primitive(mid)+kernel_range(t,order)*(t-mid)
        lo=max(natural.lo,center.lo);hi=min(natural.hi,center.hi)
        if lo>hi:raise ValueError('disjoint kernel primitive enclosures')
        return I(lo,hi)
    return enclosed(b)-enclosed(a)

def weighted(coefficient,a,b,order):
    coefficient,a,b=map(I,(coefficient,a,b));anchor=I(I((Q(coefficient.lo)+Q(coefficient.hi))/2).lo)
    support=I(a.lo,b.hi);error=I((coefficient-anchor).bound)*I(kernel_range(support,order).bound)*(I(b.hi)-I(a.lo))
    return anchor*kernel_integral(a,b,order)+I(error.hi.copy_negate(),error.hi)

from functools import lru_cache

@lru_cache(maxsize=128)
def _absolute_mass(lo,hi,order,data,context):
    from tools.opaque_high_peano import _power,_bernstein
    from tools.opaque_peano import _bernstein_absolute_mass
    nodes=[I(a,b) for a,b,_,_ in data];edges=[I(0),*nodes,I(1)];total=I(0)
    for segment,(left,right) in enumerate(zip(edges,edges[1:])):
        a,b=max(lo,left.lo),min(hi,right.hi)
        if a>=b:continue
        # Uncertain node slivers deliberately overlap. Nonnegative mass may
        # be counted twice, never omitted, including the discontinuous K1.
        coeff=[I(*v) for v in _power(order,data[segment:])]
        total=total+_bernstein_absolute_mass(_bernstein(I(a),I(b),coeff),I(b)-I(a))
    return total.hi


@lru_cache(maxsize=128)
def _signed_kernel_cached(alo,ahi,blo,bhi,order,data,context,evaluate):
    value=evaluate(I(alo,ahi),I(blo,bhi),order)
    return value.lo,value.hi


@lru_cache(maxsize=128)
def _kernel_range_cached(lo,hi,order,data,context,evaluate):
    value=evaluate(I(lo,hi),order)
    return value.lo,value.hi


def weighted_mass(coefficient,a,b,order):
    """Signed anchor plus certified absolute mass for varying coefficients.

    The full uncertain root support remains. Bernstein positivity and exact
    basis integrals enclose integral |K|; one algebraic bisection tightens
    mixed signs. This changes no crossing, coefficient or accepted tolerance.
    """
    import sys
    import tools.opaque_shading_intervals as scalar
    coefficient,a,b=map(I,(coefficient,a,b))
    if type(order)!=int or not 1<=order<=4 or min(a.lo,b.lo)<0 or max(a.hi,b.hi)>1 or a.lo>b.hi:raise ValueError('ordered normalized support and order1..4 required')
    if coefficient.zero or a.lo==a.hi==b.lo==b.hi:return I(0)
    anchor=I(I((Q(coefficient.lo)+Q(coefficient.hi))/2).lo);delta=coefficient-anchor
    nodes,weights=gauss5();data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights))
    flint=sys.modules.get('flint');context=(scalar.PRECISION,I.__init__,None if flint is None else (flint.ctx.prec,flint.ctx.cap))
    # Immutable endpoint tuples only; geometry and arithmetic context identify
    # the proof. No coefficient, contribution or integral acceptance is cached.
    signed=anchor*I(*_signed_kernel_cached(a.lo,a.hi,b.lo,b.hi,order,data,context,kernel_integral))
    if delta.zero:return signed
    rectangle=I(I(*_kernel_range_cached(a.lo,b.hi,order,data,context,kernel_range)).bound)*(I(b.hi)-I(a.lo))
    mass=I(min(rectangle.hi,_absolute_mass(a.lo,b.hi,order,data,context)))
    error=I(delta.bound)*mass
    return signed+I(error.hi.copy_negate(),error.hi)


class TransportMaterial(HighMaterial):
    def __init__(self,correction,*,order=6):super().__init__(correction,order=order,refine_roots=True)
    def integrated_bounds(self,box,axis):
        rows=self.axis_bounds(box,axis);c=self.c;other=1-axis;derivatives=[]
        for active in [axis,other]:
            args=[Jet(v,order=1) for v in box];args[active]=Jet.variable(box[active],order=1);derivatives.append([q[1,0] for q in c.inputs(*args)[:2]])
        h=I(box[axis].hi)-I(box[axis].lo);edgevalues=[]
        for edge in [box[other].lo,box[other].hi]:
            vals=[]
            for x in [box[axis].lo,box[axis].hi,I((Q(box[axis].lo)+Q(box[axis].hi))/2)]:
                args=[Jet(v) for v in box];args[other]=Jet(I(edge));args[axis]=Jet(I(x));vals.append(c.inputs(*args)[:2])
            edgevalues.append(vals)
        for witness in rows[1]['jumps']:
            c.checkpoint();role=witness['role'];inner,outer=[v[role] for v in derivatives]
            if any(v.lo<=0<=v.hi for v in [inner,outer]):continue
            roots=[];knot=self.index.cn[witness['knot']]
            for edge in edgevalues:
                root=clamped_crossing(box[axis],*[v[role][0,0] for v in edge],inner,knot)
                t=(root-I(box[axis].lo))/h;roots.append(I(max(I(0).lo,t.lo),min(I(1).hi,t.hi)))
            if (-outer/inner).hi<0:roots.reverse()
            ratio=inner/outer
            if ratio.hi<0:ratio=-ratio
            witness['transport_roots']=roots;witness['transport_ratio']=ratio;witness['transport_derivatives']=(inner,outer)
            witness['transverse_integral']=sum((weighted(jump*ratio*h**(j+2),*roots,j+1) for j,jump in enumerate(witness['values']) if not jump.zero),I(0))
        return rows
    def __call__(self,box,*,integrated_axis=0):
        self.c.error_breakdown=None
        def directions(rows):
            self.c.preferred_split_axis=0 if max(v.width for v in rows[0])>=max(v.width for v in rows[1]) else 1
            self.c.error_breakdown=[[dict(width=str(v.width),bounds=str(v)) for v in row] for row in rows]
        return high_integral(self.ordinary,box,getattr(self,"conditioned_bounds",self.axis_bounds),checkpoint=self.checkpoint,integrated_axis=integrated_axis,error_components=directions,integrated_bounds=self.integrated_bounds)
