"""Gauss5 correction proof with all fourth- or sixth-order jump terms.

Same 25 samples, uniform positive transverse operators and one shared clock.
The caller supplies ordinary derivatives on all smooth pieces and all crossings.
A constant ordinary fourth derivative can be removed because Gauss5 is exact
on quartics. This does not remove any material jump or parameter contribution.
"""
from fractions import Fraction as Q
from functools import lru_cache
import sys
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano import gauss5,kernel_local_range
from tools.opaque_high_peano import _mass,kernel_range

@lru_cache(maxsize=128)
def _fourth_range(lo,hi,context,order=4):
    v=kernel_range(I(lo,hi),order);return v.lo,v.hi

def integral(field,box,axis_bounds,*,checkpoint=lambda:None,error_components=None,conditioned=False,integrated_axis=1,direct_conditioned=False,condition_parts=2,line_bounds=None):
    import tools.opaque_shading_intervals as scalar
    if type(integrated_axis)!=int or integrated_axis not in (0,1):raise ValueError('explicit integrated axis0or1 required')
    if type(conditioned)!=bool:raise ValueError('explicit conditioning switch required')
    if type(direct_conditioned)!=bool or direct_conditioned and not conditioned:raise ValueError('direct proof requires conditioned identity')
    if type(condition_parts)!=int or condition_parts not in (2,4):raise ValueError('two or four complete transverse proof domains required')
    if line_bounds is not None and not callable(line_bounds):raise ValueError('callable quadrature-line proof required')
    if len(box)!=2:raise ValueError('two-dimensional correction required')
    box=tuple(map(I,box));h=[I(q.hi)-I(q.lo) for q in box]
    if any(v.lo<=0 for v in h):raise ValueError('positive box required')
    nodes,weights=gauss5();data=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights));mass=I(_mass(4,data));accum=None
    for x,w in zip(nodes,weights):
        for y,z in zip(nodes,weights):
            checkpoint();values=field(Jet(I(box[0].lo)+h[0]*x),Jet(I(box[1].lo)+h[1]*y))
            if not isinstance(values,(list,tuple)) or not 1<=len(values)<=4 or any(not isinstance(v,Jet) or v.order or not v[0,1].zero for v in values):raise ValueError('fixed value and ordinary tangent channels required')
            if accum is None:accum=[I(0)]*len(values)
            if len(values)!=len(accum):raise ValueError('matching channels required')
            accum=[a+v[0,0]*w*z*h[0]*h[1] for a,v in zip(accum,values)]
    errors=[I(0)]*len(accum);directions=[]
    flint=sys.modules.get('flint');context=(scalar.PRECISION,I.__init__,None if flint is None else (flint.ctx.prec,flint.ctx.cap))
    def enclose(row,domain,transverse_mass):
        length=I(domain.hi)-I(domain.lo)
        derivative=I(row['derivative'])
        order=row.get('order')
        if type(order)!=int or order not in (4,6) or derivative.lo<0:raise ValueError('fourth/sixth-order nonnegative evidence required')
        if 'derivative_range' in row:
            signed=I(row['derivative_range'])
            if signed.bound>derivative.hi:raise ValueError('signed fourth derivative exceeds absolute bound')
            anchor=I(I((Q(signed.lo)+Q(signed.hi))/2).lo);derivative=I(0,min(derivative.hi,(signed-anchor).bound))
        radius=derivative*length**(order+1)*(mass if order==4 else I(_mass(order,data)));error=I(-radius.hi,radius.hi)
        witnesses=row['jumps']
        if not isinstance(witnesses,list) or len(witnesses)>16:raise ValueError('bounded complete jump witnesses required')
        for witness in witnesses:
            checkpoint();position=I(witness['position']);values=list(map(I,witness['values']))
            if not domain.lo<=position.lo<=position.hi<=domain.hi or len(values)!=order or type(witness['required'])!=bool:raise ValueError('complete signed crossing evidence required')
            normalized=(position-I(domain.lo))/length;normalized=I(max(I(0).lo,normalized.lo),min(I(1).hi,normalized.hi));term=I(0)
            for j,v in enumerate(values):
                if v.zero:continue
                kernel=kernel_local_range(normalized,j+1) if j<3 else I(*_fourth_range(normalized.lo,normalized.hi,context,j+1))
                term=term+v*length**(j+1)*kernel
            if not witness['required']:term=I(min(I(0).lo,term.lo),max(I(0).hi,term.hi))
            supplied=witness.get('transverse_integral');measure=witness.get('transverse_measure')
            # A continuous proof strip has this exact measure. Quadrature-line
            # weights are not domain measures and must never consume this proof.
            if supplied is not None and measure is not None and measure.lo==transverse_mass.lo and measure.hi==transverse_mass.hi:
                original=term*transverse_mass;lo=max(original.lo,supplied.lo);hi=min(original.hi,supplied.hi)
                if lo>hi:raise ValueError('disjoint transverse jump evidence')
                term=I(lo,hi)/transverse_mass
            error=error+term
        return error*transverse_mass
    def bounded(jobs):
        errors=[I(0)]*len(accum);directional_errors=[[I(0)]*len(accum) for _ in (0,1)]
        for axis,proof_box,transverse_mass,continuous in jobs:
            checkpoint();rows=(axis_bounds if continuous or line_bounds is None else line_bounds)(proof_box,axis)
            if len(rows)!=len(accum):raise ValueError('matching bound channels required')
            for channel,row in enumerate(rows):
                error=enclose(row,box[axis],transverse_mass)
                errors[channel]=errors[channel]+error
                directional_errors[axis][channel]=directional_errors[axis][channel]+error
        directions=[[I(0,(I(v.width)/2).hi) for v in row] for row in directional_errors]
        return [v-e for v,e in zip(accum,errors)],directions
    # Every selected positive tensor identity is independently sufficient.
    # An explicit direct choice avoids two preliminary domains; it neither
    # skips samples nor borrows an independent error/time/work budget.
    # A continuous crossing integral cannot bound a discrete outer Gauss
    # sum. With line-specific evidence use the sufficient tensor identity
    # directly; never intersect it with two continuous-axis substitutes.
    if conditioned and line_bounds is not None:direct_conditioned=True
    ordinary=None
    if not direct_conditioned:
        ordinary,directions=bounded([(axis,box,h[1-axis],True) for axis in (0,1)])
    answer=ordinary
    if direct_conditioned or conditioned and max(v.width for v in ordinary)>I('1e-14').lo:
        jobs=[];inner=1-integrated_axis
        for node,weight in zip(nodes,weights):
            line=list(box);line[integrated_axis]=I(box[integrated_axis].lo)+h[integrated_axis]*node
            jobs.append((inner,tuple(line),h[integrated_axis]*weight,False))
        a,b=Q(box[inner].lo),Q(box[inner].hi)
        edges=[a+(b-a)*Q(i,condition_parts) for i in range(condition_parts+1)]
        for lower,upper in zip(edges,edges[1:]):
            strip=list(box);strip[inner]=I(I(lower).lo,I(upper).hi)
            jobs.append((integrated_axis,tuple(strip),I(upper-lower),True))
        refined,directions=bounded(jobs)
        answer=refined if ordinary is None else [I(max(a.lo,b.lo),min(a.hi,b.hi)) for a,b in zip(ordinary,refined)]
    if error_components is not None:error_components(directions)
    return answer
