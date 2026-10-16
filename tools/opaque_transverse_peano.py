"""Independent fixed-geometry boundary integration with certified coordinate changes.

QxQy-IxIy = Qy(Ex)+Ix(Ey): condition Ex on each actual outer Gauss node,
then integrate Ey's monotone crossing contribution by changing x to its root y.
No sampled convergence, unbounded subdivision or moving-parameter claim.
"""
from fractions import Fraction as Q
import math
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_peano import gauss5,kernel_bounds,kernel_local_bounds,kernel_local_ranges,kernel_local_range,kernel_local_mass,crossing_location


def kernel_integral(a,b,order):
    a,b=map(I,(a,b))
    if type(order)!=int or not 1<=order<=3 or a.lo<0 or b.hi>1 or a.hi>1 or b.lo<0 or a.lo>b.hi:raise ValueError('ordered normalized endpoints and kernel order1..3 required')
    nodes,weights=gauss5()
    def primitive(t):
        return (1-t)**(order+1)/math.factorial(order+1)-sum((w*I(max(I(0).lo,(x-t).lo),max(I(0).hi,(x-t).hi))**order/math.factorial(order) for x,w in zip(nodes,weights)),I(0))
    def enclosed(t):
        natural=primitive(t)
        if t.lo==t.hi:return natural
        # P is absolutely continuous and P'=K almost everywhere,including
        # across K1 jumps. The interval secant formula is therefore valid.
        anchor=I(I((Q(t.lo)+Q(t.hi))/2).lo)
        derivative=kernel_local_range(t,order)
        centered=primitive(anchor)+derivative*(t-anchor)
        lo=max(natural.lo,centered.lo);hi=min(natural.hi,centered.hi)
        if derivative.lo>=0:
            lo=max(lo,primitive(I(t.lo)).lo);hi=min(hi,primitive(I(t.hi)).hi)
        elif derivative.hi<=0:
            lo=max(lo,primitive(I(t.hi)).lo);hi=min(hi,primitive(I(t.lo)).hi)
        if lo>hi:raise ValueError('disjoint primitive enclosures')
        return I(lo,hi)
    return enclosed(b)-enclosed(a)


def weighted_kernel_parts(coefficient,a,b,order):
    """Integrate C(t)K(t),given a bound on C over the entire possible support.

    A constant anchor integrates by the exact piecewise primitive. The remainder
    is bounded by coefficient variation times a certified absolute kernel mass.
    Endpoint uncertainty and absence via coincident clipped endpoints survive.
    """
    coefficient,a,b=map(I,(coefficient,a,b));mass=kernel_integral(a,b,order)
    anchor=I(I((Q(coefficient.lo)+Q(coefficient.hi))/2).lo)
    variation=(coefficient-anchor).bound;support=I(a.lo,b.hi)
    absolute=kernel_local_mass(support,order)
    error=I(variation)*absolute
    anchor_mass=anchor*mass
    return dict(value=anchor_mass+I(error.hi.copy_negate(),error.hi),coefficient=coefficient,roots=[a,b],anchor=anchor,mass=mass,anchor_mass=anchor_mass,variation_radius=error,absolute_mass=absolute)


def weighted_kernel_integral(coefficient,a,b,order):
    return weighted_kernel_parts(coefficient,a,b,order)['value']


def clamped_crossing(domain,left,right,middle,derivative,knot):
    """Enclose the monotone root clipped to the physical integration domain.

    Endpoint comparisons retain clipped roots even when there is no interior
    crossing. All six supplied intervals must cover the same true function.
    """
    domain,left,right,middle,derivative,knot=map(I,(domain,left,right,middle,derivative,knot))
    if derivative.lo<=0<=derivative.hi:raise ValueError('strict monotonicity required')
    root=crossing_location(domain,middle,derivative,knot);possibilities=[] if root is None else [root]
    low_possible=left.hi>=knot.lo if derivative.lo>0 else left.lo<=knot.hi
    high_possible=right.lo<=knot.hi if derivative.lo>0 else right.hi>=knot.lo
    if low_possible:possibilities.append(I(domain.lo))
    if high_possible:possibilities.append(I(domain.hi))
    if not possibilities:raise ValueError('inconsistent root enclosure')
    return I(min(v.lo for v in possibilities),max(v.hi for v in possibilities))


def _error(row,domain,mass,integrated):
    h=I(domain.hi)-I(domain.lo);kernels,l1=kernel_bounds()
    third=I(row['third']);jumps=list(map(I,row['jumps']))
    if len(jumps)!=3 or any(v.lo<0 for v in [third,*jumps]):raise ValueError('nonnegative derivative and jump bounds required')
    ordinary=third*h**4*l1[2]*mass
    if not integrated and 'third_range' in row:
        # At each fixed outer Gauss node the inner K3 still has zero moment.
        # One constant anchor spans the FULL inner interval; jumps stay below.
        from tools.opaque_smooth_peano import ordinary_bounds
        ordinary=ordinary_bounds(row,domain,mass)['effective']
    if integrated and 'ordinary_integral' in row:
        bound=I(row['ordinary_integral'])
        if bound.lo<0:raise ValueError('nonnegative integrated ordinary bound required')
        ordinary=I(0,min(ordinary.hi,bound.hi))
    ordinary_center=I(0)
    if (not integrated and 'ordinary_signed' in row) or (integrated and 'ordinary_integral_signed' in row):
        # Integrated evidence already includes both physical lengths.
        supplied=I(row['ordinary_integral_signed']) if integrated else I(row['ordinary_signed'])*mass
        lo=max(supplied.lo,ordinary.hi.copy_negate());hi=min(supplied.hi,ordinary.hi)
        if lo>hi:raise ValueError('disjoint signed and conservative ordinary bounds')
        enclosure=I(lo,hi)
        ordinary_center=I(I((Q(lo)+Q(hi))/2).lo)
        ordinary=I(0,(enclosure-ordinary_center).bound)
    global_error=sum((v*h**(j+1)*kernels[j] for j,v in enumerate(jumps)),I(0))*mass
    if integrated and 'global_integral' in row:
        supplied=I(row['global_integral'])
        if supplied.lo<0:raise ValueError('nonnegative integrated jump bound required')
        global_error=I(0,min(global_error.hi,supplied.hi))
    global_center=I(0)
    if not integrated and 'global_signed' in row:
        supplied=I(row['global_signed'])*mass
        lo=max(supplied.lo,global_error.hi.copy_negate());hi=min(supplied.hi,global_error.hi)
        if lo>hi:raise ValueError('disjoint aggregate signed jump bounds')
        enclosure=I(lo,hi);global_center=I(I((Q(lo)+Q(hi))/2).lo)
        global_error=I(0,(enclosure-global_center).bound)
    local_error=I(0);signed=I(0);witnesses=row.get('localized_jumps',[])
    if not isinstance(witnesses,list) or len(witnesses)>16:raise ValueError('at most16localized witnesses')
    for witness in witnesses:
        position=I(witness['position']);values=list(map(I,witness['jumps']))
        if position.lo<domain.lo or position.hi>domain.hi or len(values)!=3 or any(v.lo<0 for v in values):raise ValueError('invalid localized witness')
        normalized=(position-I(domain.lo))/h;normalized=I(max(I(0).lo,normalized.lo),min(I(1).hi,normalized.hi))
        if 'signed_jumps' in witness:
            signs=list(map(I,witness['signed_jumps']))
            if witness.get('max_crossings')!=1 or type(witness.get('required')) is not bool or len(signs)!=3 or any(v.bound>a.hi for v,a in zip(signs,values)):raise ValueError('single signed crossing witness required')
            local=kernel_local_ranges(normalized);term=sum((v*h**(j+1)*local[j] for j,v in enumerate(signs)),I(0))
            if not witness['required']:term=I(min(I(0).lo,term.lo),max(I(0).hi,term.hi))
            term=term*mass
            if integrated and 'transverse_integral' in witness:
                other=I(witness['transverse_integral']);lo=max(term.lo,other.lo);hi=min(term.hi,other.hi)
                if lo>hi:raise ValueError('disjoint transverse and uniform witness bounds')
                term=I(lo,hi)
            signed=signed+term
        else:
            if 'transverse_integral' in witness:raise ValueError('integrated witness needs certified signed crossing')
            local=kernel_local_bounds(normalized);local_error=local_error+sum((v*h**(j+1)*local[j] for j,v in enumerate(values)),I(0))*mass
    symmetric=ordinary+global_error+local_error
    return signed+ordinary_center+global_center+I(symmetric.hi.copy_negate(),symmetric.hi),dict(ordinary=ordinary,global_jumps=global_error,localized_jumps=local_error+I(signed.width)/2,signed_correction=signed+ordinary_center+global_center)


def conditioned_integral(field,box,axis_bounds,integrated_bounds,*,checkpoint=lambda:None,error_components=None,error_breakdown=None,integrated_axis=1):
    if type(integrated_axis)!=int or integrated_axis not in (0,1):raise ValueError('integrated axis0or1 required')
    conditioned_axis=1-integrated_axis
    if len(box)!=2:raise ValueError('two-dimensional box required')
    box=tuple(map(I,box));h=[I(v.hi)-I(v.lo) for v in box]
    if any(v.lo<=0 for v in h):raise ValueError('positive box widths required')
    nodes,weights=gauss5();accum=None
    for x,w in zip(nodes,weights):
        checkpoint()
        for y,v in zip(nodes,weights):
            values=field(Jet(I(box[0].lo)+h[0]*x),Jet(I(box[1].lo)+h[1]*y))
            if not isinstance(values,(list,tuple)) or not 1<=len(values)<=4 or any(not isinstance(q,Jet) or q.order or not q[0,1].zero for q in values):raise ValueError('one to four fixed-geometry jet channels required')
            if accum is None:accum=[I(0)]*len(values)
            if len(accum)!=len(values):raise ValueError('channel count changed')
            accum=[a+q[0,0]*w*v*h[0]*h[1] for a,q in zip(accum,values)]
    errors=[[I(0)]*len(accum) for _ in (0,1)];parts=[[dict(ordinary=I(0),global_jumps=I(0),localized_jumps=I(0),signed_correction=I(0)) for _ in accum] for _ in (0,1)]
    for y,w in zip(nodes,weights):
        checkpoint();line=list(box);line[integrated_axis]=I(box[integrated_axis].lo)+h[integrated_axis]*y;rows=axis_bounds(tuple(line),conditioned_axis)
        if len(rows)!=len(accum):raise ValueError('axis channel count changed')
        for k,row in enumerate(rows):
            value,detail=_error(row,box[conditioned_axis],h[integrated_axis]*w,False);errors[conditioned_axis][k]=errors[conditioned_axis][k]+value
            for key,v in detail.items():parts[conditioned_axis][k][key]=parts[conditioned_axis][k][key]+v
    checkpoint();rows=integrated_bounds(box,integrated_axis)
    if len(rows)!=len(accum):raise ValueError('integrated channel count changed')
    for k,row in enumerate(rows):errors[integrated_axis][k],parts[integrated_axis][k]=_error(row,box[integrated_axis],h[conditioned_axis],True)
    if error_components:error_components(tuple(tuple(I(0,(I(v.width)/2).hi) for v in row) for row in errors))
    if error_breakdown:error_breakdown(tuple(tuple(row) for row in parts))
    return [value-errors[0][k]-errors[1][k] for k,value in enumerate(accum)]


def adaptive_conditioned_integral(field,box,axis_bounds,integrated_bounds,*,checkpoint=lambda:None,error_components=None,error_breakdown=None,preferred_axis=1,selection=None):
    """Choose a tensor identity by its integrated uncertainty, never acceptance.

    Evaluate both continuous contributions once, then reuse the selected rows.
    The chosen identity still evaluates all five opposite-axis bounds and its
    full quadrature. No minimum of incomplete bounds becomes a certificate.
    Unsupported alternatives may fall back; cancellation always propagates.
    """
    if type(preferred_axis)!=int or preferred_axis not in (0,1):raise ValueError('preferred axis0or1 required')
    if len(box)!=2:raise ValueError('two-dimensional box required')
    box=tuple(map(I,box));h=[I(v.hi)-I(v.lo) for v in box]
    if any(v.lo<=0 for v in h):raise ValueError('positive box widths required')
    candidates=[];failure=None
    for axis in (preferred_axis,1-preferred_axis):
        checkpoint()
        try:
            rows=integrated_bounds(box,axis)
            if not isinstance(rows,(tuple,list)) or not 1<=len(rows)<=4:raise ValueError('one to four integrated channels required')
            score=max(_error(row,box[axis],h[1-axis],True)[0].width for row in rows)
            candidates.append((score,axis,rows))
        except Unresolved:raise
        except ValueError as error:failure=error
    if not candidates:raise ValueError('both tensor orders unavailable') from failure
    _,axis,rows=min(candidates,key=lambda v:v[0])  # Stable tie keeps preferred order.
    if selection:selection(axis)
    def reuse(b,a):
        if a!=axis or any((x.lo,x.hi)!=(y.lo,y.hi) for x,y in zip(b,box)):raise ValueError('integrated rows belong to one box and axis')
        return rows
    return conditioned_integral(field,box,axis_bounds,reuse,checkpoint=checkpoint,error_components=error_components,error_breakdown=error_breakdown,integrated_axis=axis)
