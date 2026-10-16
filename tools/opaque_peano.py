"""Independent Gauss5 residual quadrature with explicit piecewise jump errors.

For E=Q-I, K_m(t)=E[(x-t)_+^(m-1)/(m-1)!]. Piecewise Taylor's
formula gives E(f)=integral K_3 f^(3) + sum_k sum_{j=0}^2 [f^(j)]_k K_{j+1}(k).
Each interval's kernel is a polynomial; its Bernstein hull encloses its supremum.
Physical-length powers and the positive tensor-product rule preserve this bound.

Caller must enclose every ordinary third derivative and the sum of absolute
jumps on EVERY line through the box. A spatial derivative is not a moving-domain
parameter derivative. Only fixed-geometry value corrections are supported here.
Background: https://www.math.utah.edu/~feldman/teaching_files/F18/natwo.pdf §13.2.
The displayed piecewise identity follows by adding the truncated Taylor jump
terms before applying E; it is not an assumption of continuity at binary knots.
"""
from fractions import Fraction as Q
import math
from functools import lru_cache
from tools.opaque_shading_intervals import I,Jet

@lru_cache(maxsize=1)
def gauss5():
    a=(I(5)+2*I(Q(10,7)).sqrt()).sqrt()/6;b=(I(5)-2*I(Q(10,7)).sqrt()).sqrt()/6
    nodes=[I('.5')-a,I('.5')-b,I('.5'),I('.5')+b,I('.5')+a]
    u=(I(322)-13*I(70).sqrt())/1800;v=(I(322)+13*I(70).sqrt())/1800
    return nodes,[u,v,I(Q(64,225)),v,u]

@lru_cache(maxsize=1)
def kernel_bounds():
    nodes,weights=gauss5();edges=[I(0),*nodes,I(1)];sup=[];integrals=[]
    for m in (1,2,3):
        bound=I(0);integral=I(0)
        for segment,(left,right) in enumerate(zip(edges,edges[1:])):
            t=Jet(left,order=m,coeff={(0,0):left,(1,0):right-left})
            poly=sum((((t.lift(x)-t)**(m-1))*w/math.factorial(m-1) for x,w in zip(nodes[segment:],weights[segment:])),t.lift(0))-(1-t)**m/math.factorial(m)
            bernstein=[sum((poly[j,0]*Q(math.comb(k,j),math.comb(m,j)) for j in range(k+1)),I(0)) for k in range(m+1)]
            leaves=[bernstein]
            # Exact de Casteljau subdivision of a degree<=3 polynomial.
            # These96 fixed algebraic kernel pieces are not integrand samples.
            for _ in range(4):
                children=[]
                for row in leaves:
                    levels=[row]
                    while len(levels[-1])>1:levels.append([(a+b)/2 for a,b in zip(levels[-1],levels[-1][1:])])
                    children.extend([[r[0] for r in levels],[r[-1] for r in reversed(levels)]])
                leaves=children
            for row in leaves:
                upper=I(max(z.bound for z in row));bound=I(max(bound.hi,upper.hi));integral=integral+upper*(right-left)/16
        sup.append(bound);integrals.append(I(0,integral.hi))
    return tuple(sup),tuple(integrals)


def kernel_suprema():return kernel_bounds()[0]


def crossing_location(domain,center_value,derivative,knot):
    """Enclose every root of q(x,y)=k for all transverse y in the caller's box.

    Caller supplies q at the exact domain midpoint and dq/dx over the full box.
    The mean value identity x=mid+(k-q(mid,y))/q_x(xi,y) justifies intersection.
    A derivative containing zero retains the whole domain; no root is discarded.
    """
    domain,center_value,derivative,knot=map(I,(domain,center_value,derivative,knot))
    if domain.width<=0:raise ValueError('positive crossing domain required')
    if derivative.lo<=0<=derivative.hi:return domain
    mid=I((Q(domain.lo)+Q(domain.hi))/2)
    root=mid+(knot-center_value)/derivative
    lo=max(domain.lo,root.lo);hi=min(domain.hi,root.hi)
    return None if lo>hi else I(lo,hi)


def kernel_local_bounds(position):
    """Absolute hull from the identical signed hull and existing128-entry cache.

    Clipping the signed hull to the symmetric global bound then taking its
    absolute maximum gives the same min(local absolute maximum,global bound).
    No extra cache, changed polynomial arithmetic or larger work allowance.
    """
    return tuple(I(value.bound) for value in kernel_local_ranges(position))


@lru_cache(maxsize=18)
def _kernel_power_cached(order,data):
    """Immutable coefficients of K_m(t), keyed by all node/weight endpoints.

    Six Gauss segments times three orders fit eighteen entries. Constants are
    outward enclosures, not floating approximations treated as exact. Caller
    mutation cannot alter retained Decimal endpoint tuples.
    """
    if type(order)!=int or not 1<=order<=3:raise ValueError('kernel order1..3 required')
    pairs=[(I(a,b),I(c,d)) for a,b,c,d in data]
    s0=sum((w for x,w in pairs),I(0))
    if order==1:coeff=[s0-1,I(1)]
    else:
        s1=sum((w*x for x,w in pairs),I(0))
        if order==2:coeff=[s1-I('.5'),1-s0,I('-.5')]
        else:
            s2=sum((w*x*x for x,w in pairs),I(0))
            coeff=[s2/2-I(Q(1,6)),I('.5')-s1,(s0-1)/2,I(Q(1,6))]
    return tuple((v.lo,v.hi) for v in coeff)


def _kernel_segment_bernstein(lo,hi,nodes,weights,order):
    """Exact power-to-Bernstein identity through degree three, rounded outward.

    The unchanged polynomial is sum w*(x-t)^(m-1)/(m-1)!-(1-t)^m/m!.
    Reuse only its constant power coefficients; every requested domain still
    receives its own full Bernstein hull, including both sides of Gauss jumps.
    """
    key=tuple((x.lo,x.hi,w.lo,w.hi) for x,w in zip(nodes,weights))
    c=list(map(lambda v:I(*v),_kernel_power_cached(order,key)))
    if order==1:return [c[0]+c[1]*lo,c[0]+c[1]*hi]
    if order==2:return [c[0]+lo*(c[1]+lo*c[2]),c[0]+c[1]*(lo+hi)/2+c[2]*lo*hi,c[0]+hi*(c[1]+hi*c[2])]
    ll=lo*lo;hh=hi*hi;lh=lo*hi
    return [c[0]+lo*(c[1]+lo*(c[2]+lo*c[3])),
            c[0]+c[1]*(2*lo+hi)/3+c[2]*(ll+2*lh)/3+c[3]*ll*hi,
            c[0]+c[1]*(lo+2*hi)/3+c[2]*(2*lh+hh)/3+c[3]*lo*hh,
            c[0]+hi*(c[1]+hi*(c[2]+hi*c[3]))]


def _kernel_local_ranges_uncached(position,order=None):
    """Signed K1,K2,K3 enclosures on every overlapping Gauss segment.

    K1 can jump at a node: include both polynomial pieces, never interpolate
    across it. Intersection with a valid global absolute bound stays outward.
    """
    position=I(position)
    if position.lo<0 or position.hi>1:raise ValueError('normalized kernel domain must lie in[0,1]')
    if order is not None and (type(order)!=int or not 1<=order<=3):raise ValueError('kernel order1..3 required')
    nodes,weights=gauss5();edges=[I(0),*nodes,I(1)];result=[]
    for m in ((1,2,3) if order is None else (order,)):
        global_bound=kernel_suprema()[m-1]
        hull=[]
        for segment,(left,right) in enumerate(zip(edges,edges[1:])):
            lo=max(position.lo,left.lo);hi=min(position.hi,right.hi)
            if lo>hi:continue
            hull.extend(_kernel_segment_bernstein(I(lo),I(hi),nodes[segment:],weights[segment:],m))
        result.append(I(max(min(v.lo for v in hull),global_bound.hi.copy_negate()),min(max(v.hi for v in hull),global_bound.hi)))
    return tuple(result)


@lru_cache(maxsize=128)
def _kernel_local_ranges_cached(lo,hi):
    # Three private lazy slots per position, no more storage than the old
    # all-order tuple. Populated slots contain immutable endpoint tuples.
    return [None,None,None]


def _kernel_local_values(position,order=None):
    position=I(position)
    if position.lo<0 or position.hi>1:raise ValueError('normalized kernel domain must lie in[0,1]')
    slots=_kernel_local_ranges_cached(position.lo,position.hi)
    orders=(1,2,3) if order is None else (order,)
    if order is None and all(v is None for v in slots):
        # Retain the old full-order calculation when every order is needed.
        values=_kernel_local_ranges_uncached(position)
        slots[:]=[(v.lo,v.hi) for v in values]
    else:
        for m in orders:
            if slots[m-1] is None:
                value=_kernel_local_ranges_uncached(position,m)[0]
                slots[m-1]=(value.lo,value.hi)
    return tuple(I(*slots[m-1]) for m in orders)


def kernel_local_ranges(position):
    """All three orders with exact reuse within the shared128-position cache."""
    return _kernel_local_values(position)


def kernel_local_range(position,order):
    """One order; fill only its missing slot in the same128-position cache.

    All-order callers retain the same outward operations. Returned intervals
    are fresh, so callers cannot mutate private cached endpoint tuples.
    """
    if type(order)!=int or not 1<=order<=3:raise ValueError('kernel order1..3 required')
    return _kernel_local_values(position,order)[0]


def _bernstein_absolute_mass(row,width):
    """Positive Bernstein basis integrates each coefficient with weight1/(n+1).

    Triangle inequality bounds the absolute polynomial integral. One exact
    de Casteljau bisection tightens mixed-sign controls; it is algebra, not
    integrand sampling. Uncertain coefficients remain outward intervals.
    """
    def bound(values):return sum((I(v.bound) for v in values),I(0))*width/len(values)
    coarse=bound(row)
    if all(v.lo>=0 for v in row) or all(v.hi<=0 for v in row):return I(0,coarse.hi)
    levels=[row]
    while len(levels[-1])>1:levels.append([(a+b)/2 for a,b in zip(levels[-1],levels[-1][1:])])
    fine=(bound([v[0] for v in levels])+bound([v[-1] for v in reversed(levels)]))/2
    return I(0,min(coarse.hi,fine.hi))


@lru_cache(maxsize=128)
def _kernel_local_mass_cached(lo,hi,order):
    nodes,weights=gauss5();edges=[I(0),*nodes,I(1)];total=I(0);local_sup=I(0).hi
    for segment,(left,right) in enumerate(zip(edges,edges[1:])):
        a=max(lo,left.lo);b=min(hi,right.hi)
        if a>=b:continue
        # Node-uncertainty slivers overlap: summing nonnegative masses may
        # overcount but cannot omit either true polynomial side of a knot.
        coefficients=_kernel_segment_bernstein(I(a),I(b),nodes[segment:],weights[segment:],order)
        local_sup=max(local_sup,max(v.bound for v in coefficients))
        total=total+_bernstein_absolute_mass(coefficients,I(b)-I(a))
    rectangle=I(min(local_sup,kernel_bounds()[0][order-1].hi))*(I(hi)-I(lo))
    return min(total.hi,rectangle.hi,kernel_bounds()[1][order-1].hi)


def kernel_local_mass(position,order):
    """Certified absolute kernel mass over the complete normalized support.

    At most128 immutable upper endpoints; return a fresh interval. Existing
    range-cache capacity and worker memory/time limits remain unchanged.
    """
    position=I(position)
    if type(order)!=int or not 1<=order<=3:raise ValueError('kernel order1..3 required')
    if position.lo<0 or position.hi>1:raise ValueError('normalized kernel domain must lie in[0,1]')
    return I(0,_kernel_local_mass_cached(position.lo,position.hi,order))


def residual_integral(field,box,axis_bounds,*,checkpoint=lambda:None,error_components=None,error_breakdown=None):
    """Enclose1..4 fixed-geometry channels; axis bounds include all crossings.

    Each crossing must be charged either in global jumps or in a localized
    witness (position interval and summed absolute value/slope/curvature jumps).
    The caller proves coverage; this routine validates shape and propagates it.
    Optional signed_jumps require a single possible crossing and required=True
    only when every transverse line has that crossing; otherwise zero is included.
    """
    if len(box)!=2:raise ValueError('two-dimensional box required')
    box=tuple(map(I,box));h=[I(b.hi)-I(b.lo) for b in box]
    if any(z.lo<=0 for z in h):raise ValueError('positive box widths required')
    nodes,weights=gauss5();accum=None
    for n,w in zip(nodes,weights):
        checkpoint()
        for v,z in zip(nodes,weights):
            values=field(Jet(I(box[0].lo)+h[0]*n),Jet(I(box[1].lo)+h[1]*v))
            if not isinstance(values,(list,tuple)) or not 1<=len(values)<=4 or any(not isinstance(a,Jet) for a in values):raise ValueError('one to four jet channels required')
            if any(a.order or not a[0,1].zero for a in values):raise ValueError('fixed-geometry values only; tangent proof required separately')
            if accum is None:accum=[I(0)]*len(values)
            if len(accum)!=len(values):raise ValueError('channel count changed')
            accum=[a+b[0,0]*w*z*h[0]*h[1] for a,b in zip(accum,values)]
    kernels,l1=kernel_bounds();errors=[I(0)]*len(accum);directions=[[I(0)]*len(accum) for _ in (0,1)]
    breakdown=[[],[]]
    for axis in (0,1):
        bounds=axis_bounds(box,axis)
        if len(bounds)!=len(accum):raise ValueError('axis channel count changed')
        for channel,row in enumerate(bounds):
            third=I(row['third']);jumps=list(map(I,row['jumps']))
            if len(jumps)!=3 or any(v.lo<0 for v in [third,*jumps]):raise ValueError('nonnegative third-derivative and jump bounds required')
            ordinary_error=third*h[axis]**4*l1[2]
            if 'third_range' in row:
                # Gauss5 is exact for cubics: the exact K3 has zero integral.
                # Subtract one constant from the ordinary derivative over the
                # FULL axis. Every material jump below remains charged.
                from tools.opaque_smooth_peano import ordinary_bounds
                ordinary_error=ordinary_bounds(row,box[axis],I(1))['effective']
            global_error=sum((j*h[axis]**(k+1)*kernels[k] for k,j in enumerate(jumps)),I(0));local_error=I(0);signed_error=I(0)
            localized=row.get('localized_jumps',[])
            if not isinstance(localized,list) or len(localized)>16:raise ValueError('at most16localized jump witnesses required')
            for witness in localized:
                position=I(witness['position']);values=list(map(I,witness['jumps']))
                if position.lo<box[axis].lo or position.hi>box[axis].hi:raise ValueError('crossing position outside box')
                if len(values)!=3 or any(v.lo<0 for v in values):raise ValueError('nonnegative localized jump bounds required')
                normalized=(position-I(box[axis].lo))/h[axis]
                normalized=I(max(I(0).lo,normalized.lo),min(I(1).hi,normalized.hi))
                if 'signed_jumps' in witness:
                    signed=list(map(I,witness['signed_jumps']))
                    if witness.get('max_crossings')!=1 or type(witness.get('required')) is not bool:raise ValueError('signed witness requires one possible crossing and Boolean presence')
                    if len(signed)!=3 or any(v.bound>a.hi for v,a in zip(signed,values)):raise ValueError('signed jumps must fit their absolute bounds')
                    local=kernel_local_ranges(normalized)
                    contribution=sum((j*h[axis]**(k+1)*local[k] for k,j in enumerate(signed)),I(0))
                    if not witness['required']:contribution=I(min(I(0).lo,contribution.lo),max(I(0).hi,contribution.hi))
                    signed_error=signed_error+contribution
                else:
                    local=kernel_local_bounds(normalized)
                    local_error=local_error+sum((j*h[axis]**(k+1)*local[k] for k,j in enumerate(values)),I(0))
            symmetric=ordinary_error+global_error+local_error
            error=(signed_error+I(symmetric.hi.copy_negate(),symmetric.hi))*h[1-axis]
            # QxQy-IxIy = Qy(Ex)+Ix(Ey). Both transverse operators are
            # positive and have mass h_other, so their uniform signed enclosures
            # add. The field integral is quadrature MINUS that signed error.
            breakdown[axis].append(dict(ordinary=ordinary_error*h[1-axis],global_jumps=global_error*h[1-axis],localized_jumps=(local_error+I(signed_error.width)/2)*h[1-axis],signed_correction=signed_error*h[1-axis]))
            directions[axis][channel]=I(0,(I(error.width)/2).hi)
            errors[channel]=errors[channel]+error
    if error_breakdown is not None:error_breakdown(tuple(tuple(row) for row in breakdown))
    if error_components is not None:error_components(tuple(tuple(row) for row in directions))
    return [a-e for a,e in zip(accum,errors)]
