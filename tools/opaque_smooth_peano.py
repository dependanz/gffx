"""Centered ordinary Peano remainder; all lower-order jumps stay separate.

Gauss5 is exact on cubics,so the full-interval integral of K3 is exactly zero.
For any exact constant A,integral f''' K3 equals integral (f'''-A) K3.
The signed range must enclose the regular third derivative on every smooth
piece,not just sampled points. This needs no smoothness across known jumps:
value,slope and curvature jumps must still be charged by the caller.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I
from tools.opaque_peano import kernel_bounds


def ordinary_bounds(row,domain,mass):
    domain,mass=map(I,(domain,mass));third=I(row['third'])
    if domain.lo>=domain.hi or mass.lo<0 or third.lo<0:raise ValueError('positive domain,nonnegative mass and derivative bound required')
    factor=(I(domain.hi)-I(domain.lo))**4*kernel_bounds()[1][2]*mass
    unanchored=third*factor
    if 'third_range' not in row:return dict(unanchored=unanchored,centered=unanchored,effective=unanchored,anchor=None,variation=third)
    signed=I(row['third_range'])
    if signed.bound>third.hi:raise ValueError('signed third derivative exceeds absolute evidence')
    anchor=I(I((Q(signed.lo)+Q(signed.hi))/2).lo);variation=I((signed-anchor).bound)
    centered=variation*factor
    return dict(unanchored=unanchored,centered=centered,effective=I(0,min(unanchored.hi,centered.hi)),anchor=anchor,variation=variation)


def partitioned_ordinary(third_range,box,axis,*,checkpoint=lambda:None):
    """Four fixed derivative boxes,never extra quadrature samples or recursion.

    Symmetry of the exact Gauss5 rule makes K3(1-t)=-K3(t),hence each
    normalized half has half the certified full absolute mass. A constant
    anchor may differ by OUTER half,but is shared by both INNER halves:
    only the full inner kernel integral vanishes. Rounded physical midpoint
    intervals overlap safely while integration weights remain exact halves.
    """
    if type(axis)!=int or axis not in (0,1) or len(box)!=2:raise ValueError('two-dimensional box and axis0/1 required')
    box=tuple(map(I,box))
    if any(v.lo>=v.hi for v in box):raise ValueError('positive box widths required')
    halves=[]
    for v in box:
        midpoint=I((Q(v.lo)+Q(v.hi))/2)
        halves.append((I(v.lo,midpoint.hi),I(midpoint.lo,v.hi)))
    factor=(I(box[axis].hi)-I(box[axis].lo))**4*(I(box[1-axis].hi)-I(box[1-axis].lo))*kernel_bounds()[1][2]/4
    raw=I(0);centered=I(0);ranges=[]
    for outer in (0,1):
        pair=[]
        for inner in (0,1):
            checkpoint();sub=list(box);sub[axis]=halves[axis][inner];sub[1-axis]=halves[1-axis][outer]
            value=I(third_range(tuple(sub),axis));pair.append(value)
        hull=I(min(v.lo for v in pair),max(v.hi for v in pair));anchor=I(I((Q(hull.lo)+Q(hull.hi))/2).lo)
        for value in pair:
            raw=raw+I(value.bound)*factor;centered=centered+I((value-anchor).bound)*factor
        ranges.append(dict(thirds=pair,anchor=anchor))
    # On a half interval K3 has nonzero signed mass. Keep that mass,
    # and bound only the variation about one exact anchor per derivative box.
    # The absolute mass of either normalized half is half the full bound
    # by antisymmetry; rounded physical boxes overlap only to enclose data.
    from tools.opaque_transverse_peano import kernel_integral
    masses=(kernel_integral(I(0),I('.5'),3),kernel_integral(I('.5'),I(1),3))
    scale=(I(box[axis].hi)-I(box[axis].lo))**4*(I(box[1-axis].hi)-I(box[1-axis].lo))/2
    absolute=kernel_bounds()[1][2]/2;signed=I(0)
    for row in ranges:
        for value,mass in zip(row['thirds'],masses):
            anchor=I(I((Q(value.lo)+Q(value.hi))/2).lo)
            error=I((value-anchor).bound)*absolute
            signed=signed+(anchor*mass+I(error.hi.copy_negate(),error.hi))*scale
    effective=I(0,min(raw.hi,centered.hi))
    low=max(signed.lo,effective.hi.copy_negate());high=min(signed.hi,effective.hi)
    if low>high:raise ValueError('disjoint signed partition bounds')
    return dict(unanchored=raw,centered=centered,effective=effective,signed=I(low,high),ranges=ranges,evaluations=4)
