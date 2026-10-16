"""Exact fixed-geometry light support from perspective-normal numerators.

For positive clip weights, normal normalization and light normalization preserve
sign. A linear numerator therefore clips a triangle exactly; no epsilon is used.
This certifies fixed colour/albedo tangents only, not a moving-boundary gradient.
"""
from fractions import Fraction as Q
from tools.opaque_pixel_oracle import _Budget,_triangle,_clip,_area,_eval

def partition(part,source,origin,direction,*,checkpoint=lambda:None):
    checkpoint();budget=_Budget(checkpoint)
    if len(direction)!=3 or not any(Q(v) for v in direction):raise ValueError('nonzero light direction required')
    iw=[Q(1)/Q(w) if Q(w)>0 else None for w in source['clip_w']]
    if any(w is None for w in iw):raise ValueError('positive perspective weights required')
    points=[(Q(p[0])-Q(origin[0]),Q(p[1])-Q(origin[1]),iw[i]*sum(Q(n)*Q(l) for n,l in zip(source['normal'][i],direction))) for i,p in enumerate(source['corners'])]
    base=_triangle(tuple(points))
    if base is None:raise ValueError('nondegenerate source triangle required')
    line=base[1];tri=tuple(tuple(Q(x) for x in p) for p in part['triangle']);area=_area(tri)
    if area<=0:raise ValueError('positive original piece area required')
    values=[_eval(line,p,budget) for p in tri]
    if max(values)<=0:lit=();dark=tri
    elif min(values)>=0:lit=tri;dark=()
    else:lit=_clip(tri,line,budget);dark=_clip(tri,tuple(-v for v in line),budget)
    lit_area=_area(lit) if lit else Q(0);dark_area=_area(dark) if dark else Q(0)
    if lit_area+dark_area!=area:raise ValueError('light partition area mismatch')
    triangles=[(lit[0],lit[i],lit[i+1]) for i in range(1,len(lit)-1)] if lit else []
    triangles=[t for t in triangles if _area(t)>0]
    if len(triangles)>2 or sum((_area(t) for t in triangles),Q(0))!=lit_area:raise ValueError('bounded light fan mismatch')
    if any(_eval(line,p,budget)<0 for t in triangles for p in t):raise ValueError('uncertified lit support')
    return [dict(face=part['face'],triangle=t) for t in triangles],dict(line=list(map(str,line)),corner_values=list(map(str,values)),original_area=str(area),lit_area=str(lit_area),dark_area=str(dark_area),components=len(triangles),proof='Exact linear normal-dot-light numerator; positive perspective denominator. Dark contribution to fixed colour/albedo derivative is identically zero.')

def allocate(tolerance,parts):
    from decimal import localcontext,ROUND_FLOOR
    from tools.opaque_shading_intervals import I
    areas=[_area(tuple(tuple(Q(x) for x in p) for p in row['triangle'])) for row in parts];total=sum(areas,Q(0))
    if not total:return []
    with localcontext() as ctx:
        ctx.prec=50;ctx.rounding=ROUND_FLOOR
        shares=[+(I(tolerance)*I(a/total)).lo for a in areas]
    if any(v<=0 for v in shares) or sum((I(v) for v in shares),I(0)).hi>I(tolerance).lo:raise ValueError('light components exceed piece error allocation')
    return shares
