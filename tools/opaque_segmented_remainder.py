"""Signed regular-third-derivative remainder on a finite physical partition.

Each root uncertainty zone is integrated too. Only the full-domain constant
anchor cancels: every subinterval retains its signed K3 mass. Slabs use exact
Decimal endpoints and cover the domain without gaps; endpoint values have
measure zero. All lower-order jumps are still required separately by the caller.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I
from tools.opaque_transverse_peano import weighted_kernel_parts


def allowed_segments(candidates,witnesses,slab,role):
    """Certified strictly monotone crossings constrain smooth OPEN slabs.

    Branch restriction requires a crossing on EVERY represented line.
    Possible-but-absent crossings and the uncertainty zones retain all
    candidate branches; monotonicity alone does not prove root existence.
    """
    selected=list(candidates);slab=I(slab)
    for w in witnesses:
        if w['role']!=role or w.get('required') is not True:continue
        p=I(w['position']);sign=w.get('orientation',0)
        if sign not in (-1,1):continue
        below=(slab.hi<=p.lo and sign==1) or (slab.lo>=p.hi and sign==-1)
        above=(slab.lo>=p.hi and sign==1) or (slab.hi<=p.lo and sign==-1)
        if below:selected=[i for i in selected if i<w['knot']]
        if above:selected=[i for i in selected if i>=w['knot']]
    if not selected:raise ValueError('inconsistent certified material segment')
    return selected


def segmented_remainder(third_range,domain,roots,*,full_range,checkpoint=lambda:None):
    domain=I(domain);full_range=I(full_range)
    if domain.lo>=domain.hi:raise ValueError('positive remainder domain')
    if not isinstance(roots,(list,tuple)) or len(roots)>16:raise ValueError('at most16root enclosures')
    edges={domain.lo,domain.hi}
    for root in roots:
        root=I(root)
        if root.lo<domain.lo or root.hi>domain.hi:raise ValueError('root outside remainder domain')
        edges.update((root.lo,root.hi))
    edges=sorted(edges)
    if len(edges)-1>16:return None  # No additional inputs; retain prior bound.
    anchor=I(I((Q(full_range.lo)+Q(full_range.hi))/2).lo)
    h=I(domain.hi)-I(domain.lo);value=I(0);parts=[]
    for a,b in zip(edges,edges[1:]):
        checkpoint();slab=I(a,b);coefficient=I(third_range(slab))
        lo=max(coefficient.lo,full_range.lo);hi=min(coefficient.hi,full_range.hi)
        if lo>hi:raise ValueError('disjoint regular derivative enclosures')
        coefficient=I(lo,hi)-anchor
        def normalized(x):
            v=(I(x)-I(domain.lo))/h
            return I(max(I(0).lo,v.lo),min(I(1).hi,v.hi))
        part=weighted_kernel_parts(coefficient,normalized(a),normalized(b),3)
        value=value+part['value'];parts.append(dict(domain=slab,**part))
    return dict(value=value*h**4,anchor=anchor,parts=parts,evaluations=len(parts))
