"""Outward angular classifiers with an explicit displaced-boundary error.

An affine classifier L approximates a smooth cosine q with |q-L|<=delta.
Different branch choices can occur only inside |L-k|<=delta, and there
|q-k|<=delta. Exact polygon band area times a Taylor bound on the adjacent
branch difference encloses that error, including binary value/slope jumps.
This is a fixed-geometry integration aid, never a modified appearance model.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_pixel_oracle import _bounded


def area(poly):
    if len(poly)<3:return Q(0)
    return abs(_bounded(sum((_bounded(a[0]*b[1]-a[1]*b[0]) for a,b in zip(poly,poly[1:]+poly[:1])),Q(0))))/2


def clip(poly,plane):
    if not poly:return []
    if len(poly)>64:raise ValueError('polygon vertex cap64')
    def value(p):return _bounded(sum((_bounded(a*b) for a,b in zip(plane,(*p,Q(1)))),Q(0)))
    result=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        va,vb=value(a),value(b)
        if va>=0:result.append(a)
        if va*vb<0:
            t=_bounded(va/(va-vb));result.append(tuple(_bounded(x+t*(y-x)) for x,y in zip(a,b)))
    unique=[]
    for p in result:
        if not unique or p!=unique[-1]:unique.append(p)
    if len(unique)>1 and unique[-1]==unique[0]:unique.pop()
    if len(unique)>64:raise ValueError('polygon vertex cap64')
    return unique if area(unique) else []


def partition(triangle,planes,*,checkpoint=lambda:None):
    checkpoint()
    if len(triangle)!=3 or any(len(p)!=2 for p in triangle):raise ValueError('triangle shape')
    if len(planes)>16:raise ValueError('plane cap16')
    cells=[[( _bounded(Q(x)),_bounded(Q(y))) for x,y in triangle]]
    if not area(cells[0]):raise ValueError('positive triangle area')
    for plane in planes:
        checkpoint()
        if len(plane)!=3:raise ValueError('plane shape')
        plane=tuple(_bounded(Q(q)) for q in plane)
        if not any(plane[:2]):continue # constant classifiers never split ownership
        parts=[]
        for poly in cells:
            for p in (plane,tuple(-z for z in plane)):
                child=clip(poly,p)
                if child:parts.append(child)
        if len(parts)>256:raise Unresolved('angular partition cell cap256')
        cells=parts
    return cells


def band_area(triangle,plane,delta):
    delta=_bounded(Q(delta))
    if delta<0:raise ValueError('nonnegative classifier error')
    a,b,c=map(Q,plane)
    return area(clip(clip(list(triangle),(a,b,c+delta)),(-a,-b,-c+delta)))


def affine_enclosure(field,triangle):
    """Taylor's second-order remainder over a convex triangle's entire box."""
    tri=[tuple(map(Q,p)) for p in triangle]
    if len(tri)!=3 or any(len(p)!=2 for p in tri) or not area(tri):raise ValueError('positive triangle required')
    centre=[sum(p[j] for p in tri)/3 for j in (0,1)]
    box=[I(I(min(p[j] for p in tri)).lo,I(max(p[j] for p in tri)).hi) for j in (0,1)]
    radius=[I(max(abs(p[j]-centre[j]) for p in tri)) for j in (0,1)]
    at=field(Jet(centre[0]),Jet(centre[1]))
    if any(not q[0,1].zero for q in at):raise ValueError('fixed geometry only')
    dx=field(Jet.variable(I(centre[0]),order=1),Jet(I(centre[1]),order=1))
    dy=field(Jet(I(centre[0]),order=1),Jet.variable(I(centre[1]),order=1))
    hxx=field(Jet.variable(box[0],order=2),Jet(box[1],order=2,tangent=1))
    hyy=field(Jet(box[0],order=2),Jet.variable(box[1],order=2))
    result=[]
    for q,gx,gy,xx,yy in zip(at,dx,dy,hxx,hyy):
        coeff=[(Q(v.lo)+Q(v.hi))/2 for v in (q[0,0],gx[1,0],gy[1,0])]
        rounding=I((q[0,0]-I(coeff[0])).bound)+I((gx[1,0]-I(coeff[1])).bound)*radius[0]+I((gy[1,0]-I(coeff[2])).bound)*radius[1]
        error=rounding+I(xx[2,0].bound)*radius[0]**2+I(xx[1,1].bound)*radius[0]*radius[1]+I(yy[2,0].bound)*radius[1]**2
        plane=tuple(map(_bounded,(coeff[1],coeff[2],coeff[0]-coeff[1]*centre[0]-coeff[2]*centre[1])))
        result.append(dict(plane=plane,error=I(0,error.hi)))
    return result


def join_bound(difference,knot,delta):
    """Bound a smooth adjacent-branch difference around its common join."""
    delta=I(delta)
    if delta.lo<0:raise ValueError('nonnegative join radius')
    k=I(knot);at=difference(Jet.variable(k,order=1));around=difference(Jet.variable(k+I(delta.hi.copy_negate(),delta.hi),order=2))
    if len(at)!=len(around):raise ValueError('join channel count')
    values=[I(a[0,0].bound)+I(a[1,0].bound)*delta+I(b[2,0].bound)*delta**2 for a,b in zip(at,around)]
    return [I(0,v.hi) for v in values]


def refine_partition(part,assess,allowance,*,max_depth=10,tick=lambda:None):
    """Charge every attempted classifier to the caller's shared cell budget."""
    if type(max_depth)!=int or not 0<=max_depth<=10:raise ValueError('partition depth limit')
    allowance=I(allowance)
    if allowance.lo<=0:raise ValueError('positive partition allowance')
    pending=[(part,allowance,0)];accepted=[];visits=0
    while pending:
        tick();visits+=1
        if visits>256:raise Unresolved('partition cell budget')
        current,budget,depth=pending.pop()
        try:
            certificate=assess(current)
            if I(certificate['error']).hi<=(budget/2).lo:
                accepted.append(dict(part=current,certificate=certificate,allowance=budget,depth=depth));continue
        except Unresolved as e:
            if e.details['reason'] not in ('angular classifier knot cap8','join extension reaches logarithm singularity'):raise
        if depth==max_depth:raise Unresolved('partition depth budget',depth=depth,visits=visits)
        tri=[tuple(map(Q,p)) for p in current['triangle']]
        i,j=max(((0,1),(1,2),(2,0)),key=lambda pair:sum((tri[pair[0]][k]-tri[pair[1]][k])**2 for k in (0,1)))
        k=3-i-j;mid=tuple(_bounded((a+b)/2) for a,b in zip(tri[i],tri[j]))
        children=[(tri[i],mid,tri[k]),(mid,tri[j],tri[k])]
        for triangle in reversed(children):pending.append((dict(current,triangle=triangle),budget/2,depth+1))
    return accepted
