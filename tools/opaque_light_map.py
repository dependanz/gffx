"""Exact unit-square endpoint flattening for fixed-geometry checker trials.

x=u**4,y=v**4 is a bijection on the unit square with Jacobian16*u**3*v**3.
It changes the integration coordinates, not the domain, radiance or tolerance.
Every strictly positive coordinate/area remains positive; no cutoff is used.
"""

def coordinates(u,v,power=1):
    if type(power) is not int or power not in (1,4):raise ValueError('explicit light-map power1or4 required')
    if power==1:return u,v,u.lift(1)
    return u**4,v**4,16*(u**3)*(v**3)


def align_triangle(part,line):
    """Cyclic relabeling puts exact zero-light corners at the flattened edges.

    For the radial triangle chart a*(1-u)+u*((1-v)*b+v*c), zeros
    at a,b make light proportional to u*v; a sole zero at a makes
    it u times a positive affine factor. Cyclic order preserves the
    signed determinant. No coordinates, area or positive values change.
    """
    from fractions import Fraction as Q
    points=part['triangle']
    if len(points)!=3 or len(line)!=3 or any(len(p)!=2 for p in points):raise ValueError('triangle and affine light line required')
    t=[tuple(map(Q,p)) for p in points];a,b,c=t
    if (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])==0:raise ValueError('nondegenerate lit triangle required')
    line=tuple(map(Q,line));values=[line[0]*x+line[1]*y+line[2] for x,y in t]
    if min(values)<0 or max(values)<=0:raise ValueError('exactly clipped positive light support required')
    zeros=[i for i,v in enumerate(values) if v==0]
    if not zeros:return part
    start=zeros[0] if len(zeros)==1 else next(i for i in zeros if (i+1)%3 in zeros)
    if start==0:return part
    return dict(part,triangle=tuple(points[(start+i)%3] for i in range(3)))
