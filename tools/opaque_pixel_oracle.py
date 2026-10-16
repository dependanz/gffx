"""Bounded exact pixel integration for constant-colour, affine-depth triangles.

Development reference only: no gffx, torch, NumPy, raster fragments or candidate
edge-estimator code. Coordinates are pixel-space x right/y down; smaller z wins.
Every visible polygon is integrated exactly, including subpixel geometry.
"""
from fractions import Fraction as Q
import math
import time


class _Budget:
    def __init__(self, checkpoint):
        self.checkpoint=checkpoint
        self.remaining=200000
        self.deadline=time.monotonic()+90
        self.tick()

    def tick(self):
        self.remaining-=1
        if self.remaining<0:raise ValueError('oracle operation budget exceeded')
        if self.remaining%64==0 or self.remaining==199999:
            self.checkpoint()
            if time.monotonic()>=self.deadline:raise TimeoutError('oracle local deadline')


def _bounded(q, bits=8192):
    if q.numerator.bit_length()>bits or q.denominator.bit_length()>bits:
        raise ValueError('oracle rational complexity budget exceeded')
    return q


def _number(value):
    if type(value) not in (int,float,Q):raise TypeError('oracle expects rational numeric coordinates')
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('oracle finite values required')
    q=_bounded(Q(value),512)
    if abs(q)>16:raise ValueError('oracle coordinate/colour magnitude exceeds finite profile')
    return q


def _eval(line,p,budget):
    budget.tick()
    return _bounded(line[0]*p[0]+line[1]*p[1]+line[2])


def _clip(poly,line,budget):
    """Intersect a convex polygon with ax+by+c >= 0, using exact crossings."""
    if not poly:return ()
    result=[]
    previous=poly[-1];dp=_eval(line,previous,budget)
    for current in poly:
        dc=_eval(line,current,budget)
        if (dp>=0)!=(dc>=0):
            t=_bounded(dp/(dp-dc))
            result.append(tuple(_bounded(a+t*(b-a)) for a,b in zip(previous,current)))
        if dc>=0:result.append(current)
        previous,dp=current,dc
    # Duplicate boundary vertices do not alter area, but remove them to bound work.
    clean=[]
    for point in result:
        if not clean or point!=clean[-1]:clean.append(point)
    if len(clean)>1 and clean[0]==clean[-1]:clean.pop()
    if len(clean)>128:raise ValueError('oracle polygon complexity budget exceeded')
    return tuple(clean) if len(clean)>=3 else ()


def _area(poly):
    return abs(_bounded(sum((a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])),Q(0))))/2


def _triangle(points):
    a,b,c=points
    ux,uy=b[0]-a[0],b[1]-a[1]
    vx,vy=c[0]-a[0],c[1]-a[1]
    det=ux*vy-uy*vx
    if det==0:return None
    dzb,dzc=b[2]-a[2],c[2]-a[2]
    dzdx=_bounded((dzb*vy-dzc*uy)/det)
    dzdy=_bounded((ux*dzc-vx*dzb)/det)
    depth=(dzdx,dzdy,_bounded(a[2]-dzdx*a[0]-dzdy*a[1]))
    xy=[p[:2] for p in points]
    if det<0:xy.reverse()
    lines=[]
    for p,q in zip(xy,xy[1:]+xy[:1]):
        dx,dy=q[0]-p[0],q[1]-p[1]
        lines.append((-dy,dx,dy*p[0]-dx*p[1]))
    return tuple(lines),depth


def _subtract(poly,occluder,budget):
    """Partition subject minus a convex occluder into disjoint convex pieces."""
    outside=[];remaining=poly
    for line in occluder:
        part=_clip(remaining,tuple(-v for v in line),budget)
        if part and _area(part)>0:outside.append(part)
        remaining=_clip(remaining,line,budget)
        if not remaining or _area(remaining)==0:break
    return outside


def _subtract_if_intersects(poly,occluder,budget):
    """Keep a convex subject intact only after exact zero-area overlap proof.

    Every clipping operation uses the caller's existing rational/work budget.
    Positive overlap of any size uses the unchanged complement partition.
    """
    overlap=poly
    for line in occluder:
        overlap=_clip(overlap,line,budget)
        if not overlap or _area(overlap)==0:return [poly]
    return _subtract(poly,occluder,budget)


def _integrate(vertices,faces,colours,*,height,width,origin=(0,0),background=(0,0,0),checkpoint=lambda:None,affine=False):
    """Return exact mask/colour tuples for <=8x8 pixels and <=32 triangles.

    Coordinates and constant face colours are converted to exact Fractions.
    Identical depth planes use original face order; degenerate faces cover no area.
    This is a visibility oracle, not a varying-shading or clipping reference.
    """
    if type(height)!=int or type(width)!=int or not 0<height<=8 or not 0<width<=8:
        raise ValueError('oracle extent limit is 8x8')
    if not isinstance(vertices,(list,tuple)) or not isinstance(faces,(list,tuple)) or len(vertices)>64 or len(faces)>32:
        raise ValueError('oracle geometry extent')
    if any(not isinstance(row,(tuple,list)) or len(row)!=3 for row in vertices):raise ValueError('vertex shape')
    if any(not isinstance(face,(tuple,list)) or len(face)!=3 or any(type(i)!=int or not 0<=i<len(vertices) for i in face) for face in faces):
        raise ValueError('triangle index shape/range')
    if not isinstance(colours,(tuple,list)) or len(colours)!=len(faces) or any(not isinstance(c,(tuple,list)) or len(c)!=3 for c in colours):
        raise ValueError('one RGB colour per face required')
    if affine and any(any(not isinstance(c,(tuple,list)) or len(c)!=3 for c in face) for face in colours):raise ValueError('three RGB corners per face required')
    if not isinstance(origin,(tuple,list)) or len(origin)!=2:raise ValueError('origin shape')
    if not isinstance(background,(tuple,list)) or len(background)!=3:raise ValueError('background RGB required')
    if not callable(checkpoint):raise TypeError('checkpoint must be callable')
    v=tuple(tuple(_number(x) for x in row) for row in vertices)
    rgb=tuple(tuple(tuple(_number(x) for x in c) for c in row) for row in colours) if affine else tuple(tuple(_number(x) for x in row) for row in colours)
    ox,oy=map(_number,origin);bg=tuple(map(_number,background))
    budget=_Budget(checkpoint)
    triangles=tuple(_triangle(tuple(v[i] for i in face)) for face in faces)
    planes=tuple(tuple(_triangle(tuple((v[index][0],v[index][1],rgb[i][corner][channel]) for corner,index in enumerate(face)))[1] for channel in range(3)) if triangles[i] is not None else None for i,face in enumerate(faces)) if affine else None
    masks=[];images=[]
    for y in range(height):
        mr=[];cr=[]
        for x in range(width):
            budget.tick()
            left,top=ox+x,oy+y
            pixel=((left,top),(left+1,top),(left+1,top+1),(left,top+1))
            coverage=Q(0);colour=[Q(0),Q(0),Q(0)]
            for i,triangle in enumerate(triangles):
                if triangle is None:continue
                lines,depth=triangle
                polygon=pixel
                for line in lines:polygon=_clip(polygon,line,budget)
                if not polygon or _area(polygon)==0:continue
                visible=[polygon]
                for j,other in enumerate(triangles):
                    if i==j or other is None:continue
                    other_lines,other_depth=other
                    difference=tuple(a-b for a,b in zip(depth,other_depth))
                    if difference==(0,0,0) and j>i:continue
                    occluder=other_lines+(() if difference==(0,0,0) else (difference,))
                    pieces=[]
                    for part in visible:
                        pieces.extend(_subtract(part,occluder,budget))
                        if len(pieces)>4096:raise ValueError('oracle visible-piece budget exceeded')
                    visible=pieces
                    if not visible:break
                area=sum((_area(p) for p in visible),Q(0))
                coverage=_bounded(coverage+area)
                if affine:
                    # Independent triangulation/centroid quadrature: exact for
                    # degree-one radiance, distinct from native polygon moments.
                    for part in visible:
                        for k in range(1,len(part)-1):
                            tri=(part[0],part[k],part[k+1]);a=_area(tri)
                            centre=tuple(sum(p[axis] for p in tri)/3 for axis in range(2))
                            colour=[_bounded(c+a*_eval(line,centre,budget)) for c,line in zip(colour,planes[i])]
                else:
                    colour=[_bounded(c+area*k) for c,k in zip(colour,rgb[i])]
            if not 0<=coverage<=1:raise ArithmeticError('oracle partition violated pixel area')
            mr.append(coverage)
            cr.append(tuple(_bounded(c+(1-coverage)*b) for c,b in zip(colour,bg)))
        masks.append(tuple(mr));images.append(tuple(cr))
    checkpoint()
    return {'mask':tuple(masks),'colour':tuple(images)}


def integrate(vertices,faces,colours,**kwargs):
    """Original constant-face-radiance pixel reference; bounds unchanged."""
    return _integrate(vertices,faces,colours,**kwargs)

def integrate_affine(vertices,faces,corner_colours,**kwargs):
    """Exact integration of screen-affine RGB specified at three face corners.

    This is neither perspective interpolation nor nonlinear shader quadrature.
    Visibility and all resource/complexity bounds match the original oracle.
    """
    return _integrate(vertices,faces,corner_colours,affine=True,**kwargs)
