"""Independent local moving-visibility reference with exact branch certificates.

Vertex trajectories are linear in one parameter. Homogeneous supporting lines
and intersections retain exact polynomial identities: a zero predicate is accepted
only when it is identically zero, never merely because its first derivative is zero.
No native renderer, Torch graph, sampled topology or finite-difference derivative.
"""
from fractions import Fraction as Q
import time
from tools import opaque_pixel_oracle as geom
from tools.opaque_shading_intervals import Interval as I,Jet,integrate_triangle,Unresolved
from tools.opaque_principled_reference import perspective_field

class Poly:
    """Ascending exact rational coefficients, bounded degree32 and8192-bit terms."""
    def __init__(self,value=0):
        if isinstance(value,Poly):self.c=value.c;return
        v=list(value) if isinstance(value,(tuple,list)) else [value]
        v=[geom._bounded(Q(x)) for x in v]
        while len(v)>1 and v[-1]==0:v.pop()
        if not v:v=[Q(0)]
        if len(v)>33:raise ValueError('polynomial degree budget')
        self.c=tuple(v)
    def __getitem__(self,k):return self.c[k] if k<len(self.c) else Q(0)
    @property
    def zero(self):return self.c==(0,)
    def __add__(self,b):
        b=Poly(b);return Poly([self[i]+b[i] for i in range(max(len(self.c),len(b.c)))])
    __radd__=__add__
    def __neg__(self):return Poly([-x for x in self.c])
    def __sub__(self,b):return self+-Poly(b)
    def __rsub__(self,b):return Poly(b)+-self
    def __mul__(self,b):
        b=Poly(b)
        if self.zero or b.zero:return Poly(0)
        if len(self.c)+len(b.c)>34:raise ValueError('polynomial degree budget')
        result=[Q(0)]*(len(self.c)+len(b.c)-1)
        for i,a in enumerate(self.c):
            for j,v in enumerate(b.c):result[i+j]=geom._bounded(result[i+j]+a*v)
        return Poly(result)
    __rmul__=__mul__

P=Poly

def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum((x*y for x,y in zip(a,b)),P(0))

class BranchCertificate:
    """Prove every strict predicate keeps its sign in a common rational window."""
    def __init__(self,checkpoint=lambda:None):
        self.budget=geom._Budget(checkpoint);self.radius=Q(1,1024);self.strict={};self.identities=0
    def classify(self,p):
        self.budget.tick();p=P(p)
        if p.zero:self.identities+=1;return 0
        if not p[0]:raise Unresolved('topology transition: nonidentity predicate vanishes at parameter zero')
        if p.c not in self.strict:
            radius=self.radius
            for _ in range(65):
                self.budget.tick()
                if sum(abs(x)*radius**i for i,x in enumerate(p.c) if i)<abs(p[0]):break
                radius/=2
            else:raise Unresolved('topology certificate radius budget')
            self.radius=min(self.radius,radius);self.strict[p.c]=1 if p[0]>0 else -1
        return 1 if p[0]>0 else -1
    def report(self):
        return dict(radius=str(self.radius),identities=self.identities,strict=[dict(coefficients=[str(x) for x in k],sign=v) for k,v in self.strict.items()],operations=200000-self.budget.remaining)

class Point:
    def __init__(self,line_a,line_b,certificate):
        self.h=cross(line_a,line_b)
        sign=certificate.classify(self.h[2])
        if sign==0:raise Unresolved('parallel or coincident supporting lines')
        if sign<0:self.h=tuple(-p for p in self.h)
    def xy(self):return tuple(geom._bounded(p[0]/self.h[2][0]) for p in self.h[:2])
    def tangent(self):
        w=self.h[2];return tuple(geom._bounded((p[1]*w[0]-p[0]*w[1])/w[0]**2) for p in self.h[:2])
    def same(self,other,cert):
        diffs=[a*other.h[2]-b*self.h[2] for a,b in zip(self.h[:2],other.h[:2])]
        if all(p.zero for p in diffs):return True
        # Merely coincident today is a potential change in polygon topology.
        if all(p[0]==0 for p in diffs):raise Unresolved('topology: coincident nonidentity vertices')
        return False


def clip(poly,line,cert):
    if not poly:return []
    result=[];previous=poly[-1][0];dp=cert.classify(dot(line,previous.h))
    for current,incoming in poly:
        dc=cert.classify(dot(line,current.h))
        if (dp>=0)!=(dc>=0):
            result.append((Point(incoming,line,cert),incoming if dp>=0 else line))
        if dc>=0:result.append((current,incoming))
        previous,dp=current,dc
    clean=[]
    for item in result:
        if not clean or not item[0].same(clean[-1][0],cert):clean.append(item)
    if len(clean)>1 and clean[0][0].same(clean[-1][0],cert):
        clean[0]=(clean[0][0],clean[-1][1]);clean.pop()
    if len(clean)>128:raise Unresolved('polygon vertex budget')
    return clean if len(clean)>=3 else []


def fan(poly,cert):
    result=[]
    for k in range(1,len(poly)-1):
        points=[poly[j][0] for j in (0,k,k+1)]
        numerator=dot(points[0].h,cross(points[1].h,points[2].h))
        sign=cert.classify(numerator)
        if not sign:continue
        if sign<0:raise ArithmeticError('polygon orientation invariant')
        result.append(points)
    return result


def subtract(poly,lines,cert):
    outside=[];remaining=poly
    for line in lines:
        part=clip(remaining,tuple(-v for v in line),cert)
        if part and fan(part,cert):outside.append(part)
        remaining=clip(remaining,line,cert)
        if not remaining or not fan(remaining,cert):break
    return outside


def moving_fragments(vertices,faces,motion,*,origin=(0,0),checkpoint=lambda:None):
    # Existing exact geometry input contract, with no changed limits.
    geom.integrate(vertices,faces,[(0,0,0)]*len(faces),height=1,width=1,origin=origin,checkpoint=checkpoint)
    if len(motion)!=len(vertices) or any(len(q)!=3 for q in motion):raise ValueError('vertex motion shape')
    v=[[P([geom._number(x),geom._number(d)]) for x,d in zip(p,m)] for p,m in zip(vertices,motion)]
    cert=BranchCertificate(checkpoint);triangles=[]
    for face in faces:
        points=[v[i] for i in face];a,b,c=points
        det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        sign=cert.classify(det)
        if not sign:triangles.append(None);continue
        if sign<0:points.reverse();a,b,c=points;det=-det
        xy=[(p[0],p[1],P(1)) for p in points];lines=tuple(cross(p,q) for p,q in zip(xy,xy[1:]+xy[:1]))
        ux,uy,uz=[b[j]-a[j] for j in range(3)];vx,vy,vz=[c[j]-a[j] for j in range(3)]
        zx=uz*vy-vz*uy;zy=ux*vz-vx*uz;zc=a[2]*det-zx*a[0]-zy*a[1]
        triangles.append((lines,(zx,zy,zc),det))
    ox,oy=map(Q,origin)
    lines=[tuple(map(P,q)) for q in [(1,0,-ox),(0,1,-oy),(-1,0,ox+1),(0,-1,oy+1)]]
    # Counterclockwise boundary: left/top, top/right, right/bottom, bottom/left.
    pixel=[(Point(lines[i],lines[(i+1)%4],cert),lines[i]) for i in range(4)]
    parts=[]
    for i,tri in enumerate(triangles):
        if tri is None:continue
        edges,depth,den=tri;poly=pixel
        for line in edges:poly=clip(poly,line,cert)
        if not poly or not fan(poly,cert):continue
        visible=[poly]
        for j,other in enumerate(triangles):
            if i==j or other is None:continue
            other_edges,z,other_den=other;delta=tuple(a*other_den-b*den for a,b in zip(depth,z))
            if all(q.zero for q in delta):
                if j>i:continue
                occluder=other_edges
            else:
                if all(q[0]==0 for q in delta):raise Unresolved('topology: changing depth tie')
                occluder=other_edges+(delta,)
            visible=[q for p in visible for q in subtract(p,occluder,cert)]
            if len(visible)>4096:raise Unresolved('visible-piece budget')
            if not visible:break
        for poly in visible:
            for points in fan(poly,cert):
                parts.append(dict(face=i,triangle=[p.xy() for p in points],motion=[p.tangent() for p in points]))
                if len(parts)>32:raise Unresolved('pixel triangle-piece budget')
    return parts,cert.report()


def integrate_moving_pixel(vertices,faces,motion,fields,*,origin=(0,0),background=(0,0,0),tolerance='1e-12',max_cells=256,checkpoint=lambda:None):
    if type(max_cells)!=int or not 1<=max_cells<=256:raise ValueError('pixel cell budget')
    if len(fields)!=len(faces) or len(background)!=3:raise ValueError('field/background shape')
    bg=list(map(geom._number,background));requested=I(tolerance)
    if requested.lo<=0 or requested.hi>I('1e-6').hi:raise ValueError('reference tolerance domain')
    started=time.monotonic()
    def check():
        checkpoint()
        if time.monotonic()-started>90:raise Unresolved('pixel deadline')
    parts,certificate=moving_fragments(vertices,faces,motion,origin=origin,checkpoint=check)
    value=[I(0)]*3;derivative=[I(0)]*3;mask=Q(0);dm=Q(0);cells=0;rows=[]
    tol=I(tolerance)/max(1,len(parts))
    for part in parts:
        check()
        if cells>=max_cells:raise Unresolved('pixel cell budget')
        a,b,c=part['triangle'];da,db,dc=part['motion'];u=[b[i]-a[i] for i in range(2)];v=[c[i]-a[i] for i in range(2)];du=[db[i]-da[i] for i in range(2)];dv=[dc[i]-da[i] for i in range(2)]
        mask=geom._bounded(mask+(u[0]*v[1]-u[1]*v[0])/2);dm=geom._bounded(dm+(du[0]*v[1]+u[0]*dv[1]-du[1]*v[0]-u[1]*dv[0])/2)
        q=integrate_triangle(fields[part['face']],part['triangle'],motion=part['motion'],tolerance=str(tol.lo),max_cells=max_cells-cells,checkpoint=check,gauss_points=5)
        if len(q['value'])!=3:raise ValueError('RGB field required')
        cells+=q['cells']
        for k in range(3):value[k]=value[k]+q['value'][k];derivative[k]=derivative[k]+q['derivative'][k]
        rows.append(dict(**part,**q))
    if not 0<=mask<=1:raise ArithmeticError('pixel mask invariant')
    for k in range(3):value[k]=value[k]+I(1-mask)*bg[k];derivative[k]=derivative[k]-I(dm)*bg[k]
    check()
    return dict(value=value,derivative=derivative,mask=mask,mask_derivative=dm,pieces=rows,cells=cells,certificate=certificate,seconds=time.monotonic()-started,tolerance=tolerance,scope='local first derivative within certified stable visibility; shader smoothness remains conditional')


def moving_perspective_field(curve,triangle,motion,**attributes):
    """Transport the original perspective field with its moving triangle.

    At parameter zero its material coordinates equal screen coordinates. Their
    first tangent subtracts interpolated vertex velocity. Existing attribute/w
    tangents are independently supplied to perspective_field via attributes.
    """
    base=perspective_field(curve,triangle,**attributes)
    if len(motion)!=3 or any(len(p)!=3 for p in motion):raise ValueError('triangle motion shape')
    velocity=[[geom._number(x) for x in p[:2]] for p in motion]
    a,b,c=[[Q(x) for x in p[:2]] for p in triangle];det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def field(x,y):
        q=((x-a[0])*(c[1]-a[1])-(y-a[1])*(c[0]-a[0]))/det
        r=((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))/det
        beta=[1-q-r,q,r];coords=[]
        for axis,point in enumerate((x,y)):
            shift=sum((w*velocity[i][axis] for i,w in enumerate(beta)),x.lift(0));coeff=dict(point.c)
            for k in range(point.order+1):coeff[k,1]=point[k,1]-shift[k,0]
            coords.append(Jet(order=point.order,coeff=coeff))
        return base(*coords)
    return field
