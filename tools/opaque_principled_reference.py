"""Independent bounded Principled pixel reference, standard library only.

Exact rational visibility; outward-rounded full GGX/Smith/Schlick/reciprocal-matte
algebra; perspective-varying normals, view vectors and albedo; fixed roughness
and frozen loss-resource constants. Certifies finite smooth boxes or refuses.
No native query, Torch evaluation, learned renderer, grid-error estimator or
physical-table admission occurs here. Texture-cell/knot crossings remain open.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
import time
from tools.opaque_shading_intervals import Interval as I,Jet,integrate_triangle,Unresolved
from tools import opaque_pixel_oracle as geom

def minimum(a,b):
    a,b=I(a),I(b);return I(min(a.lo,b.lo),min(a.hi,b.hi))
def maximum(a,b):
    a,b=I(a),I(b);return I(max(a.lo,b.lo),max(a.hi,b.hi))
def absolute(a):
    return a if a.lo>=0 else -a if a.hi<=0 else I(0,a.bound)
def bernstein(b,t):
    s=1-t;return s**3*b[0]+3*s*s*t*b[1]+3*s*t*t*b[2]+t**3*b[3]

class PreparedPrincipled:
    """Enclose independent curve preparation from exact stored binary64 inputs.

    Freezes roughness. Its derivative is not a supported tangent in this stage.
    Constants retain the prior mathematical oracle's stored pi/logs/mean weights.
    This bounds pixel integration of that declared resource model, not the
    resource's physical approximation error or production floating arithmetic.
    """
    def __init__(self,data,roughness):
        rn=list(map(I,data['roughness_nodes']));rx=list(map(I,data['_rlog']));self.cn=list(map(I,data['cosine_nodes'][1:]));self.x=list(map(I,data['_clog']));n=len(self.x);r=I(roughness)
        if not 2<=len(rn)<=129 or not 2<=n<=512 or not rn[0].lo<=r.lo<=rn[-1].hi or r.lo!=r.hi:raise ValueError('fixed roughness/resource domain')
        self.roughness=r;self.r4=r**4;self.pi=I(float.fromhex('0x1.921fb54442d18p+1'))
        index=min(sum(q.lo<=r.lo for q in rn)-1,len(rn)-2);width=rx[index+1]-rx[index];u=(r.ln()-rx[index])/width
        def interpolate(y0,y1,m0,m1):
            a,b,c,d=map(I,(y0,y1,m0,m1));return bernstein([a,a+width*c/3,b-width*d/3,b],u).exp()
        values=[[interpolate(data['_logs'][index][j][c],data['_logs'][index+1][j][c],data['_slopes'][index][j][c],data['_slopes'][index+1][j][c]) for c in range(2)] for j in range(n)]
        self.horizon=[interpolate(data['_hlogs'][index],data['_hlogs'][index+1],data['_hslopes'][index],data['_hslopes'][index+1]),I(0)]
        x=self.x
        def slope(i,c):
            sec=lambda j:(values[j+1][c]-values[j][c])/(x[j+1]-x[j])
            before,after=sec(max(0,i-1)),sec(min(i,n-2))
            if n==2:return before
            if i==0:
                h0,h1=x[1]-x[0],x[2]-x[1];raw=((2*h0+h1)*before-h0*sec(1))/(h0+h1)
            elif i==n-1:
                h0,h1=x[-1]-x[-2],x[-2]-x[-3];raw=((2*h0+h1)*after-h0*sec(n-3))/(h0+h1)
            else:
                hp,hn=x[i]-x[i-1],x[i+1]-x[i];raw=(hn*before+hp*after)/(hp+hn)
            if before.lo>0 and after.lo>0:sign=1
            elif before.hi<0 and after.hi<0:sign=-1
            elif before.hi<=0<=after.lo or after.hi<=0<=before.lo:return I(0)
            else:
                bound=(I('1.5')*minimum(absolute(before),absolute(after))).hi;return I(bound.copy_negate(),bound)
            return sign*minimum(maximum(sign*raw,0),I('1.5')*minimum(absolute(before),absolute(after)))
        slopes=[]
        for i in range(n):
            m=[slope(i,c) for c in range(2)];delta=m[0]-m[1];gap=values[i][0]-values[i][1]
            risk=maximum(maximum(delta*(x[i]-x[i-1]) if i else I(0),-delta*(x[i+1]-x[i]) if i<n-1 else I(0)),0)
            if risk.lo>0:
                scale=minimum(1,3*gap/risk)
                if scale.hi<1:scale=scale*(I(1)-8*I(Q(1,2**52)))
                elif scale.lo<1:scale=I((scale*(I(1)-8*I(Q(1,2**52)))).lo,1)
            elif risk.hi==0:scale=I(1)
            else:scale=I(0,1)
            slopes.append([v*scale for v in m])
        self.controls=[[[values[i][c],values[i][c]+(x[i+1]-x[i])*slopes[i][c]/3,values[i+1][c]-(x[i+1]-x[i])*slopes[i+1][c]/3,values[i+1][c]] for c in range(2)] for i in range(n-1)]
        self.values=values
        self.mean=[self.cn[0]**2*(self.horizon[c]+2*values[0][c])/3+sum((2*self.cn[i+1]**2*(x[i+1]-x[i])*self.controls[i][c][k]*I(data['_weights'][i][k]) for i in range(n-1) for k in range(4)),I(0)) for c in range(2)]
        if any(q.lo<=0 for q in self.mean):raise ValueError('positive mean not enclosed')
    def loss(self,c):
        value=c[0,0]
        if value.lo<=0 or value.hi>1:raise ValueError('strictly front-facing unit cosine profile')
        if value.hi<self.cn[0].lo:return [c.lift(self.horizon[j])+c/c.lift(self.cn[0])*(self.values[0][j]-self.horizon[j]) for j in range(2)]
        indices=[i for i in range(len(self.cn)-1) if value.lo>=self.cn[i].lo and value.hi<=self.cn[i+1].hi]
        if not indices:raise ValueError('cosine knot crossing requires qualified partition')
        i=indices[0];t=(c.ln()-c.lift(self.x[i]))/c.lift(self.x[i+1]-self.x[i])
        return [bernstein([c.lift(v) for v in self.controls[i][j]],t) for j in range(2)]
    def radiance(self,n,v,l,albedo,f0):
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),n[0].lift(0))
        def unit(a):
            inv=dot(a,a).sqrt().reciprocal();return [q*inv for q in a]
        nn,vv,ll=unit(n),unit(v),unit(l);hh=unit([a+b for a,b in zip(vv,ll)])
        nv,nl,nh,vh=dot(nn,vv),dot(nn,ll),dot(nn,hh),dot(vv,hh)
        # Cauchy-Schwarz gives this exact constraint for actual normalized vectors.
        for q in (nv,nl,nh,vh):
            lo=max(D(-1),q[0,0].lo);hi=min(D(1),q[0,0].hi)
            if lo>hi:raise ValueError('unit-vector enclosure inconsistent')
            q.c[0,0]=I(lo,hi)
        if nv[0,0].lo<=0 or nl[0,0].lo<=0:raise ValueError('horizon/two-sided transition unqualified')
        view,light=self.loss(nv),self.loss(nl);r4=n[0].lift(self.r4);pi=n[0].lift(self.pi)
        distribution=r4/((1-nh)*(1+nh)+r4*nh*nh)**2/pi
        visibility=n[0].lift('.5')/(nl*(nv*nv*(1-r4)+r4).sqrt()+nv*(nl*nl*(1-r4)+r4).sqrt())
        result=[]
        for colour,f in zip(albedo,f0):
            mix=lambda z:(1-f)*z[0]+f*z[1]
            matte=colour*mix(view)*mix(light)/(pi*mix([n[0].lift(q) for q in self.mean]))
            result.append((distribution*visibility*(f+(1-f)*(1-vh)**5)+matte)*nl)
        return result

def perspective_field(curve,triangle,*,inverse_w,normals,views,albedo,f0,direction,tangent=None):
    """One face, directional unit irradiance, perspective-correct corner fields."""
    xy=[[Q(p[0]),Q(p[1])] for p in triangle];a,b,c=xy
    det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if not det or len(inverse_w)!=3 or min(inverse_w)<=0:raise ValueError('positive clip reciprocal and nondegenerate face required')
    if any(len(rows)!=3 or any(len(q)!=3 for q in rows) for rows in (normals,views,albedo)):raise ValueError('three vector corners required')
    if len(f0)!=3 or len(direction)!=3:raise ValueError('RGB/direction shape')
    for q in inverse_w+[z for rows in (normals,views,albedo) for p in rows for z in p]+f0+direction:I(q)
    tangent={} if tangent is None else tangent
    if set(tangent)-{'inverse_w','normals','views','albedo','f0','direction'}:raise ValueError('unsupported reference tangent')
    def field(x,y):
        beta1=((x-a[0])*(c[1]-a[1])-(y-a[1])*(c[0]-a[0]))/det
        beta2=((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))/det
        beta=[1-beta1-beta2,beta1,beta2]
        iw=[x.lift(v,tangent=tangent.get('inverse_w',[0]*3)[i]) for i,v in enumerate(inverse_w)]
        raw=[p*q for p,q in zip(beta,iw)];den=sum(raw,x.lift(0))
        if den[0,0].lo<=0:raise ValueError('perspective denominator not certified positive')
        weights=[p/den for p in raw]
        def attribute(name,rows):
            corners=[[x.lift(rows[i][j],tangent=tangent.get(name,[[0]*3]*3)[i][j]) for j in range(3)] for i in range(3)]
            return [corners[0][j]+(raw[1]*(corners[1][j]-corners[0][j])+raw[2]*(corners[2][j]-corners[0][j]))/den for j in range(3)]
        return curve.radiance(attribute('normals',normals),attribute('views',views),[x.lift(q,tangent=tangent.get('direction',[0]*3)[i]) for i,q in enumerate(direction)],attribute('albedo',albedo),[x.lift(q,tangent=tangent.get('f0',[0]*3)[i]) for i,q in enumerate(f0)])
    return field

def visible_fragments(vertices,faces,origin=(0,0),checkpoint=lambda:None,*,preserve_disjoint=False):
    if type(preserve_disjoint)!=bool:raise ValueError("explicit disjoint reference selection required")
    subtract=geom._subtract_if_intersects if preserve_disjoint else geom._subtract
    # Reuse the already qualified exact-rational geometry, never native fragments.
    geom.integrate(vertices,faces,[(0,0,0)]*len(faces),height=1,width=1,origin=origin,checkpoint=checkpoint)
    v=[[geom._number(x) for x in p] for p in vertices];tris=[geom._triangle([v[i] for i in f]) for f in faces];budget=geom._Budget(checkpoint)
    ox,oy=map(Q,origin);pixel=((ox,oy),(ox+1,oy),(ox+1,oy+1),(ox,oy+1));parts=[]
    for i,tri in enumerate(tris):
        if tri is None:continue
        lines,depth=tri;p=pixel
        for line in lines:p=geom._clip(p,line,budget)
        if not p or geom._area(p)==0:continue
        visible=[p]
        for j,other in enumerate(tris):
            if i==j or other is None:continue
            edges,z=other;delta=tuple(a-b for a,b in zip(depth,z))
            if delta==(0,0,0) and j>i:continue
            visible=[q for p in visible for q in subtract(p,edges+(() if delta==(0,0,0) else (delta,)),budget)]
            if len(visible)>4096:raise ValueError('visible-piece budget')
        for p in visible:
            for k in range(1,len(p)-1):
                triangle=(p[0],p[k],p[k+1])
                if geom._area(triangle):parts.append((i,triangle))
    return parts

def integrate_pixel(vertices,faces,fields,*,origin=(0,0),tolerance='1e-12',max_cells=256,checkpoint=lambda:None,corrections=None,preparers=None):
    if preparers is not None and len(preparers)!=len(fields):raise ValueError('one preparer per field')
    if corrections is not None and len(corrections)!=len(fields):raise ValueError('one correction per field')
    if len(fields)!=len(faces):raise ValueError('one field per face')
    start=time.monotonic()
    def check():
        checkpoint()
        if time.monotonic()-start>90:raise Unresolved('pixel deadline')
    pieces=visible_fragments(vertices,faces,origin,check)
    if len(pieces)>32:raise Unresolved('pixel triangle-piece budget')
    value=[I(0)]*3;derivative=[I(0)]*3;cells=0;rows=[]
    tol=I(tolerance)/max(1,len(pieces))
    for face,triangle in pieces:
        if cells>=max_cells:raise Unresolved('pixel cell budget',cells=cells)
        q=integrate_triangle(fields[face],triangle,tolerance=str(tol.lo),max_cells=max_cells-cells,checkpoint=check,gauss_points=5,correction=None if corrections is None else corrections[face],prepare=None if preparers is None else preparers[face]);cells+=q['cells']
        for c in range(3):value[c]=value[c]+q['value'][c];derivative[c]=derivative[c]+q['derivative'][c]
        rows.append(dict(face=face,triangle=triangle,**q))
    return dict(value=[[value]],derivative=[[derivative]],pieces=rows,cells=cells,seconds=time.monotonic()-start,tolerance=tolerance,scope='fixed roughness, directional illumination, smooth perspective fields; analytic field tangent on fixed exact visibility')
