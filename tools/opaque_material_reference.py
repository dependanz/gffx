"""Independent bounded material branches, mixed roughness jets and texture cells.
Development reference only. Exact line partition, outward Decimal60, no native code.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
import time,math
from tools.opaque_shading_intervals import I,Jet,integrate_triangle,Unresolved
from tools.opaque_moving_reference import P,Point,BranchCertificate,clip,fan
from tools.opaque_principled_reference import bernstein

def exp_jet(a):
    if isinstance(a,RangeJet):return RangeJet(exp_jet(a.plain()),exp_jet(a.centre),a.radius)
    # exp(a(z,theta))=exp(a(z,0))*(1+theta*d_theta a(z,0)).
    b=[a[0,0].exp()]
    for n in range(1,a.order+1):b.append(sum((k*a[k,0]*b[n-k] for k in range(1,n+1)),I(0))/n)
    return Jet(order=a.order,coeff={(i,j):b[i] if j==0 else sum((b[k]*a[i-k,1] for k in range(i+1)),I(0)) for i in range(a.order+1) for j in range(2)})

class RangeJet(Jet):
    """A roughness jet with a centred mean-value enclosure for its zeroth term.

    The first Taylor coefficient encloses f' on the whole root interval. Thus
    f(root_mid)+f'(root_interval)*(root-root_mid) is a second valid enclosure.
    Intersecting it with the natural enclosure controls cancellation dependency.
    This is used only for pure one-dimensional roughness, without mixed tangents.
    """
    def __init__(self,jet,centre,radius):
        super().__init__(order=jet.order,coeff=jet.c);self.centre=centre if isinstance(centre,Jet) else Jet.variable(centre,order=jet.order);self.radius=I(radius)
        if self.order and any(k[1] for k in self.c):raise ValueError('centred roughness jet is univariate')
        for k in range(self.order-1,-1,-1):
            a=self[k,0];b=self.centre[k,0]+(k+1)*self[k+1,0]*self.radius
            self.c[k,0]=I(max(a.lo,b.lo),min(a.hi,b.hi))
            if self.c[k,0].zero:self.c.pop((k,0))
    def plain(self):return Jet(order=self.order,coeff=self.c)
    def lift(self,value,tangent=0):
        if tangent:raise ValueError('centred curve has no parameter tangent')
        return RangeJet(Jet(value,order=self.order),Jet(value,order=self.order),self.radius)
    def other(self,b):
        if isinstance(b,RangeJet):
            if b.order!=self.order or b.radius.lo!=self.radius.lo or b.radius.hi!=self.radius.hi:raise ValueError('centred jet mismatch')
            return b
        if isinstance(b,Jet):raise ValueError('mixed centred/plain expression')
        return self.lift(b)
    def __add__(self,b):
        b=self.other(b);return RangeJet(self.plain()+b.plain(),self.centre+b.centre,self.radius)
    __radd__=__add__
    def __neg__(self):return RangeJet(-self.plain(),-self.centre,self.radius)
    def __sub__(self,b):return self+-self.other(b)
    def __rsub__(self,b):return self.other(b)+-self
    def __mul__(self,b):
        b=self.other(b);return RangeJet(self.plain()*b.plain(),self.centre*b.centre,self.radius)
    __rmul__=__mul__
    def reciprocal(self):return RangeJet(self.plain().reciprocal(),self.centre.reciprocal(),self.radius)
    def ln(self):return RangeJet(self.plain().ln(),self.centre.ln(),self.radius)
    def sqrt(self):return RangeJet(self.plain().sqrt(),self.centre.sqrt(),self.radius)

def select(a,b,kind):
    if a is b:return a
    b=a.other(b);delta=(a-b)[0,0]
    if delta.hi<0:return a if kind=='min' else b
    if delta.lo>0:return b if kind=='min' else a
    if all(a[k].lo==a[k].hi==b[k].lo==b[k].hi for k in a.c.keys()|b.c.keys()):return a
    raise ValueError('material branch not certified: '+kind+'; delta='+str(delta)+'; centre='+str(getattr(a-b,'centre',None)))

def abs_jet(a):
    if a[0,0].lo>0:return a
    if a[0,0].hi<0:return -a
    if not a.c:return a
    raise ValueError('material branch not certified: absolute')

def partition(cuts,*,checkpoint=lambda:None):
    """Partition one pixel by homogeneous polynomial lines, including motion.

    Each coefficient is its full polynomial trajectory, not merely a tangent.
    Every region carries the selected side of each line, and certified vertex
    velocities. The32-triangle,90s and symbolic budgets remain unchanged.
    """
    if len(cuts)>32:raise Unresolved('material cut budget')
    cert=BranchCertificate(checkpoint)
    lines=[tuple(map(P,q)) for q in [(1,0,0),(0,1,0),(-1,0,1),(0,-1,1)]]
    pixel=[(Point(lines[i],lines[(i+1)%4],cert),lines[i]) for i in range(4)]
    polygons=[(pixel,())]
    for cut in cuts:
        checkpoint()
        if isinstance(cut,dict):
            for guard in cut['positive']:
                if cert.classify(guard)<=0:raise Unresolved('material projection denominator/orientation')
            cut=cut['line']
        if len(cut)!=3:raise ValueError('homogeneous line shape')
        line=tuple(map(P,cut))
        if all(q.zero for q in line):raise ValueError('identically zero material cut')
        output=[]
        for poly,signs in polygons:
            for sign in (-1,1):
                part=clip(poly,tuple(sign*q for q in line),cert)
                if part and fan(part,cert):output.append((part,signs+(sign,)))
        if sum(len(p)-2 for p,s in output)>32:raise Unresolved('material piece budget')
        polygons=output
    parts=[dict(triangle=[p.xy() for p in points],motion=[p.tangent() for p in points],sides=signs) for poly,signs in polygons for points in fan(poly,cert)]
    return dict(pieces=parts,certificate=cert.report())

def integrate_partition(regions,field,*,tolerance='1e-12',max_cells=256,checkpoint=lambda:None,gauss_points=5):
    if type(max_cells)!=int or not 1<=max_cells<=256:raise ValueError('pixel cell budget')
    requested=I(tolerance)
    if requested.lo<=0 or requested.hi>D('1e-6'):raise ValueError('reference tolerance domain')
    parts=regions['pieces']
    if not 1<=len(parts)<=32:raise ValueError('material piece budget')
    start=time.monotonic()
    def check():
        checkpoint()
        if time.monotonic()-start>90:raise Unresolved('material pixel deadline')
    value=[I(0)]*3;derivative=[I(0)]*3;cells=0;rows=[];tol=requested/len(parts)
    for p in parts:
        check()
        if cells>=max_cells:raise Unresolved('pixel cell budget')
        q=integrate_triangle(field(p['sides']),p['triangle'],motion=p['motion'],tolerance=str(tol.lo),max_cells=max_cells-cells,checkpoint=check,gauss_points=gauss_points)
        if len(q['value'])!=3:raise ValueError('RGB field required')
        cells+=q['cells'];rows.append(dict(**p,**q))
        for k in range(3):value[k]=value[k]+q['value'][k];derivative[k]=derivative[k]+q['derivative'][k]
    check()
    return dict(value=value,derivative=derivative,pieces=rows,cells=cells,certificate=regions['certificate'],seconds=time.monotonic()-start,tolerance=tolerance)

def bilinear(texels,u,v,*,tangent=None):
    """One declared cell; callers partition texel-centre/wrap/clamp boundaries.

    Cell extrapolation is needed for interval boxes extending beyond a triangular
    cell's bounding rectangle. Cell ownership must come from exact partitioning,
    never an interval midpoint or this evaluator. Texel and UV tangents are kept.
    """
    if len(texels)!=2 or any(len(row)!=2 for row in texels):raise ValueError('2x2 texture cell required')
    n=len(texels[0][0])
    if not 1<=n<=4 or any(len(q)!=n for row in texels for q in row):raise ValueError('texture channels')
    tangent=[[[0]*n]*2]*2 if tangent is None else tangent
    return [sum((u.lift(texels[j][i][k],tangent=tangent[j][i][k])*(u if i else 1-u)*(v if j else 1-v) for j in range(2) for i in range(2)),u.lift(0)) for k in range(n)]

class RoughPrincipled:
    """Full stored-resource expression with roughness as an interval mixed jet.

    A selected roughness segment may include its closed endpoints. All internal
    slope/limiter branches must be constant on the enclosure or evaluation refuses.
    This independently constructs every angular value/control and the whole mean.
    """
    def __init__(self,data,r,*,segment=None):
        self.roughness=r;self.r4=r**4;self.pi=I(float.fromhex('0x1.921fb54442d18p+1'))
        rn=list(map(I,data['roughness_nodes']));self.cn=list(map(I,data['cosine_nodes'][1:]));self.x=list(map(I,data['_clog']));rx=list(map(I,data['_rlog']));n=len(self.x)
        if not 2<=len(rn)<=129 or not 2<=n<=512:raise ValueError('material resource size')
        if segment is None:
            choices=[i for i in range(len(rn)-1) if rn[i].lo<=r[0,0].lo<=r[0,0].hi<=rn[i+1].hi]
            if not choices:raise ValueError('roughness knot crossing requires partition')
            segment=choices[0]
        if type(segment)!=int or not 0<=segment<len(rn)-1:raise ValueError('roughness segment')
        i=segment
        if r[0,0].lo<rn[i].lo or r[0,0].hi>rn[i+1].hi:raise ValueError('roughness outside selected segment')
        width=rx[i+1]-rx[i];u=(r.ln()-rx[i])/width
        def interp(a,b,c,d):
            a,b,c,d=map(I,(a,b,c,d));return exp_jet(bernstein([a,a+width*c/3,b-width*d/3,b],u))
        def log_controls(j,c):
            a,b,da,db=map(I,(data['_logs'][i][j][c],data['_logs'][i+1][j][c],data['_slopes'][i][j][c],data['_slopes'][i+1][j][c]))
            return [a,a+width*da/3,b-width*db/3,b]
        logs=[[log_controls(j,c) for c in range(2)] for j in range(n)]
        values=[[exp_jet(bernstein(logs[j][c],u)) for c in range(2)] for j in range(n)]
        horizon=[interp(data['_hlogs'][i],data['_hlogs'][i+1],data['_hslopes'][i],data['_hslopes'][i+1]),r.lift(0)]
        x=self.x;secs=[[values[j][c]*(exp_jet(bernstein([b-a for a,b in zip(logs[j][c],logs[j+1][c])],u))-1)/(x[j+1]-x[j]) for c in range(2)] for j in range(n-1)]
        def slope(j,c):
            before,after=secs[max(0,j-1)][c],secs[min(j,n-2)][c]
            if n==2:return before
            if j==0:
                h0,h1=x[1]-x[0],x[2]-x[1];raw=(before*(2*h0+h1)-secs[1][c]*h0)/(h0+h1)
            elif j==n-1:
                h0,h1=x[-1]-x[-2],x[-2]-x[-3];raw=(after*(2*h0+h1)-secs[n-3][c]*h0)/(h0+h1)
            else:
                hp,hn=x[j]-x[j-1],x[j+1]-x[j];raw=(before*hn+after*hp)/(hp+hn)
            if before[0,0].lo>0 and after[0,0].lo>0:sign=1
            elif before[0,0].hi<0 and after[0,0].hi<0:sign=-1
            elif before[0,0].hi<0<after[0,0].lo or after[0,0].hi<0<before[0,0].lo:return r.lift(0)
            else:raise ValueError('material branch not certified: secant sign')
            bound=select(abs_jet(before),abs_jet(after),'min')*I('1.5') if before is not after else abs_jet(before)*I('1.5')
            return sign*select(select(sign*raw,0,'max'),bound,'min')
        slopes=[]
        for j in range(n):
            try:m=[slope(j,c) for c in range(2)]
            except ValueError as error:raise ValueError('angular node '+str(j)+': '+str(error)) from error
            delta=m[0]-m[1];gap=values[j][0]-values[j][1]
            if delta[0,0].lo>0:risk=delta*(x[j]-x[j-1] if j else I(0))
            elif delta[0,0].hi<0:risk=-delta*(x[j+1]-x[j] if j<n-1 else I(0))
            elif not delta.c:risk=r.lift(0)
            else:raise ValueError('material branch not certified: coupled slope')
            scale=select(r.lift(1),3*gap/risk,'min') if risk[0,0].lo>0 else r.lift(1) if not risk.c else None
            if scale is None:raise ValueError('material branch not certified: coupled risk')
            if scale[0,0].hi<1:scale=scale*(I(1)-8*I(Q(1,2**52)))
            elif scale[0,0].lo<1:raise ValueError('material branch not certified: coupled scale')
            slopes.append([v*scale for v in m])
        self.controls=[[[values[j][c],values[j][c]+slopes[j][c]*(x[j+1]-x[j])/3,values[j+1][c]-slopes[j+1][c]*(x[j+1]-x[j])/3,values[j+1][c]] for c in range(2)] for j in range(n-1)]
        self.values=values;self.horizon=horizon
        self.mean=[(horizon[c]+2*values[0][c])*self.cn[0]**2/3+sum((self.controls[j][c][k]*(2*self.cn[j+1]**2*(x[j+1]-x[j])*I(data['_weights'][j][k])) for j in range(n-1) for k in range(4)),r.lift(0)) for c in range(2)]
        if any(q[0,0].lo<=0 for q in self.mean):raise ValueError('positive mean not enclosed')
    def loss(self,c):
        value=c[0,0]
        if value.lo<=0 or value.hi>1:raise ValueError('strictly front-facing unit cosine profile')
        if value.hi<self.cn[0].lo:return [self.horizon[j]+c/self.cn[0]*(self.values[0][j]-self.horizon[j]) for j in range(2)]
        indices=[i for i in range(len(self.cn)-1) if value.lo>=self.cn[i].lo and value.hi<=self.cn[i+1].hi]
        if not indices:raise ValueError('cosine knot crossing requires qualified partition')
        i=indices[0];t=(c.ln()-self.x[i])/(self.x[i+1]-self.x[i]);return [bernstein(self.controls[i][j],t) for j in range(2)]
    def radiance(self,n,v,l,albedo,f0):
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),n[0].lift(0))
        def unit(a):
            inv=dot(a,a).sqrt().reciprocal();return [q*inv for q in a]
        nn,vv,ll=unit(n),unit(v),unit(l);hh=unit([a+b for a,b in zip(vv,ll)])
        nv,nl,nh,vh=dot(nn,vv),dot(nn,ll),dot(nn,hh),dot(vv,hh)
        for q in (nv,nl,nh,vh):q.c[0,0]=I(max(D(-1),q[0,0].lo),min(D(1),q[0,0].hi))
        if nv[0,0].lo<=0 or nl[0,0].lo<=0:raise ValueError('horizon transition requires partition')
        view,light=self.loss(nv),self.loss(nl);r4=self.r4;pi=self.pi
        distribution=r4/((1-nh)*(1+nh)+r4*nh*nh)**2/pi
        visibility=n[0].lift('.5')/(nl*(nv*nv*(1-r4)+r4).sqrt()+nv*(nl*nl*(1-r4)+r4).sqrt())
        result=[]
        for colour,f in zip(albedo,f0):
            mix=lambda z:(1-f)*z[0]+f*z[1]
            matte=colour*mix(view)*mix(light)/(mix(self.mean)*pi)
            result.append((distribution*visibility*(f+(1-f)*(1-vh)**5)+matte)*nl)
        return result


class CurveEnvelope:
    """Taylor enclosures in roughness, amortized over integration samples.

    Centre coefficients through order6 and interval seventh derivatives bound
    every evaluated coefficient via Taylor's theorem. Supports spatial order6
    plus one tangent (Gauss3); no claim for higher-order consumers.
    """
    def __init__(self,data,lo,hi,*,checkpoint=lambda:None):
        self.lo,self.hi=I(lo).lo,I(hi).hi
        self.centre=(I(lo)+I(hi))/2
        if self.centre.lo!=self.centre.hi:raise ValueError('exact centre required')
        checkpoint();self.bound=RoughPrincipled(data,RangeJet(Jet.variable(I(self.lo,self.hi),order=7),self.centre,I(self.lo,self.hi)-self.centre));checkpoint()
        self.base=object.__new__(RoughPrincipled)
        self.base.pi,self.base.cn,self.base.x=self.bound.pi,self.bound.cn,self.bound.x
        self.base.values=[[q.centre for q in row] for row in self.bound.values]
        self.base.horizon=[q.centre for q in self.bound.horizon]
        self.base.controls=[[[q.centre for q in row] for row in group] for group in self.bound.controls]
        self.base.mean=[q.centre for q in self.bound.mean];checkpoint()
    def at(self,r):
        if r.order>6:raise ValueError('roughness envelope supports Gauss3/order6 only')
        if r[0,0].lo<self.lo or r[0,0].hi>self.hi:raise ValueError('roughness outside certified envelope')
        def lift(a,b):
            d=r[0,0]-self.centre;coeff=[]
            for k in range(8):
                value=sum((a[j,0]*math.comb(j,k)*d**(j-k) for j in range(k,7)),I(0))+b[7,0]*math.comb(7,k)*d**(7-k)
                coeff.append(value)
            delta=Jet(order=r.order,coeff={k:v for k,v in r.c.items() if k!=(0,0)});result=r.lift(0);power=r.lift(1)
            for k in range(r.order+2):result=result+power*coeff[k];power=power*delta
            return result
        out=object.__new__(RoughPrincipled);out.roughness=r;out.r4=r**4;out.pi=self.base.pi;out.cn=self.base.cn;out.x=self.base.x
        class Lazy:
            def __init__(self,count,get):self.count,self.get=count,get
            def __len__(self):return self.count
            def __getitem__(self,i):return self.get(i)
        out.values=Lazy(len(self.base.values),lambda i:[lift(a,b) for a,b in zip(self.base.values[i],self.bound.values[i])])
        out.horizon=[lift(a,b) for a,b in zip(self.base.horizon,self.bound.horizon)]
        out.controls=Lazy(len(self.base.controls),lambda i:[[lift(a,b) for a,b in zip(x,y)] for x,y in zip(self.base.controls[i],self.bound.controls[i])])
        out.mean=[lift(a,b) for a,b in zip(self.base.mean,self.bound.mean)]
        return out


def perspective_boundary(triangle,inverse_w,values,threshold,*,motion=None,iw_motion=None,value_motion=None):
    """Exact moving isoline of a perspective-correct scalar field.

    Homogeneous barycentric numerators avoid rational functions. Input corner
    positions, reciprocal W and attributes follow their declared linear paths;
    their products retain ALL polynomial coefficients for the topology proof.
    Positive denominator over the pixel is established from its four corners.
    """
    from tools.opaque_moving_reference import cross,dot
    if len(triangle)!=3 or any(len(p)!=2 for p in triangle) or len(inverse_w)!=3 or len(values)!=3:raise ValueError('perspective triangle fields')
    motion=[(0,0)]*3 if motion is None else motion;iw_motion=[0]*3 if iw_motion is None else iw_motion;value_motion=[0]*3 if value_motion is None else value_motion
    points=[(P([p[0],d[0]]),P([p[1],d[1]]),P(1)) for p,d in zip(triangle,motion)]
    edges=[cross(points[(i+1)%3],points[(i+2)%3]) for i in range(3)];det=dot(points[0],edges[0])
    if not det[0]:raise ValueError('degenerate perspective face')
    sign=1 if det[0]>0 else -1;edges=[tuple(sign*p for p in edge) for edge in edges]
    iw=[P([v,d]) for v,d in zip(inverse_w,iw_motion)];value=[P([v,d])-Q(threshold) for v,d in zip(values,value_motion)]
    denominator=tuple(sum((iw[i]*edges[i][k] for i in range(3)),P(0)) for k in range(3))
    guards=[sign*det,*iw,*[denominator[0]*x+denominator[1]*y+denominator[2] for x,y in [(0,0),(1,0),(1,1),(0,1)]]]
    return dict(line=tuple(sum((value[i]*iw[i]*edges[i][k] for i in range(3)),P(0)) for k in range(3)),positive=guards)

def perspective_value(x,y,triangle,inverse_w,values,*,motion=None,iw_motion=None,value_motion=None):
    motion=[(0,0)]*3 if motion is None else motion;iw_motion=[0]*3 if iw_motion is None else iw_motion;value_motion=[0]*3 if value_motion is None else value_motion
    p=[[x.lift(v,tangent=d) for v,d in zip(q,m)] for q,m in zip(triangle,motion)];a,b,c=p
    det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);beta1=((x-a[0])*(c[1]-a[1])-(y-a[1])*(c[0]-a[0]))/det;beta2=((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))/det
    raw=[q*x.lift(v,tangent=d) for q,v,d in zip([1-beta1-beta2,beta1,beta2],inverse_w,iw_motion)];den=sum(raw,x.lift(0));a=[x.lift(v,tangent=d) for v,d in zip(values,value_motion)]
    if den[0,0].lo<=0:raise ValueError('perspective denominator')
    return a[0]+(raw[1]*(a[1]-a[0])+raw[2]*(a[2]-a[0]))/den

def texture_cell(texels,u,v,column,row,*,wrap_u='clamp',wrap_v='clamp',tangent=None):
    """Declared normalized-UV bilinear cell of one level, with exact wrap indices.

    No implicit LOD, cell discovery, derivative discontinuity averaging or gamma.
    Border/mip/anisotropic modes require separate qualification.
    """
    h=len(texels);w=len(texels[0]);channels=len(texels[0][0])
    if not 1<=h<=16 or not 1<=w<=16 or any(len(p)!=w for p in texels):raise ValueError('bounded rectangular texture')
    if wrap_u not in ('clamp','repeat') or wrap_v not in ('clamp','repeat'):raise ValueError('texture wrap profile')
    if type(column)!=int or type(row)!=int or max(abs(column),abs(row))>64:raise ValueError('texture cell range')
    tangent=[[[0]*channels for _ in range(w)] for _ in range(h)] if tangent is None else tangent
    def index(q,size,mode):return q%size if mode=='repeat' else min(size-1,max(0,q))
    cells=[];d=[]
    for j in range(row,row+2):
        yy=index(j,h,wrap_v);cells.append([texels[yy][index(i,w,wrap_u)] for i in range(column,column+2)]);d.append([tangent[yy][index(i,w,wrap_u)] for i in range(column,column+2)])
    return bilinear(cells,u*w-Q(1,2)-column,v*h-Q(1,2)-row,tangent=d)
