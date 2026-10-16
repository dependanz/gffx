"""Bounded development-only outward intervals and validated triangle quadrature.

Three-point Gauss in each Duffy coordinate. Sixth derivatives enclose its error;
agreement between sampled grids is never used as a certificate. Decimal basic
operations round outward; correctly-rounded sqrt/ln/exp are widened by one ulp.
Trust boundary: Python Decimal, this checked implementation, and a smooth supplied
field over every accepted box. Not a production renderer or an external proof.
"""
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction as Q
import time,math
PRECISION=60

def dec(x):
    if isinstance(x,D):return x
    if isinstance(x,float):return D.from_float(x)
    if isinstance(x,bool):raise TypeError('bool is not a reference number')
    return D(x)

def _product_endpoints(a,b):
    """Monotone endpoint extrema; zero endpoints retain the original tie order.

    Directed rounding is monotone, so selecting real product extrema before
    rounding gives the same outward endpoints as rounding all four products.
    Both mixed-sign intervals need two candidates per side; no terms are dropped
    from an uncertain-sign extremum. No cache or precision change is involved.
    """
    al,ah,bl,bh=a.lo,a.hi,b.lo,b.hi
    if 0 in (al,ah,bl,bh):
        pairs=((al,bl),(al,bh),(ah,bl),(ah,bh));return pairs,pairs
    if al>0:
        if bl>0:return ((al,bl),),((ah,bh),)
        if bh<0:return ((ah,bl),),((al,bh),)
        return ((ah,bl),),((ah,bh),)
    if ah<0:
        if bl>0:return ((al,bh),),((ah,bl),)
        if bh<0:return ((ah,bh),),((al,bl),)
        return ((al,bh),),((al,bl),)
    if bl>0:return ((al,bh),),((ah,bh),)
    if bh<0:return ((ah,bl),),((al,bl),)
    return ((al,bh),(ah,bl)),((al,bl),(ah,bh))

class Interval:
    __slots__=('lo','hi')
    def __init__(self,lo,hi=None):
        if isinstance(lo,Interval):self.lo,self.hi=lo.lo,lo.hi;return
        if isinstance(lo,D):
            self.lo=lo;self.hi=lo if hi is None else hi if isinstance(hi,D) else dec(hi)
        elif isinstance(lo,Q) and hi is None:
            self.lo=self.rounded(lambda:D(lo.numerator)/D(lo.denominator),ROUND_FLOOR)
            self.hi=self.rounded(lambda:D(lo.numerator)/D(lo.denominator),ROUND_CEILING)
        else:self.lo=dec(lo);self.hi=self.lo if hi is None else dec(hi)
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo>self.hi:raise ValueError('finite ordered interval required')
    @staticmethod
    def rounded(fn,mode):
        with localcontext() as c:c.prec=PRECISION;c.rounding=mode;return fn()
    @property
    def width(self):return self.rounded(lambda:self.hi-self.lo,ROUND_CEILING)
    @property
    def bound(self):return max(self.lo.copy_abs(),self.hi.copy_abs())
    @property
    def zero(self):return self.lo==0 and self.hi==0
    def __repr__(self):return '['+str(self.lo)+','+str(self.hi)+']'
    def __add__(self,b):
        b=b if isinstance(b,Interval) else Interval(b)
        if self.zero:return Interval(b)
        if b.zero:return self
        with localcontext() as c:
            c.prec=PRECISION;c.rounding=ROUND_FLOOR;lo=self.lo+b.lo
            c.rounding=ROUND_CEILING;hi=self.hi+b.hi
        return Interval(lo,hi)
    __radd__=__add__
    def __neg__(self):return Interval(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,b):return self+-Interval(b)
    def __rsub__(self,b):return Interval(b)+-self
    def __mul__(self,b):
        b=b if isinstance(b,Interval) else Interval(b)
        if self.zero or b.zero:return Interval(0)
        # Select only mathematically possible extrema, retaining outward rounding.
        lower,upper=_product_endpoints(self,b)
        with localcontext() as c:
            c.prec=PRECISION;c.rounding=ROUND_FLOOR
            low=min(x*y for x,y in lower)
            c.rounding=ROUND_CEILING
            high=max(x*y for x,y in upper)
        return Interval(low,high)
    __rmul__=__mul__
    def __truediv__(self,b):
        b=b if isinstance(b,Interval) else Interval(b)
        if b.lo<=0<=b.hi:raise ValueError('interval denominator includes zero')
        with localcontext() as c:
            c.prec=PRECISION;c.rounding=ROUND_FLOOR;lo=1/b.hi
            c.rounding=ROUND_CEILING;hi=1/b.lo
        return self*Interval(lo,hi)
    def __rtruediv__(self,b):return Interval(b)/self
    def __pow__(self,n):
        if type(n)!=int or not 0<=n<=16:raise ValueError('bounded nonnegative integer power required')
        a=Interval(1)
        for _ in range(n):a=a*self
        return a
    def _trans(self,name):
        if name=='sqrt' and self.lo<0 or name=='ln' and self.lo<=0:raise ValueError('transcendental domain')
        with localcontext() as c:
            c.prec=PRECISION
            a=getattr(self.lo,name)();b=getattr(self.hi,name)()
            lo=a.next_minus();hi=b.next_plus()
            if name=='sqrt' and self.lo==0:lo=D(0)
        return Interval(lo,hi)
    def sqrt(self):return self._trans('sqrt')
    def ln(self):return self._trans('ln')
    def exp(self):return self._trans('exp')

I=Interval
ZERO=I(0)

class Jet:
    """Spatial Taylor coefficients through order6, crossed with one tangent."""
    def __init__(self,value=0,*,order=0,coeff=None,tangent=0):
        self.order=order
        if type(order)!=int or not 0<=order<=10:raise ValueError('jet order')
        self.c={(0,0):I(value)} if coeff is None else {k:I(v) for k,v in coeff.items()}
        if tangent:self.c[0,1]=I(tangent)
        self.c={k:v for k,v in self.c.items() if not v.zero}
    @staticmethod
    def _from_owned(order,coeff):
        # Internal arithmetic has already produced fresh intervals. Transfer
        # that dictionary without copying each interval again. External
        # coefficient dictionaries still go through the copying constructor.
        value=object.__new__(Jet);value.order=order
        value.c={k:v for k,v in coeff.items() if not v.zero}
        return value
    def __getitem__(self,key):return self.c.get(key,ZERO)
    @staticmethod
    def variable(value,axis=0,order=6):
        q=Jet(value,order=order);q.c[(1,0) if axis==0 else (0,1)]=I(1);return q
    def lift(self,value,tangent=0):return Jet(value,order=self.order,tangent=tangent)
    def other(self,b):
        if not isinstance(b,Jet):return self.lift(b)
        if b.order!=self.order:raise ValueError('jet order mismatch')
        return b
    def __add__(self,b):
        b=self.other(b);out={}
        # Same separately directed coefficient additions, one context setup.
        # Copy either zero-side branch so returned intervals stay private.
        with localcontext() as context:
            context.prec=PRECISION
            for key in self.c.keys()|b.c.keys():
                a,v=self[key],b[key]
                if v.zero:out[key]=I(a)
                elif a.zero:out[key]=I(v)
                else:
                    context.rounding=ROUND_FLOOR;lo=a.lo+v.lo
                    context.rounding=ROUND_CEILING;hi=a.hi+v.hi
                    out[key]=I(lo,hi)
        return Jet._from_owned(self.order,out)
    __radd__=__add__
    def __neg__(self):return Jet._from_owned(self.order,{k:-v for k,v in self.c.items()})
    def __sub__(self,b):return self+-self.other(b)
    def __rsub__(self,b):return self.other(b)+-self
    def __mul__(self,b):
        b=self.other(b);out={}
        # Preserve convolution order and each separate outward product/add.
        # One local context removes setup work; no Decimal fused multiply-add,
        # reassociation, precision change, or thresholding is permitted here.
        with localcontext() as ctx:
            ctx.prec=PRECISION
            for (i,j),a in self.c.items():
                for (k,l),v in b.c.items():
                    if i+k>self.order or j+l>1:continue
                    key=(i+k,j+l)
                    if a.zero or v.zero:lo=hi=D(0)
                    else:
                        lower,upper=_product_endpoints(a,v)
                        ctx.rounding=ROUND_FLOOR;lo=min(x*y for x,y in lower)
                        ctx.rounding=ROUND_CEILING;hi=max(x*y for x,y in upper)
                    prior=out.get(key)
                    if prior is not None and not prior.zero:
                        if lo==0 and hi==0:continue
                        ctx.rounding=ROUND_FLOOR;lo=prior.lo+lo
                        ctx.rounding=ROUND_CEILING;hi=prior.hi+hi
                    out[key]=I(lo,hi)
        return Jet._from_owned(self.order,out)
    __rmul__=__mul__
    def reciprocal(self):
        if not any(k!=(0,0) for k in self.c):return self.lift(I(1)/self[0,0])
        out={(0,0):I(1)/self[0,0]}
        # The absent tangent is identically zero under these recurrences.
        # Every present tangent, however tiny, keeps the original two-axis path.
        tangents=2 if any(j for i,j in self.c) else 1
        for i,j in sorted(((i,j) for i in range(self.order+1) for j in range(tangents) if i+j),key=lambda k:(sum(k),k)):
            v=ZERO
            for (k,l),a in self.c.items():
                if k+l and k<=i and l<=j:v=v+a*out.get((i-k,j-l),ZERO)
            out[i,j]=-v*out[0,0]  # Identical outward reciprocal, reused locally.
        return Jet._from_owned(self.order,out)
    def __truediv__(self,b):return self*self.other(b).reciprocal()
    def __rtruediv__(self,b):return self.other(b)*self.reciprocal()
    def __pow__(self,n):
        if type(n)!=int or not 0<=n<=16:raise ValueError('jet power')
        a=self.lift(1)
        for _ in range(n):a=a*self
        return a
    def sqrt(self):
        if not any(k!=(0,0) for k in self.c):return self.lift(self[0,0].sqrt())
        out={(0,0):self[0,0].sqrt()}
        inverse_denominator=I(1)/(2*out[0,0])
        # The absent tangent is identically zero under these recurrences.
        # Every present tangent, however tiny, keeps the original two-axis path.
        tangents=2 if any(j for i,j in self.c) else 1
        for i,j in sorted(((i,j) for i in range(self.order+1) for j in range(tangents) if i+j),key=lambda k:(sum(k),k)):
            cross=ZERO
            for (k,l),a in list(out.items()):
                if k+l and k<=i and l<=j:cross=cross+a*out.get((i-k,j-l),ZERO)
            out[i,j]=(self[i,j]-cross)*inverse_denominator
        return Jet._from_owned(self.order,out)
    def ln(self):
        if not any(k!=(0,0) for k in self.c):return self.lift(self[0,0].ln())
        u=Jet(order=self.order,coeff={k:v for k,v in self.c.items() if k!=(0,0)})/self.lift(self[0,0]);power=self.lift(1);out=self.lift(self[0,0].ln())
        for k in range(1,self.order+2):power=power*u;out=out+power*I(Q((-1)**(k+1),k))
        return out

class Unresolved(ValueError):
    def __init__(self,reason,**details):
        self.details=dict(reason=reason,**details);super().__init__('unresolved integral: '+reason+'; '+str(details))

def integrate_triangle(field,triangle,*,motion=None,tolerance='1e-12',max_cells=256,max_depth=10,checkpoint=lambda:None,gauss_points=3,correction=None,prepare=None):
    """Enclose RGB (or <=4 channels) and one analytic directional derivative.

    Exact input triangle; optional coordinate tangent differentiates its mapping
    and Jacobian. Smoothness must be established by the field; interval domain
    refusals cause subdivision and eventually a preserved unresolved result.
    """
    tol=I(tolerance)
    if tol.lo<=0 or tol.hi>D('1e-6'):raise ValueError('positive tolerance at most1e-6 required')
    if type(max_cells)!=int or not 1<=max_cells<=4096 or type(max_depth)!=int or not 0<=max_depth<=12:raise ValueError('quadrature work limits')
    if len(triangle)!=3 or any(len(p)!=2 for p in triangle):raise ValueError('triangle xy shape')
    tri=[[Q(x) for x in p] for p in triangle]
    if any(abs(x)>16 for p in tri for x in p):raise ValueError('triangle finite coordinate profile')
    velocity=[[Q(0),Q(0)] for _ in range(3)] if motion is None else [[Q(x) for x in p] for p in motion]
    if len(velocity)!=3 or any(len(p)!=2 for p in velocity) or any(abs(x)>16 for p in velocity for x in p):raise ValueError('triangle tangent profile')
    det=(tri[1][0]-tri[0][0])*(tri[2][1]-tri[0][1])-(tri[1][1]-tri[0][1])*(tri[2][0]-tri[0][0])
    if det==0:raise ValueError('degenerate integration triangle')
    sign=1 if det>0 else -1
    if gauss_points==3:
        root=(I(Q(3,5)).sqrt())/2;nodes=[I('.5')-root,I('.5'),I('.5')+root];weights=[I(Q(5,18)),I(Q(4,9)),I(Q(5,18))]
    elif gauss_points==5:
        a=(I(5)+2*I(Q(10,7)).sqrt()).sqrt()/6;b=(I(5)-2*I(Q(10,7)).sqrt()).sqrt()/6
        nodes=[I('.5')-a,I('.5')-b,I('.5'),I('.5')+b,I('.5')+a]
        u=(I(322)-13*I(70).sqrt())/1800;v=(I(322)+13*I(70).sqrt())/1800
        weights=[u,v,I(Q(64,225)),v,u]
    else:raise ValueError('qualified Gauss rule must be3 or5')
    order=2*gauss_points
    remainder_factor=I(Q(math.factorial(gauss_points)**4,(order+1)*math.factorial(order)**2))
    start=time.monotonic();pending=[(Q(0),Q(1),Q(0),Q(1),0)];total=None;count=0;accepted=0;max_seen=0;last=None;leaves=[];trace=[]
    def evaluate(s,t,fn=field,weighted=True):
        p=[[s.lift(tri[i][j],tangent=velocity[i][j]) for j in range(2)] for i in range(3)]
        u=[p[1][j]-p[0][j] for j in range(2)];v=[p[2][j]-p[0][j] for j in range(2)]
        x,y=[p[0][j]+s*u[j]+(1-s)*t*v[j] for j in range(2)]
        jac=sign*(u[0]*v[1]-u[1]*v[0])*(1-s)
        values=fn(x,y)
        if not isinstance(values,(list,tuple)) or not 1<=len(values)<=4 or any(not isinstance(q,Jet) for q in values):raise ValueError('field returns1..4 matching jets')
        return [q*jac for q in values] if weighted else values
    while pending:
        try:checkpoint()
        except Unresolved as e:raise Unresolved(e.details['reason'],cells=count,accepted=accepted,last=last,trace=trace) from e
        if time.monotonic()-start>90:raise Unresolved('local deadline',cells=count,last=last)
        if count==max_cells:raise Unresolved('cell budget',cells=count,accepted=accepted,last=last,trace=trace)
        a,b,c,d,depth=pending.pop();count+=1;max_seen=max(max_seen,depth);hs=I(b-a);ht=I(d-c)
        split_s=b-a>=d-c
        try:
            if prepare is not None:
                ss=(a+b)/2;tt=(c+d)/2
                px,py=[tri[0][j]+ss*(tri[1][j]-tri[0][j])+(1-ss)*tt*(tri[2][j]-tri[0][j]) for j in range(2)]
                prepare(px,py)  # Freeze one smooth extension for this entire cell.
            accum=None
            for ns,ws in zip(nodes,weights):
                for nt,wt in zip(nodes,weights):
                    vals=evaluate(Jet(I(a)+hs*ns),Jet(I(c)+ht*nt))
                    if accum is None:accum=[[ZERO,ZERO] for _ in vals]
                    for k,q in enumerate(vals):
                        for j in range(2):accum[k][j]=accum[k][j]+q[0,j]*ws*wt*hs*ht
            ds=evaluate(Jet.variable(I(I(a).lo,I(b).hi),order=order),Jet(I(I(c).lo,I(d).hi),order=order))
            dt=evaluate(Jet(I(I(a).lo,I(b).hi),order=order),Jet.variable(I(I(c).lo,I(d).hi),order=order))
            bounds=[];error_s=D(0);error_t=D(0)
            for k in range(len(accum)):
                channel=[]
                for j in range(2):
                    es=ht*hs**(order+1)*ds[k][order,j].bound*remainder_factor;et=hs*ht**(order+1)*dt[k][order,j].bound*remainder_factor
                    error_s=max(error_s,es.hi);error_t=max(error_t,et.hi);error=es+et
                    channel.append(accum[k][j]+I(error.hi.copy_negate(),error.hi))
                bounds.append(channel)
            base_widths=[[str(v.width) for v in q] for q in bounds];correction_widths=[]
            if correction is not None:
                # The correction needs only interval value/first-tangent bounds,
                # not high spatial derivatives. Integrate the nonnegative Duffy
                # weight exactly, and enclose the residual over the whole box.
                delta=evaluate(Jet(I(I(a).lo,I(b).hi)),Jet(I(I(c).lo,I(d).hi)),correction,False)
                if len(delta)!=len(bounds):raise ValueError('correction channel count')
                ux,uy=tri[1][0]-tri[0][0],tri[1][1]-tri[0][1]
                vx,vy=tri[2][0]-tri[0][0],tri[2][1]-tri[0][1]
                dux,duy=velocity[1][0]-velocity[0][0],velocity[1][1]-velocity[0][1]
                dvx,dvy=velocity[2][0]-velocity[0][0],velocity[2][1]-velocity[0][1]
                factor=(b-a)*(d-c)*(1-(a+b)/2)
                mass=Jet(abs(det)*factor,tangent=sign*(dux*vy+ux*dvy-duy*vx-uy*dvx)*factor)
                for k,q in enumerate(delta):
                    residual=q*mass
                    correction_widths.append([str(residual[0,j].width) for j in range(2)])
                    for j in range(2):bounds[k][j]=bounds[k][j]+residual[0,j]
            split_s=error_s>=error_t
            target=(2*tol*hs*ht).lo
            last=dict(widths=[[str(v.width) for v in q] for q in bounds],base_widths=base_widths,correction_widths=correction_widths,depth=depth)
            okay=all(v.width<=target for q in bounds for v in q)
        except ValueError as e:
            okay=False;last=dict(domain=str(e),depth=depth)
        if len(trace)<32:trace.append(dict(cell=count,box=[str(z) for z in (a,b,c,d)],okay=okay,split_s=split_s,**last))
        if okay:
            if total is None:total=[[ZERO,ZERO] for _ in bounds]
            for k,q in enumerate(bounds):
                for j in range(2):total[k][j]=total[k][j]+q[j]
            accepted+=1;leaves.append([str(x) for x in (a,b,c,d)])
        else:
            if depth==max_depth:raise Unresolved('depth budget',cells=count,last=last)
            if split_s:
                mid=(a+b)/2;pending.extend([(mid,b,c,d,depth+1),(a,mid,c,d,depth+1)])
            else:
                mid=(c+d)/2;pending.extend([(a,b,mid,d,depth+1),(a,b,c,mid,depth+1)])
    return dict(value=[q[0] for q in total],derivative=[q[1] for q in total],cells=count,accepted=accepted,max_depth=max_seen,leaves=leaves,tolerance=str(tol.lo),method='outward Decimal60 + tensor Gauss'+str(gauss_points)+' + derivative order'+str(order),gauss_points=gauss_points,seconds=time.monotonic()-start)
