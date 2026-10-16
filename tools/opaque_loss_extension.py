"""Smooth angular extension plus outward piecewise residual bounds.
Image and angle-independent material tangents only; horizons still refuse.
No high derivatives are taken across a spline knot.
"""
import math
from decimal import Decimal as D
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_principled_reference import PreparedPrincipled,bernstein

class LossExtension:
    def __init__(self,curve):self.curve=curve;self.indices=[None,None];self.cursor=0;self.coefficient_cache={}
    def __getattr__(self,name):return getattr(self.curve,name)
    def at(self,q,index):
        u=(q.ln()-q.lift(self.x[index]))/q.lift(self.x[index+1]-self.x[index])
        return [bernstein([q.lift(b) for b in controls],u) for controls in self.controls[index]]
    def at_centered(self,q,index):
        """The same cubic in log cosine, in a centered power basis.

        Difference coefficients cancel constant/linear control trends before
        interval evaluation. This changes only the enclosure, not the spline,
        knot residual, parameter tangent, or accepted tolerance. All coefficients
        are outward intervals; no small term is dropped.
        """
        if index==-1:return self.at(q,index)
        from fractions import Fraction as Q
        logarithm=q.ln()
        # Freeze an exact midpoint for this evaluation. Polynomial translation
        # is an identity for every anchor, so the anchor is not differentiated.
        anchor=I((Q(logarithm[0,0].lo)+Q(logarithm[0,0].hi))/2)
        z=logarithm-q.lift(anchor);result=[]
        for channel in range(2):
            key=('centered',index,channel,anchor.lo,anchor.hi)
            if key not in self.coefficient_cache:
                if len(self.coefficient_cache)>=128:self.coefficient_cache.pop(next(iter(self.coefficient_cache)))
                self.coefficient_cache[key]=self.power(index,channel,anchor)
            coefficients=self.coefficient_cache[key];value=q.lift(coefficients[-1])
            for c in reversed(coefficients[:-1]):value=q.lift(c)+z*value
            result.append(value)
        return result
    def loss(self,q):
        role=self.cursor;self.cursor+=1
        if role>=2:raise ValueError('exactly view/light loss roles')
        if self.indices[role] is None:
            value=q[0,0]
            if value.lo<self.cn[0].lo or value.hi>1:raise ValueError('extension horizon/range')
            self.indices[role]=min(max(sum(n.lo<=value.lo for n in self.cn)-1,0),len(self.cn)-2)
        return self.at(q,self.indices[role])
    def radiance(self,*args):
        self.cursor=0;return PreparedPrincipled.radiance(self,*args)
    def power(self,index,channel,anchor):
        b=self.controls[index][channel];h=self.x[index+1]-self.x[index]
        p=[b[0],3*(b[1]-b[0])/h,3*(b[0]-2*b[1]+b[2])/h**2,(-b[0]+3*b[1]-3*b[2]+b[3])/h**3]
        shift=anchor-self.x[index]
        return [sum((math.comb(j,k)*p[j]*shift**(j-k) for j in range(k,4)),I(0)) for k in range(4)]
    def delta(self,q,index):
        if any(j==1 and not v.zero for (i,j),v in q.c.items()):raise ValueError('angle tangent requires separate knot transport qualification')
        value=q[0,0]
        if value.lo<self.cn[0].lo or value.hi>1:raise ValueError('extension horizon/range')
        chosen=[i for i in range(len(self.cn)-1) if value.hi>=self.cn[i].lo and value.lo<=self.cn[i+1].hi]
        if not chosen or len(chosen)>8:raise ValueError('bounded angular-segment residual profile')
        result=[]
        for channel in range(2):
            pieces=[]
            for i in chosen:
                if i==index:pieces.append(I(0));continue
                # Centre at the separating knot, retaining the small value/slope
                # differences without interval cancellation against large terms.
                key=(i,index,channel)
                if key not in self.coefficient_cache:
                    anchor=self.x[i if i>index else i+1]
                    coefficients=[a-b for a,b in zip(self.power(i,channel,anchor),self.power(index,channel,anchor))]
                    self.coefficient_cache[key]=(anchor,coefficients)
                anchor,coefficients=self.coefficient_cache[key]
                variable=I(max(value.lo,self.cn[i].lo),min(value.hi,self.cn[i+1].hi)).ln()-anchor
                p=coefficients[-1]
                for coefficient in reversed(coefficients[:-1]):p=coefficient+variable*p
                pieces.append(p)
            result.append(I(min(p.lo for p in pieces),max(p.hi for p in pieces)))
        return result
    def _residual(self,q,index):return [q.lift(v) for v in self.delta(q,index)]
    def correction_radiance(self,n,v,l,albedo,f0,*,cosine_bounds=None):
        if any(i is None for i in self.indices):raise ValueError('seed smooth branches before correction')
        dot=lambda a,b:sum((p*q for p,q in zip(a,b)),n[0].lift(0))
        def unit(a):
            inverse=dot(a,a).sqrt().reciprocal();return [q*inverse for q in a]
        nn,vv,ll=unit(n),unit(v),unit(l);nv,nl=dot(nn,vv),dot(nn,ll)
        for q in (nv,nl):
            lo=max(D(-1),q[0,0].lo);hi=min(D(1),q[0,0].hi)
            if lo>hi:raise ValueError('unit-vector enclosure inconsistent')
            q.c[0,0]=I(lo,hi)
        if cosine_bounds is not None:
            for q,b in zip((nv,nl),cosine_bounds):q.c[0,0]=I(max(q[0,0].lo,b.lo),min(q[0,0].hi,b.hi))
        if nv[0,0].lo<=0 or nl[0,0].lo<=0:raise ValueError('horizon residual unqualified')
        av,bl=self.at(nv,self.indices[0]),self.at(nl,self.indices[1]);dv=self._residual(nv,self.indices[0]);dl=self._residual(nl,self.indices[1])
        result=[]
        for a,f in zip(albedo,f0):
            mix=lambda row:(1-f)*row[0]+f*row[1]
            difference=mix(dv)*mix(bl)+mix(av)*mix(dl)+mix(dv)*mix(dl)
            result.append(a*difference*nl/(n[0].lift(self.pi)*mix([n[0].lift(q) for q in self.mean])))
        return result


def bounded_cosines(field,x,y,*,directional=None,monotone=False,partial=False):
    """Intersect natural range with the two-dimensional mean-value enclosure.

    Derivatives are enclosed on the complete coordinate rectangle. This requires
    a smooth cosine field there; it does not differentiate the piecewise table.
    """
    from fractions import Fraction as Q
    x,y=I(x),I(y);mx=I((Q(x.lo)+Q(x.hi))/2);my=I((Q(y.lo)+Q(y.hi))/2)
    if type(monotone)!=bool:raise ValueError('monotone range flag must be Boolean')
    if type(partial)!=bool or (partial and not monotone):raise ValueError('partial monotonicity requires explicit monotone bounds')
    natural=field(Jet(x),Jet(y));centre=field(Jet(mx),Jet(my))
    dx=field(Jet.variable(x,order=1),Jet(y,order=1));dy=field(Jet(x,order=1),Jet.variable(y,order=1))
    if directional is not None:directional([[c[1,0].bound*(I(x.hi)-I(x.lo)),d[1,0].bound*(I(y.hi)-I(y.lo))] for c,d in zip(dx,dy)])
    out=[]
    for a,b,c,d in zip(natural,centre,dx,dy):
        mean=b[0,0]+c[1,0]*(x-mx)+d[1,0]*(y-my)
        out.append(I(max(a[0,0].lo,mean.lo),min(a[0,0].hi,mean.hi)))
    if monotone:
        # Signed first-derivative enclosures hold over the WHOLE rectangle.
        # They prove where extrema occur; evaluating corners alone proves nothing.
        corners={}
        def at(a,b):
            a,b=I(a),I(b);key=(a.lo,a.hi,b.lo,b.hi)
            if key not in corners:corners[key]=field(Jet(a),Jet(b))
            return corners[key]
        for k,(c,d) in enumerate(zip(dx,dy)):
            signs=[1 if q.lo>=0 else -1 if q.hi<=0 else 0 for q in (c[1,0],d[1,0])]
            if signs==[0,0] or (not partial and 0 in signs):continue
            # Proved axes attain extrema at endpoints. An unproved axis must
            # retain its ENTIRE interval; corner sampling is not a bound.
            low=[v if sign==0 else v.lo if sign>0 else v.hi for v,sign in zip((x,y),signs)]
            high=[v if sign==0 else v.hi if sign>0 else v.lo for v,sign in zip((x,y),signs)]
            lo=at(*low)[k][0,0].lo;hi=at(*high)[k][0,0].hi
            out[k]=I(max(out[k].lo,lo),min(out[k].hi,hi))
    return out

def correction_field(extension,triangle,**attrs):
    """Perspective residual with dependency-aware angular ranges; fixed geometry.

    Mean-value bounds restrict values only. The original angular tangent remains
    present and is still rejected by delta if nonzero.
    """
    from types import SimpleNamespace
    from tools.opaque_principled_reference import perspective_field
    def cosines(n,v,l,a,f):
        dot=lambda a,b:sum((p*q for p,q in zip(a,b)),n[0].lift(0))
        return [dot(n,z)/(dot(n,n)*dot(z,z)).sqrt() for z in (v,l)]
    clean=dict(attrs);clean.pop('tangent',None)
    angle_field=perspective_field(SimpleNamespace(radiance=cosines),triangle,**clean)
    def evaluate(x,y):
        bounds=bounded_cosines(angle_field,x[0,0],y[0,0])
        def corrected(*args):return extension.correction_radiance(*args,cosine_bounds=bounds)
        return perspective_field(SimpleNamespace(radiance=corrected),triangle,**attrs)(x,y)
    return evaluate
