"""Slow independent scalar reference for test fixtures, standard library only.

Not a renderer or public gffx API. Uses direct Hermite polynomials and adaptive
segment integration, not the candidate's tensor/Bezier/analytic-weight code.
Accepts explicit loss payloads; consistency with them is not physical admission.
"""
import bisect
import math


def _number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('expected finite scalar')
    return float(x)


def _range(x, low=0., high=1.):
    x = _number(x)
    if not low <= x <= high:
        raise ValueError('scalar outside declared domain')
    return x


def _vector(v):
    if len(v) != 3:
        raise ValueError('expected three components')
    return [_number(x) for x in v]


def _rgb(v):
    if isinstance(v,(list,tuple)):
        if len(v)!=3:raise ValueError('expected RGB triple')
        return [_range(x) for x in v]
    return [_range(v)]*3


def _normal(v):
    v = _vector(v)
    length = math.hypot(*v)
    if not math.isfinite(length) or length == 0:
        raise ValueError('nonrepresentable direction')
    return [x/length for x in v]


def _dot(a,b):
    value = math.fsum(x*y for x,y in zip(a,b))
    if abs(value) > 1+8*math.ulp(1.):
        raise ValueError('normalized dot outside roundoff allowance')
    return min(1., max(-1., value))


def _slopes(xs, rows):
    """Lagrange-polynomial derivatives, then explicit monotone/control bounds."""
    size=len(xs)
    secants=[[(b[k]-a[k])/(xs[i+1]-xs[i]) for k in range(2)] for i,(a,b) in enumerate(zip(rows,rows[1:]))]
    if size == 2:
        return [secants[0][:],secants[0][:]]
    slopes=[]
    for i,x in enumerate(xs):
        start=max(0,min(i-1,size-3)); indices=range(start,start+3)
        raw=[]
        for channel in range(2):
            value=0.
            for j in indices:
                others=[k for k in indices if k!=j]
                k,l=others
                value += rows[j][channel]*(2*x-xs[k]-xs[l])/((xs[j]-xs[k])*(xs[j]-xs[l]))
            before=secants[max(0,i-1)][channel]; after=secants[min(i,size-2)][channel]
            if before*after <= 0:
                value=0.
            else:
                sign=1. if before>0 else -1.
                value=sign*min(max(sign*value,0.),1.5*min(abs(before),abs(after)))
            raw.append(value)
        # Both endpoint fields must have ordered Bezier controls.
        delta=raw[0]-raw[1]; gap=rows[i][0]-rows[i][1]
        risk=delta*(xs[i]-xs[i-1]) if delta>0 and i>0 else -delta*(xs[i+1]-xs[i]) if delta<0 and i<size-1 else 0.
        scale=min(1.,3*gap/risk) if risk>0 else 1.
        if scale<1: scale *= 1-8*math.ulp(1.)
        slopes.append([x*scale for x in raw])
    return slopes


def _hermite(xs, rows, slopes, x):
    i=max(0,min(bisect.bisect_right(xs,x)-1,len(xs)-2))
    h=xs[i+1]-xs[i]; u=(x-xs[i])/h
    return [(2*u**3-3*u*u+1)*rows[i][k]+(u**3-2*u*u+u)*h*slopes[i][k]
            +(-2*u**3+3*u*u)*rows[i+1][k]+(u**3-u*u)*h*slopes[i+1][k] for k in range(2)]


def _integrate(f,a,b,tolerance=1e-12,depth=18):
    middle=(a+b)/2; fa,fm,fb=f(a),f(middle),f(b)
    whole=(b-a)*(fa+4*fm+fb)/6
    def refine(a,b,fa,fm,fb,whole,tol,left):
        mid=(a+b)/2; fl,fr=f((a+mid)/2),f((mid+b)/2)
        lo=(mid-a)*(fa+4*fl+fm)/6; hi=(b-mid)*(fm+4*fr+fb)/6
        error=lo+hi-whole
        if abs(error)<=15*tol:
            return lo+hi+error/15
        if left==0:
            raise ValueError('reference integral did not converge within work limit')
        return refine(a,mid,fa,fl,fm,lo,tol/2,left-1)+refine(mid,b,fm,fr,fb,hi,tol/2,left-1)
    return refine(a,b,fa,fm,fb,whole,tolerance,depth)


class LossCurve:
    """Copied bounded-cubic payload. No file loading, tensor code or admission flag."""
    def __init__(self,payload):
        if set(payload) != {'roughness_nodes','cosine_nodes','losses'}:
            raise ValueError('unexpected payload fields')
        r,c=payload['roughness_nodes'],payload['cosine_nodes']
        if not 2<=len(r)<=129 or not 2<=len(c)<=513:
            raise ValueError('axis work limit')
        self.r=tuple(_range(x) for x in r); self.c=tuple(_range(x) for x in c)
        if self.r[0]<=0 or self.c[0]!=0 or self.c[-1]!=1 or any(a>=b for axis in (self.r,self.c) for a,b in zip(axis,axis[1:])):
            raise ValueError('invalid axes')
        losses=payload['losses']
        if len(losses)!=len(r) or any(len(row)!=len(c) for row in losses):
            raise ValueError('loss shape')
        self.rows=tuple(tuple(tuple(_range(x) for x in pair) for pair in row) for row in losses)
        for row in self.rows:
            for j,pair in enumerate(row):
                if len(pair)!=2 or pair[0]<=0 or pair[1]>pair[0] or (j==0 and pair[1]!=0) or (j>0 and pair[1]<=0):
                    raise ValueError('invalid endpoint losses')

    def _angular(self,roughness):
        r=_range(roughness,self.r[0],self.r[-1]); xs=[math.log(x) for x in self.r]
        angular=[]
        for j in range(len(self.c)):
            # Horizon F1 is identically zero; use equal dummy fields only for log interpolation.
            rows=[[math.log(row[j][0]),math.log(row[j][1] if j else row[j][0])] for row in self.rows]
            pair=[math.exp(x) for x in _hermite(xs,rows,_slopes(xs,rows),math.log(r))]
            if j==0:pair[1]=0.
            angular.append(pair)
        logs=[math.log(x) for x in self.c[1:]]
        slopes=_slopes(logs,angular[1:]) if len(logs)>1 else None
        return angular,logs,slopes

    def loss(self,cosine,roughness,f0):
        c=_range(cosine); f=_range(f0); rows,logs,slopes=self._angular(roughness)
        if c<self.c[1] or len(logs)==1:
            u=c/self.c[1]; pair=[(1-u)*rows[0][k]+u*rows[1][k] for k in range(2)]
        else:
            pair=_hermite(logs,rows[1:],slopes,math.log(c))
        return (1-f)*pair[0]+f*pair[1]

    def mean(self,roughness,f0):
        f=_range(f0); rows,logs,slopes=self._angular(roughness)
        pair=[self.c[1]**2*(rows[0][k]+2*rows[1][k])/3 for k in range(2)]
        for a,b in zip(logs,logs[1:]):
            for k in range(2):
                pair[k]+=_integrate(lambda x:2*math.exp(2*x)*_hermite(logs,rows[1:],slopes,x)[k],a,b,1e-12/max(1,len(logs)-1))
        return (1-f)*pair[0]+f*pair[1]


def _geometry(n,v,l,two_sided):
    if not isinstance(two_sided,bool):raise ValueError('two_sided must be bool')
    n,v,l=_normal(n),_normal(v),_normal(l)
    if two_sided and _dot(n,v)<0:n=[-x for x in n]
    nv,nl=_dot(n,v),_dot(n,l)
    if nv<=0 or nl<=0:return None
    h=_normal([a+b for a,b in zip(v,l)])
    return nv,nl,_dot(n,h),_dot(v,h)


def classic(n,v,l,*,albedo,specular,exponent,two_sided=False):
    a,s=_rgb(albedo),_rgb(specular); e=_number(exponent)
    if e<0:raise ValueError('negative exponent')
    geometry=_geometry(n,v,l,two_sided)
    if geometry is None:return [0.,0.,0.]
    nh=geometry[2]
    return [x/math.pi+y*(e+8)/(8*math.pi)*nh**e for x,y in zip(a,s)]


def principled(n,v,l,*,albedo,roughness,f0,budget,two_sided=False):
    a,f=_rgb(albedo),_rgb(f0); r=_range(roughness,budget.r[0],budget.r[-1])
    r4=r**4
    if r4==0:raise ValueError('roughness divisor underflow')
    geometry=_geometry(n,v,l,two_sided)
    if geometry is None:return [0.,0.,0.]
    nv,nl,nh,vh=geometry
    d=r4/(math.pi*((1-nh*nh)+r4*nh*nh)**2)
    visibility=1/(2*(nl*math.sqrt(r4+(1-r4)*nv*nv)+nv*math.sqrt(r4+(1-r4)*nl*nl)))
    result=[]
    for colour,reflectance in zip(a,f):
        fresnel=reflectance+(1-reflectance)*(1-vh)**5
        lv,ll,mean=budget.loss(nv,r,reflectance),budget.loss(nl,r,reflectance),budget.mean(r,reflectance)
        if mean<=0:raise ValueError('invalid positive loss mean')
        result.append(d*visibility*fresnel+colour*lv*ll/(math.pi*mean))
    if not all(math.isfinite(x) and x>=0 for x in result):raise ValueError('unrepresentable BRDF')
    return result


def bilinear(texture,uv):
    """Independent half-texel repeat sampler, v=0 at the first stored row."""
    h=len(texture); w=len(texture[0]) if h else 0
    if not h or not w or any(len(row)!=w for row in texture):raise ValueError('invalid texture')
    if len(uv)!=2:raise ValueError('expected UV pair')
    x=(_number(uv[0])%1)*w-.5; y=(_number(uv[1])%1)*h-.5
    i,j=math.floor(x),math.floor(y); u,v=x-i,y-j
    corners=[_vector(texture[j%h][i%w]),_vector(texture[j%h][(i+1)%w]),
             _vector(texture[(j+1)%h][i%w]),_vector(texture[(j+1)%h][(i+1)%w])]
    return [(1-v)*((1-u)*corners[0][k]+u*corners[1][k])+v*((1-u)*corners[2][k]+u*corners[3][k]) for k in range(3)]


def direct_buffer(normals,views,covered,direction,irradiance,*,model,**parameters):
    """Shade explicit shared input buffers; no camera/raster or autodiff claim."""
    count=len(normals)
    if count>16384 or len(views)!=count or len(covered)!=count or any(type(x) is not bool for x in covered):
        raise ValueError('buffer shape or work limit')
    if not any(covered):raise ValueError('empty covered mask')
    if model not in ('classic','principled'):raise ValueError('unknown model')
    light=_normal(direction); weight=_vector(irradiance)
    if min(weight)<0:raise ValueError('negative irradiance')
    shader=classic if model=='classic' else principled
    result=[]
    for n,v,mask in zip(normals,views,covered):
        if not mask:result.append([0.,0.,0.]);continue
        cosine=max(0.,_dot(_normal(n),light))
        if parameters.get('two_sided',False) and _dot(_normal(n),_normal(v))<0:
            cosine=max(0.,-_dot(_normal(n),light))
        result.append([x*y*cosine for x,y in zip(shader(n,v,light,**parameters),weight)])
    return result
