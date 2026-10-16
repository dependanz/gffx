"""Explicit serial-worker persistent Arb endpoint experiment; never default.

Pinned python-flint0.9.0/FLINT Arb supplies outward lower()/upper() endpoints.
Keep two exact dyadic endpoints through arithmetic; Decimal60 exports are lazy
and directed. This is a different arithmetic trust boundary, not an assertion
of identical Decimal endpoints. No broad balls, approximate-zero pruning,
changed integral criterion, or renderer imports. Caller verifies binary hashes.
"""
from contextlib import contextmanager
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction as Q
import sys
from tools import opaque_shading_intervals as core
_BASE=core.I
_active=False

@contextmanager
def backend(*,series=False):
    global _active
    if type(series)!=bool:raise ValueError("explicit series selection required")
    if _active:raise RuntimeError('nested persistent endpoint context')
    import flint
    from flint import arb,arb_series,fmpq,ctx
    if flint.__version__!='0.9.0':raise RuntimeError('pinned python-flint0.9.0 required')
    settings=ctx.prec,ctx.threads,ctx.cap
    oldmul,oldreverse=core.Jet.__mul__,core.Jet.__rmul__
    saved=[];oldzero=core.ZERO
    def number(x):
        if isinstance(x,bool):raise TypeError('bool is not a reference number')
        if isinstance(x,int):return arb(x)
        q=Q(x);return arb(fmpq(q.numerator,q.denominator))
    def exported(x,mode):
        q=x.fmpq()
        with localcontext() as c:c.prec=60;c.rounding=mode;return D(int(q.p))/D(int(q.q))
    class Endpoint(_BASE):
        __slots__=('a','b','lc','hc')
        def __init__(self,lo,hi=None):
            if isinstance(lo,Endpoint) and hi is None:self.a,self.b=lo.a,lo.b
            elif isinstance(lo,_BASE) and hi is None:self.a=number(lo.lo).lower();self.b=number(lo.hi).upper()
            else:
                if hi is not None and Q(lo)>Q(hi):raise ValueError('finite ordered interval required')
                value=number(lo);self.a=value.lower();self.b=(value if hi is None else number(hi)).upper()
            if not self.a.is_finite() or not self.b.is_finite():raise ValueError('finite interval required')
            # Authored decimal endpoints are exact mathematical inputs, even
            # when their internal dyadic enclosure has nonzero width. Preserve
            # that stronger known input bound; computed results use make() and
            # never inherit this metadata. Nonterminating fractions retain the
            # ordinary directed Decimal60 input enclosure.
            if isinstance(lo,Endpoint) and hi is None:self.lc,self.hc=lo.lc,lo.hc
            elif isinstance(lo,_BASE) and hi is None:self.lc,self.hc=lo.lo,lo.hi
            else:
                known=_BASE(lo,hi);self.lc,self.hc=known.lo,known.hi
        @classmethod
        def make(cls,a,b):
            if not a.is_finite() or not b.is_finite() or a>b:raise ValueError('finite ordered computed endpoints required')
            z=object.__new__(cls);z.a=a;z.b=b;z.lc=z.hc=None;return z
        @property
        def lo(self):
            if self.lc is None:self.lc=exported(self.a,ROUND_FLOOR)
            return self.lc
        @property
        def hi(self):
            if self.hc is None:self.hc=exported(self.b,ROUND_CEILING)
            return self.hc
        @property
        def zero(self):return self.a==0 and self.b==0
        @property
        def width(self):return _BASE.rounded(lambda:self.hi-self.lo,ROUND_CEILING)
        @property
        def bound(self):return max(self.lo.copy_abs(),self.hi.copy_abs())
        def __add__(self,v):
            v=v if isinstance(v,Endpoint) else Endpoint(v)
            return self.make((self.a+v.a).lower(),(self.b+v.b).upper())
        __radd__=__add__
        def __neg__(self):return self.make(-self.b,-self.a)
        def __sub__(self,v):return self+-Endpoint(v)
        def __rsub__(self,v):return Endpoint(v)+-self
        def __mul__(self,v):
            v=v if isinstance(v,Endpoint) else Endpoint(v)
            if self.zero or v.zero:return Endpoint(0)
            if self.a>=0:
                if v.a>=0:low=self.a*v.a;high=self.b*v.b
                elif v.b<=0:low=self.b*v.a;high=self.a*v.b
                else:low=self.b*v.a;high=self.b*v.b
            elif self.b<=0:
                if v.a>=0:low=self.a*v.b;high=self.b*v.a
                elif v.b<=0:low=self.b*v.b;high=self.a*v.a
                else:low=self.a*v.b;high=self.a*v.a
            elif v.a>=0:low=self.a*v.b;high=self.b*v.b
            elif v.b<=0:low=self.b*v.a;high=self.a*v.a
            else:
                low=min((self.a*v.b).lower(),(self.b*v.a).lower())
                high=max((self.a*v.a).upper(),(self.b*v.b).upper())
            return self.make(low.lower(),high.upper())
        __rmul__=__mul__
        def __truediv__(self,v):
            v=v if isinstance(v,Endpoint) else Endpoint(v)
            if v.a<=0<=v.b:raise ValueError('interval denominator includes zero')
            return self*self.make((1/v.b).lower(),(1/v.a).upper())
        def __rtruediv__(self,v):return Endpoint(v)/self
        def __pow__(self,n):
            if type(n)!=int or not 0<=n<=16:raise ValueError('bounded nonnegative integer power required')
            z=Endpoint(1)
            for _ in range(n):z=z*self
            return z
        def sqrt(self):
            if self.a<0:raise ValueError('sqrt domain')
            return self.make(self.a.sqrt().lower(),self.b.sqrt().upper())
        def ln(self):
            if self.a<=0:raise ValueError('ln domain')
            return self.make(self.a.log().lower(),self.b.log().upper())
        def exp(self):return self.make(self.a.exp().lower(),self.b.exp().upper())
    def convolve(a,b):
        def groups(q,j):
            rows={}
            for sign in (1,-1):
                for edge in ('a','b'):
                    coeff=[]
                    for i in range(q.order+1):
                        v=Endpoint(q[i,j]);match=v.a>=0 if sign==1 else v.b<=0
                        coeff.append(getattr(v,edge) if match else arb(0))
                    if any(v!=0 for v in coeff):rows[sign,edge]=arb_series(coeff,prec=q.order+1)
            return rows
        def product(ga,gb):
            lo=arb_series([],prec=a.order+1);hi=arb_series([],prec=a.order+1)
            for sa,sb,al,bl,ah,bh in [(1,1,'a','a','b','b'),(-1,-1,'b','b','a','a'),(1,-1,'b','a','a','b'),(-1,1,'a','b','b','a')]:
                if (sa,al) in ga and (sb,bl) in gb:lo=lo+ga[sa,al]*gb[sb,bl]
                if (sa,ah) in ga and (sb,bh) in gb:hi=hi+ga[sa,ah]*gb[sb,bh]
            return lo,hi
        ap,bp=groups(a,0),groups(b,0);lo,hi=product(ap,bp);rows=[(0,lo,hi)]
        if any(j for i,j in a.c) or any(j for i,j in b.c):
            lo,hi=product(groups(a,1),bp);u,v=product(ap,groups(b,1));rows.append((1,lo+u,hi+v))
        return core.Jet._from_owned(a.order,{(i,j):Endpoint.make(lo[i].lower(),hi[i].upper()) for j,lo,hi in rows for i in range(a.order+1)})
    def multiply(a,b):
        b=a.other(b);out={}
        if series and a.order>=3 and len(a.c)*len(b.c)>=16 and not any(Endpoint(v).a<0<Endpoint(v).b for v in [*a.c.values(),*b.c.values()]):return convolve(a,b)
        for (i,j),v in a.c.items():
            for (k,l),w in b.c.items():
                if i+k>a.order or j+l>1:continue
                key=i+k,j+l;value=Endpoint(v)*w
                out[key]=out[key]+value if key in out else value
        return core.Jet._from_owned(a.order,out)
    try:
        _active=True;ctx.prec=256;ctx.threads=1;ctx.cap=11
        for name,module in list(sys.modules.items()):
            if name.startswith('tools.') and module is not None and name!=__name__:
                for key,value in list(vars(module).items()):
                    if value is _BASE:saved.append((module,key,value));setattr(module,key,Endpoint)
                    elif value is oldzero and key=='ZERO':saved.append((module,key,value));setattr(module,key,Endpoint(0))
        core.Jet.__mul__=core.Jet.__rmul__=multiply
        yield Endpoint
    finally:
        # Imports performed inside the context inherit its factory; restore them too.
        for name,module in list(sys.modules.items()):
            if name.startswith('tools.') and module is not None and name!=__name__:
                for key,value in list(vars(module).items()):
                    if value is Endpoint:setattr(module,key,_BASE)
                    elif key=='ZERO' and isinstance(value,Endpoint):setattr(module,key,oldzero)
        for module,key,value in reversed(saved):setattr(module,key,value)
        core.Jet.__mul__,core.Jet.__rmul__=oldmul,oldreverse
        ctx.prec,ctx.threads,ctx.cap=settings;_active=False
