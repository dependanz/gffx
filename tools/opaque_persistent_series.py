"""Opt-in persistent Arb dual series for checker experiments, never production.

Keep complete spatial/first-tangent series in compiled arithmetic. Export
outward Decimal60 coefficients only on request. Ball arithmetic can widen
bounds; no endpoint identity or original-image qualification is implied.
Caller verifies the isolated python-flint0.9.0 binary manifest before entry.
"""
from contextlib import contextmanager
from collections.abc import MutableMapping
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING,MIN_EMIN,MAX_EMAX
import sys
from tools import opaque_shading_intervals as core
_BASE=core.Jet
_active=False

@contextmanager
def backend():
    global _active
    if _active:raise RuntimeError('persistent series context cannot nest')
    import flint
    from flint import arb,arb_series,fmpq,ctx
    if flint.__version__!='0.9.0':raise RuntimeError('pinned python-flint0.9.0 required')
    from tools.opaque_low_jets import LowJet
    settings=ctx.prec,ctx.cap,ctx.threads
    def ball(v):
        v=core.I(v)
        def number(x):
            return arb(str(x))
        a=number(v.lo);return a if v.lo==v.hi else a.union(number(v.hi))
    def export(v):
        if not v.is_finite():raise ValueError('finite compiled coefficient required')
        if v==0:return core.I(0)
        def endpoint(x,mode):
            m,e=x.man_exp();m,e=int(m),int(e)
            if not m:return D(0)
            with localcontext() as c:
                c.prec=60;c.Emin=MIN_EMIN;c.Emax=MAX_EMAX
                c.rounding=mode if m>0 else (ROUND_CEILING if mode==ROUND_FLOOR else ROUND_FLOOR)
                scale=D(2)**e;c.rounding=mode
                return D(m)*scale
        return core.I(endpoint(v.lower(),ROUND_FLOOR),endpoint(v.upper(),ROUND_CEILING))
    class Coefficients(MutableMapping):
        def __init__(self,owner):self.owner=owner
        def __getitem__(self,key):
            if key not in self:raise KeyError(key)
            return self.owner[key]
        def __contains__(self,key):
            i,j=key;return 0<=i<=self.owner.order and j in (0,1) and (not self.owner.low[i,j].zero if i<=self.owner.low.order else self.owner.series[j][i]!=0)
        def __iter__(self):
            for j in (0,1):
                for i in range(self.owner.order+1):
                    if (i,j) in self:yield i,j
        def __len__(self):return sum(1 for _ in self)
        def __setitem__(self,key,value):
            i,j=key
            if type(i)!=int or type(j)!=int or not 0<=i<=self.owner.order or j not in (0,1):raise ValueError('bounded dual coefficient index')
            self.owner.series[j][i]=ball(value);self.owner.cache.pop(key,None)
            if i<=self.owner.low.order:self.owner.low.c[key]=core.I(value)
            if key==(0,0):self.owner.constant=core.I(value);self.owner.cache[key]=self.owner.constant
        def __delitem__(self,key):
            if key not in self:raise KeyError(key)
            self[key]=0
    class PersistentJet(_BASE):
        def __init__(self,value=0,*,order=0,coeff=None,tangent=0):
            if type(order)!=int or not 0<=order<=10:raise ValueError('jet order')
            self.order=order;self.low=LowJet(order=min(order,2));self.series=[arb_series([],prec=order+1),arb_series([],prec=order+1)];self.cache={};self.constant=core.I(0);self.c=Coefficients(self)
            if coeff is None:self.c[0,0]=value
            else:
                for k,v in coeff.items():self.c[k]=v
            if tangent:self.c[0,1]=tangent
        @classmethod
        def make(cls,order,p,t,low):
            if any(not v[i].is_finite() for v in (p,t) for i in range(order+1)):raise ValueError('finite compiled series required')
            # The ordinary directed interval calculation independently proves this
            # value range. Intersecting that information by replacement of the
            # value coefficient is valid; higher coefficients remain outward.
            for j,series in enumerate((p,t)):
                for i in range(low.order+1):series[i]=ball(low[i,j])
            z=object.__new__(cls);z.order=order;z.series=[p,t];z.low=low;z.constant=low[0,0];z.cache={};z.c=Coefficients(z);return z
        @staticmethod
        def _from_owned(order,coeff):return PersistentJet(order=order,coeff=coeff)
        def __getitem__(self,key):
            i,j=key
            if not 0<=i<=self.order or j not in (0,1):return core.ZERO
            if i<=self.low.order:return self.low[key]
            if key not in self.cache:self.cache[key]=export(self.series[j][i])
            return self.cache[key]
        @staticmethod
        def variable(value,axis=0,order=6):
            z=PersistentJet(value,order=order);z.c[(1,0) if axis==0 else (0,1)]=1;return z
        def lift(self,value,tangent=0):return PersistentJet(value,order=self.order,tangent=tangent)
        def other(self,b):
            if not isinstance(b,_BASE):return self.lift(b)
            if b.order!=self.order:raise ValueError('jet order mismatch')
            return b if isinstance(b,PersistentJet) else PersistentJet(order=b.order,coeff=b.c)
        def __add__(self,b):
            b=self.other(b);return self.make(self.order,self.series[0]+b.series[0],self.series[1]+b.series[1],self.low+b.low)
        __radd__=__add__
        def __neg__(self):return self.make(self.order,-self.series[0],-self.series[1],-self.low)
        def __sub__(self,b):return self+-self.other(b)
        def __rsub__(self,b):return self.other(b)+-self
        def __mul__(self,b):
            b=self.other(b);a,t=self.series;p,u=b.series
            return self.make(self.order,a*p,t*p+a*u,self.low*b.low)
        __rmul__=__mul__
        def reciprocal(self):
            if self[0,0].lo<=0<=self[0,0].hi:raise ValueError('compiled reciprocal domain')
            p,t=self.series;v=p.inv();return self.make(self.order,v,-t*v*v,self.low.reciprocal())
        def __truediv__(self,b):return self*self.other(b).reciprocal()
        def __rtruediv__(self,b):return self.other(b)*self.reciprocal()
        def __pow__(self,n):
            if type(n)!=int or not 0<=n<=16:raise ValueError('jet power')
            z=self.lift(1)
            for _ in range(n):z=z*self
            return z
        def sqrt(self):
            if self[0,0].lo<=0:
                if all(s[i]==0 for s in self.series for i in range(self.order+1)):return self.lift(0)
                raise ValueError('compiled sqrt domain')
            p,t=self.series;v=p.sqrt();return self.make(self.order,v,t/(2*v),self.low.sqrt())
        def ln(self):
            if self[0,0].lo<=0:raise ValueError('compiled logarithm domain')
            p,t=self.series;return self.make(self.order,p.log(),t/p,self.low.ln())
    try:
        _active=True;ctx.prec=256;ctx.cap=11;ctx.threads=1
        for name,module in list(sys.modules.items()):
            if name.startswith('tools.') and module is not None and name!=__name__:
                for key,value in list(vars(module).items()):
                    if value is _BASE:setattr(module,key,PersistentJet)
        yield PersistentJet
    finally:
        for name,module in list(sys.modules.items()):
            if name.startswith('tools.') and module is not None and name!=__name__:
                for key,value in list(vars(module).items()):
                    if value is PersistentJet:setattr(module,key,_BASE)
        ctx.prec,ctx.cap,ctx.threads=settings;_active=False
