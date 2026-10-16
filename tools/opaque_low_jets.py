"""Isolated original directed recurrences for retained low spatial coefficients.

Used only by the opt-in persistent-series experiment. Do not patch this class
when selecting compiled high coefficients; it independently bounds orders0..2.
"""
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,ZERO,PRECISION,_product_endpoints
class LowJet:
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
        value=object.__new__(LowJet);value.order=order
        value.c={k:v for k,v in coeff.items() if not v.zero}
        return value
    def __getitem__(self,key):return self.c.get(key,ZERO)
    @staticmethod
    def variable(value,axis=0,order=6):
        q=LowJet(value,order=order);q.c[(1,0) if axis==0 else (0,1)]=I(1);return q
    def lift(self,value,tangent=0):return LowJet(value,order=self.order,tangent=tangent)
    def other(self,b):
        if not isinstance(b,LowJet):return self.lift(b)
        if b.order!=self.order:raise ValueError('jet order mismatch')
        return b
    def __add__(self,b):
        b=self.other(b);return LowJet._from_owned(self.order,{k:I(self[k]) if b[k].zero else self[k]+b[k] for k in self.c.keys()|b.c.keys()})
    __radd__=__add__
    def __neg__(self):return LowJet._from_owned(self.order,{k:-v for k,v in self.c.items()})
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
        return LowJet._from_owned(self.order,out)
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
        return LowJet._from_owned(self.order,out)
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
        return LowJet._from_owned(self.order,out)
    def ln(self):
        if not any(k!=(0,0) for k in self.c):return self.lift(self[0,0].ln())
        u=LowJet(order=self.order,coeff={k:v for k,v in self.c.items() if k!=(0,0)})/self.lift(self[0,0]);power=self.lift(1);out=self.lift(self[0,0].ln())
        for k in range(1,self.order+2):power=power*u;out=out+power*I(Q((-1)**(k+1),k))
        return out
