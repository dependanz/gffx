"""Directed Decimal intervals for mathematical Gaussian transition attribution.

Basic operations round outward. Exponential uses a positive series with a
geometric tail, reciprocal and bounded squaring, independently of Decimal.exp
and the candidate's split/Horner/scaler implementation. No production imports.
"""
from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING


class Interval:
    def __init__(self,lo,hi=None,precision=100):
        self.lo=D(lo);self.hi=D(lo if hi is None else hi);self.precision=precision
        if self.lo>self.hi:raise ValueError('reversed interval')

    def rounded(self,fn,mode):
        with localcontext() as context:
            context.prec=self.precision;context.rounding=mode
            return fn()

    def _other(self,value):
        return value if isinstance(value,Interval) else Interval(value,precision=self.precision)

    def __add__(self,other):
        other=self._other(other)
        return Interval(self.rounded(lambda:self.lo+other.lo,ROUND_FLOOR),
                        self.rounded(lambda:self.hi+other.hi,ROUND_CEILING),self.precision)

    def __neg__(self):return Interval(self.hi.copy_negate(),self.lo.copy_negate(),self.precision)
    def __sub__(self,other):return self+(-self._other(other))

    def __mul__(self,other):
        other=self._other(other);pairs=[(a,b) for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        low=min(self.rounded(lambda:a*b,ROUND_FLOOR) for a,b in pairs)
        high=max(self.rounded(lambda:a*b,ROUND_CEILING) for a,b in pairs)
        return Interval(low,high,self.precision)

    def __truediv__(self,other):
        other=self._other(other)
        if other.lo<=0<=other.hi:raise ValueError('interval denominator includes zero')
        pairs=[(a,b) for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        low=min(self.rounded(lambda:a/b,ROUND_FLOOR) for a,b in pairs)
        high=max(self.rounded(lambda:a/b,ROUND_CEILING) for a,b in pairs)
        return Interval(low,high,self.precision)

    def exp_nonpositive(self):
        if self.hi>0:raise ValueError('nonpositive interval required')
        t=-self;k=0
        while t.hi>D('.5'):
            t=t/2;k+=1
            if k>18:raise ValueError('exponent outside bounded Gaussian domain')
        total=term=Interval(1,precision=self.precision)
        for j in range(1,81):
            term=term*t/j;total=total+term
        first=term*t/81
        remainder=first/(Interval(1,precision=self.precision)-t/82)
        positive=Interval(total.lo,(total+remainder).hi,self.precision)
        result=Interval(1,precision=self.precision)/positive
        for _ in range(k):result=result*result
        return result


def reference_intervals(sigma_bits,dtype,radius,*,precision=100):
    from tools.gaussian_weights_oracle import _validate
    sigma=Interval(_validate(sigma_bits,dtype,radius,precision),precision=precision)
    zero=Interval(0,precision=precision)
    one=Interval(1,precision=precision)
    if radius==0:return dict(weights=[(D(1),D(1))],derivatives=[(D(0),D(0))])
    raw=[];slopes=[]
    for j in range(-radius,radius+1):
        square=Interval(j*j,precision=precision)
        exponent=-(square/(sigma*sigma*2))
        raw.append(exponent.exp_nonpositive())
        slopes.append(square/(sigma*sigma*sigma))
    total=zero
    for q in raw:total=total+q
    weights=[q/total for q in raw]
    mu=zero
    for w,a in zip(weights,slopes):mu=mu+w*a
    derivatives=[q*(a-mu)/total for q,a in zip(raw,slopes)]
    return dict(weights=[(v.lo,v.hi) for v in weights],derivatives=[(v.lo,v.hi) for v in derivatives])
