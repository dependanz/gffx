"""Opt-in outward binary64 convolution above spatial order two.

Trust: IEEE binary64 basic operations and math.nextafter, plus Decimal exact
conversion. Every product/add is widened separately; no FMA, threshold or
nonzero deletion. Unsupported input/output ranges fall back to the original
60-digit directed implementation. Low spatial coefficients (including their
parameter tangent) use that original convolution in original iteration order.
This global patch is only for one isolated serial qualification worker.
"""
from contextlib import contextmanager
from decimal import Decimal as D
import math,sys
from tools.opaque_shading_intervals import I,Jet
_active=False

class _Range(Exception):pass

def _endpoint(d,up):
    f=float(d)
    if not math.isfinite(f) or (f==0 and d!=0):raise _Range
    q=D.from_float(f)
    if (up and q<d) or (not up and q>d):f=math.nextafter(f,math.inf if up else -math.inf)
    if not math.isfinite(f):raise _Range
    return f

def _widen(value,up):
    result=math.nextafter(value,math.inf if up else -math.inf)
    if not math.isfinite(result):raise _Range
    return result

@contextmanager
def binary_derivatives():
    global _active
    if _active:raise RuntimeError('nested binary derivative context')
    if (sys.float_info.radix,sys.float_info.mant_dig,sys.float_info.rounds)!=(2,53,1):raise RuntimeError('binary64 nearest rounding required')
    half=2.0**-53
    if 1.0+half!=1.0 or -1.0-half!=-1.0:raise RuntimeError('nearest rounding canary failed')
    original,reverse=Jet.__mul__,Jet.__rmul__
    stats=dict(fast=0,fallback=0,ordinary=0)
    def multiply(a,b):
        b=a.other(b)
        if a.order<6 or len(a.c)*len(b.c)<36:
            stats['ordinary']+=1;return original(a,b)
        try:
            aa=[(i,j,_endpoint(v.lo,False),_endpoint(v.hi,True)) for (i,j),v in a.c.items()]
            bb=[(i,j,_endpoint(v.lo,False),_endpoint(v.hi,True)) for (i,j),v in b.c.items()]
            out={}
            for i,j,al,ah in aa:
                for k,l,bl,bh in bb:
                    if i+k<3 or i+k>a.order or j+l>1:continue
                    key=(i+k,j+l)
                    if al==ah==0 or bl==bh==0:continue
                    products=(al*bl,al*bh,ah*bl,ah*bh)
                    lo=_widen(min(products),False);hi=_widen(max(products),True)
                    if key in out:
                        pl,ph=out[key];lo=_widen(pl+lo,False);hi=_widen(ph+hi,True)
                    out[key]=(lo,hi)
            lowa=Jet._from_owned(2,{k:v for k,v in a.c.items() if k[0]<=2})
            lowb=Jet._from_owned(2,{k:v for k,v in b.c.items() if k[0]<=2})
            result=original(lowa,lowb).c
            result.update({k:I(D.from_float(lo),D.from_float(hi)) for k,(lo,hi) in out.items()})
        except (OverflowError,_Range):
            stats['fallback']+=1;return original(a,b)
        stats['fast']+=1
        return Jet._from_owned(a.order,result)
    _active=True;Jet.__mul__=Jet.__rmul__=multiply
    try:yield stats
    finally:Jet.__mul__,Jet.__rmul__=original,reverse;_active=False
