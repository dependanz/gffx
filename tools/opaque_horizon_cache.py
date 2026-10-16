"""Bounded, invocation-local exact reuse for a fixed horizon evaluator.

All input coefficients and the active arithmetic implementation/precision are
part of the key. Stored endpoints are immutable; callers receive fresh Jets.
No rounding, cutoff, acceptance or work-budget rule changes.
"""
import sys
from tools.opaque_shading_intervals import I,Jet
import tools.opaque_shading_intervals as scalar


class HorizonCache:
    def __init__(self,evaluate):
        if not callable(evaluate):raise ValueError('fixed evaluator required')
        self.evaluate=evaluate;self.entries={};self.hits=0;self.misses=0
    def __call__(self,x):
        if not isinstance(x,Jet):raise ValueError('Taylor jet required')
        flint=sys.modules.get('flint');context=None if flint is None else (flint.ctx.prec,flint.ctx.cap)
        key=(Jet.__mul__,Jet.reciprocal,Jet.sqrt,Jet.ln,scalar.PRECISION,context,x.order,tuple(sorted((k,v.lo,v.hi) for k,v in x.c.items())))
        if key in self.entries:self.hits+=1;coeff=self.entries[key]
        else:
            self.misses+=1;value=self.evaluate(x)
            if not isinstance(value,Jet) or value.order!=x.order:raise ValueError('matching output jet required')
            coeff=tuple((k,v.lo,v.hi) for k,v in value.c.items())
            if len(self.entries)>=64:self.entries.pop(next(iter(self.entries)))
            self.entries[key]=coeff
        return Jet(order=x.order,coeff={k:I(lo,hi) for k,lo,hi in coeff})
    def report(self):return dict(hits=self.hits,misses=self.misses,entries=len(self.entries),entry_cap=64)
