"""At most sixteen exact third-order inputs, scoped to one axis-bound call.

Input callbacks must be pure for this scope. Every jet coefficient and order
belongs to the key. Store immutable endpoints; return independent jets on hits.
Never reuse across preparation, geometry changes or separate axis-bound calls.
"""
from tools.opaque_shading_intervals import I,Jet

class InputReuse:
    def __init__(self,inputs,*,checkpoint):
        self.inputs=inputs;self.checkpoint=checkpoint;self._values={}
    @property
    def size(self):return len(self._values)
    def __call__(self,*args):
        self.checkpoint()
        if any(v.order!=3 for v in args):return self.inputs(*args)
        key=tuple((v.order,tuple((i,j,str(q.lo),str(q.hi)) for (i,j),q in sorted(v.c.items()))) for v in args)
        if key in self._values:
            return tuple(None if row is None else Jet(order=row[0],coeff={k:I(lo,hi) for k,lo,hi in row[1]}) for row in self._values[key])
        result=self.inputs(*args)
        if any(v is None and i!=2 for i,v in enumerate(result)):raise ValueError("only the unused half-vector may be absent")
        if len(self._values)<16:
            self._values[key]=tuple(None if v is None else (v.order,tuple((k,q.lo,q.hi) for k,q in v.c.items())) for v in result)
        return result
