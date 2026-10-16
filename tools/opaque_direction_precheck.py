"""Cheap direction work selection, never an integration certificate.

The outer transverse proof needs fixed angular monotonicity when knots overlap.
An uncertain relevant slope skips that expensive attempt; the retained complete
radial/direct proof still owns correctness. A positive precheck proves no integral.
"""
from bisect import bisect_left,bisect_right
from tools.opaque_shading_intervals import Jet
from tools.opaque_centered_gradient import CenteredInputs
from tools.opaque_value_tail import ValueTailBoundary

class DirectionBoundary(ValueTailBoundary):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.prechecks=[]
    def candidate_directions(self,box):
        c=self.c;c.checkpoint();inputs=CenteredInputs(c.inputs,checkpoint=c.checkpoint,refine_values=True)
        values=inputs(Jet(box[0],order=1),Jet.variable(box[1],order=1));index=self.proof.index
        for role,q in enumerate(values[:2]):
            d=q[1,0];v=q[0,0]
            if not d.zero and d.lo<0<d.hi and bisect_left(index.upper[:-1],v.lo)!=bisect_right(index.lower[:-1],v.hi):
                self.prechecks.append(dict(transverse=False,role=role));return (0,)
        self.prechecks.append(dict(transverse=True));return (1,0)
