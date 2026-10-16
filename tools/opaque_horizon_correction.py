"""Explicit aggregate proof for a box crossing the first angular knot.

A horizon base plus a certified crossing range selects an existing complete
aggregate Peano identity, even when the box has fewer than eight segments.
The finite-table residual representation keeps the linear horizon branch,
all ordinary derivatives, every jump and both cosine-product factors.
Selection changes neither coverage nor acceptance. No small term is dropped.
"""
from tools.opaque_shading_intervals import Jet,Unresolved
from tools.opaque_centered_gradient import CenteredInputs
from tools.opaque_aggregate_peano import AggregateMaterial
from tools.opaque_directional_correction import DirectionalMaterialCorrection

class HorizonMaterialCorrection(DirectionalMaterialCorrection):
    def __call__(self,box):
        self.checkpoint();original=self.inputs
        values=original(Jet(box[0]),Jet(box[1]));first=self.extension.cn[0]
        near=any(base==-1 and q[0,0].lo<=first.hi and q[0,0].hi>=first.lo for base,q in zip(self.extension.indices,values[:2]))
        if not near:return super().__call__(box)
        try:
            self.inputs=CenteredInputs(original,checkpoint=self.checkpoint)
            proof=AggregateMaterial(self)
            for axis in (0,1):
                self.checkpoint();self.error_breakdown=None;self.preferred_split_axis=None
                try:
                    result=proof(box,integrated_axis=axis)
                    self.last_integrated_axis=axis;self.last_method='horizon-aggregate'
                    return result
                except Unresolved:raise
                except ValueError as error:
                    if str(error)!='aggregate angular crossings need certified monotonicity':raise
        finally:self.inputs=original
        return super().__call__(box)
