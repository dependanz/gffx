"""Experimental complete range, optionally intersected with Gauss8 transport.

No estimated error admits a result. Small range or a wide angular region may
skip the optional proof; all acceptance uses the retained outward enclosure.
The fixed thirty-two-knot threshold is a work selector, not a discarded contribution.
"""
from bisect import bisect_left,bisect_right
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_high_transport import TransportMaterial

class Correction:
    def __init__(self,correction,*,tolerance,integrated_axis=0):
        if type(integrated_axis)!=int or integrated_axis not in (0,1):raise ValueError('tensor axis0or1')
        self.tolerance=I(tolerance)
        if self.tolerance.lo<=0:raise ValueError('positive tolerance')
        self.c=correction;self.proof=TransportMaterial(correction,order=6);self.axis=integrated_axis
        self.error_breakdown=None;self.preferred_split_axis=None;self.used_high=0;self.range_only=0;self.used_aggregate=0
    def __call__(self,box):
        self.c.checkpoint();self.error_breakdown=None;self.preferred_split_axis=getattr(self.c,'domain_split_axis',None)
        area=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))
        values=self.proof.ordinary(Jet(box[0]),Jet(box[1]));result=[q[0,0]*area for q in values]
        if all(q.width<=(2*self.tolerance*area).lo for q in result):self.range_only+=1;return result
        if self.knot_count(box)>32:
            from tools.opaque_aggregate_peano import AggregateMaterial
            proof=AggregateMaterial(self.c);kwargs={};self.used_aggregate+=1
        else:proof=self.proof;kwargs=dict(integrated_axis=self.axis);self.used_high+=1
        try:other=proof(box,**kwargs)
        except ValueError as e:
            if str(e)!='aggregate angular crossings need certified monotonicity':raise
            self.range_only+=1;return result
        if len(other)!=len(result):raise ValueError('channel count')
        self.preferred_split_axis=getattr(self.c,'preferred_split_axis',None)
        bounds=[]
        for a,b in zip(result,other):
            lo=max(a.lo,b.lo);hi=min(a.hi,b.hi)
            if lo>hi:raise ValueError('disjoint range and quadrature evidence')
            bounds.append(I(lo,hi))
        return bounds

    def knot_count(self,box):
        cosines=self.c.inputs(Jet(box[0]),Jet(box[1]))[:2];index=self.proof.index
        return sum(bisect_right(index.lower[:-1],v[0,0].hi)-bisect_left(index.upper[:-1],v[0,0].lo) for v in cosines)
