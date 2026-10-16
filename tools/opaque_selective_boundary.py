"""Opt-in complete proof intersection on bounded scheduled rectangles."""
import time
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Unresolved
from tools.opaque_boundary_material import BoundaryMaterial

def eligible(box):
    return Q(box[0].lo)>=Q(1,8) and Q(box[0].hi)-Q(box[0].lo)<=Q(1,16) and Q(box[1].hi)-Q(box[1].lo)<=Q(1,2)

class SelectiveBoundary:
    def __init__(self,correction):
        self.c=correction;self.proof=None;self.error_breakdown=None;self.events=[]
    def __getattr__(self,name):return getattr(self.c,name)
    def should_refine(self,box,old):return eligible(box)
    def __call__(self,box,*,integrated_axis=0):
        c=self.c;c.checkpoint();old=c.envelope_integral(box);prior=c.preferred_split_axis
        if not self.should_refine(box,old):return old
        start=time.perf_counter();row=dict(box=[[str(x.lo),str(x.hi)] for x in box],before=[str(x.width) for x in old]);self.events.append(row)
        try:
            if self.proof is None:self.proof=BoundaryMaterial(c,order=6,parts=2,localize_higher=True,algebraic=True)
            new=self.proof(box,integrated_axis=integrated_axis)
        except Unresolved:row['status']='stopped';raise
        except ValueError as e:
            if str(e) not in ('strict monotonicity required','aggregate angular crossings need certified monotonicity','strict boundary direction required','implicit denominator must exclude zero over the full rectangle'):raise
            row.update(status='fallback',reason=str(e));c.preferred_split_axis=prior;return old
        else:
            if len(new)!=len(old):raise ValueError('matching complete proof channels required')
            got=[]
            for a,b in zip(old,new):
                lo=max(a.lo,b.lo);hi=min(a.hi,b.hi)
                if lo>hi:raise ValueError('disjoint complete boundary proofs')
                got.append(I(lo,hi))
            row.update(status='intersected',after=[str(x.width) for x in got]);return got
        finally:row['seconds']=time.perf_counter()-start
