"""Opt-in tail work scheduling; only complete interval proofs can be returned.

A local error target decides whether to spend work, never whether to accept.
Known proof-domain refusals retain the direct enclosure; other failures escape.
"""
import time
from tools.opaque_twelfth_boundary import TwelfthBoundary
from tools.opaque_centered_gradient import CenteredBoundaryMaterial
from tools.opaque_shading_intervals import I,Unresolved

class TailBoundary(TwelfthBoundary):
    def __init__(self,correction,*,tolerance,piece):
        if type(piece) is not int or piece not in (11,12):raise ValueError('qualified tail piece required')
        super().__init__(correction,tolerance=tolerance);self.directions=(1,0) if piece==11 else (0,)
    def candidate_directions(self,box):return self.directions
    def __call__(self,box,*,integrated_axis=0):
        c=self.c;c.checkpoint();old=c.envelope_integral(box);prior=c.preferred_split_axis
        if not self.should_refine(box,old):return old
        if self.proof is None:self.proof=CenteredBoundaryMaterial(c,order=6,parts=2,localize_higher=True,algebraic=True)
        for axis in self.candidate_directions(box):
            c.checkpoint();start=time.perf_counter();row=dict(axis=axis,box=[[str(x.lo),str(x.hi)] for x in box],before=[str(x.width) for x in old]);self.events.append(row)
            try:
                new=self.proof(box,integrated_axis=axis)
                if len(new)!=len(old):raise ValueError('matching complete proof channels required')
                got=[]
                for a,b in zip(old,new):
                    lo=max(a.lo,b.lo);hi=min(a.hi,b.hi)
                    if lo>hi:raise ValueError('disjoint complete boundary proofs')
                    got.append(I(lo,hi))
                row.update(status='intersected',after=[str(x.width) for x in got]);return got
            except Unresolved:row['status']='stopped';raise
            except ValueError as e:
                if str(e) not in ('strict monotonicity required','aggregate angular crossings need certified monotonicity','strict boundary direction required','implicit denominator must exclude zero over the full rectangle'):raise
                row.update(status='fallback',reason=str(e));c.preferred_split_axis=prior
            finally:row['seconds']=time.perf_counter()-start
        return old
