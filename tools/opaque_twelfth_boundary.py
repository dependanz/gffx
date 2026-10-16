"""Opt-in radial boundary schedule for the saved twelfth-piece investigation.

Scheduling never accepts a region; the complete proof intersection and final
summed enclosure remain authoritative. Domains lacking monotonicity retain the
complete direct bound, including the exact horizon endpoint.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I
from tools.opaque_late_boundary import LateBoundary

class TwelfthBoundary(LateBoundary):
    def should_refine(self,box,old):
        x,y=box;hx=Q(x.hi)-Q(x.lo);hy=Q(y.hi)-Q(y.lo)
        if hx>Q(1,32) or hy>Q(1,8):return False
        target=(2*self.target*I(hx*hy)).lo
        return any(z.width>target for z in old)
    def __call__(self,box,*,integrated_axis=0):
        return super().__call__(box,integrated_axis=integrated_axis)
