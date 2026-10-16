"""Spend the complete boundary proof only after cheap spatial subdivision."""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I
from tools.opaque_selective_boundary import SelectiveBoundary

class LateBoundary(SelectiveBoundary):
    def __init__(self,correction,*,tolerance):
        self.target=I(tolerance)
        if self.target.lo<=0 or self.target.hi>I('1e-6').hi:raise ValueError('positive bounded scheduling target')
        super().__init__(correction)
    def should_refine(self,box,old):
        x,y=box;hx=Q(x.hi)-Q(x.lo);hy=Q(y.hi)-Q(y.lo)
        if Q(x.lo)<Q(1,8) or hx>Q(1,32) or hy>Q(1,8):return False
        # This target chooses work only. The sum of complete enclosures remains
        # the sole acceptance authority; no local target discards any error.
        target=(2*self.target*I(hx*hy)).lo
        return any(z.width>target for z in old)
    def __call__(self,box,*,integrated_axis=1):
        return super().__call__(box,integrated_axis=integrated_axis)
