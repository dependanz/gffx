"""Twelfth-only radial proof with optional geometric value centering."""
from tools.opaque_tail_boundary import TailBoundary
from tools.opaque_centered_gradient import ValueCenteredBoundary

class ValueTailBoundary(TailBoundary):
    def __init__(self,correction,*,tolerance,piece):
        if type(piece) is not int or piece!=11:raise ValueError('qualified twelfth piece required')
        super().__init__(correction,tolerance=tolerance,piece=piece);self.directions=(0,)
    def __call__(self,box,*,integrated_axis=0):
        if self.proof is None:self.proof=ValueCenteredBoundary(self.c,order=6,parts=2,localize_higher=True,algebraic=True)
        return super().__call__(box,integrated_axis=integrated_axis)
