"""Opt-in fixed-geometry correction using centered slopes and the first coordinate first.

The C2 mean-value slope enclosure is supplied by CenteredInputs. It changes
only derivative bounds, never the integrand, coverage, error share or acceptance.
A complete conditioned identity owns the returned interval. Only known domain
refusals fall back to the historical correction; cancellation and inconsistent
proof evidence propagate. All temporary hooks are restored before fallback.
"""
from tools.opaque_centered_gradient import CenteredInputs
from tools.opaque_peano_material import MaterialCorrection

_DOMAIN_REFUSALS=frozenset(('bounded angular segment count',
    'bounded angular crossing count','curved angular crossings need certified monotonicity'))

class CenteredMaterialCorrection(MaterialCorrection):
    def __call__(self,box):
        original=self.inputs;axis=self.integrated_axis;adaptive=self.adaptive_order
        wrapped=CenteredInputs(original,checkpoint=self.checkpoint)
        self.error_breakdown=None;self.preferred_split_axis=None
        try:
            self.inputs=wrapped;self.integrated_axis=0;self.adaptive_order=False
            try:return self._certified_integral(box)
            except ValueError as error:
                if str(error) not in _DOMAIN_REFUSALS:raise
        finally:
            self.inputs=original;self.integrated_axis=axis;self.adaptive_order=adaptive
        return super().__call__(box)
