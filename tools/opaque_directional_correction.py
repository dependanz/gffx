"""Explicit complete-identity selection; no extra cells or relaxed acceptance.

Inspect the full cosine enclosure before trying the local eight-segment proof.
Wider regions use the existing finite-table grouped proof, retaining every knot.
Compare two integrated error enclosures, reuse the selected rows, and complete
that identity with all five line bounds and all 25 quadrature samples. A score
selects work only; it never admits a value. Known domain refusals alone may use
the historical centered fallback. Cancellation and inconsistent evidence escape.
"""
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_centered_correction import CenteredMaterialCorrection,_DOMAIN_REFUSALS
from tools.opaque_centered_gradient import CenteredInputs
from tools.opaque_grouped_peano import GroupedMaterial
from tools.opaque_transverse_peano import conditioned_integral,_error

_REFUSALS=_DOMAIN_REFUSALS|{'aggregate angular crossings need certified monotonicity'}

class DirectionalMaterialCorrection(CenteredMaterialCorrection):
    def _wide(self,box):
        self.checkpoint();values=self.inputs(Jet(box[0]),Jet(box[1]));cn=self.extension.cn
        for q in values[:2]:
            v=q[0,0]
            if v.lo<0 or v.hi>1:raise ValueError('front unit cosine range')
            count=int(v.lo<=cn[0].hi)+sum(v.hi>=cn[i].lo and v.lo<=cn[i+1].hi for i in range(len(cn)-1))
            if count>8:return True
        return False

    def __call__(self,box):
        original=self.inputs;self.error_breakdown=None;self.preferred_split_axis=None
        self.last_method=None
        try:
            self.inputs=CenteredInputs(original,checkpoint=self.checkpoint)
            wide=self._wide(box);proof=GroupedMaterial(self) if wide else self
            candidates=[]
            for axis in (0,1):
                self.checkpoint()
                try:
                    rows=proof.integrated_bounds(box,axis)
                    h=I(box[1-axis].hi)-I(box[1-axis].lo)
                    score=max(_error(row,box[axis],h,True)[0].width for row in rows)
                    candidates.append((score,axis,rows))
                except Unresolved:raise
                except ValueError as error:
                    if str(error) not in _REFUSALS:raise
            for _,axis,rows in sorted(candidates,key=lambda row:row[0]):
                def reuse(b,a):
                    if a!=axis or any((x.lo,x.hi)!=(y.lo,y.hi) for x,y in zip(b,box)):raise ValueError('directional rows belong to one box and axis')
                    return rows
                def directions(errors):
                    self.directional_error_bounds=errors
                    self.preferred_split_axis=0 if max(v.hi for v in errors[0])>=max(v.hi for v in errors[1]) else 1
                def breakdown(parts):self.error_breakdown=parts
                self.error_breakdown=None;self.preferred_split_axis=None
                try:
                    result=conditioned_integral(proof.ordinary,box,proof.axis_bounds,reuse,checkpoint=self.checkpoint,error_components=directions,error_breakdown=breakdown,integrated_axis=axis)
                    self.last_integrated_axis=axis;self.last_method='grouped' if wide else 'localized'
                    return result
                except Unresolved:raise
                except ValueError as error:
                    if str(error) not in _REFUSALS:raise
        finally:self.inputs=original
        self.error_breakdown=None;self.preferred_split_axis=None;self.last_method='historical-fallback'
        return super().__call__(box)
