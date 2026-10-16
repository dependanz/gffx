"""I_outer I_inner f = Q_outer(I_inner f) - E_outer(I_inner f).

Analytic line intervals enclose I_inner f at ALL uncertain Gauss node points.
The complete independently bounded outer Peano error integrates every ordinary
and jump term over the inner domain. It cannot be discarded just because the
inner integration is accurate. No old inner quadrature samples are reused.
"""
from tools.opaque_shading_intervals import I
from tools.opaque_primitive_rectangle import Rectangle
from tools.opaque_high_transport import TransportMaterial
from tools.opaque_boundary_material import BoundaryMaterial
from tools.opaque_high_peano import high_error
from tools.opaque_peano import gauss5

class Hybrid:
    def __init__(self,correction,*,order=6,outer="transport"):
        if outer not in ("transport","boundary"):raise ValueError("qualified outer proof required")
        self.c=correction;self.lines=Rectangle(correction,order=order);self.outer=(TransportMaterial if outer=="transport" else BoundaryMaterial)(correction,order=6);self.last=None
    def __call__(self,box,*,axis=0):
        if type(axis)!=int or axis not in (0,1):raise ValueError('two-dimensional inner axis required')
        box=tuple(map(I,box));h=[I(v.hi)-I(v.lo) for v in box]
        if len(box)!=2 or any(v.lo<=0 for v in h):raise ValueError('positive rectangle required')
        c=self.c;outer=1-axis;nodes,weights=gauss5();values=[I(0),I(0)];traces=[]
        for node,w in zip(nodes,weights):
            c.checkpoint();line=list(box);line[outer]=I(box[outer].lo)+h[outer]*node;v=self.lines.trace(tuple(line),axis=axis);traces.append(v);values=[a+b*w*h[outer] for a,b in zip(values,v)]
        rows=self.outer.integrated_bounds(box,outer);errors=[high_error(r,box[outer],h[axis],checkpoint=c.checkpoint,integrated=True) for r in rows]
        self.last=dict(traces=traces,quadrature=values,outer_error=errors);return [a-b for a,b in zip(values,errors)]
