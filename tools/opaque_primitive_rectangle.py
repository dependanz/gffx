"""Conservative rectangle correction from exact angular primitives.

Each linear residual term uses a smooth OTHER base branch. The residual product
is bounded separately over the complete box. Unknown directional monotonicity
keeps the complete range; it never removes a term. Outer integration currently
uses a range enclosure, so this helper is experimental and may be too wide.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_angular_primitives import Primitives
from tools.opaque_primitive_parts import terms

class Rectangle:
    def __init__(self,correction,*,order=3):
        if type(order)!=int or not 1<=order<=6:raise ValueError("primitive order1..6 required")
        self.order=order
        self.c=correction;self.primitive=Primitives(correction.range_index or correction.extension,max_order=order,checkpoint=correction.checkpoint)
        correction.range_index=self.primitive.index;self.index=self.primitive.index;self.last=None
    def __call__(self,box,*,axis=0):return self.evaluate(box,axis=axis,line=False)
    def line(self,box,*,axis=0):return self.evaluate(box,axis=axis,line=True)
    def trace(self,box,*,axis=0):
        """Enclose 1D integrals for every transverse coordinate in its interval."""
        return self.evaluate(box,axis=axis,line=True,enclosed=True)
    def evaluate(self,box,*,axis=0,line=False,enclosed=False):
        if type(axis)!=int or axis not in (0,1):raise ValueError('two-dimensional axis required')
        c=self.c;e=c.extension;box=tuple(map(I,box));h=[I(v.hi)-I(v.lo) for v in box]
        if line:
            if h[axis].lo<=0 or (not enclosed and not h[1-axis].zero):raise ValueError('positive line and exact transverse point required')
            h[1-axis]=I(1)
        elif any(v.lo<=0 for v in h):raise ValueError('positive rectangle required')
        def inputs(position):
            args=[Jet(v,order=self.order+1) for v in box];args[axis]=Jet.variable(position,order=self.order+1);c.checkpoint();return c.inputs(*args)
        vals=inputs(box[axis]);delta=[c.mix(self.index.delta(q[0,0],base)) for q,base in zip(vals[:2],e.indices)]
        prefactor=vals[1]*vals[3]/(vals[0].lift(e.pi)*vals[0].lift(c.mean));cross=delta[0]*delta[1]*prefactor[0,0]*h[0]*h[1];parts=[];details=[]
        for role in [0,1]:
            c.checkpoint();base=e.indices[role]
            def weight(v):return v[1]*v[3]*c.mix([self.index.base_jet(v[1-role],e.indices[1-role],ch) for ch in range(2)])/(v[0].lift(e.pi)*v[0].lift(c.mean))
            if delta[role].zero:parts.append(I(0));continue
            q=vals[role];g=weight(vals)
            if q[1,0].lo<=0<=q[1,0].hi:parts.append(delta[role]*g[0,0]*h[0]*h[1]);continue
            anchor=I((Q(q[0,0].lo)+Q(q[0,0].hi))/2)
            cache={}
            def primitive(z,n):
                key=(z.lo,z.hi)
                if key not in cache:
                    if len(cache)>=3:raise ValueError("bounded box and two-edge primitive reuse required")
                    cache[key]=[c.mix(row) for row in self.primitive.residual_family(z,anchor,self.order,base)]
                return cache[key][n-1]
            _,remainder=terms(q,g,primitive,order=self.order,checkpoint=c.checkpoint)
            boundary=[]
            for edge in [box[axis].lo,box[axis].hi]:
                v=inputs(I(edge));boundary.append(terms(v[role],weight(v),primitive,order=self.order,checkpoint=c.checkpoint)[0])
            parts.append((boundary[1]-boundary[0]+remainder*h[axis])*h[1-axis]);details.append(dict(role=role,boundaries=boundary,remainder=remainder*h[axis],integrated=parts[-1]))
        self.last=dict(linear=parts,cross=[I(0),cross],details=details,line=line);return [I(0),sum(parts,cross)]
