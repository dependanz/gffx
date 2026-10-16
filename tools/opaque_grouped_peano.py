"""Signed finite-table curvature witnesses, separate from the local8segment proof.

Every active table knot contributes exactly once per angular role. Whole-line
monotonicity certifies at most one crossing; uncertain presence includes zero.
Value/slope jumps and all remaining second-product terms keep absolute bounds.
The finite table remains at most512nodes; all work is cancellable and charged.
"""
from bisect import bisect_left,bisect_right
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_aggregate_peano import AggregateMaterial
from tools.opaque_peano import crossing_location,kernel_bounds,kernel_local_range,gauss5

class GroupedMaterial(AggregateMaterial):
    def __call__(self,box,*,integrated_axis=1):
        return super().__call__(box,integrated_axis=integrated_axis)
    def signed_jumps(self,box,axis):
        c=self.c;index=self.index;c.checkpoint();args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(box[axis],order=3);values=c.inputs(*args)
        h=I(box[axis].hi)-I(box[axis].lo);kernels,_=kernel_bounds();lower=I(0);signed=I(0);points=[]
        for x in (box[axis].lo,box[axis].hi,I((Q(box[axis].lo)+Q(box[axis].hi))/2)):
            at=[Jet(v) for v in box];at[axis]=Jet(I(x));points.append(c.inputs(*at)[:2])
        for role,q in enumerate(values[:2]):
            if q[1,0].zero:continue
            sums=index.jump_sums(q[0,0]);z,s,t=[c.mix([sums[0][j],sums[1][j]]) for j in range(3)]
            if all(v.zero for v in [z,s,t]):continue
            if q[1,0].lo<0<q[1,0].hi:raise ValueError('aggregate angular crossings need certified monotonicity')
            orientation=1 if q[1,0].lo>=0 else -1
            actual=c.mix(index.actual_jet(values[1-role]));p=values[1]*actual*values[3]/(q.lift(c.extension.pi)*q.lift(c.mean));a,b,d=[I(p[j,0].bound) for j in range(3)];u=I(q[1,0].bound);uu=2*I(q[2,0].bound)
            lower=lower+z*a*h*kernels[0]+(s*u*a+z*b)*h**2*kernels[1]+(s*(uu*a+2*u*b)+2*z*d)*h**3*kernels[2]
            first=bisect_left(index.upper[:-1],q[0,0].lo);last=bisect_right(index.lower[:-1],q[0,0].hi)
            for k in range(first,last):
                c.checkpoint();knot=index.cn[k];position=crossing_location(box[axis],points[2][role][0,0],q[1,0],knot)
                if position is None:continue
                left,right=(point[role][0,0] for point in points[:2]);required=(left.hi<knot.lo and right.lo>knot.hi) if orientation>0 else (right.hi<knot.lo and left.lo>knot.hi)
                at=[Jet(v,order=1) for v in box];at[axis]=Jet.variable(position,order=1);narrow=c.inputs(*at);angular=narrow[role];other=c.mix(index.actual_jet(narrow[1-role]));p=narrow[1]*other*narrow[3]/(angular.lift(c.extension.pi)*angular.lift(c.mean))
                jump=c.mix([index.jump_values[k][ch][2] for ch in range(2)])
                normalized=(position-I(box[axis].lo))/h;normalized=I(max(I(0).lo,normalized.lo),min(I(1).hi,normalized.hi))
                term=jump*angular[1,0]*angular[1,0]*p[0,0]*h**3*kernel_local_range(normalized,3)*orientation
                if not required:term=I(min(I(0).lo,term.lo),max(I(0).hi,term.hi))
                signed=signed+term
        return signed+I(lower.hi.copy_negate(),lower.hi)
    def axis_bounds(self,box,axis):
        rows=super().axis_bounds(box,axis);rows[1]['global_signed']=self.signed_jumps(box,axis)
        from tools.opaque_segmented_remainder import segmented_remainder
        domain=box[axis];h=I(domain.hi)-I(domain.lo);roots=[I(domain.lo)+h*x for x in gauss5()[0]]
        def third(slab):
            self.checkpoint();args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(slab,order=3);return 6*self.ordinary(*args)[1][3,0]
        result=segmented_remainder(third,domain,roots,full_range=rows[1]['third_range'],checkpoint=self.checkpoint)
        if result is not None:rows[1]['ordinary_signed']=result['value']
        return rows
    def integrated_bounds(self,box,axis):
        from tools.opaque_aggregate_transport import integrated_jumps
        from tools.opaque_smooth_peano import partitioned_ordinary
        rows=AggregateMaterial.axis_bounds(self,box,axis);rows[1]['global_integral']=integrated_jumps(self,box,axis)
        def third(sub,active):
            args=[Jet(v,order=3) for v in sub];args[active]=Jet.variable(sub[active],order=3);return 6*self.ordinary(*args)[1][3,0]
        part=partitioned_ordinary(third,box,axis,checkpoint=self.checkpoint);rows[1]['ordinary_integral']=part['effective'];rows[1]['ordinary_integral_signed']=part['signed']
        return rows
