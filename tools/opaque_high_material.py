"""Full ten-order residual identity with every angular knot contribution."""
import math
from bisect import bisect_left,bisect_right
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_aggregate_peano import AggregateMaterial
from tools.opaque_smooth_branch import SmoothBaseBranches
from tools.opaque_peano_material import difference
from tools.opaque_peano import crossing_location
from tools.opaque_high_peano import high_integral

class HighMaterial(AggregateMaterial):
    def __init__(self,correction,*,order=10,refine_roots=False):
        if type(order)!=int or not 2<=order<=10:raise ValueError("qualified material order2..10 required")
        if type(refine_roots)!=bool:raise ValueError("Boolean root refinement required")
        self.refine_roots=refine_roots;self.order=order
        super().__init__(correction);self.smooth=SmoothBaseBranches(correction.extension)
    def ordinary_inputs(self,u,values):
        nv,nl,_,jac=values;e=self.c.extension;mix=self.c.mix
        dv=self.index.high_delta_jet(nv,e.indices[0]);dl=self.index.high_delta_jet(nl,e.indices[1]);av=self.smooth.at(nv,e.indices[0]);bl=self.smooth.at(nl,e.indices[1])
        return [u.lift(0),nl*(mix(dv)*mix(bl)+mix(av)*mix(dl)+mix(dv)*mix(dl))*jac/(u.lift(e.pi)*u.lift(self.c.mean))]
    def actual(self,q,base):
        return [self.index.base_jet(q,base,ch)+v for ch,v in enumerate(self.index.high_delta_jet(q,base))]
    def axis_bounds(self,box,axis,*,root_refinements=1):
        if type(root_refinements)!=int or not 0<=root_refinements<=3:raise ValueError('at most three certified root refinements required')
        c=self.c;e=c.extension;index=self.index;order=self.order;c.checkpoint();args=[Jet(v,order=order) for v in box];args[axis]=Jet.variable(box[axis],order=order);values=c.inputs(*args);ordinary=self.ordinary_inputs(args[0],values)[1];jumps=[];points=None
        for role,q in enumerate(values[:2]):
            if q[1,0].zero:continue
            first=bisect_left(index.upper[:-1],q[0,0].lo);last=bisect_right(index.lower[:-1],q[0,0].hi)
            if first==last:continue
            if q[1,0].lo<0<q[1,0].hi:raise ValueError('aggregate angular crossings need certified monotonicity')
            orientation=1 if q[1,0].lo>=0 else -1
            if points is None:
                points=[]
                for x in (box[axis].lo,box[axis].hi,I((Q(box[axis].lo)+Q(box[axis].hi))/2)):
                    at=[Jet(v) for v in box];at[axis]=Jet(I(x));points.append(c.inputs(*at)[:2])
            for k in range(first,last):
                c.checkpoint();knot=index.cn[k];position=crossing_location(box[axis],points[2][role][0,0],q[1,0],knot)
                if position is None:continue
                if self.refine_roots:
                    for _ in range(root_refinements):
                        if not position.width:break
                        at=[Jet(v,order=1) for v in box];at[axis]=Jet.variable(position,order=1);derivative=c.inputs(*at)[role][1,0]
                        mid=I((Q(position.lo)+Q(position.hi))/2);at=[Jet(v) for v in box];at[axis]=Jet(mid);center=c.inputs(*at)[role][0,0]
                        position=crossing_location(position,center,derivative,knot)
                        if position is None:break
                    if position is None:continue
                at=[Jet(v,order=order-1) for v in box];at[axis]=Jet.variable(position,order=order-1);narrow=c.inputs(*at);coeff=dict(narrow[role].c);coeff[0,0]=knot;angular=Jet(order=order-1,coeff=coeff)
                delta=difference(e,angular,k,k-1);other=c.mix(self.actual(narrow[1-role],e.indices[1-role]));light=narrow[1] if role==0 else angular
                jump=light*c.mix(delta)*other*narrow[3]/(angular.lift(e.pi)*angular.lift(c.mean));left,right=(point[role][0,0] for point in points[:2]);required=(left.hi<knot.lo and right.lo>knot.hi) if orientation>0 else (right.hi<knot.lo and left.lo>knot.hi)
                jumps.append(dict(role=role,knot=k,position=position,values=[jump[j,0]*math.factorial(j)*orientation for j in range(order)],required=required))
        return [dict(order=order,derivative=I(0),jumps=[]),dict(order=order,derivative=I(ordinary[order,0].bound)*math.factorial(order),jumps=jumps)]
    def __call__(self,box,*,integrated_axis=1):
        self.c.error_breakdown=None
        def directions(rows):self.c.preferred_split_axis=0 if max(v.width for v in rows[0])>=max(v.width for v in rows[1]) else 1
        return high_integral(self.ordinary,box,self.axis_bounds,checkpoint=self.checkpoint,integrated_axis=integrated_axis,error_components=directions)
