"""Experimental Taylor transport of the dominant curvature jump.

Only the geometry and a fixed smooth OTHER angular branch are differentiated
along the knot. The complete other-role residual is bounded separately, so its
own discontinuities never inherit a false smoothness claim. All remaining jump
orders retain the prior complete proof.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_high_transport import TransportMaterial,weighted
from tools.opaque_boundary_taylor import curve_jet,weighted_taylor,clipped_root,supports
from tools.opaque_high_peano import kernel_range
from tools.opaque_peano import crossing_location

def primary(q):return Jet(order=q.order,coeff={(i,0):v for (i,j),v in q.c.items() if not j})
def partial(q):return Jet(order=q.order,coeff={(i,0):q[i,1] for i in range(q.order+1)})

class BoundaryMaterial(TransportMaterial):
    def __init__(self,correction,*,order=6,parts=2,boundary_order=3,local_other_base=False,localize_higher=False,algebraic=False):
        if type(parts)!=int or parts not in (1,2,4):raise ValueError('one, two or four fixed proof domains required')
        if type(boundary_order)!=int or boundary_order not in (3,4):raise ValueError('qualified boundary Taylor order3or4 required')
        if type(local_other_base)!=bool:raise ValueError("explicit local extension selection required")
        if type(localize_higher)!=bool:raise ValueError("explicit higher-jump localization required")
        if type(algebraic)!=bool:raise ValueError('explicit algebraic boundary selection required')
        self.algebraic=algebraic;self.algebraic_uses=0;self.algebraic_fallbacks=0
        self.localize_higher=localize_higher
        self.local_other_base=local_other_base
        self.boundary_order=boundary_order;self.parts=parts;super().__init__(correction,order=order)
    def conditioned_bounds(self,box,axis):
        return self.axis_bounds(box,axis,root_refinements=3)
    def coefficient(self,box,axis,role,knot,x,y,jump,h,orientation,*,other_base=None):
        c=self.c;e=c.extension;index=self.index
        base=e.indices[1-role] if other_base is None else other_base
        if type(base)!=int or not -1<=base<len(index.cn)-1:raise ValueError("valid other-role base required")
        def inputs(a,b):
            args=[a,b] if axis==0 else [b,a];c.checkpoint();return c.inputs(*args)
        def normalized(a,b):
            fn=getattr(c,'role_inputs',None)
            if self.algebraic and fn is not None:
                c.checkpoint();args=[a,b] if axis==0 else [b,a];return fn(role,*args)
            return inputs(a,b)[role]
        implicit=getattr(c,'implicit_inputs',None)
        yy=None
        if self.algebraic and implicit is not None:
            def equation(a,b):
                c.checkpoint();args=[a,b] if axis==0 else [b,a];return implicit(role,knot,*args)
            try:yy=curve_jet(equation,x,y,order=self.boundary_order,checkpoint=c.checkpoint)
            except ValueError as error:
                if str(error) not in ('positive unit algebraic knot required','positive algebraic boundary branch required','implicit denominator must exclude zero over the full rectangle'):raise
                self.algebraic_fallbacks+=1
            else:self.algebraic_uses+=1
        if yy is None:yy=curve_jet(normalized,x,y,order=self.boundary_order,checkpoint=c.checkpoint)
        xx=Jet.variable(x,order=self.boundary_order);xx.c[0,1]=I(1);vals=inputs(xx,yy);qx=partial(vals[role]);vals=[primary(v) if v is not None else None for v in vals]
        xx=Jet.variable(x,order=self.boundary_order);yt=Jet(order=self.boundary_order,coeff=yy.c);yt.c[0,1]=I(1);qy=partial(inputs(xx,yt)[role]);ratio=qx/qy
        if ratio[0,0].hi<0:ratio=-ratio
        elif ratio[0,0].lo<=0:raise ValueError('strict boundary direction required')
        light=xx.lift(knot) if role==1 else vals[1]
        smooth=c.mix([index.base_jet(vals[1-role],base,ch) for ch in range(2)])
        factor=light*vals[3]*qx*qx*ratio/(xx.lift(e.pi)*xx.lift(c.mean))*h**4*orientation*jump
        delta=c.mix(index.delta(vals[1-role][0,0],base))
        return factor*smooth,factor[0,0]*delta
    def dominant_piece(self,box,axis,role,knot,jump,h,orientation,a,b,*,geometry=None):
        c=self.c;other=1-axis
        if a.lo==b.hi:return I(0)
        if a.hi>b.lo:return None
        center=I(I((Q(a.hi)+Q(b.lo))/2).lo);x=I(box[axis].lo)+h*center
        def q_at(y,order=0):
            args=[Jet(v,order=order) for v in box];args[axis]=Jet(x,order=order);args[other]=Jet.variable(y,order=order) if order else Jet(y);c.checkpoint();fn=getattr(c,'role_inputs',None);return fn(role,*args) if self.algebraic and fn is not None else c.inputs(*args)[role]
        derivative=q_at(box[other],1)[1,0]
        if derivative.lo<=0<=derivative.hi:return None
        left,right=q_at(I(box[other].lo))[0,0],q_at(I(box[other].hi))[0,0]
        exists=(left.hi<=knot.lo and right.lo>=knot.hi) if derivative.lo>0 else (right.hi<=knot.lo and left.lo>=knot.hi)
        if not exists:return None
        y=box[other]
        for _ in range(3):
            if not y.width:break
            derivative=q_at(y,1)[1,0];mid=I((Q(y.lo)+Q(y.hi))/2);yn=crossing_location(y,q_at(mid)[0,0],derivative,knot)
            if yn is None:raise ValueError('disjoint certified boundary root')
            y=yn
        support=I(a.lo,b.hi);physical=I(box[axis].lo)+h*support
        def equation(yarg):
            args=[yarg.lift(physical),yarg] if axis==0 else [yarg,yarg.lift(physical)];fn=getattr(c,'role_inputs',None);return fn(role,*args) if self.algebraic and fn is not None else c.inputs(*args)[role]
        outer_range=clipped_root(equation,box[other],knot,checkpoint=c.checkpoint)
        if geometry is not None:geometry.update(x=physical,y=outer_range)
        other_base=None
        if self.local_other_base:
            from bisect import bisect_right
            args=[Jet(physical),Jet(outer_range)] if axis==0 else [Jet(outer_range),Jet(physical)]
            c.checkpoint();other_value=c.inputs(*args)[1-role][0,0]
            midpoint=(Q(other_value.lo)+Q(other_value.hi))/2
            other_base=max(-1,min(len(self.index.cn)-2,bisect_right(self.index.lower,midpoint)-1))
        # Centre and full-support remainder MUST use the same chosen branch.
        # It need not match every true branch: the entire residual is retained.
        at,_=self.coefficient(box,axis,role,knot,x,y,jump,h,orientation,other_base=other_base)
        whole,residual=self.coefficient(box,axis,role,knot,physical,outer_range,jump,h,orientation,other_base=other_base)
        n=self.boundary_order
        dominant=weighted_taylor([at[j,0]*h**j for j in range(n)],I(whole[n,0].bound)*h**n,center,a,b,3)
        error=I(residual.bound)*I(kernel_range(support,3).bound)*(I(b.hi)-I(a.lo))
        return dominant+I(error.hi.copy_negate(),error.hi)
    def localized_higher(self,geometry,axis,role,k,knot,h,orientation,a,b):
        """All j>=3 jumps on an already certified boundary support.

        Restrict only geometry ranges; the exact angular jump and complete
        other-role branch hull retain every term, including simultaneous jumps.
        No derivative along the boundary is assumed smooth here.
        """
        if not geometry or self.order<=3:return I(0)
        import math
        from tools.opaque_peano_material import difference
        c=self.c;e=c.extension;x,y=geometry['x'],geometry['y'];n=self.order-1
        u=Jet.variable(x,order=n);v=Jet(y,order=n);args=[u,v] if axis==0 else [v,u]
        c.checkpoint();values=c.inputs(*args);q=values[role]
        u=Jet(x,order=1);v=Jet.variable(y,order=1);args=[u,v] if axis==0 else [v,u]
        c.checkpoint();qy=c.inputs(*args)[role][1,0];qx=q[1,0]
        if qy.lo<=0<=qy.hi or qx.lo<=0<=qx.hi:raise ValueError('localized jump monotonicity unavailable')
        ratio=qx/qy
        if ratio.hi<0:ratio=-ratio
        angular=Jet(order=n,coeff={**q.c,(0,0):knot})
        delta=difference(e,angular,k,k-1);other=c.mix(self.actual(values[1-role],e.indices[1-role]))
        light=values[1] if role==0 else angular
        jump=light*c.mix(delta)*other*values[3]/(angular.lift(e.pi)*angular.lift(c.mean))
        return sum((weighted(jump[j,0]*math.factorial(j)*orientation*ratio*h**(j+2),a,b,j+1) for j in range(3,self.order)),I(0))
    def integrated_bounds(self,box,axis):
        rows=super().integrated_bounds(box,axis);c=self.c;index=self.index;other=1-axis;h=I(box[axis].hi)-I(box[axis].lo);self.last_boundary_details=[]
        for w in rows[1]['jumps']:
            c.checkpoint()
            if 'transport_roots' not in w:continue
            old_roots=w['transport_roots'];roots=[];role=w['role'];inner,outer=w['transport_derivatives'];knot=index.cn[w['knot']]
            for edge in [box[other].lo,box[other].hi]:
                def equation(x):
                    args=[x,x.lift(I(edge))] if axis==0 else [x.lift(I(edge)),x];fn=getattr(c,'role_inputs',None);return fn(role,*args) if self.algebraic and fn is not None else c.inputs(*args)[role]
                root=clipped_root(equation,box[axis],knot,checkpoint=c.checkpoint);t=(root-I(box[axis].lo))/h;roots.append(I(max(I(0).lo,t.lo),min(I(1).hi,t.hi)))
            if (-outer/inner).hi<0:roots.reverse()
            a,b=roots;orientation=1 if inner.lo>0 else -1;index.jump_sums(knot);jumps=[c.mix([index.jump_values[w['knot']][ch][j] for ch in range(2)]) for j in range(3)]
            domains=[(a,b)] if self.parts==1 else supports(a,b,self.parts)
            geometries=[{} for _ in domains]
            pieces=[self.dominant_piece(box,axis,role,knot,jumps[2],h,orientation,*sub,geometry=geo) for sub,geo in zip(domains,geometries)]
            if any(v is None for v in pieces):continue
            dominant=sum(pieces,I(0))
            args=[Jet(v,order=2) for v in box];args[axis]=Jet.variable(w['position'],order=2);vals=c.inputs(*args);q=vals[role];light=Jet(order=2,coeff={**q.c,(0,0):knot}) if role==1 else vals[1]
            actual=c.mix(index.actual_jet(vals[1-role]));p=light*actual*vals[3]/(q.lift(c.extension.pi)*q.lift(c.mean))
            remaining=orientation*(jumps[1]*(2*q[2,0]*p[0,0]+2*q[1,0]*p[1,0])+jumps[0]*2*p[2,0])
            ratio=w['transport_ratio'];candidate=dominant+weighted(remaining*ratio*h**4,a,b,3)
            for j,jump in enumerate(w['values']):
                if j!=2 and not jump.zero:candidate=candidate+weighted(jump*ratio*h**(j+2),a,b,j+1)
            if self.localize_higher:
                localized=dominant+weighted(remaining*ratio*h**4,a,b,3)
                for j in (0,1):
                    if not w['values'][j].zero:localized=localized+weighted(w['values'][j]*ratio*h**(j+2),a,b,j+1)
                try:
                    for sub,geo in zip(domains,geometries):localized=localized+self.localized_higher(geo,axis,role,w['knot'],knot,h,orientation,*sub)
                except ValueError as error:
                    if str(error)!='localized jump monotonicity unavailable':raise
                else:
                    lo=max(candidate.lo,localized.lo);hi=min(candidate.hi,localized.hi)
                    if lo>hi:raise ValueError('disjoint localized higher-jump evidence')
                    candidate=I(lo,hi)
            prior=w['transverse_integral'];refined=sum((weighted(jump*ratio*h**(j+2),a,b,j+1) for j,jump in enumerate(w['values']) if not jump.zero),I(0))
            lo=max(prior.lo,candidate.lo,refined.lo);hi=min(prior.hi,candidate.hi,refined.hi)
            if lo>hi:raise ValueError('disjoint boundary Taylor evidence')
            w['transverse_integral']=I(lo,hi);self.last_boundary_details.append(dict(role=role,knot=w['knot'],old_width=str(prior.width),new_width=str(w['transverse_integral'].width),roots=[str(a),str(b)],old_roots=[str(v) for v in old_roots],refined_width=str(refined.width),taylor_width=str(candidate.width),proof_domains=self.parts))
        return rows
