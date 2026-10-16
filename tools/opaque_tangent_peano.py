"""Piecewise residual value and ordinary first-tangent integrals.

Spatial Peano bounds include every value/slope/curvature jump of BOTH channels.
Distributional parameter transport is separate and must be added by the caller.
Only fixed radial triangles with affine homogeneous fields per coordinate fit
these root-count bounds. Curved maps require certified spatial monotonicity;
the first linear angular segment is opt-in and retains its join jumps.
"""
import math
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano_material import difference
from tools.opaque_peano import residual_integral,crossing_location


def segmented_mixed(e,q,base,*,horizon=False):
    v=q[0,0];cn=e.cn
    if v.lo<(0 if horizon else cn[0].lo) or v.hi>1:raise ValueError('strict positive table domain')
    chosen=([-1] if horizon and v.lo<=cn[0].hi else [])+[i for i in range(len(cn)-1) if v.hi>=cn[i].lo and v.lo<=cn[i+1].hi]
    if not chosen or len(chosen)>8:raise ValueError('bounded angular segment count')
    rows=[]
    for i in chosen:
        coeff=dict(q.c);coeff[0,0]=I(max(v.lo,0 if i==-1 else cn[i].lo),min(v.hi,cn[0].hi if i==-1 else cn[i+1].hi))
        rows.append(difference(e,Jet(order=q.order,coeff=coeff),i,base))
    return [Jet(order=q.order,coeff={(i,j):I(min(row[c][i,j].lo for row in rows),max(row[c][i,j].hi for row in rows)) for i in range(q.order+1) for j in (0,1)}) for c in range(2)]


def channel(q,j):return Jet(order=q.order,coeff={(i,0):q[i,j] for i in range(q.order+1)})


class MixedLoss:
    """Fixed positive channel combination, evaluated as one exact cubic.

    Bernstein/power bases are linear in their controls. Combining the two
    fixed-F0 channels before evaluation retains the identical real polynomial
    and all spatial/parameter coefficients. It changes only interval enclosure
    and arithmetic cost. No normalization, jump or tiny term is removed.
    """
    def __init__(self,extension,f):
        self.e=extension;self.f=I(f);self.cache={}
        if self.f.lo<0 or self.f.hi>1:raise ValueError('unit fixed mixing coefficient required')
    def mix(self,a,b):return (1-self.f)*a+self.f*b
    def polynomial(self,q,index,anchor):
        e=self.e
        key=(index,anchor.lo,anchor.hi,self.f.lo,self.f.hi,tuple((v.lo,v.hi) for row in e.controls[index] for v in row),e.x[index].lo,e.x[index].hi,e.x[index+1].lo,e.x[index+1].hi)
        if key not in self.cache:
            rows=[e.power(index,j,anchor) for j in (0,1)]
            coefficients=[self.mix(a,b) for a,b in zip(*rows)]
            if len(self.cache)>=128:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=tuple((v.lo,v.hi) for v in coefficients)
        return [I(*v) for v in self.cache[key]]
    @staticmethod
    def evaluate(q,z,coeff):
        result=q.lift(coeff[-1])
        for c in reversed(coeff[:-1]):result=q.lift(c)+z*result
        return result
    def at(self,q,index):
        e=self.e
        if index==-1:
            horizon=self.mix(*e.horizon);slope=self.mix(e.values[0][0]-e.horizon[0],e.values[0][1]-e.horizon[1])
            return q.lift(horizon)+q/q.lift(e.cn[0])*slope
        logarithm=q.ln();anchor=I((Q(logarithm[0,0].lo)+Q(logarithm[0,0].hi))/2)
        return self.evaluate(q,logarithm-q.lift(anchor),self.polynomial(q,index,anchor))
    def difference(self,q,actual,base):
        if actual==base:return q.lift(0)
        e=self.e
        if actual<0 or base<0:return self.at(q,actual)-self.at(q,base)
        anchor=e.x[actual if actual>base else actual+1]
        coeff=[a-b for a,b in zip(self.polynomial(q,actual,anchor),self.polynomial(q,base,anchor))]
        return self.evaluate(q,q.ln()-q.lift(anchor),coeff)
    def segmented(self,q,base,*,horizon=False):
        e=self.e;v=q[0,0];cn=e.cn
        if v.lo<(0 if horizon else cn[0].lo) or v.hi>1:raise ValueError('strict positive table domain')
        chosen=([-1] if horizon and v.lo<=cn[0].hi else [])+[i for i in range(len(cn)-1) if v.hi>=cn[i].lo and v.lo<=cn[i+1].hi]
        if not chosen or len(chosen)>8:raise ValueError('bounded angular segment count')
        rows=[]
        for index in chosen:
            coeff=dict(q.c);coeff[0,0]=I(max(v.lo,0 if index==-1 else cn[index].lo),min(v.hi,cn[0].hi if index==-1 else cn[index+1].hi))
            rows.append(self.difference(Jet(order=q.order,coeff=coeff),index,base))
        return Jet(order=q.order,coeff={(i,j):I(min(row[i,j].lo for row in rows),max(row[i,j].hi for row in rows)) for i in range(q.order+1) for j in (0,1)})


class TangentCorrection:
    """Return [value residual integral, ordinary tangent residual integral]."""
    def __init__(self,extension,inputs,*,checkpoint=lambda:None,signed=True,horizon=False,curved=False,range_fallback=False,centered=False,order=3,conditioned=False,partitioned_range=False,mixed_curve=False,transverse=False,curved_transverse=False,transverse_partial=None,transverse_ratio=None,integrated_axis=1,local_crossings=False,line_order=4,crossing_support=False,root_parts=1):
        if type(root_parts)!=int or root_parts not in (1,2,4) or root_parts>1 and not (order==4 and conditioned and transverse and crossing_support and (not curved or curved_transverse)):raise ValueError('root partition requires qualified two/four-part front profile')
        self.root_parts=root_parts;self.root_partition_queries=0
        if type(crossing_support)!=bool or crossing_support and not (order==4 and conditioned and transverse and (not curved or curved_transverse)):raise ValueError('crossing support requires conditioned explicit transverse profile')
        self.crossing_support=crossing_support
        if type(line_order)!=int or line_order not in (4,6) or line_order==6 and not (order==4 and conditioned and transverse and (not curved or curved_transverse)):raise ValueError('sixth-order lines require conditioned explicitly qualified transverse profile')
        self.line_order=line_order
        if type(local_crossings)!=bool:raise ValueError('explicit local crossing flag required')
        self.local_crossings=local_crossings
        if type(range_fallback)!=bool or range_fallback and not curved:raise ValueError("range fallback requires curved proof profile")
        if type(centered)!=bool:raise ValueError('explicit centered proof switch required')
        if type(order)!=int or order not in (3,4):raise ValueError('qualified correction order3or4 required')
        if type(conditioned)!=bool or conditioned and order!=4:raise ValueError('conditioning requires fourth-order proof')
        if type(mixed_curve)!=bool:raise ValueError("explicit mixed-curve switch required")
        self.mixed=MixedLoss(extension,I(.04)) if mixed_curve else None
        if type(curved_transverse)!=bool or curved_transverse and not (curved and transverse and order==4):raise ValueError("curved transverse requires explicit fourth-order curved proof profile")
        if type(transverse)!=bool or transverse and (order!=4 or curved and not curved_transverse):raise ValueError("transverse tangent requires fourth-order front or explicitly qualified curved profile")
        # Generic change-of-variable bounds apply to strict monotone curved
        # crossings too. The opt-in preserves axis_bounds' crossing-count
        # certification and every jump, smooth term and separate motion flux.
        self.curved_transverse=curved_transverse
        if transverse_partial is not None and (not transverse or not callable(transverse_partial)):raise ValueError("exact partial proof requires transverse profile")
        if transverse_ratio is not None and (not transverse or not callable(transverse_ratio)):raise ValueError('exact ratio proof requires transverse profile')
        self.transverse_ratio=transverse_ratio
        self.transverse=transverse;self.transverse_partial=transverse_partial
        if type(integrated_axis)!=int or integrated_axis not in (0,1) or integrated_axis!=1 and not conditioned:raise ValueError("explicit conditioned integration axis required")
        self.integrated_axis=integrated_axis
        self.conditioned=conditioned
        self.order=order
        self.centered=centered
        if type(partitioned_range)!=bool or partitioned_range and not (curved and range_fallback):raise ValueError('partitioned range requires the curved fallback')
        self.partitioned_range=partitioned_range;self.range_proof_boxes=0
        self.range_fallback=range_fallback;self.range_fallbacks=0
        if type(horizon)!=bool or type(curved)!=bool:raise ValueError("explicit horizon/curved flags required")
        self.horizon=horizon;self.curved=curved
        self.signed=signed;self.e=extension;self.inputs=inputs;self.checkpoint=checkpoint;self.f=I(.04);self.mean=self.mix(extension.mean);self.preferred_split_axis=None
    def mix(self,rows):return rows[0]*(1-self.f)+rows[1]*self.f
    def ordinary_inputs(self,u,values):
        if self.mixed is not None:
            nv,nl,_,jac=values;m=self.mixed
            dv=m.segmented(nv,self.e.indices[0],horizon=self.horizon);dl=m.segmented(nl,self.e.indices[1],horizon=self.horizon);term=u.lift(0)
            if dv.c:term=dv*m.at(nl,self.e.indices[1])
            if dl.c:
                term=term+m.at(nv,self.e.indices[0])*dl
                if dv.c:term=term+dv*dl
            return nl*term*jac/(self.e.pi*self.mean)
        nv,nl,_,jac=values;e=self.e;dv=segmented_mixed(e,nv,e.indices[0],horizon=self.horizon);dl=segmented_mixed(e,nl,e.indices[1],horizon=self.horizon)
        # Empty coefficient maps certify every mixed coefficient is exactly zero.
        # Tiny and high-order nonzeros always retain the original arithmetic.
        live_v=any(q.c for q in dv);live_l=any(q.c for q in dl);term=u.lift(0)
        if live_v:
            mdv=self.mix(dv);term=mdv*self.mix(e.at(nl,e.indices[1]))
        if live_l:
            mdl=self.mix(dl);term=term+self.mix(e.at(nv,e.indices[0]))*mdl
            if live_v:term=term+mdv*mdl
        return nl*term*jac/(e.pi*self.mean)
    def ordinary(self,u,s):
        q=self.ordinary_inputs(u,self.inputs(u,s));return [channel(q,j) for j in (0,1)]
    def partition_jump(self,support,axis,role,k,order):
        self.root_partition_queries+=1
        return self.jump_on_support(support,axis,role,k,order)

    def jump_on_support(self,support,axis,role,k,order):
        narrow=[Jet(v,order=order) for v in support];narrow[axis]=Jet.variable(support[axis],order=order);vals=self.inputs(*narrow);cross=vals[role];other=vals[1-role]
        coeff=dict(cross.c);coeff[0,0]=self.e.cn[k];at_knot=Jet(order=order,coeff=coeff)
        if self.mixed is None:
            delta=self.mix(difference(self.e,at_knot,k,k-1))
            other_actual=self.mix([a+b for a,b in zip(self.e.at(other,self.e.indices[1-role]),segmented_mixed(self.e,other,self.e.indices[1-role],horizon=self.horizon))])
        else:
            delta=self.mixed.difference(at_knot,k,k-1)
            other_actual=self.mixed.at(other,self.e.indices[1-role])+self.mixed.segmented(other,self.e.indices[1-role],horizon=self.horizon)
        light=vals[1] if role==0 else at_knot;jump=light*delta*other_actual*vals[3]/(self.e.pi*self.mean)
        return jump

    def axis_bounds(self,box,axis,*,transverse=True,proof_order=None):
        order=self.order if proof_order is None else proof_order
        if proof_order is not None and not (proof_order==6 and self.line_order==6 and not transverse):raise ValueError("qualified discrete sixth-order line required")
        args=[Jet(v,order=order) for v in box];args[axis]=Jet.variable(box[axis],order=order);values=self.inputs(*args);ordinary=self.ordinary_inputs(args[0],values);cn=self.e.cn;localized=[[],[]]
        for role,q in enumerate(values[:2]):
            if q[1,0].zero:continue
            active=[k for k in range(0 if self.horizon else 1,len(cn)-1) if q[0,0].lo<=cn[k].hi and q[0,0].hi>=cn[k].lo]
            if len(active)>8:raise ValueError('bounded angular crossing count')
            if not active:continue
            slope=q[1,0]
            if self.transverse_partial is not None:
                self.checkpoint();exact=self.transverse_partial(role,box,axis);slope=I(max(slope.lo,exact.lo),min(slope.hi,exact.hi))
            center=None
            if self.local_crossings:
                # A one-dimensional mean-value enclosure retains the ENTIRE
                # transverse interval. It can rule out a spurious knot without
                # assuming monotonicity, changing derivatives, or dropping size.
                mid=[Jet(v) for v in box];mid[axis]=Jet((I(box[axis].lo)+I(box[axis].hi))/2)
                center=self.inputs(*mid)[role][0,0]
                mean=center+slope*(box[axis]-mid[axis][0,0])
                local=I(max(q[0,0].lo,mean.lo),min(q[0,0].hi,mean.hi))
                active=[k for k in active if local.lo<=cn[k].hi and local.hi>=cn[k].lo]
                if not active:continue
            monotone=slope.lo>=0 or slope.hi<=0
            if self.curved and not monotone:raise ValueError('curved angular crossings need certified monotonicity')
            count=1 if monotone else 4 if role==0 else 2
            if center is None:
                mid=[Jet(v) for v in box];mid[axis]=Jet((I(box[axis].lo)+I(box[axis].hi))/2);center=self.inputs(*mid)[role][0,0]
            for k in active:
                self.checkpoint();loc=crossing_location(box[axis],center,slope,cn[k])
                if loc is None:continue
                # Curved quadrature lines retain the original certified root
                # enclosure and every jump. Optional Newton tightening on each
                # line costs more than it saved in the full-piece profile;
                # continuous transverse strip proofs still refine their roots.
                if self.transverse and not transverse and not self.curved_transverse:
                    from tools.opaque_tangent_transverse import contract_crossing
                    loc=contract_crossing(self.inputs,box,axis,role,cn[k],loc,partial_bound=self.transverse_partial,checkpoint=self.checkpoint)
                    if loc is None:continue
                support=list(box);support[axis]=loc
                if self.crossing_support and transverse and box[1-axis].width:
                    # Bound the support of the jump, not the integration domain.
                    # Interval Newton retains EVERY possible transverse root for
                    # all inner positions. Uncertain slopes leave it unchanged.
                    from tools.opaque_tangent_transverse import contract_crossing
                    span=contract_crossing(self.inputs,tuple(support),1-axis,role,cn[k],support[1-axis],partial_bound=self.transverse_partial,checkpoint=self.checkpoint)
                    if span is None:continue
                    support[1-axis]=span
                jump=self.jump_on_support(support,axis,role,k,order)
                orientation=1 if slope.lo>0 else -1 if slope.hi<0 else 0
                required=False
                if self.signed and orientation:
                    ends=[]
                    for edge in (box[axis].lo,box[axis].hi):
                        points=[Jet(z) for z in box];points[axis]=Jet(I(edge));ends.append(self.inputs(*points)[role][0,0])
                    left,right=ends;knot=cn[k]
                    required=(left.hi<knot.lo and right.lo>knot.hi) if orientation==1 else (right.hi<knot.lo and left.lo>knot.hi)
                for j in (0,1):
                    witness=dict(position=loc,jumps=[I(jump[i,j].bound)*math.factorial(i)*count for i in range(order)],max_crossings=count,role=role,knot=k)
                    if self.signed and orientation:witness.update(signed_jumps=[jump[i,j]*math.factorial(i)*orientation for i in range(order)],required=required)
                    localized[j].append(witness)
        if order in (4,6):
            rows=[]
            for j in (0,1):
                witnesses=[]
                for w in localized[j]:
                    values=w.get('signed_jumps')
                    if values is None:values=[I(-v.hi,v.hi) for v in w['jumps']]
                    witnesses.append(dict(position=w['position'],values=values,required=w.get('required',False),**(dict(role=w['role'],knot=w['knot']) if self.transverse else {})))
                rows.append(dict(order=order,derivative=I(ordinary[order,j].bound)*math.factorial(order),**({'derivative_range':ordinary[order,j]*math.factorial(order)} if self.centered else {}),jumps=witnesses))
            if self.transverse and transverse:
                from tools.opaque_tangent_transverse import refine
                refine(rows,self.inputs,box,axis,self.e.cn,checkpoint=self.checkpoint,partial_bound=self.transverse_partial,partial_ratio=self.transverse_ratio,**(dict(root_parts=self.root_parts,jump_evaluator=self.partition_jump) if self.root_parts>1 else {}))
            return rows
        return [dict(third=I(ordinary[3,j].bound)*6,**({'third_range':ordinary[3,j]*6} if self.centered else {}),jumps=[I(0)]*3,localized_jumps=localized[j]) for j in (0,1)]
    def range_integral(self,box,*,parts=1):
        if type(parts)!=int or parts not in (1,2) or len(box)!=2:raise ValueError('one or four complete range domains required')
        edges=[]
        for q in box:
            a,b=Q(q.lo),Q(q.hi)
            if a>=b:raise ValueError('positive range domain required')
            edges.append([a+(b-a)*Q(i,parts) for i in range(parts+1)])
        result=[I(0),I(0)]
        for a,b in zip(edges[0],edges[0][1:]):
            for c,d in zip(edges[1],edges[1][1:]):
                self.checkpoint();self.range_proof_boxes+=1
                values=self.ordinary(Jet(I(I(a).lo,I(b).hi)),Jet(I(I(c).lo,I(d).hi)));mass=I((b-a)*(d-c))
                result=[v+q[0,0]*mass for v,q in zip(result,values)]
        self.checkpoint();return result

    def __call__(self,box):
        self.checkpoint();conditioned=self.conditioned;direct=False;self.last_proof='ordinary';self.preferred_split_axis=None
        if conditioned:
            # This is a complete interval integral, not a small-term cutoff.
            # Selection only trades proof cost; the whole sum alone accepts.
            mass=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))
            if mass.lo<=0:raise ValueError('positive correction box required')
            try:
                enclosure=[q[0,0]*mass for q in self.ordinary(Jet(box[0]),Jet(box[1]))]
                width=max(v.width for v in enclosure);self.checkpoint()
                # Work selection only. Narrow radial boxes with broad residuals
                # use a stronger complete identity directly, not a looser test.
                direct=width>I('1e-12').lo and I(box[0].width).hi<=I('.5').lo
                if width<=I('1e-14').lo:
                    self.last_proof='range';return enclosure
                # The actual ordinary integral enclosure decides escalation.
                # A natural-range magnitude cannot suppress the stronger proof.
            except ValueError as error:
                from tools.opaque_shading_intervals import Unresolved
                if isinstance(error,Unresolved):raise
                # Failure to certify a range cannot justify an omission.
                conditioned=True
        if conditioned:self.last_proof='conditioned'
        def directions(rows):self.preferred_split_axis=0 if max(v.hi for v in rows[0])>=max(v.hi for v in rows[1]) else 1
        integration=residual_integral
        if self.order==4:
            from tools.opaque_fourth_tangent import integral
            integration=integral
        # Continuous crossing integrals have no use on discrete Gauss lines.
        # Keep every line derivative/jump; omit only that unconsumed proof.
        extra=dict(line_bounds=lambda box,axis:self.axis_bounds(box,axis,transverse=False,**(dict(proof_order=6) if self.line_order==6 else {}))) if self.order==4 and self.transverse else {}
        try:return integration(self.ordinary,box,self.axis_bounds,checkpoint=self.checkpoint,error_components=directions,**extra,**({"conditioned":True,"integrated_axis":self.integrated_axis} if conditioned else {}),**({"direct_conditioned":True,"condition_parts":4} if direct else {}))
        except ValueError as error:
            if not self.range_fallback or str(error)!='curved angular crossings need certified monotonicity':raise
            # The range of each one-sided ordinary derivative encloses its
            # integral even when crossing count/monotonicity is unproved.
            # Distributional parameter flux is still separately required.
            self.checkpoint();rows=self.ordinary(Jet(box[0]),Jet(box[1]));mass=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))
            self.range_fallbacks+=1;self.preferred_split_axis=None
            whole=[q[0,0]*mass for q in rows]
            if not self.partitioned_range:return whole
            # Four fixed proof domains enclose the integral of the same ordinary
            # field with the parent's frozen extension. No new quadrature nodes,
            # cached result, omitted term or acceptance criterion is introduced.
            refined=self.range_integral(box,parts=2)
            return [I(max(a.lo,b.lo),min(a.hi,b.hi)) for a,b in zip(whole,refined)]
