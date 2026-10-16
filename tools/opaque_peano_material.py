"""Piecewise spatial jets and jump witnesses for the fixed-material oracle.

No tangent or full material correctness claim comes from device agreement.
A noncurved radial triangle makes normal/view numerators affine in either
coordinate with the other fixed: a view-cosine knot has at most four isolated
roots (squared quadratic over quartic); a light-cosine knot at most two.
An identically zero polynomial fixes that angle at the knot, so causes no
along-line branch switching. Curved maps instead require certified monotonicity.
"""
import math
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_peano import residual_integral,crossing_location
from tools.opaque_smooth_branch import SmoothBranches
from tools.opaque_input_reuse import InputReuse


def difference(extension,q,actual,base):
    if actual==base:return [q.lift(0),q.lift(0)]
    if actual<0 or base<0:return [a-b for a,b in zip(extension.at(q,actual),extension.at(q,base))]
    answer=[]
    for channel in range(2):
        key=(actual,base,channel)
        if key not in extension.coefficient_cache:
            anchor=extension.x[actual if actual>base else actual+1]
            extension.coefficient_cache[key]=(anchor,[a-b for a,b in zip(extension.power(actual,channel,anchor),extension.power(base,channel,anchor))])
        anchor,coeff=extension.coefficient_cache[key];z=q.ln()-q.lift(anchor);value=q.lift(coeff[-1])
        for c in reversed(coeff[:-1]):value=q.lift(c)+z*value
        answer.append(value)
    return answer


def segmented(extension,q,base,*,allowed=None):
    if any(j for i,j in q.c):raise ValueError('fixed angle only; moving knot transport required separately')
    v=q[0,0];cn=extension.cn
    if v.lo<0 or v.hi>1:raise ValueError('front unit cosine range')
    chosen=([-1] if v.lo<=cn[0].hi else [])+[i for i in range(len(cn)-1) if v.hi>=cn[i].lo and v.lo<=cn[i+1].hi]
    if allowed is not None:chosen=[i for i in chosen if i in allowed]
    if not chosen or len(chosen)>8:raise ValueError('bounded angular segment count')
    rows=[]
    for i in chosen:
        low=0 if i==-1 else cn[i].lo;high=cn[0].hi if i==-1 else cn[i+1].hi
        coeff=dict(q.c);coeff[0,0]=I(max(v.lo,low),min(v.hi,high));restricted=Jet(order=q.order,coeff=coeff)
        rows.append(difference(extension,restricted,i,base))
    return [Jet(order=q.order,coeff={(i,0):I(min(row[c][i,0].lo for row in rows),max(row[c][i,0].hi for row in rows)) for i in range(q.order+1)}) for c in range(2)]


class MaterialCorrection:
    def __init__(self,extension,inputs,*,curved,checkpoint=lambda:None,transverse=False,smooth_partition=False,segmented_conditioned=False,integrated_axis=1,adaptive_order=False,refine_roots=False,range_fallback=False,envelope_fallback=False,aggregate_fallback=False,planar_envelope=False):
        if type(smooth_partition)!=bool or smooth_partition and not transverse:raise ValueError('smooth partition requires transverse integration')
        if type(segmented_conditioned)!=bool or segmented_conditioned and not transverse:raise ValueError('segmented remainder requires transverse integration')
        if type(integrated_axis)!=int or integrated_axis not in (0,1):raise ValueError('integrated axis0or1 required')
        if type(adaptive_order)!=bool or adaptive_order and not transverse:raise ValueError('adaptive order requires transverse integration')
        if type(refine_roots)!=bool:raise ValueError('root refinement requires Boolean switch')
        if type(range_fallback)!=bool or range_fallback and not curved:raise ValueError('range fallback requires curved profile')
        if type(planar_envelope)!=bool or planar_envelope and (curved or not envelope_fallback):raise ValueError('explicit planar envelope profile required')
        if type(envelope_fallback)!=bool or envelope_fallback and not (curved or planar_envelope):raise ValueError('envelope fallback requires curved or explicit planar profile')
        if type(aggregate_fallback)!=bool or aggregate_fallback and not envelope_fallback:raise ValueError('aggregate proof requires complete envelope fallback')
        self.aggregate_fallback=aggregate_fallback
        self.envelope_fallback=envelope_fallback;self.range_index=None
        self.range_fallback=range_fallback
        self.refine_roots=refine_roots
        self.adaptive_order=adaptive_order
        self.integrated_axis=integrated_axis
        self.segmented_conditioned=segmented_conditioned
        self.smooth_partition=smooth_partition
        self.extension=extension;self.inputs=inputs;self.curved=curved;self.checkpoint=checkpoint
        self.role_inputs=getattr(inputs,'role',None);self.implicit_inputs=getattr(inputs,'implicit',None)
        self.f=I(.04);self.mean=self.mix(extension.mean);self.transverse=transverse;self.smooth=SmoothBranches(extension)
    def mix(self,row):return row[0]*(1-self.f)+row[1]*self.f
    def ordinary(self,u,s):
        return self._ordinary_inputs(u,self.inputs(u,s))
    def _ordinary_inputs(self,u,values,allowed=(None,None)):
        nv,nl,_,jac=values;e=self.extension
        dv=segmented(e,nv,e.indices[0],allowed=allowed[0]);dl=segmented(e,nl,e.indices[1],allowed=allowed[1])
        live_v=any(q.c for q in dv);live_l=any(q.c for q in dl)
        # Empty jet dictionaries certify every retained coefficient is zero.
        # Every nonzero coefficient, however tiny or high-order, takes the live path.
        term=u.lift(0)
        if live_v:
            mdv=self.mix(dv);bl=self.smooth.at(nl,e.indices[1]);term=mdv*self.mix(bl)
        if live_l:
            mdl=self.mix(dl);av=self.smooth.at(nv,e.indices[0]);term=term+self.mix(av)*mdl
            if live_v:term=term+mdv*mdl
        # Keep the same physical factors and denominator validation even at zero.
        return [u.lift(0),nl*term*jac/(e.pi*self.mean)]
    def _jump_inputs(self,box,axis,position,values,*,input_values=None):
        # Mean-value inclusion already proved every crossing lies here.
        # Re-evaluate spatial jets on that region, retaining the active axis;
        # do not differentiate the moving root or discard any jump order.
        domain=I(box[axis]);position=I(position)
        if position.lo<domain.lo or position.hi>domain.hi:raise ValueError('crossing region outside box')
        if position.lo==domain.lo and position.hi==domain.hi:return values
        self.checkpoint();args=[Jet(box[j],order=3) for j in (0,1)]
        args[axis]=Jet.variable(position,order=3)
        return (input_values or self.inputs)(*args)
    def _axis_bounds(self,box,axis,*,input_values=None):
        inputs=input_values or self.inputs
        order=3;args=[Jet(box[j],order=order) for j in (0,1)];args[axis]=Jet.variable(box[axis],order=order)
        values=inputs(*args);nv,nl,_,jac=values;e=self.extension;cn=e.cn;ordinary=self._ordinary_inputs(args[0],values);localized=[];midpoints=None;endpoints=None
        for role,q in enumerate((nv,nl)):
            if q[1,0].zero:continue
            active=[k for k in range(len(cn)-1) if q[0,0].lo<=cn[k].hi and q[0,0].hi>=cn[k].lo]
            if len(active)>8:raise ValueError('bounded angular crossing count')
            if not active:continue
            if q[1,0].lo>=0 or q[1,0].hi<=0:count=1
            elif self.curved:raise ValueError('curved angular crossings need certified monotonicity')
            else:count=4 if role==0 else 2
            if midpoints is None:
                middle=[Jet(box[j]) for j in (0,1)]
                middle[axis]=Jet(I((Q(box[axis].lo)+Q(box[axis].hi))/2))
                midpoints=inputs(*middle)[:2]
            for k in active:
                self.checkpoint()
                location=crossing_location(box[axis],midpoints[role][0,0],q[1,0],cn[k])
                if location is None:continue  # Mean-value inclusion certifies no crossing.
                narrow=self._jump_inputs(box,axis,location,values,input_values=input_values) if input_values is not None else self._jump_inputs(box,axis,location,values)
                if self.refine_roots and location.lo<location.hi and not (narrow[role][1,0].lo<=0<=narrow[role][1,0].hi):
                    # Every root already lies in this strip. One more mean-value
                    # inclusion, centered on the strip, may contract its location.
                    # The reused derivative/jump evidence covers the LARGER strip,
                    # so it remains valid; transverse uncertainty is retained.
                    self.checkpoint();points=[Jet(box[j]) for j in (0,1)]
                    points[axis]=Jet(I((Q(location.lo)+Q(location.hi))/2))
                    center=inputs(*points)[role][0,0]
                    refined=crossing_location(location,center,narrow[role][1,0],cn[k])
                    if refined is None:continue # Certified no root, not a sampled absence.
                    location=refined
                crossing=narrow[role];other=narrow[1-role]
                other_actual=[a+b for a,b in zip(self.smooth.at(other,e.indices[1-role]),segmented(e,other,e.indices[1-role]))]
                coefficients=dict(crossing.c);coefficients[0,0]=cn[k];at_knot=Jet(order=order,coeff=coefficients)
                delta=difference(e,at_knot,k,k-1)
                light=narrow[1] if role==0 else at_knot
                jump=light*self.mix(delta)*self.mix(other_actual)*narrow[3]/(e.pi*self.mean)
                witness=dict(position=location,jumps=[I(jump[j,0].bound)*math.factorial(j)*count for j in range(3)],max_crossings=count,role=role,knot=k)
                orientation=1 if q[1,0].lo>0 else -1 if q[1,0].hi<0 else 0
                if orientation:
                    if endpoints is None:
                        endpoints=[]
                        for edge in (box[axis].lo,box[axis].hi):
                            points=[Jet(box[j]) for j in (0,1)];points[axis]=Jet(I(edge))
                            endpoints.append(inputs(*points)[:2])
                    left,right=(ends[role][0,0] for ends in endpoints);knot=cn[k]
                    required=(left.hi<knot.lo and right.lo>knot.hi) if orientation==1 else (right.hi<knot.lo and left.lo>knot.hi)
                    # At a decreasing crossing, right-minus-left in the spatial
                    # coordinate is the NEGATIVE of high-angle minus low-angle.
                    # Jets already contain the signed spatial chain factors.
                    witness.update(orientation=orientation,signed_jumps=[jump[j,0]*math.factorial(j)*orientation for j in range(3)],required=required)
                localized.append(witness)
        return [dict(third=I(0),third_range=I(0),jumps=[I(0)]*3,localized_jumps=[]),dict(third=I(ordinary[1][3,0].bound)*6,third_range=ordinary[1][3,0]*6,jumps=[I(0)]*3,localized_jumps=localized)]
    def axis_bounds(self,box,axis):
        if not self.segmented_conditioned:return self._axis_bounds(box,axis)
        inputs=InputReuse(self.inputs,checkpoint=self.checkpoint)
        rows=self._axis_bounds(box,axis,input_values=inputs)
        witnesses=rows[1]['localized_jumps']
        if not witnesses or any(w.get('orientation',0)==0 for w in witnesses):return rows
        from tools.opaque_segmented_remainder import segmented_remainder,allowed_segments
        e=self.extension;cn=e.cn
        def third(slab):
            self.checkpoint();args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(slab,order=3)
            values=inputs(*args);allowed=[]
            for role,q in enumerate(values[:2]):
                v=q[0,0]
                candidates=([-1] if v.lo<=cn[0].hi else [])+[i for i in range(len(cn)-1) if v.hi>=cn[i].lo and v.lo<=cn[i+1].hi]
                allowed.append(allowed_segments(candidates,witnesses,slab,role))
            return self._ordinary_inputs(args[0],values,allowed)[1][3,0]*6
        result=segmented_remainder(third,box[axis],[w['position'] for w in witnesses],full_range=rows[1]['third_range'],checkpoint=self.checkpoint)
        if result is not None:rows[1]['ordinary_signed']=result['value']
        return rows

    def integrated_bounds(self,box,axis,*,diagnostic=None):
        """Integrate monotone knot witnesses along the transverse coordinate.

        The change of variable is admitted only when both coordinate derivatives
        strictly exclude zero over the entire box. Other witnesses keep the
        existing conservative whole-transverse bound and multiplicity.
        """
        from tools.opaque_transverse_peano import clamped_crossing,weighted_kernel_parts
        rows=self._axis_bounds(box,axis);transverse=1-axis
        derivatives=[]
        for active in (axis,transverse):
            args=[Jet(box[j],order=1) for j in (0,1)];args[active]=Jet.variable(box[active],order=1)
            derivatives.append([v[1,0] for v in self.inputs(*args)[:2]])
        h=I(box[axis].hi)-I(box[axis].lo);low=I(box[axis].lo)
        for row in rows:
            for witness in row.get('localized_jumps',[]):
                detail=dict(role=witness['role'],knot=witness['knot'],position=witness['position'],admission='fallback')
                if 'signed_jumps' not in witness:
                    if diagnostic:diagnostic({**detail,'reason':'unsigned_crossing'})
                    continue
                role=witness['role'];inner,outer=(v[role] for v in derivatives)
                if any(v.lo<=0<=v.hi for v in (inner,outer)):
                    if diagnostic:diagnostic({**detail,'reason':'derivative_contains_zero','inner':inner,'outer':outer})
                    continue
                self.checkpoint();roots=[];knot=self.extension.cn[witness['knot']]
                for edge in (box[transverse].lo,box[transverse].hi):
                    values=[]
                    for position in (box[axis].lo,box[axis].hi,I((Q(box[axis].lo)+Q(box[axis].hi))/2)):
                        args=[Jet(I(0)),Jet(I(0))];args[axis]=Jet(I(position));args[transverse]=Jet(I(edge))
                        values.append(self.inputs(*args)[role][0,0])
                    root=clamped_crossing(box[axis],*values,inner,knot)
                    t=(root-low)/h;roots.append(I(max(I(0).lo,t.lo),min(I(1).hi,t.hi)))
                # ds/dt=-q_transverse/q_inner. Orient the integration range;
                # the spatial right-minus-left jump already owns its sign.
                if (-outer/inner).hi<0:roots.reverse()
                ratio=inner/outer
                if ratio.hi<0:ratio=-ratio
                terms=[weighted_kernel_parts(jump*ratio*h**(j+2),*roots,j+1) for j,jump in enumerate(witness['signed_jumps'])]
                witness['transverse_integral']=sum((part['value'] for part in terms),I(0))
                if diagnostic:diagnostic({**detail,'admission':'integrated','inner':inner,'outer':outer,'roots':roots,'terms':terms,'value':witness['transverse_integral']})
        if self.smooth_partition:
            from tools.opaque_smooth_peano import partitioned_ordinary
            def third(sub,active):
                args=[Jet(sub[j],order=3) for j in (0,1)];args[active]=Jet.variable(sub[active],order=3)
                return self.ordinary(*args)[1][3,0]*6
            partition=partitioned_ordinary(third,box,axis,checkpoint=self.checkpoint)
            rows[1]['ordinary_integral']=partition['effective']
            rows[1]['ordinary_integral_signed']=partition['signed']
        return rows
    def __call__(self,box):
        self.preferred_split_axis=None;self.error_breakdown=None
        try:return self._certified_integral(box)
        except ValueError as error:
            if self.envelope_fallback and str(error) in ('bounded angular segment count','bounded angular crossing count','curved angular crossings need certified monotonicity'):
                return self.envelope_integral(box)
            if not self.range_fallback or str(error)!='curved angular crossings need certified monotonicity':raise
            # A complete order-zero range times exact box area is a valid
            # integral enclosure without a crossing-count or smoothness claim.
            # ordinary still enforces <=8segments and every physical factor.
            self.checkpoint();rows=self.ordinary(Jet(box[0]),Jet(box[1]))
            area=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))
            self.error_breakdown=None
            return [q[0,0]*area for q in rows]
    def envelope_integral(self,box):
        """Complete range intersected with aggregate proof; no preliminary path.

        Explicit experimental scheduling only. Every value term and physical
        factor remains; cancellation and unsupported arithmetic still propagate.
        """
        self.preferred_split_axis=None;self.error_breakdown=None
        from tools.opaque_residual_range import ResidualRangeIndex
        if self.range_index is None:self.range_index=ResidualRangeIndex(self.extension,checkpoint=self.checkpoint)
        self.checkpoint();u,s=Jet(box[0]),Jet(box[1]);nv,nl,_,jac=self.inputs(u,s);e=self.extension
        dv=self.range_index.delta(nv[0,0],e.indices[0]);dl=self.range_index.delta(nl[0,0],e.indices[1])
        av=self.smooth.at(nv,e.indices[0]);bl=self.smooth.at(nl,e.indices[1])
        term=self.mix(dv)*self.mix(bl)[0,0]+self.mix(av)[0,0]*self.mix(dl)+self.mix(dv)*self.mix(dl)
        area=(I(box[0].hi)-I(box[0].lo))*(I(box[1].hi)-I(box[1].lo))
        self.error_breakdown=None
        self.preferred_split_axis=getattr(self,'domain_split_axis',None)
        result=[I(0),nl[0,0]*term*jac[0,0]*area/(e.pi*self.mean)]
        if self.aggregate_fallback:
            from tools.opaque_aggregate_peano import AggregateMaterial
            try:other=AggregateMaterial(self)(box)
            except ValueError as error:
                if str(error)!='aggregate angular crossings need certified monotonicity':raise
            else:
                if len(other)!=len(result):raise ValueError('aggregate channel count')
                result=[I(max(a.lo,b.lo),min(a.hi,b.hi)) for a,b in zip(result,other)]
                # The intersection may have another center than either
                # proof; retain complete widths, not additive parts.
                self.error_breakdown=None
        return result
    def _certified_integral(self,box):
        self.preferred_split_axis=None
        def capture(rows):
            self.directional_error_bounds=rows
            self.preferred_split_axis=0 if max(v.hi for v in rows[0])>=max(v.hi for v in rows[1]) else 1
        def capture_breakdown(rows):self.error_breakdown=rows
        self.last_integrated_axis=self.integrated_axis
        if self.transverse:
            from tools.opaque_transverse_peano import conditioned_integral,adaptive_conditioned_integral
            if self.adaptive_order:
                def selected(axis):self.last_integrated_axis=axis
                return adaptive_conditioned_integral(self.ordinary,box,self.axis_bounds,self.integrated_bounds,checkpoint=self.checkpoint,error_components=capture,error_breakdown=capture_breakdown,preferred_axis=self.integrated_axis,selection=selected)
            return conditioned_integral(self.ordinary,box,self.axis_bounds,self.integrated_bounds,checkpoint=self.checkpoint,error_components=capture,error_breakdown=capture_breakdown,integrated_axis=self.integrated_axis)
        return residual_integral(self.ordinary,box,self.axis_bounds,checkpoint=self.checkpoint,error_components=capture,error_breakdown=capture_breakdown)


def radial_cosines(fields,direction,*,simplex=False,horizon_patch=None):
    """Reduce rational dot products exactly before normalizing homogeneous fields.

    Affine/bilinear coefficients are exact rationals from the original packed
    inputs. No normalization approximation or changed material is introduced.
    The half-vector identity v.h=sqrt((1+v.l)/2) requires nonopposite unit v,l.
    """
    from fractions import Fraction as Q
    from tools.opaque_pixel_oracle import _bounded
    prepared=fields.coefficients
    def poly(row):
        if any(len(p.c)!=1 for p in row):raise ValueError('fixed packed fields only')
        if simplex:return {(0,0):row[0][0],(1,0):row[1][0],(0,1):row[1][0]+row[2][0]}
        return {(0,0):row[0][0],(1,0):row[1][0],(1,1):row[2][0]}
    n=list(map(poly,prepared['n']));v=list(map(poly,prepared['v']));l=[{(0,0):Q(z)} for z in direction]
    def dot(a,b,*,exact=False):
        result={}
        for x,y in zip(a,b):
            for (i,j),aa in x.items():
                for (k,m),bb in y.items():result[i+k,j+m]=_bounded(result.get((i+k,j+m),Q(0))+aa*bb)
        # Convert once; do not repeatedly round large rational coefficients.
        return {key:(value if exact else I(value)) for key,value in result.items() if value}
    if horizon_patch is not None:
        if simplex:raise ValueError('radial horizon numerator requires radial coordinates')
        aa,bb,cc=horizon_patch.radial;factor=horizon_patch.original_factor
        expected={(2,j):factor*v[0] for j,v in enumerate(aa)}
        expected.update({(1,j):factor*v[0] for j,v in enumerate(bb)});expected[0,0]=factor*cc[0]
        if any(len(v.c)!=1 for v in [*aa,*bb,cc]) or dot(n,v,exact=True)!={k:v for k,v in expected.items() if v}:
            raise ValueError('horizon numerator must equal exact packed normal/view dot product')
    polynomials=[dot(n,v),dot(n,l),dot(v,l),dot(n,n),dot(v,v),dot(l,l)]
    def evaluate(t,s,*,horizon_distance=None,material_only=False,horizon_numerator=None):
        if type(material_only)!=bool:raise ValueError("explicit Boolean material-only selection required")
        if (horizon_patch is None)!=(horizon_distance is None):raise ValueError('horizon distance requires matching certified patch')
        def at(poly):
            out=t.lift(0)
            for i in range(max((i for i,j in poly),default=0),-1,-1):
                row=t.lift(0)
                for j in range(max((j for k,j in poly if k==i),default=0),-1,-1):row=row*s+t.lift(poly.get((i,j),I(0)))
                out=out*t+row
            return out
        if horizon_numerator is not None and (horizon_patch is None or not isinstance(horizon_numerator,Jet) or horizon_numerator.order!=t.order):raise ValueError('matching certified horizon numerator required')
        # The certified horizon identity replaces polynomial zero completely.
        # Do not calculate that discarded polynomial on either horizon path.
        nv=at(polynomials[0]) if horizon_patch is None else horizon_patch.horizon_numerator(horizon_distance,s) if horizon_numerator is None else horizon_numerator
        nl,nn,vv,ll=(at(polynomials[i]) for i in (1,3,4,5))
        if not material_only:vl=at(polynomials[2])
        nv=nv/(nn*vv).sqrt();nl=nl/(nn*ll).sqrt()
        if material_only:return nv,nl,None # Missing specular input is explicit, never a fake zero.
        vl=vl/(vv*ll).sqrt()
        return nv,nl,((1+vl)/2).sqrt()
    def components(role,t,s,*,horizon_distance=None,horizon_numerator=None):
        if type(role)!=int or role not in (0,1):raise ValueError('cosine role0or1 required')
        if (horizon_patch is None)!=(horizon_distance is None):raise ValueError('horizon distance requires matching certified patch')
        if horizon_numerator is not None and (horizon_patch is None or not isinstance(horizon_numerator,Jet) or horizon_numerator.order!=t.order):raise ValueError('matching certified horizon numerator required')
        def at(poly):
            out=t.lift(0)
            for i in range(max((i for i,j in poly),default=0),-1,-1):
                row=t.lift(0)
                for j in range(max((j for k,j in poly if k==i),default=0),-1,-1):row=row*s+t.lift(poly.get((i,j),I(0)))
                out=out*t+row
            return out
        if role==0:
            numerator=at(polynomials[0]) if horizon_patch is None else horizon_patch.horizon_numerator(horizon_distance,s) if horizon_numerator is None else horizon_numerator
            denominator=at(polynomials[3])*at(polynomials[4])
        else:numerator=at(polynomials[1]);denominator=at(polynomials[3])*at(polynomials[5])
        return numerator,denominator
    def role_only(role,t,s,**kwargs):
        numerator,denominator=components(role,t,s,**kwargs)
        return numerator/denominator.sqrt()
    def implicit(role,knot,t,s,**kwargs):
        knot=I(knot)
        if knot.lo<=0 or knot.hi>1:raise ValueError('positive unit algebraic knot required')
        numerator,denominator=components(role,t,s,**kwargs)
        if numerator[0,0].lo<=0 or denominator[0,0].lo<=0:raise ValueError('positive algebraic boundary branch required')
        return numerator*numerator-denominator*(knot*knot)
    evaluate.role=role_only;evaluate.implicit=implicit
    return evaluate
