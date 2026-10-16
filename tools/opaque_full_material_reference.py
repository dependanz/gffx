"""Independent fixed-geometry full-original-material pixel reference.

Exact saved depth-visible pieces,positive conic maps and actual frozen angular
loss resource. Common F0/roughness1 splits radiance into specular + albedo*matte.
This certifies RGB and constant-albedo tangents,not moving angular gradients.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
from types import SimpleNamespace
import time
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_square_reference import integrate_square
from tools.opaque_smooth_branch import SmoothBaseBranches
from tools.opaque_peano_material import MaterialCorrection,radial_cosines
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_loss_extension import LossExtension,bounded_cosines
from tools.opaque_curved_reference import CurvedTriangle,perspective_numerator,radial_homogeneous_fields,flattened_radial_map
from tools.opaque_moving_reference import P,BranchCertificate

class HorizonExtension(LossExtension):
    """First linear segment participates in the same bounded residual model."""
    def at(self,q,index):
        if index==-1:return [q.lift(self.horizon[j])+q/q.lift(self.cn[0])*(self.values[0][j]-self.horizon[j]) for j in range(2)]
        return super().at(q,index)
    def delta(self,q,index):
        if any(j==1 and not v.zero for (i,j),v in q.c.items()):raise ValueError('angle tangent requires qualified knot transport')
        value=q[0,0]
        if value.lo<0 or value.hi>1:raise ValueError('certified front cosine range')
        chosen=([-1] if value.lo<=self.cn[0].hi else [])+[i for i in range(len(self.cn)-1) if value.hi>=self.cn[i].lo and value.lo<=self.cn[i+1].hi]
        if not chosen or len(chosen)>8:raise ValueError('bounded angular-segment residual profile')
        result=[]
        for channel in range(2):
            pieces=[]
            for i in chosen:
                if i==index:pieces.append(I(0));continue
                lo=max(value.lo,D(0) if i==-1 else self.cn[i].lo);hi=min(value.hi,self.cn[0].hi if i==-1 else self.cn[i+1].hi);segment=I(lo,hi)
                if i>=0 and index>=0:
                    key=(i,index,channel)
                    if key not in self.coefficient_cache:
                        anchor=self.x[i if i>index else i+1];self.coefficient_cache[key]=(anchor,[a-b for a,b in zip(self.power(i,channel,anchor),self.power(index,channel,anchor))])
                    anchor,coefficients=self.coefficient_cache[key];z=segment.ln()-anchor;answer=coefficients[-1]
                    for c in reversed(coefficients[:-1]):answer=c+z*answer
                else:
                    log_index=i if i>=0 else index;anchor=self.x[log_index];coefficients=self.power(log_index,channel,anchor);coefficients[0]=coefficients[0]-self.horizon[channel]
                    z=segment.ln()-anchor;answer=coefficients[-1]
                    for c in reversed(coefficients[:-1]):answer=c+z*answer
                    answer=answer-segment*(self.values[0][channel]-self.horizon[channel])/self.cn[0]
                    if i==-1:answer=-answer
                pieces.append(answer)
            result.append(I(min(q.lo for q in pieces),max(q.hi for q in pieces)))
        return result


class CenteredHorizonExtension(HorizonExtension):
    """Equivalent base polynomial for a bounded experimental checker profile."""
    def at(self,q,index):
        if index==-1:return super().at(q,index)
        return self.at_centered(q,index)


def piece_tolerances(parts,*,work_weights=None):
    """Conserve the original half-radius across exact area times work weights.

    Specular and matte each receive sum<=5e-13. RGB=spec+albedo*matte
    therefore keeps radius<=1e-12 when 0<=albedo<=1. Rounding is downward,
    and full_pixel still checks the complete final RGB/Jacobian enclosure.
    Positive bounded rational weights redistribute the SAME pool; default1
    retains area-only allocation. Degenerate triangles keep exact zero measure.
    """
    if work_weights is None:work_weights=[Q(1)]*len(parts)
    if not isinstance(work_weights,(list,tuple)) or len(work_weights)!=len(parts) or any(type(v) not in (int,Q) or not Q(1,16)<=v<=16 for v in work_weights):
        raise ValueError('one positive exact work weight in[1/16,16] per piece required')
    weights=[]
    for part,factor in zip(parts,work_weights):
        a,b,c=[[Q(x) for x in p] for p in part['triangle']]
        weights.append(abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))*factor)
    total=sum(weights,Q(0))
    if not total:raise ValueError('positive visible area required')
    return [I(w/total/Q(2*10**12)).lo for w in weights]


def original_piece_tolerances(parts):
    """Declared thirteen-piece allocation; total basis radius remains5e-13.

    Fourth work weight5/2, all others1; normalize the complete vector exactly.
    Other shares become strictly tighter to compensate. No certificate is reused
    or admitted by this allocation: each piece and the final sum must still pass.
    """
    if len(parts)!=13:raise ValueError('exact thirteen-piece work profile required')
    return piece_tolerances(parts,work_weights=[1,1,1,Q(5,2)]+[1]*9)


def full_pixel(data,*,checkpoint=lambda:None,progress=lambda row:None,selected_piece=None,material_only_inputs=False,horizon_power=4,reference_profile="standard",shared_piece_budget=False,tight_discriminant=False,aggregate_residual=False,smooth_partition=None,stable_view_base=False,tight_light_slope=False,tight_cosine_slopes=False,slope_partition=None,streamed_piece=None,centered_correction=False,directional_correction=False,horizon_correction=False,light_map_power=1,weighted_light_residual=False,align_light_boundary=False):
    from tools.opaque_reference_profile import profile
    limits=profile(reference_profile)
    remaining=reference_profile in ('remaining33-offline-two-hour-v1','remaining28-offline-two-hour-v1')
    if type(light_map_power) is not int or light_map_power not in (1,4) or light_map_power!=1 and not remaining:raise ValueError('explicit remaining flat light-map profile required')
    if type(weighted_light_residual) is not bool or weighted_light_residual and not remaining:raise ValueError('explicit remaining weighted-light profile required')
    if type(align_light_boundary) is not bool or align_light_boundary and (not remaining or light_map_power!=4):raise ValueError('explicit remaining power4 light alignment required')
    if light_map_power!=1 and slope_partition is not None:raise ValueError('mapped slope partition is not qualified')
    streamed=remaining or reference_profile in ('streamed22-offline-v1','streamed22-offline-extended-v1','streamed22-offline-hour-v1','streamed22-offline-two-hour-v1')
    if type(centered_correction)!=bool or centered_correction and not streamed:raise ValueError('centered correction requires explicit streamed profile')
    if type(directional_correction)!=bool or directional_correction and not (streamed and centered_correction):raise ValueError('directional correction requires explicit centered streamed profile')
    if type(horizon_correction)!=bool or horizon_correction and not (streamed and centered_correction and directional_correction):raise ValueError('horizon correction requires explicit directional streamed profile')
    if streamed != (streamed_piece is not None):raise ValueError('explicit streamed profile and selector required together')
    if streamed:
        if remaining:
            from tools.opaque_remaining_reference_contract import validate_input,allocation,scope
            contract_args=dict(reference_profile=reference_profile);piece_count=scope(reference_profile)['pieces']
        else:
            from tools.opaque_streamed_reference_contract import validate_input,allocation
            contract_args={};piece_count=22
        validate_input(data,**contract_args)
        if type(streamed_piece)!=int or not 0<=streamed_piece<piece_count or selected_piece is not None or shared_piece_budget:raise ValueError('one streamed piece without other campaign modes required')
    qualified_mode=shared_piece_budget or streamed
    if type(tight_discriminant)!=bool:raise ValueError('explicit tight discriminant flag')
    if type(tight_cosine_slopes)!=bool or tight_cosine_slopes and not qualified_mode:raise ValueError('tight cosine slopes require explicit shared pixel mode')
    if type(tight_light_slope)!=bool or tight_light_slope and not qualified_mode:raise ValueError('tight light slope requires explicit shared pixel mode')
    if type(stable_view_base)!=bool or stable_view_base and not (qualified_mode and aggregate_residual):raise ValueError('stable view base requires shared aggregate residual mode')
    if type(aggregate_residual)!=bool or aggregate_residual and not qualified_mode:raise ValueError('aggregate residual requires explicit shared pixel mode')
    if type(shared_piece_budget)!=bool or shared_piece_budget and selected_piece is not None:raise ValueError("shared whole-pixel pool cannot mix with selected-piece campaign")
    if type(horizon_power)!=int or horizon_power not in (4,8):raise ValueError("qualified horizon power4or8 required")
    if type(material_only_inputs)!=bool:raise ValueError("explicit material-only input selection required")
    # The approved offline campaign selects one original piece without changing
    # its geometry,material or share of the complete thirteen-piece tolerance.
    if selected_piece is not None and (type(selected_piece) is not int or not 0<=selected_piece<13):
        raise ValueError('approved piece index0..12 required')
    if smooth_partition is not None:
        from tools.opaque_local_smooth import validate
        validate(smooth_partition)
        if not qualified_mode:raise ValueError('localized smooth bounds require explicit shared pixel mode')
    if slope_partition is not None:
        from tools.opaque_cosine_slopes import validate_partition
        validate_partition(slope_partition)
        if not (qualified_mode and tight_cosine_slopes):raise ValueError('slope partition requires shared exact cosine slopes')
    slope_boxes=0
    material=data['material']
    if material!={'roughness':1.,'f0':[.04]*3,'albedo':[.2,.4,.6],'direction':[1.,1.,2.]}:raise ValueError('explicit fixed original material profile')
    start=time.monotonic();cpu_start=time.process_time();cells=0;rows=[];values=[I(0),I(0)];curve=PreparedPrincipled(data['budget'],1.);chosen=data['selected'];parts=chosen['visible_parts'];ox,oy=chosen['origin']
    if not 1<=len(parts)<=(piece_count if remaining else 32):raise ValueError('visible piece cap32 or explicit frozen remaining scope')
    tolerances=original_piece_tolerances(parts) if len(parts)==13 else piece_tolerances(parts)
    if streamed:tolerances=allocation(data,**contract_args)
    if selected_piece is not None and len(parts)!=13:raise ValueError("approved thirteen-piece input required")
    def check():
        checkpoint()
        if time.monotonic()-start>=limits.reference_seconds:raise Unresolved('complete material reference deadline',cells=cells,completed_pieces=len(rows))
    shared_range=None
    if aggregate_residual:
        from tools.opaque_residual_range import ResidualRangeIndex
        shared_range=ResidualRangeIndex(HorizonExtension(curve),checkpoint=check)
    def prepare_piece(number,part):
        # Only exact light-crossing pieces need this endpoint map. Preserve
        # fully lit pieces' original coordinates, bounds and execution path.
        piece_power=light_map_power if any(row['piece']==number and Q(row['dark_area'])>0 for row in light_partitions) else 1
        if align_light_boundary and piece_power!=1:
            from tools.opaque_light_map import align_triangle
            proof=next(row for row in light_partitions if row['piece']==number)
            part=align_triangle(part,proof['line'])
        check();source=data['faces'][part['face']];tri=[(Q(p[0])-ox,Q(p[1])-oy) for p in source['corners']];iw=[1/Q(w) for w in source['clip_w']];visible=[[Q(x) for x in p] for p in part['triangle']]
        coeff=perspective_numerator(tri,inverse_w=iw,normals=source['normal'],views=source['view']);cert=BranchCertificate(check)
        def at(p):
            x,y=p;return sum((a*b for a,b in zip(coeff,(x*x,x*y,y*y,x,y,1))),P(0))
        corners=[at(p) for p in visible];controls=corners+[2*at([(visible[i][k]+visible[j][k])/2 for k in range(2)])-(corners[i]+corners[j])*Q(1,2) for i,j in ((0,1),(0,2),(1,2))]
        signs=[cert.classify(q) for q in controls]
        if tolerances[number]==0 or all(q<0 for q in signs):
            row=dict(piece=number,face=source['face'],classification='certified back-facing black',value=[I(0),I(0)],cells=0,certificate=cert.report());return source,False,None,row
        curved=not all(q>0 for q in signs)
        if curved and piece_power!=1:raise ValueError("light map currently requires view-front component")
        if curved:patch=CurvedTriangle(visible,coeff,checkpoint=check)
        else:patch=SimpleNamespace(triangle=[[P(z) for z in p] for p in visible])
        if curved and tight_discriminant:
            from tools.opaque_quadratic_sqrt import QuadraticSqrt
            patch.sqrt_discriminant=QuadraticSqrt(patch.discriminant,normalize=True)
        fields=radial_homogeneous_fields(patch,tri,inverse_w=iw,fields={'n':source['normal'],'v':source['view']});cosines=radial_cosines(fields,material['direction'],horizon_patch=patch if curved else None)
        a,b,c=visible;det=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]));extension=HorizonExtension(curve);extension.bounds=None;input_cache={}
        slope_proof=None
        if tight_cosine_slopes:
            from tools.opaque_cosine_slopes import FixedCosineSlopes
            slope_proof=FixedCosineSlopes(fields.coefficients,material['direction'])
        def refine_cosine(role,q,t,s):
            if slope_proof is not None and q.order:
                bound=slope_proof.bound(role,t,s);old=q[1,0]
                q.c[1,0]=I(max(old.lo,bound.lo),min(old.hi,bound.hi))
            return q
        def refine_light(q,t,s):
            if tight_light_slope and q.order:
                from tools.opaque_normal_light import normal_light_spatial_bound
                # Exact cross-product cancellation bounds the same coefficient.
                # All higher/mixed coefficients retain the original full Jet.
                bound=normal_light_spatial_bound(fields.coefficients['n'],t,s)
                old=q[1,0];q.c[1,0]=I(max(old.lo,bound.lo),min(old.hi,bound.hi))
            return q
        def inputs(u,s,*,material_only=False,_localize=True):
            key=(u[0,0].lo,u[0,0].hi,s[0,0].lo,s[0,0].hi,material_only) if u.order==0 and not u[0,1].bound and not s[0,1].bound else None
            if key is not None and key in input_cache:return input_cache[key]
            map_jac=u.lift(1)
            if piece_power!=1:
                from tools.opaque_light_map import coordinates
                u,s,map_jac=coordinates(u,s,piece_power)
            # Flatten the endpoint near the horizon without approximating it.
            if curved:t,jac,distance,numerator=flattened_radial_map(patch,u,s,power=horizon_power,with_numerator=True)
            else:t=u;jac=u*det
            if piece_power!=1:jac=jac*map_jac
            nv,nl,vh=cosines(t,s,horizon_distance=distance,material_only=material_only,horizon_numerator=numerator) if curved else cosines(t,s,material_only=material_only)
            nv=refine_cosine(0,nv,t,s);nl=refine_cosine(1,refine_light(nl,t,s),t,s)
            if slope_partition is not None and _localize and u.order and any(q[1,0].lo<0<q[1,0].hi for q in (nv,nl)):
                from tools.opaque_cosine_slopes import subdivided_slopes
                def slope_values(x,y):
                    nonlocal slope_boxes
                    slope_boxes+=1
                    return inputs(x,y,material_only=True,_localize=False)[:2]
                for q,bound in zip((nv,nl),subdivided_slopes(slope_values,u,s,parts=slope_partition,checkpoint=check)):
                    old=q[1,0];q.c[1,0]=I(max(old.lo,bound.lo),min(old.hi,bound.hi))
            # Exact Bernstein/conic sign certificates and Cauchy-Schwarz justify
            # these VALUE intersections. Spatial derivative coefficients stay.
            nv.c[0,0]=I(max(D(0),nv[0,0].lo),min(D(1),nv[0,0].hi))
            for q in ((nl,) if material_only else (nl,vh)):q.c[0,0]=I(max(D(-1),q[0,0].lo),min(D(1),q[0,0].hi))
            if remaining:nl.c[0,0]=I(max(D(0),nl[0,0].lo),nl[0,0].hi)
            elif nl[0,0].lo<=0:raise ValueError('strict light horizon profile')
            if extension.bounds is not None:
                for q,bound in zip((nv,nl),extension.bounds):q.c[0,0]=I(max(q[0,0].lo,bound.lo),min(q[0,0].hi,bound.hi))
            if key is not None:input_cache[key]=(nv,nl,vh,jac)
            return nv,nl,vh,jac
        def cosine_map(u,s):
            if curved:
                t,_,distance,numerator=flattened_radial_map(patch,u,s,power=horizon_power,with_numerator=True)
                return t,s,dict(horizon_distance=distance,horizon_numerator=numerator)
            if piece_power!=1:
                from tools.opaque_light_map import coordinates
                u,s,_=coordinates(u,s,piece_power)
            return u,s,{}
        def role_inputs(role,u,s):
            if slope_partition is not None and u.order:return inputs(u,s,material_only=True)[role]
            t,s,kwargs=cosine_map(u,s);q=cosines.role(role,t,s,**kwargs)
            q.c[0,0]=I(max(D(0) if role==0 else D(-1),q[0,0].lo),min(D(1),q[0,0].hi))
            q=refine_cosine(role,q,t,s)
            if role==1:q=refine_light(q,t,s)
            if role==1 and remaining:q.c[0,0]=I(max(D(0),q[0,0].lo),q[0,0].hi)
            elif role==1 and q[0,0].lo<=0:raise ValueError('strict light horizon profile')
            if extension.bounds is not None:
                bound=extension.bounds[role];q.c[0,0]=I(max(q[0,0].lo,bound.lo),min(q[0,0].hi,bound.hi))
            return q
        def implicit_inputs(role,knot,u,s):
            t,s,kwargs=cosine_map(u,s);return cosines.implicit(role,knot,t,s,**kwargs)
        inputs.role=role_inputs;inputs.implicit=implicit_inputs
        def prepare(u,s,box):
            certified_correction.domain_split_axis=None;directions=[]
            input_cache.clear();extension.bounds=None;limits=bounded_cosines(lambda u,s:list(inputs(u,s)[:2]),*box,directional=directions.append,monotone=curved);extension.bounds=limits;input_cache.clear();nv,nl,_,_=inputs(Jet(box[0]),Jet(box[1]));extension.indices=[]
            candidates=[];cn=curve.cn
            for role,v in enumerate(limits):
                count=int(v.lo<=cn[0].hi)+sum(v.hi>=cn[i].lo and v.lo<=cn[i+1].hi for i in range(len(cn)-1))
                if count>8:candidates.extend((weight.hi,axis) for axis,weight in enumerate(directions[0][role]))
            if candidates:certified_correction.domain_split_axis=max(candidates)[1]

            for role,q in enumerate((nv,nl)):
                value=q[0,0];middle=(Q(value.lo)+Q(value.hi))/2;index=-1 if value.lo<=curve.cn[0].hi else min(sum(Q(z.lo)<=middle for z in curve.cn)-1,len(curve.cn)-2)
                # A base branch is an algebraic decomposition, not a changed
                # material. Near the curved horizon avoid extrapolating log(q)
                # over a wide positive range. The complete signed correction
                # still owns every difference from the actual spline.
                if stable_view_base and curved and role==0 and value.hi>=4*value.lo:index=-1
                extension.indices.append(index)
        f=I(.04);mix=lambda z:z[0]*(1-f)+z[1]*f;mean=mix(curve.mean)
        def smooth(u,s,branch=None):
            at=certified_correction.smooth.at if branch is None else branch
            nv,nl,vh,jac=inputs(u,s);av=at(nv,extension.indices[0]);bl=at(nl,extension.indices[1])
            spec=nl*((1-vh)**5*(1-f)+f)/((nv+nl)*(2*curve.pi))
            matte=nl*mix(av)*mix(bl)/(curve.pi*mean)
            return [spec*jac,matte*jac]
        def refinement_field(u,s):
            # Historical Bernstein derivative scale guides refinement only.
            # Centered smooth bounds and all signed corrections own acceptance.
            return smooth(u,s,extension.at)
        def correction(u,s):
            nv,nl,_,jac=inputs(u,s);av=extension.at(nv,extension.indices[0]);bl=extension.at(nl,extension.indices[1]);dv=[u.lift(q) for q in extension.delta(nv,extension.indices[0])];dl=[u.lift(q) for q in extension.delta(nl,extension.indices[1])]
            return [u.lift(0),nl*(mix(dv)*mix(bl)+mix(av)*mix(dl)+mix(dv)*mix(dl))*jac/(curve.pi*mean)]
        # Both identities are analytically qualified. The fourth piece starts
        # with coordinate0 as its tie preference and chooses per box using the
        # integrated uncertainty; the complete selected bound owns acceptance.
        material_inputs=(lambda u,s:inputs(u,s,material_only=True)) if material_only_inputs else inputs
        if material_only_inputs:material_inputs.role=role_inputs;material_inputs.implicit=implicit_inputs
        correction_type=MaterialCorrection
        if centered_correction:
            from tools.opaque_centered_correction import CenteredMaterialCorrection
            correction_type=CenteredMaterialCorrection
        if directional_correction:
            from tools.opaque_directional_correction import DirectionalMaterialCorrection
            correction_type=DirectionalMaterialCorrection
        if horizon_correction:
            from tools.opaque_horizon_correction import HorizonMaterialCorrection
            correction_type=HorizonMaterialCorrection
        certified_correction=correction_type(extension,material_inputs,curved=curved,checkpoint=check,transverse=True,smooth_partition=True,segmented_conditioned=True,integrated_axis=0 if len(parts)==13 and number==3 else 1,adaptive_order=len(parts)==13 and number==3,refine_roots=False,range_fallback=curved,envelope_fallback=curved or aggregate_residual,aggregate_fallback=curved or aggregate_residual,planar_envelope=aggregate_residual and not curved)
        certified_correction.weighted_light_residual=weighted_light_residual and (light_map_power==1 or piece_power!=1)
        if shared_range is not None:certified_correction.range_index=shared_range
        certified_correction.smooth=SmoothBaseBranches(extension)  # One shared two-branch cache.
        return source,curved,dict(field=smooth,depth=0,prepare=prepare,correction_integrator=certified_correction,refinement_field=None if curved or smooth_partition is not None else refinement_field,**({'remainder_partition':smooth_partition} if smooth_partition is not None else {})),None
    if streamed:selected_piece=streamed_piece
    selected=enumerate(parts) if selected_piece is None else [(selected_piece,parts[selected_piece])]
    charts=[];work=[];light_partitions=[]
    for number,part in selected:
        if remaining:
            from tools.opaque_light_partition import partition,allocate
            components,certificate=partition(part,data['faces'][part['face']],chosen['origin'],material['direction'],checkpoint=check)
            light_partitions.append(dict(piece=number,**certificate))
            if not components:
                row=dict(piece=number,face=data['faces'][part['face']]['face'],classification='certified light-back-facing black',value=[I(0),I(0)],cells=0,certificate=certificate);rows.append(row);progress(row)
            work.extend((number,component,tol) for component,tol in zip(components,allocate(tolerances[number],components)))
        else:work.append((number,part,tolerances[number]))
    for number,part,component_tolerance in work:
        source,curved,chart,zero=prepare_piece(number,part)
        if zero is not None:rows.append(zero);progress(zero);continue
        if shared_piece_budget:
            charts.append(chart);rows.append(dict(piece=number,face=source['face'],curved=curved));continue
        try:result=integrate_square(chart['field'],tolerance=str(component_tolerance),max_cells=limits.cells-cells,max_depth=limits.depth,reference_profile=reference_profile,checkpoint=check,prepare=chart['prepare'],correction_integrator=chart['correction_integrator'],refinement_field=chart['refinement_field'],remainder_partition=chart.get('remainder_partition'),progress=progress)
        except Unresolved as e:raise Unresolved('complete original material unresolved',completed_pieces=len(rows),total_cells=cells,seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,piece=number,face=source['face'],cause=e.details) from e
        cells+=result['cells'];values=[a+b for a,b in zip(values,result['value'])];row=dict(piece=number,face=source['face'],curved=curved,**result);rows.append(row);progress(row)
    shared=None
    if shared_piece_budget and charts:
        # Independent visible pieces retain the original per-piece depth origin.
        # Every evaluation still consumes one shared whole-pixel work/time pool.
        try:
            shared=integrate_square(None,chart_cover=charts,disjoint_roots=True,tolerance='5e-13',max_cells=limits.cells,max_depth=limits.depth,reference_profile=reference_profile,checkpoint=check,progress=progress)
        except Unresolved as e:raise Unresolved('complete shared original material unresolved',pieces=len(parts),charts=len(charts),seconds=time.monotonic()-start,cause=e.details) from e
        values=shared['value'];cells=shared['cells']
    check()
    rgb=[values[0]+I(a)*values[1] for a in material['albedo']]
    if any(q.width>D('2e-12') for q in rgb+[values[1]]):raise Unresolved('complete material total radius')
    return dict(align_light_boundary=align_light_boundary,light_map_power=light_map_power,weighted_light_residual=weighted_light_residual,**(dict(light_partitions=light_partitions) if remaining else {}),horizon_correction=horizon_correction,directional_correction=directional_correction,centered_correction=centered_correction,streamed_piece=streamed_piece,allocated_basis_radius=str(tolerances[streamed_piece]) if streamed else None,slope_partition=slope_partition,slope_boxes=slope_boxes,tight_cosine_slopes=tight_cosine_slopes,tight_light_slope=tight_light_slope,stable_view_base=stable_view_base,smooth_partition=smooth_partition,shared_piece_budget=shared_piece_budget,shared_enclosure=shared,reference_profile=limits.record(),selected_piece=selected_piece,value=rgb,albedo_derivative=values[1],basis=values,pieces=rows,cells=cells,seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,tolerance='1e-12',scope=('Complete fixed packed-geometry pixel' if selected_piece is None else 'Only one selected original visible piece')+'; actual original material and constant RGB albedo Jacobian only. Moving geometry/view/normal and their knot transport are not certified by this run.')
