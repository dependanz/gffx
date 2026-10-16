"""Independent original-piece normal/view and attached screen-translation tangents.

Curved view horizons use a certified root map and its complete first Jacobian.
Screen translation is a projected-input direction, NOT world vertex/camera motion.
Every reference is fresh; partial cover enclosures do not qualify a gradient.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
from types import SimpleNamespace
import time
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_moving_reference import P,BranchCertificate,moving_fragments
from tools.opaque_curved_reference import CurvedTriangle,perspective_numerator,radial_homogeneous_fields,flattened_radial_map
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_full_material_reference import HorizonExtension
from tools.opaque_loss_extension import bounded_cosines
from tools.opaque_tangent_peano import TangentCorrection,MixedLoss
from tools.opaque_knot_transport import material_transport
from tools.opaque_square_reference import integrate_square


def horizon_split(a,b,*,inside):
    """Exact graded partition near the radial horizon, never an acceptance test."""
    if type(inside)!=bool or not 0<=a<b<=1:raise ValueError('unit interval and explicit horizon side required')
    a,b=Q(a),Q(b)
    lo,hi=(1-b,1-a) if inside else (a,b)
    point=hi/4 if lo==0 else 2*lo*hi/(lo+hi)
    return 1-point if inside else point


def prepare(data,*,piece,motion,anchor=1,horizon_power=4,checkpoint=lambda:None,tight_discriminant=False,world_profile=None):
    if type(tight_discriminant)!=bool:raise ValueError("explicit discriminant candidate flag")
    if type(horizon_power)!=int or horizon_power not in (4,8):raise ValueError('qualified horizon power required')
    if type(piece)!=int or not 0<=piece<13 or len(data['selected']['visible_parts'])!=13:raise ValueError('thirteen-piece original profile')
    world=motion in ('world_x','camera_x','world_shear_xz')
    if motion not in ('normal_tilt','view_x','screen_x','world_x','camera_x','world_shear_xz') or type(anchor)!=int or anchor not in (0,1,2):raise ValueError('declared motion/cyclic anchor')
    if world!=(world_profile is not None):raise ValueError('world motion requires its explicit source profile only')
    if data['material']!={'roughness':1.,'f0':[.04]*3,'albedo':[.2,.4,.6],'direction':[1.,1.,2.]}:raise ValueError('original material profile')
    part=data['selected']['visible_parts'][piece];source=data['faces'][part['face']];ox,oy=data['selected']['origin'];tri=[(Q(p[0])-ox,Q(p[1])-oy) for p in source['corners']];iw=[1/Q(w) for w in source['clip_w']];visible=[[Q(v) for v in p] for p in part['triangle']]
    dn=[[Q(.01)*Q(z),0,-Q(.01)*Q(x)] for x,y,z in source['normal']] if motion=='normal_tilt' else [[0]*3 for _ in range(3)]
    dv=[[Q(.01),0,0] for _ in range(3)] if motion=='view_x' else [[0]*3 for _ in range(3)]
    velocity=[[Q(0),Q(0)] for _ in range(3)];visibility=None
    if motion=='screen_x':
        chosen=data['selected'];parts,visibility=moving_fragments(chosen['screen_vertices'],chosen['faces'],[[Q(.01),0,0] for _ in chosen['screen_vertices']],checkpoint=checkpoint)
        matches=[p for p in parts if p['face']==part['face'] and set(map(tuple,p['triangle']))==set(map(tuple,visible))]
        if len(matches)!=1:raise Unresolved('moving piece correspondence',piece=piece,matches=len(matches))
        row=matches[0];vel={tuple(x):v for x,v in zip(row['triangle'],row['motion'])};velocity=[[vel[tuple(p)][0]-Q(.01),vel[tuple(p)][1]] for p in visible]
    projected=None;world_motion=None
    if world:
        from tools.opaque_projected_motion import aligned_world_motion,ProjectedFields
        world_motion=aligned_world_motion(data,world_profile,kind=motion,input_sha256='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a')
        chosen=data['selected'];parts,visibility=moving_fragments(chosen['screen_vertices'],chosen['faces'],world_motion['screen_motion'],checkpoint=checkpoint)
        matches=[row for row in parts if row['face']==part['face'] and set(map(tuple,row['triangle']))==set(map(tuple,visible))]
        if len(matches)!=1:raise Unresolved('world moving piece correspondence',piece=piece,matches=len(matches))
        row=matches[0];vel={tuple(x):v for x,v in zip(row['triangle'],row['motion'])};velocity=[list(vel[tuple(point)][:2]) for point in visible]
        fields=world_motion['faces'][part['face']];dn=fields['normal_motion'];dv=fields['view_motion']
        projected=ProjectedFields(triangle=tri,inverse_w=iw,normals=source['normal'],views=source['view'],checkpoint=checkpoint,**fields)
    visible=visible[anchor:]+visible[:anchor];velocity=velocity[anchor:]+velocity[:anchor]
    co=projected.numerator() if projected is not None else perspective_numerator(tri,inverse_w=iw,normals=source['normal'],views=source['view'],normal_motion=dn,view_motion=dv);cert=BranchCertificate(checkpoint)
    moving=[[P([x,d]) for x,d in zip(p,v)] for p,v in zip(visible,velocity)]
    def at(p):
        x,y=p;return sum((a*b for a,b in zip(co,(x*x,x*y,y*y,x,y,1))),P(0))
    corners=[at(p) for p in moving];controls=corners+[2*at([(moving[i][k]+moving[j][k])*Q(1,2) for k in range(2)])-(corners[i]+corners[j])*Q(1,2) for i,j in ((0,1),(0,2),(1,2))];signs=[cert.classify(v) for v in controls]
    back=all(s<0 for s in signs);curved=not back and not all(s>0 for s in signs)
    patch=CurvedTriangle(visible,co,motion=velocity,checkpoint=checkpoint) if curved else SimpleNamespace(triangle=moving)
    if curved and tight_discriminant:
        from tools.opaque_quadratic_sqrt import QuadraticSqrt
        patch.sqrt_discriminant=QuadraticSqrt(patch.discriminant,**({"normalize":True} if world else {}))
    fields=projected.radial(patch) if projected is not None else radial_homogeneous_fields(patch,tri,inverse_w=iw,fields={'n':[[P([Q(x),Q(d)]) for x,d in zip(n,m)] for n,m in zip(source['normal'],dn)],'v':[[P([Q(x),Q(d)]) for x,d in zip(v,m)] for v,m in zip(source['view'],dv)]})
    a,b,c=moving;det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);det=cert.classify(det)*det
    return dict(world_motion=world_motion,horizon_power=horizon_power,piece=piece,face=source['face'],motion=motion,anchor=anchor,source=source,triangle=tri,inverse_w=iw,normal_motion=dn,view_motion=dv,patch=patch,fields=fields,det=det,curved=curved,back=back,visibility=visibility,certificate=cert.report())


def unflattened_box(box,*,inside,power=4):
    if type(power)!=int or power not in (4,8):raise ValueError('qualified horizon power required')
    if type(inside)!=bool:raise ValueError('explicit curved side required')
    a,b=map(Q,(box[0].lo,box[0].hi))
    if not 0<=a<=b<=1:raise ValueError('unit radial box required')
    z=lambda u:1-(1-u)**power if inside else u**power
    return I(I(z(a)).lo,I(z(b)).hi),box[1]


def mapped_inputs(p,u,s,*,flatten=True,material_only=False):
    if type(flatten)!=bool or type(material_only)!=bool:raise ValueError('explicit coordinate chart and material-only switch')
    if p['curved']:
        if flatten:t,jac,distance,numerator=flattened_radial_map(p['patch'],u,s,power=p['horizon_power'],with_numerator=True)
        else:
            t,jac=p['patch'].radial_map(u,s);distance=1-u if p['patch'].inside else u;numerator=p['patch'].horizon_numerator(distance,s)
    else:t=u;jac=u*u.lift(p['det'][0],tangent=p['det'][1]);numerator=None
    fields=p['fields'](t,s);n,v=fields['n'],fields['v'];l=[u.lift(k) for k in (1,1,2)];dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0));nn,vv,ll=dot(n,n),dot(v,v),dot(l,l)
    nv=(dot(n,v) if numerator is None else numerator)/(nn*vv).sqrt();nl=dot(n,l)/(nn*ll).sqrt();vh=u.lift(0) if material_only else ((1+dot(v,l)/(vv*ll).sqrt())/2).sqrt()
    return nv,nl,vh,jac


def mapped_angle(p,u,s):
    """Unflattened outside-horizon view cosine for inverse-root work only.

    Preserve the exact arithmetic of mapped_inputs' view channel. Light,
    half-vector and their square roots are not needed by the inverse recurrence.
    The physical Jacobian is still evaluated by the final mapped_inputs call.
    """
    if not p['curved'] or p['patch'].inside:raise ValueError('outside curved radial profile required')
    t,_=p['patch'].radial_map(u,s)
    numerator=p['patch'].horizon_numerator(u,s)
    fields=p['fields'](t,s);n,v=fields['n'],fields['v']
    dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
    nn,vv=dot(n,n),dot(v,v)
    return numerator/(nn*vv).sqrt()


def radial_angle_derivative(p,u,s):
    """Analytic partial derivative in the unflattened radial coordinate.

    All quantities remain Jets, including horizon motion and normal/view
    trajectories. The stable horizon numerator is differentiated algebraically;
    this is not a finite difference or a derivative of clipped intervals.
    """
    if not p['curved'] or p['patch'].inside:raise ValueError('outside curved radial profile required')
    patch=p['patch'];root,disc=patch._root_discriminant(s);width=1-root;t=root+u*width
    fields=p['fields'](t,s);n,v=fields['n'],fields['v']
    def derivative(name):
        return [width*(u.lift(b[0],tangent=b[1])+s*u.lift(c[0],tangent=c[1])) for a,b,c in p['fields'].coefficients[name]]
    dn,dv=derivative('n'),derivative('v')
    dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
    nn,vv=dot(n,n),dot(v,v);den=(nn*vv).sqrt()
    aa=patch.lift(s,patch.radial[0][-1])
    for c in reversed(patch.radial[0][:-1]):aa=aa*s+patch.lift(s,c)
    delta=width*u;numerator=patch.original_factor*delta*(disc+aa*delta)
    derivative_numerator=patch.original_factor*width*(disc+2*aa*delta)
    return derivative_numerator/den-(numerator/den)*(dot(n,dn)/nn+dot(v,dv)/vv)


def reference(data,*,piece,motion,anchor=1,tolerance='5e-13',progress=lambda row:None,diagnostic_boxes=None,horizon_power=4,domain_only=False,graded=False,knot_targeted=False,cut_diagnostic_boxes=None,curved_knot=False,entry_knot=False,correction_order=3,conditioned_correction=False,mixed_curve=False,initial_leaves=None,transverse_correction=False,correction_axis=1,material_inputs=False,exact_motion=False,refinement_axis=None,polynomial_inputs=False,world_profile=None,line_order=4,crossing_support=False,root_parts=1):
    if type(root_parts)!=int or root_parts not in (1,2,4) or root_parts>1 and not (correction_order==4 and conditioned_correction and transverse_correction and crossing_support):raise ValueError('root partition requires qualified two/four-part front profile')
    if type(crossing_support)!=bool or crossing_support and not (correction_order==4 and conditioned_correction and transverse_correction):raise ValueError('crossing support requires conditioned transverse front profile')
    if type(line_order)!=int or line_order not in (4,6) or line_order==6 and not (correction_order==4 and conditioned_correction and transverse_correction):raise ValueError('sixth-order lines require conditioned transverse front profile')
    if type(polynomial_inputs)!=bool or polynomial_inputs and not (conditioned_correction and transverse_correction and material_inputs):raise ValueError('polynomial inputs require selected conditioned material-only transverse profile')
    if refinement_axis is not None and (type(refinement_axis)!=int or refinement_axis not in (0,1) or not (conditioned_correction and transverse_correction)):raise ValueError('explicit front refinement axis requires selected conditioned transverse profile')
    if type(exact_motion)!=bool or exact_motion and not (material_inputs and transverse_correction):raise ValueError('exact motion requires material-only transverse profile')
    if type(material_inputs)!=bool or material_inputs and not conditioned_correction:raise ValueError('material-only queries require selected front correction')
    if type(correction_axis)!=int or correction_axis not in (0,1) or correction_axis!=1 and not conditioned_correction or type(correction_order)!=int or correction_order not in (3,4) or type(conditioned_correction)!=bool or conditioned_correction and correction_order!=4 or type(mixed_curve)!=bool or mixed_curve and not conditioned_correction or type(transverse_correction)!=bool or transverse_correction and not conditioned_correction:raise ValueError('explicit original correction profile required')
    if type(entry_knot)!=bool or (entry_knot and (not curved_knot or not domain_only)):raise ValueError('entry knot cover is curved-domain-preflight only')
    if type(curved_knot)!=bool or (curved_knot and not domain_only):raise ValueError('curved knot cover is domain-preflight only')
    if type(knot_targeted)!=bool or (knot_targeted and graded):raise ValueError('explicit exclusive knot partition flag')
    if knot_targeted and not domain_only and cut_diagnostic_boxes is None:raise ValueError('original knot profile requires a qualified domain cover before accuracy trial')
    if type(graded)!=bool:raise ValueError('explicit graded split flag')
    if graded and not domain_only:raise ValueError('graded split is domain-preflight only until integrator qualification')
    if type(domain_only)!=bool:raise ValueError('explicit domain preflight flag')
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('original motion reference deadline')
    p=prepare(data,piece=piece,motion=motion,anchor=anchor,horizon_power=horizon_power,checkpoint=check,**({'world_profile':world_profile} if world_profile is not None else {}))
    if polynomial_inputs and p['curved']:raise ValueError('polynomial inputs require noncurved physical map')
    if refinement_axis is not None and p['curved']:raise ValueError('front refinement axis requires noncurved chart')
    if (correction_order!=3 or conditioned_correction or mixed_curve) and p['curved']:raise ValueError('selected correction extension currently requires front-facing geometry')
    if p['back']:return dict(piece=piece,motion=motion,classification='locally back-facing black',value=[I(0)]*3,derivative=[I(0)]*3,cells=0,certificate=p['certificate'])
    curve=PreparedPrincipled(data['budget'],1.);e=HorizonExtension(curve);bounds=None;cache={};guidance=dict(calls=0,field_evaluations=0,cuts=[],curved_attempts=0,curved_cuts=[],curved_refusals=[],entry_cuts=[],entry_attempts=0)
    polynomial=None;front_material_cache=None
    if polynomial_inputs:
        from tools.opaque_polynomial_inputs import PolynomialCosines
        polynomial=PolynomialCosines(p['fields'].coefficients)
        from tools.opaque_prepared_angle import MaterialInputCache
        front_material_cache=MaterialInputCache()
    def polynomial_material(u,s):
        # Cache raw field tuples only. Region-specific clamps stay outside.
        return (*polynomial(u,s,material_only=True),u*u.lift(p['det'][0],tangent=p['det'][1]))
    def inputs(u,s,*,flatten=True,material_only=False):
        if polynomial is None:nv,nl,vh,jac=mapped_inputs(p,u,s,flatten=flatten,**(dict(material_only=True) if material_only else {}))
        elif material_only:
            nv,nl,vh,jac=front_material_cache('front_polynomial_material',polynomial_material,u,s)
        else:
            nv,nl,vh=polynomial(u,s,material_only=material_only)
            jac=u*u.lift(p['det'][0],tangent=p['det'][1])
            # Raw order-zero sample fields share exact arithmetic with material-only queries.
            # Snapshot before any regional clamps; the highlight is unused there.
            if u.order==0:front_material_cache.seed('front_polynomial_material',polynomial_material,u,s,(nv,nl,u.lift(0),jac))
        if material_only and exact_motion:
            for role,q in enumerate((nv,nl)):
                exact=partial_model.motion_bound(role,u,s);old=q[0,1];value=I(max(old.lo,exact.lo),min(old.hi,exact.hi))
                if value.zero:q.c.pop((0,1),None)
                else:q.c[0,1]=value
        for role,q in enumerate((nv,nl,vh)):
            lo=max(D(0),q[0,0].lo);hi=min(D(1),q[0,0].hi)
            if bounds is not None and role<2:lo=max(lo,bounds[role].lo);hi=min(hi,bounds[role].hi)
            q.c[0,0]=I(lo,hi)
        if nl[0,0].lo<=0:raise ValueError('strict light-horizon profile')
        return nv,nl,vh,jac
    def correction_inputs(u,s,*,flatten=True):return inputs(u,s,flatten=flatten,material_only=material_inputs)
    def prep(u,s,box):
        nonlocal bounds
        cache.clear();bounds=None;directions=[];bounds=bounded_cosines(lambda a,b:list(correction_inputs(a,b)[:2]),*box,directional=directions.append,monotone=p['curved'],partial=p['curved']);correction.domain_split_axis=None;e.indices=[];candidates=[];correction.segment_counts=[]
        for role,value in enumerate(bounds):
            if value.lo<=curve.cn[0].hi:index=-1
            else:index=min(max(sum(k.lo<=(value.lo+value.hi)/2 for k in curve.cn)-1,0),len(curve.cn)-2)
            e.indices.append(index)
            count=int(value.lo<=curve.cn[0].hi)+sum(value.lo<=curve.cn[k+1].hi and value.hi>=curve.cn[k].lo for k in range(len(curve.cn)-1))
            correction.segment_counts.append(count)
            if count>8:candidates.extend((weight.hi,axis) for axis,weight in enumerate(directions[0][role]))
        if candidates:correction.domain_split_axis=max(candidates)[1]
    f=I(.04);mix=lambda z:z[0]*(1-f)+z[1]*f;mean=mix(curve.mean)
    combined=MixedLoss(e,f) if mixed_curve else None
    def smooth(u,s):
        nv,nl,vh,jac=inputs(u,s)
        if combined is not None:return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*combined.at(nv,e.indices[0])*combined.at(nl,e.indices[1])*jac/(curve.pi*mean)]
        return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*mix(e.at_centered(nv,e.indices[0]))*mix(e.at_centered(nl,e.indices[1]))*jac/(curve.pi*mean)]
    partial=None
    if transverse_correction:
        from tools.opaque_tangent_transverse import CosinePartials
        partial_model=CosinePartials(p['fields'].coefficients);partial=partial_model.bound
    tight=TangentCorrection(e,correction_inputs,checkpoint=check,horizon=True,curved=p['curved'],range_fallback=p['curved'],line_order=line_order,crossing_support=crossing_support,root_parts=root_parts,**(dict(centered=True,order=correction_order,conditioned=conditioned_correction,mixed_curve=mixed_curve,transverse=transverse_correction,transverse_partial=partial,transverse_ratio=partial_model.ratio if transverse_correction else None,integrated_axis=correction_axis) if correction_order!=3 or conditioned_correction or mixed_curve else {}))
    def pair(box):
        key=tuple((v.lo,v.hi) for v in box)
        if key not in cache:cache[key]=tight(box)
        return cache[key]
    def correction(box):
        q=pair(box);correction.preferred_split_axis=tight.preferred_split_axis if refinement_axis is None else refinement_axis;return [I(0),q[0]]
    def tangent(box):
        check();chart=unflattened_box(box,inside=p['patch'].inside,power=horizon_power) if p['curved'] else box
        angle=lambda a,b:list(correction_inputs(a,b,flatten=not p['curved'])[:2])
        q=material_transport(curve,angle,chart,[1]*3,[.04]*3,horizon=True)['bound'][0]
        jac=correction_inputs(Jet(chart[0]),Jet(chart[1]),flatten=not p['curved'])[3][0,0];bound=q*I(jac.bound)
        ordinary=pair(box)[1]
        if conditioned_correction:tangent.preferred_split_axis=tight.preferred_split_axis if refinement_axis is None else refinement_axis
        return [I(0),ordinary+I(-bound.hi,bound.hi)]
    def choose_cut(box,reason):
        from tools.opaque_knot_partition import knot_cut
        if not knot_targeted or max(correction.segment_counts,default=0)<=8:return None
        check();guidance['calls']+=1
        axis=correction.domain_split_axis
        if axis is None:return None
        # A centre-line angle target need not balance the transverse shape.
        # In the curved preflight keep exact transverse bisection; both child
        # domains are still independently certified and fully charged.
        if curved_knot and axis==1:return None
        role=max(range(2),key=lambda k:correction.segment_counts[k])
        def sample(x,y):
            guidance['field_evaluations']+=1
            return inputs(Jet(x),Jet(y))[role][0,0]
        row=knot_cut(curve.cn,sample,box,axis=axis,checkpoint=check)
        if row is None:return None
        if len(guidance['cuts'])<32:guidance['cuts'].append(dict(box=list(map(str,box)),reason=reason,**row))
        return row['axis'],row['point']
    if cut_diagnostic_boxes is not None:
        if not knot_targeted or not 1<=len(cut_diagnostic_boxes)<=4:raise ValueError('one to four targeted cut diagnostics')
        rows=[]
        for a,b,c,d in cut_diagnostic_boxes:
            box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));prep((a+b)/2,(c+d)/2,box);parent=correction.segment_counts.copy();cut=choose_cut((a,b,c,d),'bounded angular segment count');children=[]
            if cut is not None:
                axis,t=cut
                pairs=[(a,t,c,d),(t,b,c,d)] if axis==0 else [(a,b,c,t),(a,b,t,d)]
                for aa,bb,cc,dd in pairs:
                    prep((aa+bb)/2,(cc+dd)/2,(I(I(aa).lo,I(bb).hi),I(I(cc).lo,I(dd).hi)));children.append(dict(box=list(map(str,(aa,bb,cc,dd))),counts=correction.segment_counts.copy()))
            rows.append(dict(box=list(map(str,(a,b,c,d))),parent_counts=parent,cut=None if cut is None else [cut[0],str(cut[1])],children=children))
        return dict(status='cut_diagnostic_only',rows=rows,guidance=guidance,seconds=time.monotonic()-start)
    if domain_only:
        # This proves only a complete cover satisfying the segment-count domain.
        # It does not perform quadrature, admit derivatives, or reuse its work budget.
        pending=[(Q(0),Q(1),Q(0),Q(1),0)];leaves=[];cells=0
        while pending:
            check()
            if cells>=256:raise Unresolved('domain preflight cell budget',cells=cells,leaves=len(leaves),guidance=guidance)
            a,b,c,d,depth=pending.pop();cells+=1;box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));axis=0 if b-a>=d-c else 1
            try:
                prep((a+b)/2,(c+d)/2,box)
                if max(correction.segment_counts)<=8:
                    leaves.append(dict(box=list(map(str,(a,b,c,d))),depth=depth,counts=correction.segment_counts.copy()));continue
                reason='bounded angular segment count';axis=correction.domain_split_axis if correction.domain_split_axis is not None else axis
            except ValueError as error:reason=str(error)
            if curved_knot and depth<10 and correction.segment_counts and correction.segment_counts[0]>8 and correction.segment_counts[1]<=8:
                from tools.opaque_knot_partition import knot_region
                guidance['curved_attempts']+=1
                try:
                    if bounds is None:raise ValueError('missing parent cosine bounds')
                    eligible=[k for k in curve.cn if bounds[0].lo<k.lo and k.hi<bounds[0].hi]
                    if not eligible:raise ValueError('no interior angular target')
                    target=eligible[len(eligible)//2]
                    def angle(u,s):
                        guidance['field_evaluations']+=1
                        return mapped_inputs(p,u,s)[0]
                    ranges=[I(bounds[0].lo,target.hi),I(target.lo,bounds[0].hi)]
                    counts=[int(v.lo<=curve.cn[0].hi)+sum(v.lo<=curve.cn[k+1].hi and v.hi>=curve.cn[k].lo for k in range(len(curve.cn)-1)) for v in ranges]
                    if max(counts)>8:raise ValueError('curved children exceed segment cap')
                    try:root=knot_region(angle,a,b,Jet.variable(box[1],order=1),target,checkpoint=check,tight_endpoints=True)
                    except ValueError as root_error:
                        if isinstance(root_error,Unresolved) or not entry_knot or depth>8:raise
                        from tools.opaque_knot_partition import KnotEntryChart
                        guidance['entry_attempts']+=1
                        chart=KnotEntryChart(angle,(a,b,c,d),target,checkpoint=check)
                        # Binary tree: pre/post split, then the two post-entry
                        # strips. Charge four nodes, three leaves, two depths.
                        if cells+4>256:raise Unresolved('entry child cell budget',cells=cells,parent=list(map(str,(a,b,c,d))),guidance=guidance)
                        cells+=4;entry_id=len(guidance['entry_cuts'])
                        guidance['entry_cuts'].append(dict(parent=list(map(str,(a,b,c,d))),depth=depth,knot=str(target),entry=str(chart.entry),entry_parameter=str(chart.entry_tangent),extended_lower=str(chart.extended_lower),counts=counts,evaluations=chart.evaluations,coverage='Pre-entry slab plus both post-entry strips; complete parent, only shared edges overlap.'))
                        for part,(side,extra_depth) in enumerate([(1,1),(0,2),(1,2)]):leaves.append(dict(entry_cut=entry_id,part=part,angular_side=side,depth=depth+extra_depth,counts=[counts[side],correction.segment_counts[1]]))
                        continue
                    if cells+2>256:raise Unresolved('curved child cell budget',cells=cells,guidance=guidance)
                    cells+=2;cut_id=len(guidance['curved_cuts'])
                    guidance['curved_cuts'].append(dict(parent=list(map(str,(a,b,c,d))),depth=depth,knot=str(target),root=str(root['root'][0,0]),root_parameter=str(root['root'][0,1]),orientation=1 if root['radial_derivative'].lo>0 else -1,counts=counts,evaluations=root['evaluations'],coverage='The unique interior root partitions this parent completely; children share only the boundary.'))
                    for side,count in enumerate(counts):leaves.append(dict(curved_cut=cut_id,angular_side=side,depth=depth+1,counts=[count,correction.segment_counts[1]]))
                    continue
                except ValueError as error:
                    if isinstance(error,Unresolved):raise
                    guidance['last_curved_refusal']=dict(box=list(map(str,(a,b,c,d))),depth=depth,reason=str(error))
                    if len(guidance['curved_refusals'])<32:guidance['curved_refusals'].append(guidance['last_curved_refusal'])
            if depth==10:raise Unresolved('domain preflight depth budget',cells=cells,box=list(map(str,(a,b,c,d))),domain=reason,counts=correction.segment_counts,leaves=len(leaves),guidance=guidance)
            cut=choose_cut((a,b,c,d),reason) if knot_targeted else None
            if cut is not None:axis=cut[0]
            guidance['last_split_parent']=dict(box=list(map(str,(a,b,c,d))),depth=depth,axis=axis,counts=correction.segment_counts.copy())
            if axis==0:
                mid=horizon_split(a,b,inside=p['patch'].inside) if graded and p['curved'] else (a+b)/2 if cut is None else cut[1];pending.extend([(mid,b,c,d,depth+1),(a,mid,c,d,depth+1)])
            else:
                mid=(c+d)/2 if cut is None else cut[1];pending.extend([(a,b,mid,d,depth+1),(a,b,c,mid,depth+1)])
        return dict(status='domain_only',entry_knot=entry_knot,curved_knot=curved_knot,graded=graded,knot_targeted=knot_targeted,guidance=guidance,piece=piece,motion=motion,horizon_power=horizon_power,cells=cells,leaves=leaves,seconds=time.monotonic()-start,scope='Complete segment-domain cover only; no image or gradient accuracy acceptance.')
    if diagnostic_boxes is not None:
        if not 1<=len(diagnostic_boxes)<=3:raise ValueError('one to three saved diagnostic boxes')
        rows=[]
        for a,b,c,d in diagnostic_boxes:
            box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));prep((a+b)/2,(c+d)/2,box)
            rows.append(dict(box=list(map(str,(a,b,c,d))),ordinary=pair(box),combined_tangent=tangent(box),range_fallbacks=tight.range_fallbacks))
        return dict(status='diagnostic_only',piece=piece,motion=motion,rows=rows,front_material_cache=None if front_material_cache is None else front_material_cache.report(),seconds=time.monotonic()-start)
    try:
        if initial_leaves is None:
            result=integrate_square(smooth,tolerance=tolerance,max_cells=256,max_depth=10,checkpoint=check,prepare=prep,correction_integrator=correction,tangent_correction_integrator=tangent,progress=progress,domain_split=choose_cut if knot_targeted else None)
        else:
            if p['curved'] or knot_targeted:raise ValueError('scheduled front reference requires the plain front domain')
            result=integrate_square(None,tolerance=tolerance,max_cells=256,max_depth=10,checkpoint=check,progress=progress,initial_leaves=initial_leaves,chart_cover=[dict(field=smooth,prepare=prep,correction_integrator=correction,tangent_correction_integrator=tangent,depth=0)])
    except Unresolved as error:raise Unresolved('original motion unresolved',piece=piece,motion=motion,curved=p['curved'],guidance=guidance,front_material_cache=None if front_material_cache is None else front_material_cache.report(),seconds=time.monotonic()-start,cause=error.details) from error
    value=[result['value'][0]+I(a)*result['value'][1] for a in data['material']['albedo']];derivative=[result['derivative'][0]+I(a)*result['derivative'][1] for a in data['material']['albedo']]
    if any(v.width>D('2e-12') for v in value+derivative):raise Unresolved('combined motion RGB radius')
    return dict(root_parts=root_parts,root_partition_queries=tight.root_partition_queries,front_material_cache=None if front_material_cache is None else front_material_cache.report(),crossing_support=crossing_support,line_order=line_order,polynomial_inputs=polynomial_inputs,refinement_axis=refinement_axis,exact_motion=exact_motion,material_inputs=material_inputs,correction_axis=correction_axis,transverse_correction=transverse_correction,correction_order=correction_order,conditioned_correction=conditioned_correction,mixed_curve=mixed_curve,knot_targeted=knot_targeted,guidance=guidance,horizon_power=horizon_power,piece=piece,face=p['face'],motion=motion,anchor=anchor,curved=p['curved'],value=value,derivative=derivative,basis=result,range_fallbacks=tight.range_fallbacks,certificate=p['certificate'],visibility=p['visibility'],curved_certificate=p['patch'].certificate if p['curved'] else None,seconds=time.monotonic()-start,scope='One original visible piece and declared packed-input direction. Screen translation keeps attributes attached; no world-vertex/camera-pose or recomputed-normal admission. Curved maps include local horizon and Jacobian motion.')


def interior_band_reference(data,*,box,knot_index,motion='normal_tilt',domain_only=True):
    """Bounded local integration experiment, never a full-piece certificate.

    Two adjacent exact view-angle knots bound an interior strip. One certified
    light-angle branch is required throughout; boundary entry/exit is refused.
    The root map carries view-boundary transport, so no duplicate flux is added.
    """
    from tools.opaque_knot_partition import knot_band
    if type(domain_only)!=bool:raise ValueError('explicit band domain flag')
    start=time.monotonic();evaluations=0;cache={}
    def check():
        if time.monotonic()-start>=90:raise Unresolved('interior band deadline',evaluations=evaluations)
    a,b,c,d=map(Q,box)
    if not 0<=a<b<=1 or not 0<=c<d<=1:raise ValueError('unit interior band box')
    p=prepare(data,piece=8,motion=motion,anchor=0,horizon_power=4,checkpoint=check)
    curve=PreparedPrincipled(data['budget'],1.);extension=HorizonExtension(curve)
    if type(knot_index)!=int or not 0<=knot_index<len(curve.cn)-1:raise ValueError('adjacent view knots required')
    low,high=curve.cn[knot_index:knot_index+2]
    def angle(u,s):
        nonlocal evaluations
        check();evaluations+=1;return mapped_inputs(p,u,s)[0]
    def inputs(t,y):
        check();s=c+(d-c)*y;key=(s.order,tuple(sorted((ij,str(v.lo),str(v.hi)) for ij,v in s.c.items())))
        if key not in cache:
            if len(cache)>=64:cache.pop(next(iter(cache)))
            cache[key]=knot_band(angle,a,b,t.lift(0),s,low,high,checkpoint=check)
        band=cache[key];u=band['coordinate']+t*band['jacobian']
        u.c[0,0]=I(max(u[0,0].lo,I(a).lo),min(u[0,0].hi,I(b).hi))
        nv,nl,vh,jac=mapped_inputs(p,u,s)
        nv.c[0,0]=I(max(nv[0,0].lo,low.lo),min(nv[0,0].hi,high.hi))
        for q in (nl,vh):q.c[0,0]=I(max(D(0),q[0,0].lo),min(D(1),q[0,0].hi))
        return nv,nl,vh,jac*band['jacobian']*(d-c)
    try:
        whole=inputs(Jet(I(0,1)),Jet(I(0,1)));light=whole[1][0,0]
        candidates=[i for i in range(len(curve.cn)-1) if light.lo>=curve.cn[i].lo and light.hi<=curve.cn[i+1].hi]
        if light.lo>0 and light.hi<curve.cn[0].lo:candidates=[-1]
        if len(candidates)!=1:raise Unresolved('single light branch not certified',light=str(light),evaluations=evaluations)
        light_index=candidates[0]
        report=dict(status='domain_only',scope='One local interior angular band of piece8, not full piece/pixel accuracy.',box=list(map(str,box)),knot_index=knot_index,light_index=light_index,view_range=str(whole[0][0,0]),light_range=str(light))
        if not domain_only:
            f=I(.04);mix=lambda z:z[0]*(1-f)+z[1]*f;mean=mix(curve.mean)
            def field(t,y):
                nv,nl,vh,jac=inputs(t,y)
                # Domain membership is certified globally above. Intersection
                # changes only the value enclosure; all derivatives survive.
                nl.c[0,0]=I(max(nl[0,0].lo,light.lo),min(nl[0,0].hi,light.hi))
                return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*mix(extension.at(nv,knot_index))*mix(extension.at(nl,light_index))*jac/(curve.pi*mean)]
            result=integrate_square(field,tolerance='5e-13',max_cells=256,max_depth=10,checkpoint=check)
            report.update(status='local_band_integrated',basis=result)
        report.update(evaluations=evaluations,seconds=time.monotonic()-start);return report
    except ValueError as error:
        raise Unresolved('interior band unresolved',cause=getattr(error,'details',str(error)),evaluations=evaluations,seconds=time.monotonic()-start) from error



def angular_cover_reference(data,*,motion='normal_tilt',domain_only=True,angle_coordinate=False):
    """Fresh complete angular cover; accuracy remains a separate guarded gate."""
    from tools.opaque_knot_partition import MonotoneAngularCover
    if domain_only is not True:raise ValueError('angular cover is domain-preflight only')
    if type(angle_coordinate)!=bool:raise ValueError('explicit angle coordinate flag')
    start=time.monotonic()
    def check():
        if time.monotonic()-start>=90:raise Unresolved('angular cover reference deadline')
    p=prepare(data,piece=8,motion=motion,anchor=0,horizon_power=4,checkpoint=check)
    if not p['curved'] or p['patch'].inside:raise ValueError('original outside horizon profile required')
    curve=PreparedPrincipled(data['budget'],1.)
    angle=lambda u,s:mapped_angle(p,u,s)
    outer=angle(Jet(1),Jet(0))[0,0]
    indices=[i for i in range(6,len(curve.cn),6) if curve.cn[i].hi<outer.lo]
    cover=MonotoneAngularCover(angle,[curve.cn[i] for i in indices],checkpoint=check)
    bounds=bounded_cosines(lambda u,s:list(mapped_inputs(p,u,s,flatten=False)[:2]),I(0,1),I(0,1),monotone=True,partial=True)
    count=lambda v:int(v.lo<=curve.cn[0].hi)+sum(v.lo<=curve.cn[i+1].hi and v.hi>=curve.cn[i].lo for i in range(len(curve.cn)-1))
    rows=[]
    for i,chart in enumerate(cover.charts):
        check();row=cover.map_angle_cover(i,Jet(I(0,1)),Jet(I(0,1)),lambda u,s:radial_angle_derivative(p,u,s)) if angle_coordinate else cover.map(i,Jet(I(0,1)),Jet(I(0,1)))
        physical=row['coordinate_kind']=='physical_angle'
        if physical:
            radial=row['radius'];fields=p['fields'](radial,row['s']);n,v=fields['n'],fields['v'];l=[t.lift(k) for k in (1,1,2)]
            dot=lambda a,b:sum((x*y for x,y in zip(a,b)),t.lift(0))
            nn,vv,ll=dot(n,n),dot(v,v),dot(l,l)
            nv=row['angle'];nl=dot(n,l)/(nn*ll).sqrt();vh=((1+dot(v,l)/(vv*ll).sqrt())/2).sqrt()
            jac=radial*radial.lift(p['patch'].det[0],tangent=p['patch'].det[1])
        else:nv,nl,vh,jac=mapped_inputs(p,row['u'],row['s'],flatten=False)
        interval=row['angular_range'];counts=[count(interval),count(bounds[1])]
        if max(counts)>8:raise Unresolved('angular chart segment cap',chart=i,counts=counts,cells=cover.cells)
        physical=jac*row['jacobian']
        rows.append(dict(coordinate_kind=row.get('coordinate_kind','radial'),chart=i,band=chart['band'],depth=chart['depth'],counts=counts,view=str(interval),radial=str(row['u'][0,0]),transverse=str(row['s'][0,0]),jacobian=str(row['jacobian'][0,0]),physical_jacobian=str(physical[0,0]),physical_parameter=str(physical[0,1])))
    return dict(status='complete_angular_domain_cover',angle_coordinate=angle_coordinate,piece=8,motion=motion,cells=cover.cells,depth=cover.depth,charts=len(cover.charts),bands=len(cover.levels)+1,level_indices=indices,events={str(indices[i]):dict(location=str(v[0,0]),parameter=str(v[0,1])) for i,v in cover.events.items()},radial_slope=str(cover.radial),outer_edge_slope=str(cover.edge_slope),rows=rows,evaluations=cover.evaluations,seconds=time.monotonic()-start,scope='Complete original piece8 angular-domain cover and first moving physical-Jacobian enclosures; no quadrature or installed gradient acceptance.')


def angular_chart_reference(data,*,chart_index,diagnostic_box=None,angle_coordinate=False,tight_discriminant=False):
    """One bounded integration diagnostic, not a complete-piece certificate.

    Construction charges the entire cover tree. Only refinements of this
    selected leaf can use the remaining cell/depth allowance. The existing
    smooth-extension residual and conservative jump-transport bounds remain
    separate from the moving coordinate Jacobian. No full trial is enabled.
    """
    from tools.opaque_knot_partition import MonotoneAngularCover
    if type(angle_coordinate)!=bool:raise ValueError('explicit angle coordinate flag')
    start=time.monotonic();cover=None;last=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('angular chart reference deadline')
    p=prepare(data,piece=8,motion='normal_tilt',anchor=0,horizon_power=4,checkpoint=check,tight_discriminant=tight_discriminant)
    from tools.opaque_full_material_reference import CenteredHorizonExtension
    curve=PreparedPrincipled(data['budget'],1.);e=CenteredHorizonExtension(curve)
    angle=lambda u,s:mapped_angle(p,u,s)
    outer=angle(Jet(1),Jet(0))[0,0];indices=[i for i in range(6,len(curve.cn),6) if curve.cn[i].hi<outer.lo]
    cover=MonotoneAngularCover(angle,[curve.cn[i] for i in indices],checkpoint=check)
    if type(chart_index)!=int or not 0<=chart_index<len(cover.charts):raise ValueError('qualified chart index required')
    chart=cover.charts[chart_index];global_bounds=bounded_cosines(lambda u,s:list(mapped_inputs(p,u,s,flatten=False)[:2]),I(0,1),I(0,1),monotone=True,partial=True);bounds=None;cache={}
    def inputs(t,y):
        row=cover.map_angle(chart_index,t,y,lambda u,s:radial_angle_derivative(p,u,s)) if angle_coordinate else cover.map(chart_index,t,y)
        physical=row['coordinate_kind']=='physical_angle'
        if physical:
            radial=row['radius'];fields=p['fields'](radial,row['s']);n,v=fields['n'],fields['v'];l=[t.lift(k) for k in (1,1,2)]
            dot=lambda a,b:sum((x*y for x,y in zip(a,b)),t.lift(0))
            nn,vv,ll=dot(n,n),dot(v,v),dot(l,l)
            nv=row['angle'];nl=dot(n,l)/(nn*ll).sqrt();vh=((1+dot(v,l)/(vv*ll).sqrt())/2).sqrt()
            jac=radial*radial.lift(p['patch'].det[0],tangent=p['patch'].det[1])
        else:nv,nl,vh,jac=mapped_inputs(p,row['u'],row['s'],flatten=False)
        # The certified inverse makes the angle exactly this affine target,
        # including its zero transverse/motion derivatives. Other fields and
        # the complete inverse/physical Jacobian retain all mixed coefficients.
        if angle_coordinate:nv=row['angle']
        for role,(q,bound) in enumerate(zip((nv,nl,vh),(row['angular_range'],global_bounds[1],I(0,1)))):
            lo=max(q[0,0].lo,bound.lo);hi=min(q[0,0].hi,bound.hi)
            if bounds is not None and role<2:lo=max(lo,bounds[role].lo);hi=min(hi,bounds[role].hi)
            q.c[0,0]=I(lo,hi)
        return nv,nl,vh,jac*row['jacobian']
    def prep(t,y,box):
        nonlocal bounds
        bounds=None;cache.clear();bounds=bounded_cosines(lambda a,b:list(inputs(a,b)[:2]),*box);e.indices=[]
        for value in bounds:
            count=int(value.lo<=curve.cn[0].hi)+sum(value.lo<=curve.cn[k+1].hi and value.hi>=curve.cn[k].lo for k in range(len(curve.cn)-1))
            if count>8:raise ValueError('bounded angular segment count')
            e.indices.append(-1 if value.lo<=curve.cn[0].hi else min(max(sum(k.lo<=(value.lo+value.hi)/2 for k in curve.cn)-1,0),len(curve.cn)-2))
    f=I(.04);mix=lambda z:z[0]*(1-f)+z[1]*f;mean=mix(curve.mean)
    combined=MixedLoss(e,f) if p.get('mixed_curve',False) else None
    def smooth(t,y):
        nv,nl,vh,jac=inputs(t,y)
        if combined is not None:
            return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*combined.at(nv,e.indices[0])*combined.at(nl,e.indices[1])*jac/(curve.pi*mean)]
        return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*mix(e.at_centered(nv,e.indices[0]))*mix(e.at_centered(nl,e.indices[1]))*jac/(curve.pi*mean)]
    tight=TangentCorrection(e,inputs,checkpoint=check,horizon=True,curved=True,range_fallback=True,centered=p.get('gradient_guidance',False),order=p.get('correction_order',3),conditioned=p.get('conditioned_correction',False),partitioned_range=p.get('conditioned_correction',False),mixed_curve=p.get('mixed_curve',False))
    def pair(box):
        key=tuple((v.lo,v.hi) for v in box)
        if key not in cache:cache[key]=tight(box)
        return cache[key]
    def correction(box):return [I(0),pair(box)[0]]
    def tangent(box):
        q=material_transport(curve,lambda t,y:list(inputs(t,y)[:2]),box,[1]*3,[.04]*3,horizon=True)['bound'][0]
        jac=inputs(Jet(box[0]),Jet(box[1]))[3][0,0];bound=q*I(jac.bound)
        # Conservative jump uncertainty is not an added signed boundary term.
        # It may also bound zero flux at a mapped fixed-level edge; no boundary
        # motion is added twice to the central derivative.
        ordinary=pair(box)[1]
        if p.get('gradient_guidance'):tangent.preferred_split_axis=tight.preferred_split_axis
        return [I(0),ordinary+I(-bound.hi,bound.hi)]
    def progress(row):
        nonlocal last
        last=row
    if diagnostic_box is not None:
        import traceback
        a,b,c,d=map(Q,diagnostic_box)
        if not 0<=a<b<=1 or not 0<=c<d<=1:raise ValueError('unit diagnostic box required')
        box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));rows=[]
        def discriminant():
            y=Jet.variable(box[1],order=10);row=cover.map(chart_index,Jet(box[0],order=0),Jet(box[1],order=0));sv=Jet.variable(row['s'][0,0],order=10)
            co=p['patch'].discriminant;q=p['patch'].lift(sv,co[-1])
            for c in reversed(co[:-1]):q=q*sv+p['patch'].lift(sv,c)
            return [q]
        stages=[('discriminant',discriminant),('prepare',lambda:prep((a+b)/2,(c+d)/2,box)),('center',lambda:smooth(Jet((a+b)/2),Jet((c+d)/2))),('radial10',lambda:smooth(Jet.variable(box[0],order=10),Jet(box[1],order=10))),('transverse10',lambda:smooth(Jet(box[0],order=10),Jet.variable(box[1],order=10))),('residual',lambda:pair(box)),('transport',lambda:tangent(box))]
        def factors(axis):
            args=[Jet(v,order=10) for v in box];args[axis]=Jet.variable(box[axis],order=10)
            nv,nl,vh,jac=inputs(*args)
            return [nv,nl,vh,jac,mix(e.at(nv,e.indices[0])),mix(e.at(nl,e.indices[1]))]
        stages += [('radial_factors10',lambda:factors(0)),('transverse_factors10',lambda:factors(1))]
        for name,call in stages:
            check()
            try:
                result=call();rows.append(dict(stage=name,status='evaluated',result=None if result is None else [str(v) if isinstance(v,I) else dict(value=str(v[0,0]),parameter=str(v[0,1]),high=str(v[v.order,0]),mixed=str(v[v.order,1])) for v in result]))
            except ValueError as error:
                if isinstance(error,Unresolved):raise
                rows.append(dict(stage=name,status='refused',reason=str(error),traceback=traceback.format_exc()))
                if name=='prepare':break
        return dict(status='chart_diagnostic_only',angle_coordinate=angle_coordinate,chart=chart_index,box=list(map(str,diagnostic_box)),rows=rows,evaluations=cover.evaluations,seconds=time.monotonic()-start)
    try:
        result=integrate_square(smooth,tolerance='5e-13',max_cells=256-cover.cells+1,max_depth=10-chart['depth'],checkpoint=check,prepare=prep,correction_integrator=correction,tangent_correction_integrator=tangent,progress=progress)
        return dict(status='local_angular_chart_integrated',angle_coordinate=angle_coordinate,chart=chart_index,band=chart['band'],basis=result,cover_cells=cover.cells,charged_cells=cover.cells+result['cells']-1,cover_depth=chart['depth'],evaluations=cover.evaluations,range_fallbacks=tight.range_fallbacks,seconds=time.monotonic()-start,scope='One original Principled chart and normal_tilt derivative; not complete-piece or installed-gradient acceptance.')
    except ValueError as error:
        raise Unresolved('angular chart integration unresolved',angle_coordinate=angle_coordinate,chart=chart_index,band=chart['band'],cover_cells=cover.cells,cover_depth=chart['depth'],evaluations=cover.evaluations,seconds=time.monotonic()-start,last=last,cause=getattr(error,'details',str(error))) from error



def angular_material_chart(p,curve,cover,chart_index,global_bounds,*,checkpoint=lambda:None):
    """Own mutable extension, bounds and residual cache for exactly one chart.

    Arithmetic follows the frozen local diagnostic. Shared cover root caching
    stores coordinate jets only; it does not own this chart's material state.
    """
    from tools.opaque_full_material_reference import CenteredHorizonExtension
    if type(chart_index)!=int or not 0<=chart_index<len(cover.charts):raise ValueError('qualified chart index required')
    e=CenteredHorizonExtension(curve);bounds=None;cache={};hints={};check=checkpoint
    def coordinates(t,y):
        if p.get('physical_angle'):
            from tools.opaque_physical_angle import physical_angle_chart
            row=physical_angle_chart(p['physical_angle'],cover,chart_index,t,y,lower=lambda s:(1-p['patch'].root(Jet(s[0,0])) if p.get('inside_angle') else p['patch'].root(Jet(s[0,0])))[0,0])
            if row is not None:return row
        method=cover.map_exit_angle if p.get('exit_angle') and cover.charts[chart_index]['high'] is None and cover.charts[chart_index]['low'] is not None else cover.map_angle_cover
        return method(chart_index,t,y,p['prepared_angle'].radial if 'prepared_angle' in p else (lambda u,s:radial_angle_derivative(p,u,s)))
    def calculate_inputs(t,y,*,material_only=False):
        row=coordinates(t,y)
        physical=row['coordinate_kind']=='physical_angle'
        if physical:
            radial=1-row['radius'] if p.get('inside_angle') else row['radius'];nv=row['angle']
            if 'polynomial_light_model' in p:
                nl,vh=p['polynomial_light_model'].light_highlight(radial,row['s'],material_only=material_only)
            else:
                fields=p['fields'](radial,row['s']);n,v=fields['n'],fields['v'];l=[t.lift(k) for k in (1,1,2)]
                dot=lambda a,b:sum((x*y for x,y in zip(a,b)),t.lift(0))
                nn,vv,ll=dot(n,n),(None if material_only else dot(v,v)),dot(l,l)
                nl=dot(n,l)/(nn*ll).sqrt()
                # This private placeholder is never consumed by correction/transport.
                # Smooth radiance always requests the full view-highlight jet.
                vh=t.lift(0) if material_only else ((1+dot(v,l)/(vv*ll).sqrt())/2).sqrt()
            jac=radial*radial.lift(p['patch'].det[0],tangent=p['patch'].det[1])
        else:nv,nl,vh,jac=mapped_inputs(p,1-row['u'] if p.get('inside_angle') else row['u'],row['s'],flatten=False,material_only=material_only)
        if physical or p.get('exit_angle') and cover.charts[chart_index]['high'] is None:
            from tools.opaque_normal_light import normal_light_spatial_bound,normal_light_tangent_bound
            if not physical:radial=p['patch'].radial_map(1-row['u'] if p.get('inside_angle') else row['u'],row['s'])[0]
            if not physical:
                for key,refiner in ([((1,0),normal_light_spatial_bound)] if t.order else [])+[((0,1),normal_light_tangent_bound)]:
                    refined=refiner(p['fields'].coefficients['n'],radial,row['s']);original=nl[key]
                    nl.c[key]=I(max(original.lo,refined.lo),min(original.hi,refined.hi))
            if row['coordinate_kind'] in ('angle','physical_angle') and not p.get('inside_angle'):
                for key in ([ (1,0) ] if t.order else [])+[(0,1)]:
                    # The cover proves both physical coordinates stay in the unit domain.
                    radius=Jet(order=radial.order,coeff=radial.c);radius.c[0,0]=I(max(I(0).lo,radial[0,0].lo),min(I(1).hi,radial[0,0].hi))
                    refined=p['angle_light_differential'].bound(radius,row['s'],row['angle'],key);original=nl[key]
                    nl.c[key]=I(max(original.lo,refined.lo),min(original.hi,refined.hi))
        if p.get('full_light_differential') and physical and t.order:
            radius=Jet(order=radial.order,coeff=radial.c)
            radius.c[0,0]=I(max(I(0).lo,radial[0,0].lo),min(I(1).hi,radial[0,0].hi))
            for key,refined in p['angle_light_differential'].coefficient_bounds(radius,row['s'],row['angle']).items():
                original=nl[key];nl.c[key]=I(max(original.lo,refined.lo),min(original.hi,refined.hi))
        # The certified inverse makes the angle exactly this affine target,
        # including its zero transverse/motion derivatives. Other fields and
        # the complete inverse/physical Jacobian retain all mixed coefficients.
        if row['coordinate_kind'] in ('angle','physical_angle'):nv=row['angle']
        for role,(q,bound) in enumerate(zip((nv,nl,vh),(row['angular_range'],global_bounds[1],I(0,1)))):
            lo=max(q[0,0].lo,bound.lo);hi=min(q[0,0].hi,bound.hi)
            q.c[0,0]=I(lo,hi)
        return nv,nl,vh,jac*row['jacobian']
    from functools import partial
    material_calculation=partial(calculate_inputs,material_only=True)
    def inputs(t,y,*,material_only=False):
        check()
        memo=p.get('material_inputs_cache')
        evaluate=material_calculation if material_only else calculate_inputs
        # Coordinate/material calculation is independent of the current proof
        # region. Cache its globally clipped, immutable endpoints; apply every
        # region's freshly certified bounds only to the returned private copy.
        # Evaluator identity distinguishes full from material-only queries.
        result=evaluate(t,y) if memo is None else memo(chart_index,evaluate,t,y)
        if memo is not None and not material_only and p.get('material_inputs') and t.order==0:
            # Before regional clamps, the full sample has the same view/light
            # and Jacobian arithmetic as the material-only query. Its unused
            # highlight becomes the private exact-zero placeholder only.
            unused=t.lift(0);unused.c[0,0]=I(0) # Same explicit global clamp as calculate_inputs.
            memo.seed(chart_index,material_calculation,t,y,(result[0],result[1],unused,result[3]))
        if bounds is not None:
            for q,bound in zip(result[:2],bounds):
                value=q[0,0];q.c[0,0]=I(max(value.lo,bound.lo),min(value.hi,bound.hi))
        check();return result
    def correction_inputs(t,y):return inputs(t,y,material_only=p.get('material_inputs',False))
    def prep(t,y,box):
        nonlocal bounds
        bounds=None;cache.clear();hints.clear();bounds=bounded_cosines(lambda a,b:list(correction_inputs(a,b)[:2]),*box);e.indices=[]
        for value in bounds:
            count=int(value.lo<=curve.cn[0].hi)+sum(value.lo<=curve.cn[k+1].hi and value.hi>=curve.cn[k].lo for k in range(len(curve.cn)-1))
            if count>8:raise ValueError('bounded angular segment count')
            e.indices.append(-1 if value.lo<=curve.cn[0].hi else min(max(sum(k.lo<=(value.lo+value.hi)/2 for k in curve.cn)-1,0),len(curve.cn)-2))
    f=I(.04);mix=lambda z:z[0]*(1-f)+z[1]*f;mean=mix(curve.mean)
    combined=MixedLoss(e,f) if p.get('mixed_curve',False) else None
    def smooth(t,y):
        nv,nl,vh,jac=inputs(t,y)
        if combined is not None:
            return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*combined.at(nv,e.indices[0])*combined.at(nl,e.indices[1])*jac/(curve.pi*mean)]
        return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*mix(e.at_centered(nv,e.indices[0]))*mix(e.at_centered(nl,e.indices[1]))*jac/(curve.pi*mean)]
    tight=TangentCorrection(e,correction_inputs,checkpoint=check,horizon=True,curved=True,range_fallback=True,centered=p.get('gradient_guidance',False),order=p.get('correction_order',3),conditioned=p.get('conditioned_correction',False),partitioned_range=p.get('conditioned_correction',False),mixed_curve=p.get('mixed_curve',False),transverse=p.get('curved_transverse',False),curved_transverse=p.get('curved_transverse',False),integrated_axis=p.get('correction_axis',1),local_crossings=p.get('world_motion') is not None,crossing_support=p.get('crossing_support',False),root_parts=p.get('root_parts',1))
    def pair(box):
        key=tuple((v.lo,v.hi) for v in box)
        if key not in cache:
            cache[key]=tight(box)
            if p.get('exit_angle'):hints[key]=tight.preferred_split_axis
        if p.get('exit_angle'):tight.preferred_split_axis=hints[key]
        return cache[key]
    def correction(box):
        result=pair(box)
        if p.get('exit_angle'):correction.preferred_split_axis=tight.preferred_split_axis
        return [I(0),result[0]]
    def physical_flux(role,box,knots):
        if not p.get('mapped_transport'):return None
        if (role!=0 and not p.get('exit_angle')) or cover.charts[chart_index]['high'] is not None:
            from tools.opaque_knot_transport import transversal_flux
            row=transversal_flux(lambda a,b:inputs(a,b)[role],box,knots)
            jac=correction_inputs(Jet(box[0]),Jet(box[1]))[3][0,0]
            row=dict(row);row['bound']=row['bound']*I(jac.bound);return row
        from tools.opaque_mapped_flux import mapped_monotone_flux
        check();t,y=[Jet(v) for v in box];mapped=coordinates(t,y)
        physical=mapped['coordinate_kind']=='physical_angle'
        physical_jac=((1-mapped['radius'] if p.get('inside_angle') else mapped['radius'])*mapped['radius'].lift(p['patch'].det[0],tangent=p['patch'].det[1]))[0,0] if physical else p['patch'].radial_map(1-mapped['u'] if p.get('inside_angle') else mapped['u'],mapped['s'])[1][0,0]
        q=inputs(t,y)[role];span=I(mapped['s'][0,0].width);slope=p['physical_angle']._slope(mapped['radius'][0,0],mapped['s'][0,0]) if physical else cover.radial
        if role==1:
            from tools.opaque_normal_light import normal_light_spatial_bound
            s0=Jet(mapped['s'][0,0],order=1)
            radius=(1-Jet.variable(mapped['radius'][0,0],order=1) if p.get('inside_angle') else Jet.variable(mapped['radius'][0,0],order=1)) if physical else p['patch'].radial_map(1-Jet.variable(mapped['u'][0,0],order=1) if p.get('inside_angle') else Jet.variable(mapped['u'][0,0],order=1),s0)[0]
            slope=normal_light_spatial_bound(p['fields'].coefficients['n'],radius,s0)
        return mapped_monotone_flux(q[0,0],q[0,1],knots,radial_slope=slope,transverse_extent=span,physical_jacobian=physical_jac)
    def tangent(box):
        q=material_transport(curve,lambda t,y:list(correction_inputs(t,y)[:2]),box,[1]*3,[.04]*3,horizon=True,flux_override=physical_flux if p.get('mapped_transport') else None,pure_flux_override=bool(p.get('mapped_transport')))['bound'][0]
        jac=correction_inputs(Jet(box[0]),Jet(box[1]))[3][0,0];bound=q if p.get('mapped_transport') else q*I(jac.bound)
        # Conservative jump uncertainty is not an added signed boundary term.
        # It may also bound zero flux at a mapped fixed-level edge; no boundary
        # motion is added twice to the central derivative.
        ordinary=pair(box)[1]
        if p.get('gradient_guidance'):tangent.preferred_split_axis=tight.preferred_split_axis
        return [I(0),ordinary+I(-bound.hi,bound.hi)]
    return dict(field=smooth,prepare=prep,correction_integrator=correction,tangent_correction_integrator=tangent,depth=cover.charts[chart_index]['depth'])


def _angular_selection(piece,motion,anchor):
    if type(piece)!=int or not 0<=piece<13 or motion not in ('normal_tilt','view_x','screen_x','world_x','camera_x','world_shear_xz') or type(anchor)!=int or anchor not in (0,1,2):
        raise ValueError('declared angular piece/motion/anchor required')


def angular_material_cover(data,*,piece=8,motion='normal_tilt',anchor=0,checkpoint=lambda:None,tight_discriminant=False,algebraic_inverse=False,mapped_transport=False,exit_angle=False,unsquared_inverse=False,physical_angle=False,smooth_partition=None,tensor_rule=None,tensor_upgrade_charts=None,compiled_series=False,correction_order=3,conditioned_correction=False,contraction_stop=False,mixed_curve=False,material_inputs=False,curved_transverse=False,polynomial_light=False,world_profile=None,correction_axis=1,crossing_support=False,root_parts=1):
    """Fresh certified geometry plus separately owned material callbacks."""
    if type(crossing_support)!=bool or crossing_support and not (curved_transverse and conditioned_correction and correction_order==4):raise ValueError('qualified curved crossing support required')
    if type(root_parts)!=int or root_parts not in (1,2,4) or root_parts>1 and not crossing_support:raise ValueError('qualified curved root partition required')
    if type(correction_axis)!=int or correction_axis not in (0,1) or correction_axis!=1 and not conditioned_correction:raise ValueError('explicit conditioned correction axis required')
    _angular_selection(piece,motion,anchor)
    if type(curved_transverse)!=bool or curved_transverse and not (physical_angle and conditioned_correction and correction_order==4):raise ValueError('curved transverse requires explicit conditioned fourth-order physical-angle profile')
    if type(polynomial_light)!=bool or polynomial_light and not (physical_angle and material_inputs and conditioned_correction and correction_order==4 and curved_transverse):raise ValueError('polynomial light requires explicit conditioned fourth-order transverse physical-angle material profile')
    from tools.opaque_knot_partition import MonotoneAngularCover
    if type(algebraic_inverse)!=bool or type(mapped_transport)!=bool or (algebraic_inverse or mapped_transport) and not tight_discriminant:raise ValueError('algebraic inverse/transport requires explicit tighter horizon')
    if type(exit_angle)!=bool or exit_angle and not (mapped_transport and algebraic_inverse):raise ValueError('exit angle requires qualified algebraic inverse and mapped transport')
    if type(unsquared_inverse)!=bool or unsquared_inverse and not algebraic_inverse:raise ValueError('unsquared inverse requires explicit algebraic candidate')
    if type(physical_angle)!=bool or physical_angle and not (unsquared_inverse and exit_angle):raise ValueError('physical angle requires qualified unsquared and exit candidate')
    if type(material_inputs)!=bool or material_inputs and not physical_angle:raise ValueError('material-only queries require physical-angle candidate')
    if type(mixed_curve)!=bool or mixed_curve and not conditioned_correction:raise ValueError('mixed material requires the conditioned physical-angle profile')
    if type(contraction_stop)!=bool or contraction_stop and not physical_angle:raise ValueError('contraction stopping requires physical-angle candidate')
    if type(conditioned_correction)!=bool or conditioned_correction and correction_order!=4:raise ValueError('conditioned correction requires fourth-order proof')
    if type(correction_order)!=int or correction_order not in (3,4) or correction_order==4 and not physical_angle:raise ValueError('qualified physical correction order required')
    if type(compiled_series)!=bool or compiled_series and not physical_angle:raise ValueError('compiled series requires explicit physical-angle candidate')
    if tensor_upgrade_charts is not None:
        if tensor_rule not in ('gauss3x12-local4','gauss4x12-local4') or not isinstance(tensor_upgrade_charts,list) or not tensor_upgrade_charts or any(type(i)!=int or i<0 or i>=128 for i in tensor_upgrade_charts) or len(set(tensor_upgrade_charts))!=len(tensor_upgrade_charts):raise ValueError('explicit unique tensor upgrade chart ids required')
    if tensor_rule is not None:
        from tools.opaque_tensor_smooth import validate
        validate(tensor_rule)
        if not physical_angle or smooth_partition is not None:raise ValueError('directional smooth proof requires physical angle and owns its remainder')
    if smooth_partition is not None:
        from tools.opaque_local_smooth import validate
        validate(smooth_partition)
        if not physical_angle:raise ValueError('localized smooth proof requires physical angle candidate')
    checkpoint();p=prepare(data,piece=piece,motion=motion,anchor=anchor,horizon_power=4,checkpoint=checkpoint,tight_discriminant=tight_discriminant,**({'world_profile':world_profile} if world_profile is not None else {}))
    if not p['curved'] or p['patch'].inside:raise ValueError('outside curved angular profile required')
    if tight_discriminant:
        from tools.opaque_horizon_cache import HorizonCache
        p['horizon_cache']=HorizonCache(p['patch'].sqrt_discriminant)
        p['patch'].sqrt_discriminant=p['horizon_cache']
        from tools.opaque_prepared_angle import PreparedAngle,MaterialInputCache
        p['prepared_angle']=PreparedAngle(p)
        p['material_inputs_cache']=MaterialInputCache()
        if algebraic_inverse:
            from tools.opaque_algebraic_angle_inverse import QuarticAngleResidual,UnsquaredAngleResidual
            p['inverse_residual']=(UnsquaredAngleResidual if unsquared_inverse else QuarticAngleResidual)(p['prepared_angle'])
    if exit_angle:
        from tools.opaque_angle_light import AngleLightDifferential
        p['angle_light_differential']=AngleLightDifferential(p['fields'].coefficients)
    if physical_angle:
        from tools.opaque_physical_angle import PhysicalAngleResidual
        p['physical_angle']=PhysicalAngleResidual(p['fields'].coefficients,compiled_series=compiled_series,contraction_stop=contraction_stop)
    p['polynomial_light']=polynomial_light
    if polynomial_light:
        from tools.opaque_polynomial_inputs import PolynomialCosines
        p['polynomial_light_model']=PolynomialCosines(p['fields'].coefficients)
    p['crossing_support']=crossing_support;p['root_parts']=root_parts
    p['correction_axis']=correction_axis
    p['curved_transverse']=curved_transverse
    p['material_inputs']=material_inputs
    p['mixed_curve']=mixed_curve
    p['conditioned_correction']=conditioned_correction
    p['correction_order']=correction_order
    p['gradient_guidance']=tensor_rule in ('gauss3x8-local2','gauss3x8-local4','gauss3x8-local4x4','gauss3x12-local4','gauss4x12-local4')
    p['full_light_differential']=False  # Saved comparison did not justify its added cost.
    p['exit_angle']=exit_angle
    p['mapped_transport']=mapped_transport
    curve=PreparedPrincipled(data['budget'],1.);angle=p.get('prepared_angle') or (lambda u,s:mapped_angle(p,u,s))
    outer=angle(Jet(1),Jet(0))[0,0];indices=[i for i in range(6,len(curve.cn),6) if curve.cn[i].hi<outer.lo]
    cover=MonotoneAngularCover(angle,[curve.cn[i] for i in indices],checkpoint=checkpoint,value_field=None if 'prepared_angle' not in p else p['prepared_angle'].value_only,inverse_residual=p.get('inverse_residual'))
    bounds=bounded_cosines(lambda u,s:list(mapped_inputs(p,u,s,flatten=False)[:2]),I(0,1),I(0,1),monotone=True,partial=True)
    charts=[]
    for i in range(len(cover.charts)):
        checkpoint();charts.append(angular_material_chart(p,curve,cover,i,bounds,checkpoint=checkpoint))
        c=cover.charts[i]
        # Positive fixed-upper-angle interiors and entries; horizon/exits retain their qualified rule.
        if tensor_rule is not None and c['low'] is not None and c['high'] is not None:charts[-1]['tensor_rule']='gauss3x8-local4' if tensor_upgrade_charts is not None and i not in tensor_upgrade_charts else tensor_rule
        if smooth_partition is not None and c['low'] is not None and c['high'] is not None and c['start'] is None and c['end'] is None:charts[-1]['remainder_partition']=smooth_partition
    if tensor_upgrade_charts is not None and any(i>=len(charts) or charts[i].get('tensor_rule')!=tensor_rule for i in tensor_upgrade_charts):raise ValueError('upgrade ids must name qualified tensor charts')
    checkpoint();return p,curve,cover,charts


def angular_piece_reference(data,*,piece=8,motion='normal_tilt',anchor=0,progress=lambda row:None,tight_discriminant=False,algebraic_inverse=False,mapped_transport=False,exit_angle=False,unsquared_inverse=False,physical_angle=False,smooth_partition=None,tensor_rule=None,tensor_upgrade_charts=None,initial_leaves=None,compiled_series=False,correction_order=3,conditioned_correction=False,contraction_stop=False,mixed_curve=False,material_inputs=False,curved_transverse=False,tolerance='5e-13',polynomial_light=False,world_profile=None,correction_axis=1,crossing_support=False,root_parts=1):
    """One fresh original piece, construction and all charts under one budget.

    Complete-piece specular/matte value and the explicitly selected derivative.
    Other motions, complete-pixel accumulation and installed CPU/CUDA comparison are
    separate gates. No cached local certificate contributes to this result.
    """
    from tools.opaque_reference_profile import stronger_target
    tolerance=stronger_target(tolerance)
    if type(crossing_support)!=bool or crossing_support and not (curved_transverse and conditioned_correction and correction_order==4):raise ValueError('qualified curved crossing support required')
    if type(root_parts)!=int or root_parts not in (1,2,4) or root_parts>1 and not crossing_support:raise ValueError('qualified curved root partition required')
    if type(correction_axis)!=int or correction_axis not in (0,1) or correction_axis!=1 and not conditioned_correction:raise ValueError('explicit conditioned correction axis required')
    _angular_selection(piece,motion,anchor)
    if type(curved_transverse)!=bool or curved_transverse and not (physical_angle and conditioned_correction and correction_order==4):raise ValueError('curved transverse requires explicit conditioned fourth-order physical-angle profile')
    if type(polynomial_light)!=bool or polynomial_light and not (physical_angle and material_inputs and conditioned_correction and correction_order==4 and curved_transverse):raise ValueError('polynomial light requires explicit conditioned fourth-order transverse physical-angle material profile')
    if type(tight_discriminant)!=bool or type(algebraic_inverse)!=bool or type(mapped_transport)!=bool or (algebraic_inverse or mapped_transport) and not tight_discriminant:raise ValueError("explicit discriminant/algebraic candidate flags")
    if type(exit_angle)!=bool or exit_angle and not (mapped_transport and algebraic_inverse):raise ValueError('exit angle requires qualified algebraic inverse and mapped transport')
    if type(unsquared_inverse)!=bool or unsquared_inverse and not algebraic_inverse:raise ValueError('unsquared inverse requires explicit algebraic candidate')
    if type(physical_angle)!=bool or physical_angle and not (unsquared_inverse and exit_angle):raise ValueError('physical angle requires qualified unsquared and exit candidate')
    if type(material_inputs)!=bool or material_inputs and not physical_angle:raise ValueError('material-only queries require physical-angle candidate')
    if type(mixed_curve)!=bool or mixed_curve and not conditioned_correction:raise ValueError('mixed material requires the conditioned physical-angle profile')
    if type(contraction_stop)!=bool or contraction_stop and not physical_angle:raise ValueError('contraction stopping requires physical-angle candidate')
    if type(conditioned_correction)!=bool or conditioned_correction and correction_order!=4:raise ValueError('conditioned correction requires fourth-order proof')
    if type(correction_order)!=int or correction_order not in (3,4) or correction_order==4 and not physical_angle:raise ValueError('qualified physical correction order required')
    if type(compiled_series)!=bool or compiled_series and not physical_angle:raise ValueError('compiled series requires explicit physical-angle candidate')
    if tensor_upgrade_charts is not None:
        if tensor_rule not in ('gauss3x12-local4','gauss4x12-local4') or not isinstance(tensor_upgrade_charts,list) or not tensor_upgrade_charts or any(type(i)!=int or i<0 or i>=128 for i in tensor_upgrade_charts) or len(set(tensor_upgrade_charts))!=len(tensor_upgrade_charts):raise ValueError('explicit unique tensor upgrade chart ids required')
    if tensor_rule is not None:
        from tools.opaque_tensor_smooth import validate
        validate(tensor_rule)
        if not physical_angle or smooth_partition is not None:raise ValueError('directional smooth proof requires physical angle and owns its remainder')
    if smooth_partition is not None:
        from tools.opaque_local_smooth import validate
        validate(smooth_partition)
        if not physical_angle:raise ValueError('localized smooth proof requires physical angle candidate')
    start=time.monotonic();cover=None;last=None;p=None
    def check():
        if time.monotonic()-start>=90:raise Unresolved('shared original reference deadline')
    def update(row):
        nonlocal last
        last=row;progress(row)
    try:
        p,curve,cover,charts=angular_material_cover(data,checkpoint=check,**({"tensor_upgrade_charts":tensor_upgrade_charts} if tensor_upgrade_charts is not None else {}),**(dict(crossing_support=crossing_support,root_parts=root_parts) if crossing_support or root_parts!=1 else {}),**({'correction_axis':correction_axis} if correction_axis!=1 else {}),**({'world_profile':world_profile} if world_profile is not None else {}),**({"polynomial_light":True} if polynomial_light else {}),**({"curved_transverse":True} if curved_transverse else {}),**(dict(piece=piece,motion=motion,anchor=anchor) if (piece,motion,anchor)!=(8,'normal_tilt',0) else {}),**({"material_inputs":True} if material_inputs else {}),**({"mixed_curve":True} if mixed_curve else {}),**({"contraction_stop":True} if contraction_stop else {}),**({"conditioned_correction":True} if conditioned_correction else {}),**({"correction_order":correction_order} if correction_order!=3 else {}),**({"compiled_series":True} if compiled_series else {}),**({"tensor_rule":tensor_rule} if tensor_rule is not None else {}),**({"smooth_partition":smooth_partition} if smooth_partition is not None else {}),**({"tight_discriminant":True} if tight_discriminant else {}),**({"algebraic_inverse":True} if algebraic_inverse else {}),**({"mapped_transport":True} if mapped_transport else {}),**({"exit_angle":True} if exit_angle else {}),**({"unsquared_inverse":True} if unsquared_inverse else {}),**({"physical_angle":True} if physical_angle else {}))
        result=integrate_square(None,chart_cover=charts,tolerance=tolerance,max_cells=256,max_depth=10,checkpoint=check,progress=update,**({'initial_leaves':initial_leaves} if initial_leaves is not None else {}))
        check()
        values=[result['value'][0]+I(a)*result['value'][1] for a in data['material']['albedo']]
        derivatives=[result['derivative'][0]+I(a)*result['derivative'][1] for a in data['material']['albedo']]
        if any(v.width>I('2e-12').lo for v in values+derivatives):raise Unresolved('complete piece RGB radius')
        return dict(tensor_upgrade_charts=tensor_upgrade_charts,crossing_support=crossing_support,root_parts=root_parts,correction_axis=correction_axis,polynomial_light=polynomial_light,tolerance=tolerance,status='complete_original_piece_integrated',curved_transverse=curved_transverse,compiled_series=compiled_series,correction_order=correction_order,conditioned_correction=conditioned_correction,contraction_stop=contraction_stop,mixed_curve=mixed_curve,material_inputs=material_inputs,initial_schedule=initial_leaves,tensor_rule=tensor_rule,smooth_partition=smooth_partition,tight_discriminant=tight_discriminant,algebraic_inverse=algebraic_inverse,mapped_transport=mapped_transport,exit_angle=exit_angle,unsquared_inverse=unsquared_inverse,physical_angle=physical_angle,physical_inverse=None if p is None or not p.get("physical_angle") else p["physical_angle"].report(),inverse_residual=None if p is None or 'inverse_residual' not in p else p['inverse_residual'].report(),material_inputs_cache=None if p is None or 'material_inputs_cache' not in p else p['material_inputs_cache'].report(),prepared_angle=None if p is None or 'prepared_angle' not in p else p['prepared_angle'].report(),horizon_cache=None if p is None or 'horizon_cache' not in p else p['horizon_cache'].report(),piece=piece,face=p['face'],motion=motion,anchor=anchor,basis=result,value=values,derivative=derivatives,cover_cells=cover.cells,cover_depth=cover.depth,evaluations=cover.evaluations,seconds=time.monotonic()-start,scope=f'Fresh complete original piece{piece} value and {motion} reference; no installed or whole-pixel gradient acceptance.')
    except ValueError as error:
        raise Unresolved('shared original piece unresolved',crossing_support=crossing_support,root_parts=root_parts,polynomial_light=polynomial_light,piece=piece,motion=motion,anchor=anchor,curved_transverse=curved_transverse,compiled_series=compiled_series,correction_order=correction_order,conditioned_correction=conditioned_correction,contraction_stop=contraction_stop,mixed_curve=mixed_curve,material_inputs=material_inputs,tensor_rule=tensor_rule,smooth_partition=smooth_partition,tight_discriminant=tight_discriminant,algebraic_inverse=algebraic_inverse,mapped_transport=mapped_transport,exit_angle=exit_angle,unsquared_inverse=unsquared_inverse,physical_angle=physical_angle,physical_inverse=None if p is None or not p.get("physical_angle") else p["physical_angle"].report(),inverse_residual=None if p is None or 'inverse_residual' not in p else p['inverse_residual'].report(),material_inputs_cache=None if p is None or 'material_inputs_cache' not in p else p['material_inputs_cache'].report(),prepared_angle=None if p is None or 'prepared_angle' not in p else p['prepared_angle'].report(),horizon_cache=None if p is None or 'horizon_cache' not in p else p['horizon_cache'].report(),cause=getattr(error,'details',str(error)),cover_cells=None if cover is None else cover.cells,evaluations=None if cover is None else cover.evaluations,seconds=time.monotonic()-start,last=last) from error
