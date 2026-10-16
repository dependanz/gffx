"""Independent material quadrature using certified approximate angular partitions."""
from fractions import Fraction as Q
from types import SimpleNamespace
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_branch_partition import affine_enclosure,partition,band_area,join_bound,area,refine_partition
from tools.opaque_peano_material import radial_cosines,difference
from tools.opaque_full_material_reference import HorizonExtension
from tools.opaque_curved_reference import radial_homogeneous_fields
from tools.opaque_moving_reference import P


def material_partitions(data,part,curve,*,checkpoint=lambda:None):
    checkpoint();source=data['faces'][part['face']];ox,oy=data['selected']['origin']
    tri=[(Q(p[0])-ox,Q(p[1])-oy) for p in source['corners']];iw=[1/Q(w) for w in source['clip_w']]
    visible=[tuple(Q(x) for x in p) for p in part['triangle']]
    patch=SimpleNamespace(triangle=[[P(x) for x in p] for p in visible])
    fields=radial_homogeneous_fields(patch,tri,inverse_w=iw,fields={'n':source['normal'],'v':source['view']})
    cosines=radial_cosines(fields,data['material']['direction'],simplex=True)
    domain=[(Q(0),Q(0)),(Q(1),Q(0)),(Q(0),Q(1))]
    models=affine_enclosure(lambda x,y:list(cosines(x,y)[:2]),domain)
    extension=HorizonExtension(curve)
    # The coefficients 1-F0 and F0 are the exact mathematical mixture of the
    # frozen binary F0; decimal .96 would silently change that mixture.
    mix=lambda row:row[0]*(1-I(.04))+row[1]*I(.04)
    planes=[];loss_bounds=[];integrated_errors=[];point_errors=[];rows=[];ranges=[]
    for role,model in enumerate(models):
        checkpoint();a,b,c=model['plane'];delta=model['error'];values=[a*x+b*y+c for x,y in domain]
        raw=I(I(min(values)).lo,I(max(values)).hi)+I(delta.hi.copy_negate(),delta.hi)
        lo=max(I(0).lo,raw.lo);hi=min(I(1).hi,raw.hi)
        if lo>hi:raise Unresolved('cosine classifier empty front range')
        if role and lo<=0:raise Unresolved('light horizon profile')
        limits=I(lo,hi);ranges.append(limits)
        active=[k for k in range(len(curve.cn)-1) if curve.cn[k].hi>=lo and curve.cn[k].lo<=hi]
        if len(active)>8:raise Unresolved('angular classifier knot cap8',role=role,knots=len(active))
        integrated=I(0);point=I(0);joins=[]
        for k in active:
            knot=curve.cn[k]
            if knot.lo!=knot.hi:raise Unresolved('exact binary cosine knot required')
            plane=(a,b,c-Q(knot.lo));band=band_area(domain,plane,Q(delta.hi))
            if min(values)<Q(knot.lo)<max(values):planes.append(plane)
            if not band:continue
            if (knot-delta).lo<=0:raise Unresolved('join extension reaches logarithm singularity',role=role,knot=k)
            bound=mix(join_bound(lambda q:difference(extension,q,k,k-1),Q(knot.lo),delta))
            point=point+bound;integrated=integrated+bound*I(band)
            joins.append(dict(knot=k,cosine=str(knot),classifier_error=str(delta),band_area=str(band),loss_error=str(bound)))
        chosen=([-1] if lo<=curve.cn[0].hi else [])+[k for k in range(len(curve.cn)-1) if hi>=curve.cn[k].lo and lo<=curve.cn[k+1].hi]
        bounds=[]
        for k in chosen:
            v=I(max(lo,I(0).lo if k<0 else curve.cn[k].lo),min(hi,curve.cn[0].hi if k<0 else curve.cn[k+1].hi))
            bounds.append(mix([q[0,0] for q in extension.at(Jet(v),k)]).bound)
        loss_bounds.append(I(max(bounds)));integrated_errors.append(integrated);point_errors.append(point);rows.append(dict(role=role,model=model,joins=joins))
    det=2*area(visible)
    error=I(det)*ranges[1]/(curve.pi*mix(curve.mean))*(loss_bounds[1]*integrated_errors[0]+(loss_bounds[0]+point_errors[0])*integrated_errors[1])
    polygons=partition(domain,planes,checkpoint=checkpoint);pieces=[]
    for poly in polygons:
        centre=[sum(p[j] for p in poly)/len(poly) for j in (0,1)];indices=[]
        for model in models:
            v=sum(a*b for a,b in zip(model['plane'],(*centre,Q(1))))
            indices.append(min(sum(Q(k.lo)<=v for k in curve.cn)-1,len(curve.cn)-2))
        for i in range(1,len(poly)-1):
            points=[poly[0],poly[i],poly[i+1]]
            physical=[tuple(visible[0][k]+x*(visible[1][k]-visible[0][k])+y*(visible[2][k]-visible[0][k]) for k in (0,1)) for x,y in points]
            if area(physical):pieces.append(dict(triangle=physical,indices=indices))
    if len(pieces)>256:raise Unresolved('angular integration cell cap256')
    return dict(pieces=pieces,error=I(0,error.hi),models=rows,physical_area=area(visible),cosine_ranges=ranges,plane_count=len(planes))


def full_pixel_partitioned(data,*,checkpoint=lambda:None,progress=lambda row:None):
    import time
    from tools.opaque_curved_reference import CurvedTriangle,perspective_numerator
    from tools.opaque_square_reference import integrate_square
    from tools.opaque_principled_reference import PreparedPrincipled
    from tools.opaque_moving_reference import BranchCertificate
    from tools.opaque_loss_extension import bounded_cosines
    material=data['material']
    if material!={'roughness':1.,'f0':[.04]*3,'albedo':[.2,.4,.6],'direction':[1.,1.,2.]}:raise ValueError('explicit fixed original material profile')
    start=time.monotonic();cells=0;rows=[];values=[I(0),I(0)];curve=PreparedPrincipled(data['budget'],1.)
    parts=data['selected']['visible_parts'];ox,oy=data['selected']['origin']
    if not 1<=len(parts)<=32:raise ValueError('visible piece cap32')
    def check():
        checkpoint()
        if time.monotonic()-start>90:raise Unresolved('complete material reference90s',cells=cells,completed_pieces=len(rows))
    def tick():
        nonlocal cells
        check()
        if cells>=256:raise Unresolved('complete reference cell budget',cells=cells)
        cells+=1
    piece_tolerance=I('1e-12')/(2*len(parts))
    for number,part in enumerate(parts):
        try:
            check();source=data['faces'][part['face']];tri=[(Q(p[0])-ox,Q(p[1])-oy) for p in source['corners']];iw=[1/Q(w) for w in source['clip_w']]
            coeff=perspective_numerator(tri,inverse_w=iw,normals=source['normal'],views=source['view']);subtotal=[I(0),I(0)];results=[];subparts=[];error=I(0);before=cells
            def assess(current):
                visible=[[Q(x) for x in p] for p in current['triangle']];cert=BranchCertificate(check)
                def at(p):
                    x,y=p;return sum((a*b for a,b in zip(coeff,(x*x,x*y,y*y,x,y,1))),P(0))
                corners=[at(p) for p in visible];controls=corners+[2*at([(visible[i][k]+visible[j][k])/2 for k in (0,1)])-(corners[i]+corners[j])*Q(1,2) for i,j in ((0,1),(0,2),(1,2))]
                if all(cert.classify(q)<0 for q in controls):return dict(pieces=[],error=I(0),classification='certified back-facing black')
                return material_partitions(data,current,curve,checkpoint=check)
            leaves=refine_partition(part,assess,piece_tolerance,tick=tick)
            for leaf in leaves:
                certificate=leaf['certificate'];local_error=certificate['error'];error=error+local_error
                pieces=certificate['pieces']
                if not pieces:continue
                target=str(((leaf['allowance']-I(local_error.hi))/len(pieces)).lo)
                subparts.extend(dict(sub,target=target,depth=leaf['depth']) for sub in pieces)
            subtotal[1]=I(error.hi.copy_negate(),error.hi)
            for sub in subparts:
                check();visible=sub['triangle'];cert=BranchCertificate(check)
                def at(p):
                    x,y=p;return sum((a*b for a,b in zip(coeff,(x*x,x*y,y*y,x,y,1))),P(0))
                corners=[at(p) for p in visible];controls=corners+[2*at([(visible[i][k]+visible[j][k])/2 for k in (0,1)])-(corners[i]+corners[j])*Q(1,2) for i,j in ((0,1),(0,2),(1,2))]
                signs=[cert.classify(q) for q in controls]
                if all(q<0 for q in signs):continue
                curved=not all(q>0 for q in signs)
                patch=CurvedTriangle(visible,coeff,checkpoint=check) if curved else SimpleNamespace(triangle=[[P(z) for z in p] for p in visible])
                fields=radial_homogeneous_fields(patch,tri,inverse_w=iw,fields={'n':source['normal'],'v':source['view']});cosines=radial_cosines(fields,material['direction'])
                det=2*area(visible);extension=HorizonExtension(curve);extension.indices=sub['indices'];limits=None
                def inputs(u,s):
                    if curved:
                        z=1-(1-u)**4 if patch.inside else u**4;dz=4*(1-u)**3 if patch.inside else 4*u**3
                        t,jac=patch.radial_map(z,s);jac=jac*dz
                    else:t=u;jac=u*det
                    nv,nl,vh=cosines(t,s)
                    nv.c[0,0]=I(max(I(0).lo,nv[0,0].lo),min(I(1).hi,nv[0,0].hi))
                    for q in (nl,vh):q.c[0,0]=I(max(I(-1).lo,q[0,0].lo),min(I(1).hi,q[0,0].hi))
                    if limits is not None:
                        for q,b in zip((nv,nl),limits):q.c[0,0]=I(max(q[0,0].lo,b.lo),min(q[0,0].hi,b.hi))
                    if nl[0,0].lo<=0:raise ValueError('strict light horizon profile')
                    return nv,nl,vh,jac
                def prepare(u,s,box):
                    nonlocal limits
                    limits=None;limits=bounded_cosines(lambda x,y:list(inputs(x,y)[:2]),*box)
                f=I(.04);mix=lambda v:v[0]*(1-f)+v[1]*f;mean=mix(curve.mean)
                def smooth(u,s):
                    nv,nl,vh,jac=inputs(u,s)
                    av=extension.at(nv,extension.indices[0]);bl=extension.at(nl,extension.indices[1])
                    spec=nl*((1-vh)**5*(1-f)+f)/((nv+nl)*(2*curve.pi))
                    matte=nl*mix(av)*mix(bl)/(curve.pi*mean)
                    return [spec*jac,matte*jac]
                if cells>=256:raise Unresolved('complete reference cell budget',cells=cells)
                result=integrate_square(smooth,tolerance=sub['target'],max_cells=256-cells,max_depth=10-sub['depth'],checkpoint=check,prepare=prepare)
                cells+=result['cells'];subtotal=[a+b for a,b in zip(subtotal,result['value'])];results.append(result)
            row=dict(piece=number,face=source['face'],value=subtotal,partition_error=error,partition_count=len(subparts),cells=cells-before,classifier_cells=len(leaves)*2-1,seconds=time.monotonic()-start)
            rows.append(row);progress(row);values=[a+b for a,b in zip(values,subtotal)]
        except Unresolved as e:
            raise Unresolved('complete partitioned original unresolved',piece=number,completed_pieces=len(rows),cells=cells,seconds=time.monotonic()-start,cause=e.details) from e
    check();rgb=[values[0]+I(a)*values[1] for a in material['albedo']]
    if any(v.width>I('2e-12').lo for v in rgb+[values[1]]):raise Unresolved('complete material total radius')
    return dict(value=rgb,albedo_derivative=values[1],basis=values,pieces=rows,cells=cells,seconds=time.monotonic()-start,tolerance='1e-12',scope='Fixed packed original geometry and actual material,including bounded angular classifier mismatch. No moving-geometry claim.')
