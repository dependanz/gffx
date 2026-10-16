"""Independent fixed-visible-piece normal motion; horizons refuse explicitly.

A first bounded bridge from frozen original packed inputs to nonzero derivatives.
Ordinary derivatives and possible finite-table jump transport are both enclosed.
No native renderer, finite differences, horizon pruning, or limit expansion.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
from types import SimpleNamespace
import time
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_principled_reference import PreparedPrincipled
from tools.opaque_curved_reference import radial_homogeneous_fields,perspective_numerator
from tools.opaque_moving_reference import P,BranchCertificate
from tools.opaque_loss_extension import bounded_cosines
from tools.opaque_knot_transport import MovingLossExtension,material_transport
from tools.opaque_square_reference import integrate_square


def prepare_piece(data,piece=0,*,anchor=0):
    if type(anchor)!=int or anchor not in (0,1,2):raise ValueError("cyclic anchor must be 0,1,2")
    if type(piece)!=int or not 0<=piece<13 or len(data['selected']['visible_parts'])!=13:raise ValueError('frozen original thirteen-piece profile')
    if data['material']!={'roughness':1.,'f0':[.04]*3,'albedo':[.2,.4,.6],'direction':[1.,1.,2.]}:raise ValueError('original material profile')
    part=data['selected']['visible_parts'][piece];source=data['faces'][part['face']];ox,oy=data['selected']['origin']
    tri=[(Q(p[0])-ox,Q(p[1])-oy) for p in source['corners']];iw=[1/Q(w) for w in source['clip_w']];visible=[[Q(v) for v in p] for p in part['triangle']]
    visible=visible[anchor:]+visible[:anchor]
    motion=[[Q(.01)*Q(z),Q(0),-Q(.01)*Q(x)] for x,y,z in source['normal']]
    coeff=perspective_numerator(tri,inverse_w=iw,normals=source['normal'],views=source['view'],normal_motion=motion);cert=BranchCertificate()
    def at(p):
        x,y=p;return sum((a*b for a,b in zip(coeff,(x*x,x*y,y*y,x,y,1))),P(0))
    corners=[at(p) for p in visible];controls=corners+[2*at([(visible[i][k]+visible[j][k])/2 for k in range(2)])-(corners[i]+corners[j])*Q(1,2) for i,j in ((0,1),(0,2),(1,2))]
    if not all(cert.classify(v)>0 for v in controls):raise Unresolved('moving horizon map required',piece=piece)
    patch=SimpleNamespace(triangle=[[P(x) for x in p] for p in visible])
    fields=radial_homogeneous_fields(patch,tri,inverse_w=iw,fields={'n':[[P([Q(x),dx]) for x,dx in zip(n,d)] for n,d in zip(source['normal'],motion)],'v':source['view']})
    a,b,c=visible;det=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    return dict(anchor=anchor,piece=piece,face=source['face'],source=source,triangle=tri,visible=visible,inverse_w=iw,motion=motion,fields=fields,det=det,front_certificate=cert.report())


def cosine_field(prepared):
    def angles(u,s):
        d=prepared['fields'](u,s);n,v=d['n'],d['v'];l=[u.lift(k) for k in (1,1,2)]
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),u.lift(0))
        nn,vv,ll=dot(n,n),dot(v,v),dot(l,l)
        nv=dot(n,v)/(nn*vv).sqrt();nl=dot(n,l)/(nn*ll).sqrt()
        vh=((1+dot(v,l)/(vv*ll).sqrt())/2).sqrt()
        return nv,nl,vh
    return angles


def normal_piece(data,*,piece=0,progress=lambda row:None,diagnostic_boxes=None,anchor=0,max_cells=256,seconds=90):
    if type(max_cells)!=int or not 1<=max_cells<=256 or not 0<seconds<=90:raise ValueError("bounded original tangent limits")
    started=time.monotonic();p=prepare_piece(data,piece,anchor=anchor);curve=PreparedPrincipled(data['budget'],1.);e=MovingLossExtension(curve);angles=cosine_field(p);bounds=None;cache={}
    def check():
        if time.monotonic()-started>=seconds:raise Unresolved('normal tangent reference deadline')
    def inputs(u,s):
        nv,nl,vh=angles(u,s)
        for role,q in enumerate((nv,nl,vh)):
            lo=max(D(0),q[0,0].lo);hi=min(D(1),q[0,0].hi)
            if bounds is not None and role<2:lo=max(lo,bounds[role].lo);hi=min(hi,bounds[role].hi)
            q.c[0,0]=I(lo,hi)
        if nv[0,0].lo<=curve.cn[0].hi or nl[0,0].lo<=curve.cn[0].hi:raise ValueError('moving horizon/first segment requires qualification')
        return nv,nl,vh,u*p['det']
    def prepare(u,s,box):
        nonlocal bounds
        cache.clear();bounds=None;bounds=bounded_cosines(lambda u,s:list(angles(u,s)[:2]),*box)
        if any(q.lo<=curve.cn[0].hi or q.hi>1 for q in bounds):raise ValueError('moving horizon/first segment requires qualification')
        e.indices=[min(max(sum(n.lo<=(q.lo+q.hi)/2 for n in curve.cn)-1,0),len(curve.cn)-2) for q in bounds]
    f=I(.04);mix=lambda q:q[0]*(1-f)+q[1]*f;mean=mix(curve.mean)
    def smooth(u,s):
        nv,nl,vh,jac=inputs(u,s);av=e.at(nv,e.indices[0]);bl=e.at(nl,e.indices[1])
        return [nl*((1-vh)**5*(1-f)+f)*jac/((nv+nl)*(2*curve.pi)),nl*mix(av)*mix(bl)*jac/(curve.pi*mean)]
    def residual(u,s):
        nv,nl,_,jac=inputs(u,s);av=e.at(nv,e.indices[0]);bl=e.at(nl,e.indices[1]);dv=e._residual(nv,e.indices[0]);dl=e._residual(nl,e.indices[1])
        return [u.lift(0),nl*(mix(dv)*mix(bl)+mix(av)*mix(dl)+mix(dv)*mix(dl))*jac/(curve.pi*mean)]
    from tools.opaque_tangent_peano import TangentCorrection
    tight=TangentCorrection(e,inputs,checkpoint=check)
    def integrated(box):
        key=tuple((v.lo,v.hi) for v in box)
        if key not in cache:
            cache.clear();cache[key]=tight(box)
        return cache[key]
    def correction_integral(box):
        result=integrated(box);correction_integral.preferred_split_axis=tight.preferred_split_axis
        return [I(0),result[0]]
    def flux(box):
        check();q=material_transport(curve,lambda u,s:list(angles(u,s)[:2]),box,[1]*3,[.04]*3)['bound'][0]
        bound=q*(I(box[0].hi)*I(p['det']));return [I(0),integrated(box)[1]+I(-bound.hi,bound.hi)]
    if diagnostic_boxes is not None:
        if len(diagnostic_boxes)>3:raise ValueError('at most three saved diagnostic boxes')
        rows=[]
        for a,b,c,d in diagnostic_boxes:
            box=(I(I(a).lo,I(b).hi),I(I(c).lo,I(d).hi));prepare((a+b)/2,(c+d)/2,box);before=time.monotonic();pair=integrated(box);elapsed=time.monotonic()-before
            mass=(I(b)-I(a))*(I(d)-I(c));wide=residual(Jet(box[0]),Jet(box[1]))[1]
            old=[wide[0,j]*mass for j in (0,1)]
            if any(x.lo>y.hi or y.lo>x.hi for x,y in zip(pair,old)):raise ArithmeticError('independent residual enclosures disagree')
            rows.append(dict(box=list(map(str,[a,b,c,d])),range_bound=old,peano_bound=pair,width_ratio=[str(x.width/y.width) for x,y in zip(pair,old)],seconds=elapsed))
        return dict(status='diagnostic_only',regions=rows,seconds=time.monotonic()-started,scope='Three saved residual regions, not a complete piece or renderer acceptance.')
    try:result=integrate_square(smooth,tolerance='5e-13',max_cells=max_cells,max_depth=10,checkpoint=check,prepare=prepare,correction_integrator=correction_integral,tangent_correction_integrator=flux,progress=progress)
    except Unresolved as error:raise Unresolved('original normal tangent unresolved',piece=piece,face=p['face'],seconds=time.monotonic()-started,cause=error.details) from error
    value=[result['value'][0]+I(a)*result['value'][1] for a in data['material']['albedo']];derivative=[result['derivative'][0]+I(a)*result['derivative'][1] for a in data['material']['albedo']]
    if any(v.width>D('2e-12') for v in value+derivative):raise Unresolved('combined RGB normal-tangent radius')
    return dict(anchor=anchor,piece=piece,face=p['face'],value=value,derivative=derivative,basis=result,front_certificate=p['front_certificate'],seconds=time.monotonic()-started,scope='One fixed visible original piece, normal tilt only; strict front region. No whole pixel, moving visibility, horizon or recomputed-normal admission.')
