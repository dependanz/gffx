"""Bounded first angular derivatives and transversal jump transport.

Development reference only. Pointwise derivatives alone do not certify an
integral across discontinuities. Flux bounds require monotonicity on a box.
"""
from decimal import Decimal as D
from tools.opaque_shading_intervals import I,Jet,Unresolved
from tools.opaque_loss_extension import LossExtension,bounded_cosines

def residual_first(extension,q,index):
    if q.order:raise ValueError('piecewise residual supports value/first tangent only')
    value=q[0,0];values=extension.delta(Jet(value),index)
    chosen=[i for i in range(len(extension.cn)-1) if value.hi>=extension.cn[i].lo and value.lo<=extension.cn[i+1].hi]
    answer=[]
    for channel,base in enumerate(values):
        pieces=[]
        for i in chosen:
            if i==index:pieces.append(I(0));continue
            anchor,coefficients=extension.coefficient_cache[i,index,channel]
            segment=I(max(value.lo,extension.cn[i].lo),min(value.hi,extension.cn[i+1].hi))
            z=segment.ln()-anchor
            derivative=(coefficients[1]+z*(2*coefficients[2]+z*3*coefficients[3]))/segment*q[0,1]
            pieces.append(derivative)
        tangent=I(min(v.lo for v in pieces),max(v.hi for v in pieces))
        answer.append(Jet(coeff={(0,0):base,(0,1):tangent}))
    return answer

def transversal_flux(field,box,knots):
    """Bound integral jump transport on ANY subset of this coordinate rectangle.

    If dq/dx has fixed sign, each knot has at most one x root per y; integrating
    the delta term bounds it by sup|jump| sup|q_theta| / inf|q_x| times y extent.
    Summing both roles/regions is conservative. Nontransversal crossings refuse.
    The field must be a smooth jet expression on the entire box.
    """
    if len(box)!=2 or len(knots)>512:raise ValueError('bounded transport profile')
    x,y=map(I,box);q=field(Jet(x),Jet(y));value=bounded_cosines(lambda a,b:[field(a,b)],x,y)[0]
    active=[(I(k),I(j)) for k,j in knots if value.lo<=I(k).hi and value.hi>=I(k).lo]
    if len(active)>8:raise ValueError('bounded angular crossing count')
    if not active or q[0,1].zero:return dict(bound=I(0),axis=None,crossings=len(active),range=value,tangent=q[0,1])
    dx=field(Jet.variable(x,order=1),Jet(y,order=1))[1,0];dy=field(Jet(x,order=1),Jet.variable(y,order=1))[1,0]
    lower=lambda z:z.lo if z.lo>0 else -z.hi if z.hi<0 else D(0)
    axis=0 if lower(dx)>=lower(dy) else 1;minimum=lower((dx,dy)[axis])
    if minimum<=0:raise ValueError('knot crossing is not certified transversal')
    width=(y if axis==0 else x).width
    bound=sum((I(j.bound)*q[0,1].bound/minimum*width for _,j in active),I(0))
    return dict(bound=I(0,bound.hi),axis=axis,crossings=len(active),range=value,tangent=q[0,1])

def knot_jump(extension,k,channel):
    """Actual right-minus-left at the binary cosine knot, not a C1 assumption."""
    anchor=extension.x[k];coeff=[a-b for a,b in zip(extension.power(k,channel,anchor),extension.power(k-1,channel,anchor))]
    z=extension.cn[k].ln()-anchor;p=coeff[-1]
    for c in reversed(coeff[:-1]):p=c+z*p
    return p

class MovingLossExtension(LossExtension):
    """Ordinary derivative residual; caller must add the separately certified flux."""
    def _residual(self,q,index):return residual_first(self,q,index)

def material_transport(curve,angle_field,box,albedo,f0,*,horizon=False,flux_override=None,pure_flux_override=False):
    """Constant F0/roughness; supplied albedo bounds cover the complete box."""
    if type(pure_flux_override)!=bool or pure_flux_override and flux_override is None:raise ValueError('explicit pure flux callback contract required')
    if flux_override is not None and not callable(flux_override):raise ValueError('qualified flux override required')
    if len(albedo)!=3 or len(f0)!=3:raise ValueError('RGB transport shape')
    albedo,f0=list(map(I,albedo)),list(map(I,f0))
    if any(a.lo<0 or a.hi>1 for a in albedo+f0):raise ValueError('bounded unit material')
    if type(horizon)!=bool:raise ValueError("explicit horizon flag required")
    if horizon:
        from tools.opaque_full_material_reference import HorizonExtension
        extension=HorizonExtension(curve)
    else:extension=LossExtension(curve)
    x,y=map(I,box);ranges=bounded_cosines(angle_field,x,y)
    if len(ranges)!=2 or any(q.lo<(0 if horizon else curve.cn[0].lo) or q.hi>1 for q in ranges):raise ValueError('transport horizon/range')
    maxima=[]
    for value in ranges:
        indices=([-1] if horizon and value.lo<=curve.cn[0].hi else [])+[i for i in range(len(curve.cn)-1) if value.hi>=curve.cn[i].lo and value.lo<=curve.cn[i+1].hi]
        if not indices or len(indices)>8:raise ValueError('bounded angular transport segments')
        rows=[extension.at(Jet(I(max(value.lo,0 if i==-1 else curve.cn[i].lo),min(value.hi,curve.cn[0].hi if i==-1 else curve.cn[i+1].hi))),i) for i in indices]
        maxima.append([I(max(q[ch][0,0].bound for q in rows)) for ch in range(2)])
    flux=[I(0)]*3;roles=[];equal_channels={}
    for role in range(2):
        for channel in range(3):
            # Identical represented material intervals require identical rounded
            # proofs. Share only within this call; stateful override callbacks
            # retain all six original invocations unless the caller explicitly proves
            # purity. Equal inputs reuse only within this call; no approximate key.
            key=(role,albedo[channel].lo,albedo[channel].hi,f0[channel].lo,f0[channel].hi)
            if (flux_override is None or pure_flux_override) and key in equal_channels:
                row=equal_channels[key];flux[channel]=flux[channel]+row['bound'];roles.append(dict(role=role,channel=channel,**row));continue
            f=f0[channel];mix=lambda z:(1-f)*z[0]+f*z[1]
            factor=albedo[channel]*I(ranges[1].hi)*mix(maxima[1-role])/(curve.pi*mix(curve.mean))
            knots=[(curve.cn[k],factor*mix([(extension.at(Jet(curve.cn[0]),0)[c]-extension.at(Jet(curve.cn[0]),-1)[c])[0,0] if k==0 else knot_jump(extension,k,c) for c in range(2)])) for k in range(0 if horizon else 1,len(curve.cn)-1) if ranges[role].lo<=curve.cn[k].hi and ranges[role].hi>=curve.cn[k].lo]
            row=None if flux_override is None else flux_override(role,box,knots)
            if row is None:row=transversal_flux(lambda a,b,r=role:angle_field(a,b)[r],box,knots)
            if not isinstance(row,dict) or not isinstance(row.get('bound'),I) or row['bound'].lo<0:raise ValueError('nonnegative certified flux bound required')
            if flux_override is None or pure_flux_override:equal_channels[key]=row
            flux[channel]=flux[channel]+row['bound'];roles.append(dict(role=role,channel=channel,**row))
    return dict(bound=[I(0,q.hi) for q in flux],roles=roles)

def finish_transport(result,flux,tolerance='1e-12'):
    """Add all jump-flux uncertainty; refuse if combined derivative radius fails."""
    if len(result['derivative'])!=len(flux):raise ValueError('transport channels')
    result=dict(result);result['derivative']=[q+I(-b.hi,b.hi) for q,b in zip(result['derivative'],flux)]
    if any(q.width>2*I(tolerance).lo for q in result['value']+result['derivative']):raise Unresolved('combined angular transport error budget')
    result['jump_flux_bound']=flux;return result
