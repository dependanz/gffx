"""Conservative jump flux after a certified geometric coordinate change.

Caller proves an injective chart on its interior (boundary collapse is allowed),
F_r>0 on its physical image, and encloses that image in a transverse interval.
Changing variables cancels the chart Jacobian in the coarea integral. Each
physical transverse line has at most one root per knot, hence the bound is
sum |jump| sup|q_theta(chart)| sup J_physical * transverse_extent / inf F_r.
The supplied motion is the COMPLETE chart motion, including coordinate motion.
This is an uncertainty bound, not a signed boundary contribution to add twice.
"""
from tools.opaque_shading_intervals import I

def mapped_monotone_flux(value,tangent,knots,*,radial_slope,transverse_extent,physical_jacobian):
    if any(not isinstance(v,I) for v in (value,tangent,radial_slope,transverse_extent,physical_jacobian)) or radial_slope.lo<=0 or transverse_extent.lo<0 or physical_jacobian.lo<0:raise ValueError('certified positive physical slope and nonnegative measure required')
    active=[(I(k),I(j)) for k,j in knots if value.lo<=I(k).hi and value.hi>=I(k).lo]
    if len(active)>8:raise ValueError('bounded angular crossing count')
    total=sum((I(j.bound)*I(tangent.bound)*physical_jacobian*transverse_extent/radial_slope.lo for k,j in active),I(0))
    return dict(bound=I(0,total.hi),axis='physical-radial',crossings=len(active),range=value,tangent=tangent)
