"""Fixed-geometry integration-by-parts terms for a monotone smooth coordinate.

L(h)=d(h/qprime)/dx. For F_kprime=F_(k-1), F_0=f:
 integral f(q)g dx = [sum_(k=1)^n (-1)^(k-1)F_k(q)L^(k-1)(g)/qprime]
                    + (-1)^n integral F_n(q)L^n(g) dx.
One more coordinate derivative than integrations is required. The supplied
primitive may belong to a discontinuous f: no derivative across its knots is
used here. A later quadrature must separately prove its own regularity/error.
"""
from tools.opaque_shading_intervals import I,Jet

def crop(v,order):return Jet(order=order,coeff={k:z for k,z in v.c.items() if k[0]<=order})
def derivative(v):
    if not v.order:raise ValueError('missing spatial derivative')
    return Jet(order=v.order-1,coeff={(i,0):(i+1)*v[i+1,0] for i in range(v.order)})

def terms(coordinate,weight,primitive,*,order=3,checkpoint=lambda:None):
    if type(order)!=int or not 1<=order<=6:raise ValueError('parts order1..6 required')
    if not isinstance(coordinate,Jet) or not isinstance(weight,Jet) or coordinate.order<order+1 or weight.order<order:raise ValueError('complete coordinate ordern+1 and weight ordern required')
    if any(j and not v.zero for q in [coordinate,weight] for (i,j),v in q.c.items()):raise ValueError('fixed geometry only')
    checkpoint();dq=derivative(crop(coordinate,order+1))
    if dq[0,0].lo<=0<=dq[0,0].hi:raise ValueError('certified nonzero coordinate derivative required')
    h=crop(weight,order);boundary=I(0);last=None
    for k in range(1,order+1):
        checkpoint();density=h/crop(dq,h.order);last=I(primitive(coordinate[0,0],k));boundary=boundary+(-1)**(k-1)*last*density[0,0];h=derivative(density)
    return boundary,(-1)**order*last*h[0,0]
