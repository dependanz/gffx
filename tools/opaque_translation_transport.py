"""Independent exact translation/Leibniz derivative reference.

For a fixed, bounded, piecewise continuous RGB field F, continuous at the
endpoints, I(theta)=integral[a+a'*theta,b+b'*theta] F(x+s*theta) dx.
Substitution t=x+s*theta gives I'(0)=(s+b')F(b)-(s+a')F(a).
Internal jumps, including opaque horizons and resource knots, are included;
pointwise differentiation alone omits their transport. This proves a derivative,
not the integral value. Callers must establish the stated field/endpoint scope.
No renderer, autograd, finite difference or quadrature result enters acceptance.
"""
from fractions import Fraction as Q
from decimal import Decimal as D
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_principled_reference import PreparedPrincipled

def _rational(x):
    if type(x) not in (int,float,Q,D):raise ValueError('finite exact scalar required')
    try:return Q(x)
    except (ValueError,OverflowError,ZeroDivisionError) as e:raise ValueError('finite exact scalar required') from e

def translation_derivative(endpoint,lower,upper,*,speed,lower_speed=0,upper_speed=0):
    a,b,s,da,db=map(_rational,(lower,upper,speed,lower_speed,upper_speed))
    if a>=b:raise ValueError('strictly ordered endpoints required')
    fa,fb=list(map(I,endpoint(a))),list(map(I,endpoint(b)))
    if not 1<=len(fa)<=3 or len(fa)!=len(fb):raise ValueError('matching one to three channels required')
    return [(s+db)*v-(s+da)*u for u,v in zip(fa,fb)]

def original_endpoint(curve,x,*,role,z0,dz):
    """Full original material with one translating affine normal or view field.

    N=(0,0,1), V=(1,0,z) or N=(1,0,z), V=(0,0,1),
    z=z0+dz*(x-.5), L=(1,1,2), roughness1, F0.04, albedo.2/.4/.6.
    Nonzero vector norms, positive light cosine on the front-view region and
    nonopposite half vectors follow from these explicit vectors. Internal
    finite resource jumps and the opaque view horizon are allowed. Endpoints
    at a horizon or resource knot must refuse rather than select a derivative.
    """
    if role not in ('view','normal') or curve.roughness.lo!=1 or curve.roughness.hi!=1:raise ValueError('fixed original material role/profile')
    z=_rational(z0)+_rational(dz)*(_rational(x)-Q(1,2))
    if z==0:raise ValueError('endpoint on opaque horizon')
    if z<0:return [I(0)]*3
    length=(I(1)+I(z)*I(z)).sqrt();cosines=[I(z)/length,I(2)/I(6).sqrt() if role=='view' else (I(1)+2*I(z))/(I(6).sqrt()*length)]
    if any(q.lo<=k.hi and q.hi>=k.lo for q in cosines for k in curve.cn):raise ValueError('endpoint on resource knot requires separate one-sided evidence')
    zero=Jet(0);variable=[Jet(1),zero,Jet(z)];fixed=[zero,zero,Jet(1)]
    n,v=(fixed,variable) if role=='view' else (variable,fixed)
    return [q[0,0] for q in curve.radiance(n,v,[Jet(1),Jet(1),Jet(2)],[Jet(a) for a in (.2,.4,.6)],[Jet(.04)]*3)]

def original_translation(data,*,role,z0,dz,motion,visible_edge=None):
    """Derivative only; fixed unit-height pixel and declared affine fields."""
    if motion not in ('field','geometry'):raise ValueError('declared translation motion')
    if visible_edge is not None and (motion!='geometry' or not 0<_rational(visible_edge)<1):raise ValueError('interior attached geometry edge')
    if _rational(dz)<=0:raise ValueError('strict positive field slope')
    curve=PreparedPrincipled(data,1.)
    endpoint=lambda x:original_endpoint(curve,x,role=role,z0=z0,dz=dz)
    result=translation_derivative(endpoint,0 if visible_edge is None else visible_edge,1,speed=1 if motion=='field' else -1,lower_speed=0 if visible_edge is None else 1)
    if any(q.width>D('2e-12') for q in result):raise ValueError('reference radius exceeds 1e-12')
    return dict(derivative=result,identity='Leibniz after t=x+speed*theta; internal discontinuities included',role=role,motion=motion,visible_edge=visible_edge,z0=z0,dz=dz,scope='Declared affine translation direction only; no forward integral or arbitrary scene gradient qualification')
