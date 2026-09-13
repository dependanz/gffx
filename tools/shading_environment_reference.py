"""Independent, bounded scalar SH2 environment quadrature (not a runtime shader).

Convergence is empirical agreement, never a proof that unsampled features vanish.
Only trusted bounded BRDF callbacks; deadlines are cooperative, not process kills.
"""
import math
from time import monotonic

PROFILE = 'env-sh2-v1'
ATOL = 1e-11
RTOL = 1e-9
LEVELS = ((8,16), (16,32), (32,64), (64,128), (128,256))
MAX_EVALUATIONS = 87296
MAX_SECONDS = 30.


def _number(value):
    if type(value) not in (int,float):
        raise ValueError('expected finite real scalar')
    try: value=float(value)
    except OverflowError as error:
        raise ValueError('nonrepresentable finite scalar') from error
    if not math.isfinite(value):raise ValueError('expected finite real scalar')
    return value


def _triple(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3:
        raise ValueError('expected RGB/direction triple')
    return tuple(_number(x) for x in value)


def _normal(value):
    value=_triple(value)
    length=math.hypot(*value)
    if not math.isfinite(length) or length==0:
        raise ValueError('nonrepresentable nonzero direction required')
    return tuple(x/length for x in value)


def _sum(values):
    try: result=math.fsum(values)
    except (OverflowError,ValueError) as error:
        raise ValueError('nonrepresentable integral or radiance sum') from error
    if not math.isfinite(result):
        raise ValueError('nonrepresentable integral or radiance sum')
    return result


def _dot(a,b):
    return _sum(x*y for x,y in zip(a,b))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def gauss_legendre(order):
    """Legendre recurrence/Newton roots and analytic weights, mapped to [0,1]."""
    if type(order) is not int or not 2<=order<=128:
        raise ValueError('Gauss order must be an integer in [2,128]')
    def polynomial(z):
        before,current=1.,z
        for k in range(2,order+1):
            before,current=current,((2*k-1)*z*current-(k-1)*before)/k
        derivative=order*(z*current-before)/(z*z-1)
        return current,derivative
    pairs=[]
    for i in range(1,order+1):
        z=math.cos(math.pi*(i-.25)/(order+.5))
        for _ in range(32):
            value,derivative=polynomial(z)
            update=value/derivative
            z-=update
            if abs(update)<=4*math.ulp(z):break
        else:raise ValueError('Gauss root did not converge')
        _,derivative=polynomial(z)
        pairs.append(((1+z)/2,1/((1-z*z)*derivative*derivative)))
    pairs.sort()
    return [p[0] for p in pairs],[p[1] for p in pairs]


def _coefficients(values):
    if not isinstance(values,(list,tuple)) or len(values)!=9:
        raise ValueError('expected nine RGB SH coefficients')
    return tuple(_triple(row) for row in values)


def _radiance(direction,coefficients):
    x,y,z=direction
    c0=math.sqrt(1/(4*math.pi)); c1=math.sqrt(3/(4*math.pi))
    c2=math.sqrt(15/(4*math.pi)); c3=math.sqrt(5/(16*math.pi)); c4=math.sqrt(15/(16*math.pi))
    basis=(c0,c1*y,c1*z,c1*x,c2*x*y,c2*y*z,c3*(3*z*z-1),c2*x*z,c4*(x*x-y*y))
    return tuple(_sum(row[k]*b for row,b in zip(coefficients,basis)) for k in range(3))


def sh_radiance(direction,coefficients):
    """Signed incident radiance, not convolved irradiance or display RGB."""
    return _radiance(_normal(direction),_coefficients(coefficients))


class ReferenceConvergenceError(RuntimeError):
    """An unresolved diagnostic must never masquerade as a usable reference."""
    def __init__(self,report):
        self.report=report
        super().__init__('environment reference unresolved: '+report['reason'])


def environment_reference(normal,view,coefficients,*,brdf,max_levels=5,
                          max_evaluations=MAX_EVALUATIONS,deadline_seconds=MAX_SECONDS):
    """Integrate one positive-view, one-sided sample with the fixed env-sh2-v1 rules.

Returns JSON-compatible history and radiance only after two passing transitions.
Failure preserves history on ReferenceConvergenceError.report. No I/O or caching.
"""
    if type(max_levels) is not int or not 3<=max_levels<=len(LEVELS):
        raise ValueError('max_levels must be an integer from three to five')
    if type(max_evaluations) is not int or not 0<max_evaluations<=MAX_EVALUATIONS:
        raise ValueError('evaluation work limit outside declared range')
    deadline_seconds=_number(deadline_seconds)
    if not 0<deadline_seconds<=MAX_SECONDS:
        raise ValueError('deadline outside declared range')
    if not callable(brdf):raise ValueError('trusted BRDF callable required')
    n,v=_normal(normal),_normal(view)
    if _dot(n,v)<=0:raise ValueError('positive view hemisphere required')
    coefficients=_coefficients(coefficients)
    helper=(0.,0.,1.) if abs(n[2])<.9 else (0.,1.,0.)
    tangent=_normal(_cross(helper,n)); bitangent=_cross(n,tangent)
    history=[]; evaluations=0; streak=0; previous=None
    started=monotonic()
    def unresolved(reason):
        raise ReferenceConvergenceError(dict(profile=PROFILE,status='unresolved',reason=reason,
                                             evaluations=evaluations,history=history))
    def check_time():
        if monotonic()-started>=deadline_seconds:unresolved('time_limit')
    for n_mu,n_phi in LEVELS[:max_levels]:
        check_time()
        if evaluations+2*n_mu*n_phi>max_evaluations:unresolved('work_limit')
        nodes,weights=gauss_legendre(n_mu)
        estimates=[]
        for offset in (0.,.5):
            rings=[[],[],[]]
            for mu,w in zip(nodes,weights):
                radial=math.sqrt((1-mu)*(1+mu))
                samples=[[],[],[]]
                for j in range(n_phi):
                    phi=2*math.pi*(j+offset)/n_phi
                    a,b=radial*math.cos(phi),radial*math.sin(phi)
                    light=tuple(a*t+b*s+mu*z for t,s,z in zip(tangent,bitangent,n))
                    check_time()
                    evaluations+=1
                    value=brdf(n,v,light)
                    check_time()
                    value=_triple(value)
                    if min(value)<0:raise ValueError('negative BRDF sample')
                    radiance=_radiance(light,coefficients)
                    weight=mu*2*math.pi*w/n_phi
                    for k in range(3):
                        product=value[k]*radiance[k]*weight
                        if not math.isfinite(product):raise ValueError('nonrepresentable weighted sample')
                        samples[k].append(product)
                for k in range(3):rings[k].append(_sum(samples[k]))
            estimates.append([_sum(channel) for channel in rings])
        mean=[a/2+b/2 for a,b in zip(*estimates)]
        thresholds=[ATOL+RTOL*abs(x) for x in mean]
        offset_change=[_number(abs(a-b)) for a,b in zip(*estimates)]
        change=None
        if previous is not None:
            change=[_number(max(abs(estimates[0][k]-previous[0][k]),abs(estimates[1][k]-previous[1][k]))) for k in range(3)]
            good=all(a<=t and b<=t for a,b,t in zip(change,offset_change,thresholds))
            streak=streak+1 if good else 0
        history.append(dict(n_mu=n_mu,n_phi=n_phi,estimates=estimates,radiance=mean,
                            change=change,offset_change=offset_change,threshold=thresholds,
                            evaluations=evaluations,streak=streak))
        check_time()
        if streak==2:
            return dict(profile=PROFILE,status='converged',radiance=mean,evaluations=evaluations,history=history)
        previous=estimates
    unresolved('level_limit')
