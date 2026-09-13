"""Restricted constant-light isotropic reference; no public shader calls.

Two half-azimuth Gauss partitions retain the original sample/time ceilings and
two-transition convergence rule. This does not implement general SH lighting.
"""
import math
import time
from tools import shading_environment_reference as base
from tools import shading_independent_evaluator as scalar

PROFILE='constant-env-split-v1'


def azimuth(order,split):
    if split not in (.25,.5):raise ValueError('undeclared azimuth partition')
    nodes,weights=base.gauss_legendre(order)
    cut=math.pi*split
    return ([a+(b-a)*x for a,b in ((0.,cut),(cut,math.pi)) for x in nodes],
            [(b-a)*w for a,b in ((0.,cut),(cut,math.pi)) for w in weights])


def radial(order,rule):
    nodes,weights=base.gauss_legendre(order)
    if rule=='linear':return nodes,weights
    if rule=='square':return [u*u for u in nodes],[2*u*w for u,w in zip(nodes,weights)]
    raise ValueError('unknown radial rule')


def matte_integral(curve,cosine):
    mean=curve.mean(1.,.04)
    if not math.isfinite(mean) or mean<=0:raise ValueError('positive independent curve mean required')
    return [a*curve.loss(cosine,1.,.04) for a in (.2,.4,.6)]


def reference_point(cosine,*,model,curve=None,max_evaluations=87296,deadline_seconds=30.,radial_rule='linear',separated=False):
    c=base._number(cosine)
    if not 0<c<=1:raise ValueError('positive view cosine required')
    if model not in ('classic','principled'):raise ValueError('unknown complete material')
    if model=='principled' and curve is None:raise ValueError('explicit reference curve required')
    if radial_rule not in ('linear','square'):raise ValueError('unknown radial rule')
    profile=PROFILE if radial_rule=='linear' else 'constant-env-split-square-v1'
    if type(separated) is not bool:raise ValueError('separated must be bool')
    if separated:
        if radial_rule!='square':raise ValueError('separated profile uses square radial rule')
        profile='constant-env-separated-v1'
    matte=matte_integral(curve,c) if separated and model=='principled' else [0.,0.,0.]
    if type(max_evaluations) is not int or not 0<max_evaluations<=87296:
        raise ValueError('evaluation work limit outside declared range')
    deadline_seconds=base._number(deadline_seconds)
    if not 0<deadline_seconds<=30:raise ValueError('deadline outside declared range')
    deadline=time.monotonic()+deadline_seconds
    history=[];evaluations=0;previous=None;streak=0
    n=(0.,0.,1.);v=(math.sqrt((1-c)*(1+c)),0.,c)
    def unresolved(reason):
        raise base.ReferenceConvergenceError(dict(profile=profile,status='unresolved',reason=reason,
                                                  evaluations=evaluations,history=history))
    def check():
        if time.monotonic()>=deadline:unresolved('time_limit')
    for order in (8,16,32,64,128):
        check()
        if evaluations+4*order*order>max_evaluations:unresolved('work_limit')
        nodes,weights=radial(order,radial_rule)
        estimates=[]
        for split in (.5,.25):
            angles,angle_weights=azimuth(order,split)
            trigonometry=[(math.cos(phi),math.sin(phi)) for phi in angles]
            rings=[[],[],[]]
            for mu,w in zip(nodes,weights):
                radius=math.sqrt((1-mu)*(1+mu));sums=[[],[],[]]
                for (cp,sp),aw in zip(trigonometry,angle_weights):
                    check();evaluations+=1
                    light=(radius*cp,radius*sp,mu)
                    if model=='classic':
                        value=scalar.classic(n,v,light,albedo=[.2,.4,.6],specular=.1,exponent=2.)
                    else:
                        value=scalar.principled(n,v,light,albedo=0. if separated else [.2,.4,.6],roughness=1.,f0=.04,budget=curve)
                    check()
                    for k,x in enumerate(value):
                        product=x*mu*2*w*aw
                        if not math.isfinite(product) or product<0:raise ValueError('invalid weighted sample')
                        sums[k].append(product)
                for k in range(3):rings[k].append(base._sum(sums[k]))
            estimates.append([base._sum(values)+matte[k] for k,values in enumerate(rings)])
        mean=[a/2+b/2 for a,b in zip(*estimates)]
        thresholds=[base.ATOL+base.RTOL*abs(x) for x in mean]
        delta=[abs(a-b) for a,b in zip(*estimates)]
        change=None
        if previous is not None:
            change=[max(abs(estimates[j][k]-previous[j][k]) for j in range(2)) for k in range(3)]
            streak=streak+1 if all(a<=t and b<=t for a,b,t in zip(change,delta,thresholds)) else 0
        history.append(dict(n_mu=order,n_phi=2*order,partitions=[.5,.25],estimates=estimates,
            radiance=mean,threshold=thresholds,offset_change=delta,change=change,streak=streak,evaluations=evaluations))
        check()
        if streak==2:
            return dict(profile=profile,status='converged',radiance=mean,evaluations=evaluations,history=history)
        previous=estimates
    unresolved('level_limit')
