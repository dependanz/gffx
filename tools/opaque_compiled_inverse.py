"""Opt-in Arb formal implicit series; independent checker only.

The positive root and residual slope are certified by the existing geometric
branch proof. Spatial coefficients solve the same triangular R=A-k sqrt(B)
identity. Then R_u u_theta+R_theta=0 solves the complete first-parameter series.
Every intrinsic, target and coordinate term remains. No sampled derivatives,
small-coefficient removal or claim of unchanged Decimal endpoints.
"""
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet


def solve(field,root,s,k,slope,*,with_radial=False):
    if type(with_radial)!=bool:raise ValueError('explicit radial proof switch required')
    import flint
    from flint import arb,arb_series,fmpq,ctx
    from tools import opaque_arb_jets
    if not opaque_arb_jets._active or flint.__version__!='0.9.0' or ctx.prec!=256 or ctx.threads!=1:raise ValueError('qualified serial pinned Arb context required')
    if not isinstance(root,I) or not isinstance(slope,I) or slope.lo<=0 or not 0<=root.lo<=root.hi<=1:raise ValueError('certified positive root and slope required')
    if not isinstance(s,Jet) or not isinstance(k,Jet) or s.order!=k.order or k[0,0].lo<=0:raise ValueError('matching positive-target jets required')
    n=s.order
    def ball(v):
        v=I(v)
        if v.zero:return arb(0)
        a,b=Q(v.lo),Q(v.hi);x=arb(fmpq(a.numerator,a.denominator))
        return x if a==b else x.union(arb(fmpq(b.numerator,b.denominator)))
    if opaque_arb_jets._ball_converter is not None:ball=opaque_arb_jets._ball_converter
    def export(v):
        if not v.is_finite():raise ValueError('finite inverse coefficient required')
        if v==0:return I(0)
        a,b=v.lower().fmpq(),v.upper().fmpq()
        return I(I(Q(int(a.p),int(a.q))).lo,I(Q(int(b.p),int(b.q))).hi)
    if opaque_arb_jets._interval_converter is not None:export=opaque_arb_jets._interval_converter
    if n==0:
        # The same implicit identities at spatial order zero need scalar balls,
        # not formal power-series objects. Intrinsic and coordinate motion stay.
        aa,bb=field.coefficients(s)
        a,b=[[ball(q[0,0]) for q in row] for row in (aa,bb)]
        at,bt=[[ball(q[0,1]) for q in row] for row in (aa,bb)]
        u=ball(root);kp,kt=ball(k[0,0]),ball(k[0,1])
        def polynomial(row):
            value=row[-1]
            for c in reversed(row[:-1]):value=value*u+c
            return value
        derivative=lambda row:[q*i for i,q in enumerate(row) if i]
        norm=polynomial(b).sqrt();ar=polynomial(derivative(a));br=polynomial(derivative(b))
        ru=(ar-kp*br/(2*norm)).intersection(ball(slope))
        if not ru>0:raise ValueError('positive scalar root slope required')
        rt=polynomial(at)-kt*norm-kp*polynomial(bt)/(2*norm);motion=-rt/ru
        result={(0,0):root,(0,1):export(motion)}
        if not with_radial:return result
        arr=polynomial(derivative(derivative(a)));brr=polynomial(derivative(derivative(b)))
        br_motion=polynomial(derivative(bt))+brr*motion
        ar_motion=polynomial(derivative(at))+arr*motion
        norm_motion=(polynomial(bt)+br*motion)/(2*norm)
        ru_motion=ar_motion-(kt*br+kp*br_motion)/(2*norm)+kp*br*norm_motion/(2*norm*norm)
        radial=ru/norm;radial_motion=(ru_motion-radial*norm_motion)/norm
        return result,{(0,0):export(radial),(0,1):export(radial_motion)}
    series=lambda values:arb_series(values,prec=n+1)
    component=lambda q,j:series([ball(q[i,j]) for i in range(n+1)])
    if opaque_arb_jets._series_converter is not None:component=opaque_arb_jets._series_converter
    aa,bb=field.coefficients(s)
    a,b=[[component(q,0) for q in row] for row in (aa,bb)]
    at,bt=[[component(q,1) for q in row] for row in (aa,bb)]
    kp,kt=component(k,0),component(k,1);divisor=ball(slope)
    u=series([ball(root)])
    def polynomial(row):
        value=row[-1]
        for c in reversed(row[:-1]):value=value*u+c
        return value
    # Lower coefficients already enclose the exact inverse. The next unknown
    # is zero while evaluating its residual, then divided by certified R_u.
    for i in range(1,n+1):
        u[i]=0
        residual=polynomial(a)-kp*polynomial(b).sqrt()
        u[i]=-residual[i]/divisor
    norm=polynomial(b).sqrt()
    ar=polynomial([q*i for i,q in enumerate(a) if i]);br=polynomial([q*i for i,q in enumerate(b) if i])
    ru=ar-kp*br/(2*norm)
    ru[0]=ru[0].intersection(divisor)
    if not ru[0]>0:raise ValueError('positive full-series root slope required')
    rt=polynomial(at)-kt*norm-kp*polynomial(bt)/(2*norm)
    motion=-rt/ru
    result={(i,j):(root if (i,j)==(0,0) else export((u,motion)[j][i])) for i in range(n+1) for j in (0,1)}
    if not with_radial:return result
    # At the certified implicit root, F_u=R_u/sqrt(B). Differentiate this
    # identity along the SAME complete parameter series: coefficients already
    # contain s_theta and intrinsic motion; motion supplies u_theta, kt supplies
    # target motion. No coefficient is inferred from a zero constant tangent.
    derivative=lambda row:[q*i for i,q in enumerate(row) if i]
    arr=polynomial(derivative(derivative(a)));brr=polynomial(derivative(derivative(b)))
    br_motion=polynomial(derivative(bt))+brr*motion
    ar_motion=polynomial(derivative(at))+arr*motion
    norm_motion=(polynomial(bt)+br*motion)/(2*norm)
    ru_motion=ar_motion-(kt*br+kp*br_motion)/(2*norm)+kp*br*norm_motion/(2*norm*norm)
    radial=ru/norm
    radial_motion=(ru_motion-radial*norm_motion)/norm
    return result,{(i,j):export((radial,radial_motion)[j][i]) for i in range(n+1) for j in (0,1)}
