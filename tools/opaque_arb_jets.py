"""Optional development-only Arb series adapter for the independent checker.

Trust boundary: hash-pinned python-flint0.9.0/FLINT Arb, exact rational endpoint
conversion, this dual-series algebra, and outward Decimal60 export. This is NOT
an assertion of unchanged Decimal endpoints. The caller verifies its isolated
binary manifest before import. No production renderer import selects this code.
A single serial worker may explicitly enter the context; methods/settings are
restored even after a failed test. No coefficient is discarded by magnitude.
"""
from contextlib import contextmanager
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
_active=False
_ball_converter=None
_interval_converter=None
_series_converter=None

@contextmanager
def backend(*,profile="endpoint",conversion_cache=False,fast_conversion=False,series_reuse=False):
    if profile not in ("endpoint","ball"):raise ValueError("qualified experimental profile required")
    global _active,_ball_converter,_interval_converter,_series_converter
    if type(series_reuse)!=bool or series_reuse and (profile!='ball' or not fast_conversion):raise ValueError('native series reuse requires explicit fast ball arithmetic')
    if type(fast_conversion)!=bool:raise ValueError('explicit fast-conversion switch required')
    if type(conversion_cache)!=bool:raise ValueError("explicit conversion-cache switch required")
    if _active:raise RuntimeError('compiled reference context cannot be nested')
    import flint
    from flint import arb,arb_series,fmpq,ctx
    if flint.__version__!='0.9.0':raise RuntimeError('qualified python-flint0.9.0 required')
    names=['__mul__','__rmul__','reciprocal','sqrt','ln'];old={n:getattr(Jet,n) for n in names};settings=(ctx.prec,ctx.cap,ctx.threads)
    # Cache only immutable decimal endpoints converted at the current Arb
    # precision. Reuse no integral, derivative, root or acceptance result.
    # A context owns at most4096 scalar entries and clears them on exit.
    from collections import OrderedDict
    cache=OrderedDict();report=dict(cache_capacity=4096 if conversion_cache else 0,cache_hits=0,cache_misses=0,cache_size=0,series_hits=0,series_misses=0,series_size=0)
    def ball(v):
        v=v if isinstance(v,I) else I(v)
        if v.zero:return arb(0)
        key=(ctx.prec,v.lo,v.hi)
        if conversion_cache and key in cache:
            report['cache_hits']+=1;cache.move_to_end(key);return cache[key]
        if fast_conversion:
            lo=arb(str(v.lo));result=lo if v.lo==v.hi else lo.union(arb(str(v.hi)))
        else:
            a,b=Q(v.lo),Q(v.hi);lo=arb(fmpq(a.numerator,a.denominator))
            result=lo if a==b else lo.union(arb(fmpq(b.numerator,b.denominator)))
        report['cache_misses']+=1
        if conversion_cache:
            cache[key]=result
            if len(cache)>4096:cache.popitem(last=False)
            report['cache_size']=len(cache)
        return result
    def interval(v):
        if not v.is_finite():raise ValueError('nonfinite compiled enclosure')
        if v==0:return I(0)
        if fast_conversion:
            from decimal import Decimal as D,localcontext,ROUND_FLOOR,ROUND_CEILING,MIN_EMIN,MAX_EMAX
            import tools.opaque_shading_intervals as scalar
            def endpoint(x,mode):
                mantissa,exponent=x.man_exp();mantissa,exponent=int(mantissa),int(exponent)
                if not mantissa:return D(0)
                with localcontext() as context:
                    context.prec=scalar.PRECISION;context.Emin=MIN_EMIN;context.Emax=MAX_EMAX
                    context.rounding=mode if mantissa>0 else (ROUND_CEILING if mode==ROUND_FLOOR else ROUND_FLOOR)
                    scale=D(2)**exponent;context.rounding=mode
                    return D(mantissa)*scale
            return I(endpoint(v.lower(),ROUND_FLOOR),endpoint(v.upper(),ROUND_CEILING))
        lo,hi=v.lower().fmpq(),v.upper().fmpq()
        return I(I(Q(int(lo.p),int(lo.q))).lo,I(Q(int(hi.p),int(hi.q))).hi)
    native=OrderedDict()
    def signature(q):
        import tools.opaque_shading_intervals as scalar
        # Identity detects replacing a coefficient by a new interval even
        # with identical endpoints; endpoints detect in-place mutations.
        return (ctx.prec,ctx.cap,scalar.PRECISION,q.order,tuple((ij,id(v),v.lo,v.hi) for ij,v in q.c.items()))
    def remember(q,p,t):
        native[id(q)]=(q,signature(q),(p,t));native.move_to_end(id(q))
        if len(native)>128:native.popitem(last=False)
        report['series_size']=len(native)
    def series(q,j):
        if not series_reuse:return arb_series([ball(q[i,j]) for i in range(q.order+1)],prec=q.order+1)
        key=id(q);entry=native.get(key)
        if entry is not None and entry[1]==signature(q):
            report['series_hits']+=1;native.move_to_end(key);value=entry[2][j]
        else:
            report['series_misses']+=1
            p=arb_series([ball(q[i,0]) for i in range(q.order+1)],prec=q.order+1)
            t=arb_series([ball(q[i,1]) for i in range(q.order+1)],prec=q.order+1) if tangent(q) else None
            remember(q,p,t);value=(p,t)[j]
        return value if value is not None else arb_series([],prec=q.order+1)
    def result(order,p,t=None):
        coeff={(i,0):interval(p[i]) for i in range(order+1)}
        if t is not None:coeff.update({(i,1):interval(t[i]) for i in range(order+1)})
        q=Jet._from_owned(order,coeff)
        # Keep the stronger already-certified native enclosure, rather than
        # exporting and immediately reimporting it. No coefficient is omitted.
        if series_reuse:remember(q,p,t)
        return q
    def tangent(q):return any(j for i,j in q.c)
    def multiply(a,b):
        b=a.other(b)
        if a.order<2 or len(a.c)*len(b.c)<16:return old['__mul__'](a,b)
        ap,bp=series(a,0),series(b,0);at,bt=tangent(a),tangent(b);t=None
        if at:t=series(a,1)*bp
        if bt:t=ap*series(b,1) if t is None else t+ap*series(b,1)
        return result(a.order,ap*bp,t)
    def endpoint_multiply(a,b):
        b=a.other(b)
        if a.order<2 or len(a.c)*len(b.c)<16 or any(v.lo<0<v.hi for v in [*a.c.values(),*b.c.values()]):return old['__mul__'](a,b)
        def groups(q,j):
            rows={}
            for sign in ['p','n']:
                for edge in ['lo','hi']:
                    coeff=[]
                    for i in range(q.order+1):
                        v=q[i,j];match=v.lo>=0 if sign=='p' else v.hi<=0
                        coeff.append(ball(I(getattr(v,edge))) if match else arb(0))
                    if any(x!=0 for x in coeff):rows[sign+edge]=arb_series(coeff,prec=q.order+1)
            return rows
        def convolve(ga,gb):
            low=arb_series([],prec=a.order+1);high=arb_series([],prec=a.order+1)
            for al,bl,ah,bh in [('plo','plo','phi','phi'),('nhi','nhi','nlo','nlo'),('phi','nlo','plo','nhi'),('nlo','phi','nhi','plo')]:
                if al in ga and bl in gb:low=low+ga[al]*gb[bl]
                if ah in ga and bh in gb:high=high+ga[ah]*gb[bh]
            return low,high
        ap,bp=groups(a,0),groups(b,0);lo,hi=convolve(ap,bp);rows=[(0,lo,hi)]
        if tangent(a) or tangent(b):
            lo,hi=convolve(groups(a,1),bp);otherlo,otherhi=convolve(ap,groups(b,1));rows.append((1,lo+otherlo,hi+otherhi))
        coeff={}
        for j,lo,hi in rows:
            for i in range(a.order+1):
                low,high=interval(lo[i]),interval(hi[i]);coeff[i,j]=I(low.lo,high.hi)
        return Jet._from_owned(a.order,coeff)
    def unary(q,name):
        if q.order<2 or not any(k!=(0,0) for k in q.c):return old[name](q)
        value=q[0,0]
        if (name=='reciprocal' and value.lo<=0<=value.hi) or (name=='sqrt' and value.lo<=0) or (name=='ln' and value.lo<=0):raise ValueError('compiled analytic series domain')
        p=series(q,0);t=series(q,1) if tangent(q) else None
        if name=='reciprocal':out=p.inv();t=-t*out*out if t is not None else None
        elif name=='sqrt':
            out=p.sqrt()
            # A broad formal-series square root may lose strict positivity in
            # its constant ball. The independent directed scalar bound encloses
            # the same constant, so intersect it before dividing the tangent.
            # Spatial coefficients are retained, never dropped or rescaled.
            scalar=ball(value.sqrt());out[0]=out[0].intersection(scalar)
            t=t/(2*out) if t is not None else None
        else:out=p.log();t=t/p if t is not None else None
        return result(q.order,out,t)
    try:
        _active=True;ctx.prec=256;ctx.cap=11;ctx.threads=1
        Jet.__mul__=multiply if profile=='ball' else endpoint_multiply;Jet.__rmul__=Jet.__mul__
        if profile=='ball':
            Jet.reciprocal=lambda q:unary(q,'reciprocal');Jet.sqrt=lambda q:unary(q,'sqrt');Jet.ln=lambda q:unary(q,'ln')
        _ball_converter=ball if conversion_cache or fast_conversion else None
        _interval_converter=interval if fast_conversion else None
        _series_converter=series if series_reuse else None
        yield report
    finally:
        for n,v in old.items():setattr(Jet,n,v)
        ctx.prec,ctx.cap,ctx.threads=settings;_active=False;_ball_converter=None;_interval_converter=None;_series_converter=None;cache.clear();native.clear()
