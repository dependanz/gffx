"""Independent rational and analytic controls for persistent compiled dual series."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import math,pytest

@pytest.mark.parametrize('order',[0,1,3,6,10])
def test_persistent_series_convolution_encloses_exact_rational_endpoints(order):
    from tools.opaque_persistent_series import backend
    from tools.opaque_shading_intervals import I
    with backend() as Jet:
        a={(i,j):(Q((-1)**i*(i+1),17),Q((-1)**i*(i+1),17)+Q(1,1000)) for i in range(order+1) for j in (0,1)}
        b={(i,j):(Q((-1)**j*(i+2),19),Q((-1)**j*(i+2),19)+Q(1,900)) for i in range(order+1) for j in (0,1)}
        x=Jet(order=order,coeff={k:I(I(v[0]).lo,I(v[1]).hi) for k,v in a.items()});y=Jet(order=order,coeff={k:I(I(v[0]).lo,I(v[1]).hi) for k,v in b.items()});z=x*y
        for i in range(order+1):
            for j in (0,1):
                lo=hi=Q(0)
                for (k,l),aa in a.items():
                    if k<=i and l<=j:
                        bb=b[i-k,j-l];v=[p*q for p in aa for q in bb];lo+=min(v);hi+=max(v)
                assert Q(z[i,j].lo)<=lo<=hi<=Q(z[i,j].hi)

@pytest.mark.parametrize('kind',['reciprocal','sqrt','ln'])
@pytest.mark.parametrize('order',[0,1,3,6,10])
def test_persistent_series_analytic_coefficients_and_tangent(kind,order):
    from tools.opaque_persistent_series import backend
    with backend() as Jet:
        q=Jet(2,order=order,tangent=1)
        if order:q.c[1,0]=1
        v=getattr(q,kind)()
        with localcontext() as c:
            c.prec=90;a=D(2)
            def coefficient(n):
                if kind=='reciprocal':return D((-1)**n)/a**(n+1)
                if kind=='ln':return a.ln() if n==0 else D((-1)**(n+1))/D(n)/a**n
                x=Q(1)
                for k in range(n):x*=Q(1,2)-k;x/=k+1
                return D(x.numerator)/D(x.denominator)*a.sqrt()/a**n
            for n in range(order+1):
                want=coefficient(n);tangent=D((-1)**n)/a**(n+1) if kind=='ln' else (n+1)*coefficient(n+1)
                assert v[n,0].lo<=want<=v[n,0].hi
                assert v[n,1].lo<=tangent<=v[n,1].hi

def test_persistent_series_mutable_coefficients_tiny_terms_and_order_checks():
    from tools.opaque_persistent_series import backend
    from tools.opaque_shading_intervals import I
    with backend() as Jet:
        q=Jet(2,order=4,tangent='1e-1000');z=q*q
        assert not z[0,1].zero and z[0,1].lo<=D('4e-1000')<=z[0,1].hi
        q.c[1,0]=I(3);assert q[1,0].lo==3;q.c[1,0]=I(0);assert (1,0) not in q.c
        q.c[0,0]=I(4);assert (q*q)[0,0].lo==16
        with pytest.raises(ValueError):q+Jet(1,order=3)
        for n in (-1,11,True):
            with pytest.raises(ValueError):Jet(1,order=n)
        assert not (q-q)[0,1].zero or (q-q)[0,1].lo<=0<=(q-q)[0,1].hi

def test_persistent_series_domains_context_restoration_and_nesting():
    from tools.opaque_persistent_series import backend
    from tools import opaque_shading_intervals as m
    import flint
    old=m.Jet;settings=(flint.ctx.prec,flint.ctx.cap,flint.ctx.threads)
    with pytest.raises(RuntimeError,match='deliberate'):
        with backend() as Jet:
            assert m.Jet is Jet and flint.ctx.threads==1
            for kind in ('reciprocal','sqrt','ln'):
                with pytest.raises(ValueError):getattr(Jet(m.I(-1,1),order=4),kind)()
            with pytest.raises(RuntimeError):
                with backend():pass
            raise RuntimeError('deliberate')
    assert m.Jet is old and (flint.ctx.prec,flint.ctx.cap,flint.ctx.threads)==settings

@pytest.mark.parametrize('kind',['reciprocal','sqrt','ln'])
def test_persistent_series_interval_analytic_derivatives(kind):
    from tools.opaque_persistent_series import backend
    from tools.opaque_shading_intervals import I
    with backend() as Jet:
        q=Jet.variable(I(2,3),order=10);q.c[0,1]=1;v=getattr(q,kind)()
        with localcontext() as c:
            c.prec=90
            for a in [D(2),D(3)]:
                for n in range(11):
                    if kind=='ln':value=a.ln() if not n else D((-1)**(n+1))/D(n)/a**n;tangent=D((-1)**n)/a**(n+1)
                    elif kind=='reciprocal':value=D((-1)**n)/a**(n+1);tangent=D((-1)**(n+1)*(n+1))/a**(n+2)
                    else:
                        b=Q(1)
                        for k in range(n):b*=Q(1,2)-k;b/=k+1
                        value=D(b.numerator)/D(b.denominator)*a.sqrt()/a**n;tangent=value*(D('.5')-n)/a
                    assert v[n,0].lo<=value<=v[n,0].hi and v[n,1].lo<=tangent<=v[n,1].hi

def test_persistent_series_exact_zero_and_signed_sparse_algebra():
    from tools.opaque_persistent_series import backend
    with backend() as Jet:
        z=Jet(0,order=10);assert z.sqrt()[0,0].zero and not len((z*z).c)
        q=Jet.variable(2,order=3);assert (-q+q)[0,0].zero
        assert (3-q)[0,0].lo==1 and (q/2)[1,0].lo==Q(1,2)

def test_persistent_series_retains_independent_constant_range_identity():
    from tools.opaque_persistent_series import backend
    from tools.opaque_shading_intervals import I
    with backend() as Jet:
        x=Jet.variable(I(0,1),order=6);q=x*I('.75')+I('.25')
        assert q[0,0].lo==D('.25') and q[0,0].hi==1
        q.c[0,0]=I('.1');assert q[0,0].lo==D('.1')==q[0,0].hi
        assert (q+I('.9'))[0,0].hi==1

def test_persistent_series_extreme_exponent_conversion_is_bounded_and_nonzero():
    import time
    from tools.opaque_persistent_series import backend
    from tools.opaque_shading_intervals import I
    start=time.monotonic()
    with backend() as Jet:
        q=Jet(1,order=3,coeff={(0,0):I(1),(1,0):I('1e-1000058'),(0,1):I('-1e-1000058')})
        v=q.ln();assert v[0,0].lo<=0<=v[0,0].hi
        for key,want in [((1,0),D('1e-1000058')),((0,1),D('-1e-1000058'))]:
            z=v[key];assert not z.zero and z.lo<=want<=z.hi
        assert time.monotonic()-start<2

@pytest.mark.parametrize('kind',['multiply','reciprocal','sqrt','ln'])
def test_persistent_series_preserves_original_low_coefficients_exactly(kind):
    from tools import opaque_shading_intervals as m
    from tools.opaque_persistent_series import backend
    old=m.Jet;coeff={(0,0):m.I('1.5','2'),(1,0):m.I('-.1','.2'),(2,0):m.I('.001','.002'),(0,1):m.I('1e-1000'),(1,1):m.I('-.003','.004')}
    q=old(order=6,coeff=coeff);expected=q*q if kind=='multiply' else getattr(q,kind)()
    with backend() as Jet:
        x=Jet(order=6,coeff=coeff);got=x*x if kind=='multiply' else getattr(x,kind)()
        assert all((got[i,j].lo,got[i,j].hi)==(expected[i,j].lo,expected[i,j].hi) for i in range(3) for j in (0,1))
