"""Independent exact-coefficient controls for an optional compiled proof backend."""
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
import pytest
import importlib.util
pytestmark=pytest.mark.skipif(importlib.util.find_spec("flint") is None,reason="isolated reference-only python-flint experiment is not installed")
from tools.opaque_shading_intervals import I,Jet,Unresolved

@pytest.mark.parametrize('name',['reciprocal','sqrt','ln'])
@pytest.mark.parametrize('order',[2,3,6,10])
def test_compiled_primal_and_tangent_coefficients_match_independent_formulas(name,order):
    from tools.opaque_arb_jets import backend
    with backend():
        x=Jet(2,order=order,tangent=Q(1,8));x.c[1,0]=I(1);got=getattr(x,name)()
    with localcontext() as c:
        c.prec=90
        def derivative(n):
            if name=='ln':return D(2).ln() if n==0 else D((-1)**(n-1))*D(__import__('math').factorial(n-1))/D(2)**n
            power=Q(-1) if name=='reciprocal' else Q(1,2);fall=Q(1)
            for k in range(n):fall*=power-k
            return D(fall.numerator)/D(fall.denominator)*(D(1)/2 if power==-1 else D(2).sqrt())/D(2)**n
        for n in range(order+1):
            for j in [0,1]:
                value=derivative(n+j)/__import__('math').factorial(n)/(8 if j else 1)
                assert got[n,j].lo<=value<=got[n,j].hi

@pytest.mark.parametrize('order',[2,6,10])
def test_compiled_convolution_contains_exact_rational_dual_polynomial(order):
    from tools.opaque_arb_jets import backend
    a={(i,j):Q((-1)**(i+j)*(i+1),7+j) for i in range(order+1) for j in [0,1]};b={(i,j):Q(i+2,11+j) for i in range(order+1) for j in [0,1]}
    with backend():got=Jet(order=order,coeff=a)*Jet(order=order,coeff=b)
    for i in range(order+1):
        for j in [0,1]:
            exact=sum((a[k,l]*b[i-k,j-l] for k in range(i+1) for l in range(j+1)),Q(0))
            assert Q(got[i,j].lo)<=exact<=Q(got[i,j].hi)

def test_compiled_interval_and_tiny_tangent_coefficients_are_not_dropped():
    from tools.opaque_arb_jets import backend
    with backend():
        x=Jet(I('1.5','2'),order=6,tangent=D('1e-70'));x.c[1,0]=I('1','1.1');got=x.reciprocal()
    with localcontext() as c:
        c.prec=90
        for a in [D('1.5'),D('1.75'),D(2)]:
            for b in [D(1),D('1.1')]:
                for n in range(7):
                    for j in [0,1]:
                        v=(-1)**(n+j)*D(n+1 if j else 1)*b**n/a**(n+j+1)*(D('1e-70') if j else 1)
                        assert got[n,j].lo<=v<=got[n,j].hi
        assert not got[0,1].zero

def test_compiled_context_restores_methods_settings_and_cancellation():
    from tools.opaque_arb_jets import backend
    from flint import ctx
    before=(Jet.__mul__,Jet.reciprocal,Jet.sqrt,Jet.ln,ctx.prec,ctx.cap,ctx.threads)
    with pytest.raises(Unresolved):
        with backend():
            assert ctx.threads==1
            with pytest.raises(RuntimeError):
                with backend():pass
            raise Unresolved('intentional stop')
    assert before==(Jet.__mul__,Jet.reciprocal,Jet.sqrt,Jet.ln,ctx.prec,ctx.cap,ctx.threads)

@pytest.mark.parametrize('method,interval',[('reciprocal',I(-1,1)),('sqrt',I(-1,1)),('ln',I(0,1))])
def test_compiled_invalid_domains_refuse(method,interval):
    from tools.opaque_arb_jets import backend
    with backend():
        with pytest.raises(ValueError):getattr(Jet.variable(interval,order=6),method)()

def test_endpoint_convolution_keeps_tight_signed_interval_bounds():
    from tools.opaque_arb_jets import backend
    # Four nonzero coefficients per operand select the compiled path.
    a=Jet(order=3,coeff={(0,0):I(1,2),(1,0):I(-3,-2),(2,0):I(1),(3,0):I(-1)})
    b=Jet(order=3,coeff={(0,0):I(2,3),(1,0):I(4,5),(2,0):I(2),(3,0):I(1)})
    with backend():got=a*b
    exact=[(2,6),(-5,6),(-11,-1),(-4,1)]
    for n,(lo,hi) in enumerate(exact):
        assert got[n,0].lo<=lo<=hi<=got[n,0].hi
        assert got[n,0].width<=D(hi-lo)+D('1e-50')
