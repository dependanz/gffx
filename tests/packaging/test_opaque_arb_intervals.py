"""Exact interval endpoint and analytic derivative tests for persistent Arb."""
import pytest
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext

@pytest.mark.parametrize('a,b',[((-3,-1),(-2,4)),((1,2),(3,4)),((-3,-1),(-4,-2)),((0,1),(0,1)),(('1e-400','2e-400'),('1e-400','2e-400'))])
@pytest.mark.parametrize('op',['add','sub','mul','div'])
def test_persistent_intervals_contain_exact_endpoint_arithmetic(a,b,op):
    from tools.opaque_arb_intervals import backend
    with backend() as I:
        x,y=I(*a),I(*b)
        if op=='div' and Q(str(b[0]))<=0<=Q(str(b[1])):
            with pytest.raises(ValueError):x/y
            return
        z={'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'div':lambda:x/y}[op]()
        fn={'add':lambda x,y:x+y,'sub':lambda x,y:x-y,'mul':lambda x,y:x*y,'div':lambda x,y:x/y}[op]
        values=[fn(Q(str(u)),Q(str(v))) for u in a for v in b]
        assert Q(z.lo)<=min(values)<=max(values)<=Q(z.hi)
        assert z.width>=0

@pytest.mark.parametrize('name',['sqrt','ln','reciprocal'])
@pytest.mark.parametrize('order',[0,3,6,10])
def test_persistent_analytic_jets(name,order):
    from tools.opaque_arb_intervals import backend
    from tools.opaque_shading_intervals import Jet
    import math
    with backend() as I:
        x=Jet(2,order=order,tangent=Q(1,8))
        if order:x.c[1,0]=I(1)
        z=getattr(x,name)()
    with localcontext() as c:
        c.prec=90
        for n in range(order+1):
            for j in [0,1]:
                k=n+j
                if name=='ln':v=D(2).ln() if not k else D((-1)**(k-1)*math.factorial(k-1))/D(2)**k
                else:
                    power=Q(1,2) if name=='sqrt' else Q(-1);fall=Q(1)
                    for m in range(k):fall*=power-m
                    v=D(fall.numerator)/D(fall.denominator)*(D(2).sqrt() if name=='sqrt' else D('.5'))/D(2)**k
                v=v/math.factorial(n)/(8 if j else 1)
                assert z[n,j].lo<=v<=z[n,j].hi

def test_context_tiny_zero_domains_and_restoration():
    from tools.opaque_arb_intervals import backend
    from tools import opaque_shading_intervals as m
    before=m.I,m.Interval,m.ZERO,m.Jet.__mul__
    with pytest.raises(RuntimeError,match='stop'):
        with backend() as I:
            assert I(0).zero and not (I('1e-400')*I('1e-400')).zero
            assert I(0).sqrt().zero
            for fn in [lambda:I(-1,1).sqrt(),lambda:I(0,1).ln(),lambda:I(1)/I(-1,1)]:
                with pytest.raises(ValueError):fn()
            with pytest.raises(RuntimeError,match='nested'):
                with backend():pass
            raise RuntimeError('stop')
    assert before==(m.I,m.Interval,m.ZERO,m.Jet.__mul__)

def test_exact_authored_decimal_center_keeps_its_known_identity():
    from tools.opaque_arb_intervals import backend
    value=D('0.123456789012345678901234567890123456789012345678901234567890')
    with backend() as I:
        x=I(value)
        assert x.lo==x.hi==value and x.width==0
        y=x*x
        assert Q(y.lo)<=Q(value)**2<=Q(y.hi)

def test_copy_does_not_force_decimal_export():
    from tools.opaque_arb_intervals import backend
    with backend() as I:
        x=I(1)/3
        assert x.lc is None and x.hc is None
        y=I(x)
        assert y.lc is None and y.hc is None
        assert Q(y.lo)<=Q(1,3)<=Q(y.hi)

@pytest.mark.parametrize('order',[3,6,10])
@pytest.mark.parametrize('seed',range(5))
def test_persistent_series_contains_exact_interval_convolution(order,seed):
    import random
    from tools.opaque_arb_intervals import backend
    from tools.opaque_shading_intervals import Jet
    from tests.packaging.test_opaque_binary_derivatives import exact
    r=random.Random(seed)
    with backend(series=True) as I:
        def make():
            c={}
            for i in range(order+1):
                for j in range(2):
                    v=r.randrange(-30,30);c[i,j]=I(v,v+1)
            return Jet(order=order,coeff=c)
        a,b=make(),make();truth=exact(a,b);got=a*b
        for k,(lo,hi) in truth.items():assert Q(got[k].lo)<=lo<=hi<=Q(got[k].hi)
