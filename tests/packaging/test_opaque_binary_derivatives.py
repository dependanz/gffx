"""Independent exact rational convolution; protected low derivatives and refusals."""
from decimal import Decimal as D
from fractions import Fraction as Q
import random
import pytest
from tools.opaque_shading_intervals import I,Jet

def exact(a,b):
    out={}
    for (i,j),u in a.c.items():
        for (k,l),v in b.c.items():
            if i+k>a.order or j+l>1:continue
            products=[Q(x)*Q(y) for x in (u.lo,u.hi) for y in (v.lo,v.hi)]
            lo,hi=out.get((i+k,j+l),(Q(0),Q(0)))
            out[i+k,j+l]=lo+min(products),hi+max(products)
    return out

@pytest.mark.parametrize('seed',range(20))
def test_exact_fraction_interval_convolution(seed):
    from tools.opaque_binary_derivatives import binary_derivatives
    r=random.Random(seed)
    def make():
        c={}
        for i in range(11):
            for j in range(2):
                x=r.randrange(-1000,1000);y=x+r.randrange(0,50)
                c[i,j]=I(I(Q(x,10**(i+2))).lo,I(Q(y,10**(i+2))).hi)
        return Jet(order=10,coeff=c)
    a,b=make(),make();ordinary=a*b;truth=exact(a,b)
    with binary_derivatives() as stats:result=a*b
    assert stats['fast']>0
    for key,(lo,hi) in truth.items():
        assert Q(result[key].lo)<=lo<=hi<=Q(result[key].hi)
        if key[0]<=2:assert (result[key].lo,result[key].hi)==(ordinary[key].lo,ordinary[key].hi)

@pytest.mark.parametrize('value',['1e-400','1e400','1e308','1e-200','-1e-200'])
def test_extremes_preserve_tiny_terms_or_fallback(value):
    from tools.opaque_binary_derivatives import binary_derivatives
    a=Jet(order=6,coeff={(i,j):I(value) for i in range(7) for j in range(2)})
    truth=exact(a,a)
    with binary_derivatives() as stats:result=a*a
    for key,(lo,hi) in truth.items():assert Q(result[key].lo)<=lo<=hi<=Q(result[key].hi)
    assert not result[6,1].zero
    if value in ('1e-400','1e400','1e308'):assert stats['fallback']>0

def test_zero_sparse_scalars_low_orders_and_mismatch():
    from tools.opaque_binary_derivatives import binary_derivatives
    for order in range(11):
        a=Jet.variable(I('.2','.3'),order=order) if order else Jet('.2')
        expected=a*3
        with binary_derivatives():
            assert (a*0).c=={}
            result=3*a
            assert all((result[k].lo,result[k].hi)==(v.lo,v.hi) for k,v in expected.c.items())
            with pytest.raises(ValueError,match='order mismatch'):a*Jet(order=(order+1)%11)

def test_context_restores_after_exception_and_rejects_nesting():
    from tools.opaque_binary_derivatives import binary_derivatives
    before=Jet.__mul__,Jet.__rmul__
    with pytest.raises(RuntimeError,match='deliberate'):
        with binary_derivatives():
            with pytest.raises(RuntimeError,match='nested'):
                with binary_derivatives():pass
            raise RuntimeError('deliberate')
    assert (Jet.__mul__,Jet.__rmul__)==before
