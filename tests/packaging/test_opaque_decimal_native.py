"""Exact endpoint identity, ownership and independent rational proof controls."""
import pytest,random
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet
@pytest.mark.parametrize('order',[0,1,2,3,6,10])
@pytest.mark.parametrize('seed',range(5))
def test_compiled_decimal_preserves_endpoint_and_iteration_identity(order,seed):
    from tools.opaque_decimal_convolution import decimal_convolution
    r=random.Random(seed)
    def make():
        keys=[(i,j) for i in range(order+1) for j in range(2)];r.shuffle(keys)
        return Jet(order=order,coeff={k:I(I(Q(x:=r.randrange(-20,21),17)).lo,I(Q(x+r.randrange(0,7),17)).hi) for k in keys})
    a,b=make(),make();old=a*b
    with decimal_convolution() as stats:got=a*b
    assert stats['fast']==1
    assert list(old.c)==list(got.c)
    assert all((old[k].lo,old[k].hi)==(got[k].lo,got[k].hi) for k in old.c)
    assert got is not a and got is not b

@pytest.mark.parametrize('value',['1e-400','1e400','-1e-200','0'])
def test_compiled_decimal_extremes_and_exact_zero(value):
    from tools.opaque_decimal_convolution import decimal_convolution
    from tests.packaging.test_opaque_binary_derivatives import exact
    a=Jet(order=6,coeff={(i,j):I(value) for i in range(7) for j in range(2)});truth=exact(a,a);old=a*a
    with decimal_convolution():got=a*a
    assert list(old.c)==list(got.c)
    for key,(lo,hi) in truth.items():
        assert Q(got[key].lo)<=lo<=hi<=Q(got[key].hi)
        assert (old[key].lo,old[key].hi)==(got[key].lo,got[key].hi)

def test_compiled_decimal_context_restoration_and_mismatch():
    from tools.opaque_decimal_convolution import decimal_convolution
    from decimal import getcontext
    old=Jet.__mul__,Jet.__rmul__;precision=getcontext().prec;rounding=getcontext().rounding
    with pytest.raises(RuntimeError,match='deliberate'):
        with decimal_convolution():
            with pytest.raises(RuntimeError,match='nested'):
                with decimal_convolution():pass
            with pytest.raises(ValueError):Jet(order=3)*Jet(order=4)
            raise RuntimeError('deliberate')
    assert (Jet.__mul__,Jet.__rmul__)==old
    assert (getcontext().prec,getcontext().rounding)==(precision,rounding)

@pytest.mark.parametrize('case',['order','count','duplicate','axis','tuple'])
def test_decimal_native_rejects_invalid_coefficients(case):
    from tools.opaque_decimal_convolution import library
    from decimal import localcontext,Decimal,ROUND_FLOOR,ROUND_CEILING
    rows=[(0,0,Decimal(1),Decimal(1))];order=3
    if case=='order':order=11
    elif case=='count':rows*=23
    elif case=='duplicate':rows*=2
    elif case=='axis':rows=[(0,2,Decimal(1),Decimal(1))]
    elif case=='tuple':rows=[(0,0)]
    with localcontext() as ctx:
        with pytest.raises(ValueError):library().convolve(rows,[],order,ctx,Decimal(0),ROUND_FLOOR,ROUND_CEILING)

def test_decimal_native_preserves_exception_and_restores_context():
    from tools.opaque_decimal_convolution import decimal_convolution
    from decimal import localcontext,Overflow,getcontext
    a=Jet('1e80');old=Jet.__mul__
    with localcontext() as ctx:
        ctx.Emax=100;before=getcontext().rounding
        with pytest.raises(Overflow):
            with decimal_convolution():a*a
        assert getcontext().rounding==before and Jet.__mul__==old

def test_decimal_native_rejects_manifest_tampering(monkeypatch):
    from tools.opaque_decimal_convolution import library
    monkeypatch.setenv('GFFX_PROOF_DECIMAL_MANIFEST_SHA','0'*64)
    with pytest.raises(ValueError,match='manifest hash'):library()

def test_decimal_native_keeps_scalar_overhead_on_original_path():
    from tools.opaque_decimal_convolution import decimal_convolution
    a=Jet('.3');b=Jet('.7');old=a*b
    with decimal_convolution() as stats:got=a*b
    assert stats['fast']==0 and stats['ordinary']==1
    assert (got[0,0].lo,got[0,0].hi)==(old[0,0].lo,old[0,0].hi)
