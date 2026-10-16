"""Independent exact arithmetic and unchanged protected derivatives in compiled proof."""
import pytest
from tests.packaging.test_opaque_binary_derivatives import (
 test_exact_fraction_interval_convolution,
 test_extremes_preserve_tiny_terms_or_fallback,
 test_zero_sparse_scalars_low_orders_and_mismatch,
 test_context_restores_after_exception_and_rejects_nesting)
@pytest.fixture(autouse=True)
def compiled_backend(monkeypatch):
    from tools import opaque_binary_derivatives as old
    from tools.opaque_native_convolution import native_convolution
    monkeypatch.setattr(old,'binary_derivatives',native_convolution)

@pytest.mark.parametrize('bad',['order','count','duplicate','nan','reverse','axis'])
def test_native_abi_refuses_invalid_data(bad):
    from tools.opaque_native_convolution import library,Term
    import ctypes as C
    dll,fn=library();a=(Term*2)(Term(3,0,1,2),Term(4,0,1,2));b=(Term*1)(Term(0,0,1,1));order=6;n=2
    if bad=='order':order=11
    elif bad=='count':n=23
    elif bad=='duplicate':a[1]=a[0]
    elif bad=='nan':a[0].lo=float('nan')
    elif bad=='reverse':a[0].lo=3
    elif bad=='axis':a[0].j=2
    out=(C.c_double*44)();mask=C.c_uint32();assert fn(order,a,n,b,1,out,C.byref(mask))==2

def test_native_manifest_hash_refusal(monkeypatch):
    from tools.opaque_native_convolution import library
    monkeypatch.setenv('GFFX_PROOF_NATIVE_MANIFEST_SHA','0'*64)
    with pytest.raises(ValueError,match='manifest hash'):library()

@pytest.mark.parametrize('order',[3,4,5])
@pytest.mark.parametrize('sign',[-1,1])
def test_native_short_series_contains_exact_convolution(order,sign):
    from fractions import Fraction as Q
    from tools.opaque_shading_intervals import I,Jet
    from tools.opaque_native_convolution import native_convolution
    from tests.packaging.test_opaque_binary_derivatives import exact
    a=Jet(order=order,coeff={(i,j):I(Q(sign*(i+1),13**(j+i+1))) for i in range(order+1) for j in range(2)})
    b=Jet(order=order,coeff={(i,j):I(Q((-1)**i,7**(j+i+1))) for i in range(order+1) for j in range(2)})
    old=a*b;truth=exact(a,b)
    with native_convolution() as stats:got=a*b
    assert stats['fast']==1
    for key,(lo,hi) in truth.items():
        assert Q(got[key].lo)<=lo<=hi<=Q(got[key].hi)
        if key[0]<=2:assert (got[key].lo,got[key].hi)==(old[key].lo,old[key].hi)
