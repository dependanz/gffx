"""Independent root, moment, complete-jump and resource tests for Gauss8."""
from fractions import Fraction as Q
import math
import pytest

def test_roots_are_unique_positive_weights_and_all_exact_moments():
    from tools.opaque_gauss8 import rule
    from tools.opaque_shading_intervals import I
    nodes,weights=rule()
    assert len(nodes)==len(weights)==8
    for x,w in zip(nodes,weights):
        r=2*x-1
        p=(6435*r**8-12012*r**6+6930*r**4-1260*r**2+35)/128
        assert p.lo<=0<=p.hi and 0<x.lo<x.hi<1 and w.lo>0
        assert x.width<I('1e-55').lo
    assert all(a.hi<b.lo for a,b in zip(nodes,nodes[1:]))
    for k in range(16):
        v=sum((w*x**k for x,w in zip(nodes,weights)),I(0));q=I(Q(1,k+1))
        assert v.lo<=q.lo<=q.hi<=v.hi and v.width<I('1e-50').lo
    # Rule is not exact beyond degree15: avoid claiming order18.
    v=sum((w*x**16 for x,w in zip(nodes,weights)),I(0));q=I(Q(1,17))
    assert v.hi<q.lo

def test_construction_is_cancellable_and_returns_owned_intervals():
    from tools.opaque_gauss8 import rule
    class Stop(Exception):pass
    def cancel():raise Stop
    with pytest.raises(Stop):rule(checkpoint=cancel)
    a,b=rule();expected=a[0].lo;a[0].lo=0;b.clear()
    c,d=rule();assert c[0].lo==expected and len(d)==8

def test_explicit_context_restores_and_does_not_patch_old_producer():
    from tools.opaque_gauss8 import context
    from tools import opaque_high_peano as h,opaque_peano as old
    saved=h.gauss5;old_rule=old.gauss5
    with pytest.raises(RuntimeError,match='sentinel'):
        with context():
            assert len(h.gauss5()[0])==8 and old.gauss5 is old_rule
            with pytest.raises(ValueError,match='nested'):
                with context():pass
            raise RuntimeError('sentinel')
    assert h.gauss5 is saved and old.gauss5 is old_rule

@pytest.mark.parametrize('order',[2,3,6,10])
@pytest.mark.parametrize('degree',range(13))
def test_complete_identity_keeps_exact_truncated_polynomial(degree,order):
    from tools.opaque_gauss8 import context
    from tests.packaging.test_opaque_high_peano import test_complete_high_identity_contains_exact_truncated_polynomial_integral as original
    with context():original(degree,order)

def test_fixed_work_charge_and_attempted_work_never_exceed_256():
    from tools.opaque_gauss8 import Work
    w=Work(256)
    assert w.charge==3
    for _ in range(85):w.take()
    assert (w.units,w.regions)==(255,85)
    with pytest.raises(ValueError,match='work budget'):w.take()
    assert (w.units,w.regions)==(255,85)
    for n in [0,257,True,1.5]:
        with pytest.raises(ValueError):Work(n)

def test_cannot_run_any_region_without_full_charge():
    from tools.opaque_gauss8 import Work
    w=Work(2)
    with pytest.raises(ValueError):w.take()
    assert w.units==w.regions==0
