"""Signed derivative certificates justify corner extrema, never sampled signs."""
from fractions import Fraction as Q
import pytest

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_certified_corner_ranges_contain_exact_rational_interior(sx,sy):
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_shading_intervals import I
    calls=[]
    def field(x,y):calls.append(1);return [sx*x/(x+1)+sy*y/(y+1)]
    old=bounded_cosines(field,I(1,'1.1'),I(2,'2.1'));calls.clear()
    new=bounded_cosines(field,I(1,'1.1'),I(2,'2.1'),monotone=True)
    assert len(calls)==6 and old[0].lo<=new[0].lo<=new[0].hi<=old[0].hi
    assert new[0].width<old[0].width
    for i in range(11):
        for j in range(11):
            x=Q(1)+Q(i,100);y=Q(2)+Q(j,100);v=sx*x/(x+1)+sy*y/(y+1)
            assert Q(new[0].lo)<=v<=Q(new[0].hi)

def test_unknown_sign_keeps_whole_domain_and_does_not_sample_corners():
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_shading_intervals import I
    calls=[]
    def field(x,y):calls.append(1);return [(x-I('.5'))**2+y]
    v=bounded_cosines(field,I(0,1),I(0,1),monotone=True)[0]
    assert len(calls)==4 and v.lo<=0 and v.hi>=I('1.25').hi

def test_shared_corners_are_evaluated_once_and_cancellation_propagates():
    from tools.opaque_loss_extension import bounded_cosines
    from tools.opaque_shading_intervals import I,Unresolved
    calls=[]
    def field(x,y):
        calls.append(1)
        if len(calls)>4:raise Unresolved('stop corner work')
        return [x+y,2*x+y]
    with pytest.raises(Unresolved):bounded_cosines(field,I(0,1),I(0,1),monotone=True)
    calls.clear()
    def shared(x,y):calls.append(1);return [x+y,2*x+y]
    bounded_cosines(shared,I(0,1),I(0,1),monotone=True);assert len(calls)==6
