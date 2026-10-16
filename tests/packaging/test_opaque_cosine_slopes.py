"""Independent closed controls for exact polynomial cosine slopes."""
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet
from tools.opaque_moving_reference import P


def coefficients(tiny=False):
    e=Q(1,2**60) if tiny else Q(1)
    return {'n':[[P(0),P(e),P(0)],[P(1),P(0),P(0)],[P(0)]*3],
            'v':[[P(0)]*3,[P(1),P(0),P(0)],[P(0),P(0),P(1)]]}

@pytest.mark.parametrize('tiny',[False,True])
@pytest.mark.parametrize('axis',[0,1])
def test_cancelled_slopes_enclose_closed_normalization_and_tiny_terms(tiny,axis):
    from tools.opaque_cosine_slopes import FixedCosineSlopes
    c=coefficients(tiny);proof=FixedCosineSlopes(c,(1,1,2));e=I(Q(1,2**60) if tiny else 1)
    for x,y in [(Q(1,4),Q(1,3)),(Q(3,4),Q(2,3))]:
        t,s=Jet(I(x),order=1),Jet(I(y),order=1)
        if axis==0:t=Jet.variable(I(x),order=1)
        else:s=Jet.variable(I(y),order=1)
        N=1+(t*e)**2;V=1+(t*s)**2
        closed=[1/(N*V).sqrt(),(1+t*e)/(6*N).sqrt()]
        for role,q in enumerate(closed):
            got=proof.bound(role,t,s);want=q[1,0]
            assert max(got.lo,want.lo)<=min(got.hi,want.hi)
            assert (got-want).bound<I('1e-45').lo
        if tiny and axis==0:assert proof.bound(1,t,s).lo>0

def test_cancelled_slopes_reject_intrinsic_motion_and_bad_contracts():
    from tools.opaque_cosine_slopes import FixedCosineSlopes
    c=coefficients();c['n'][0][0]=P([0,1])
    with pytest.raises(ValueError):FixedCosineSlopes(c,(1,1,2))
    p=FixedCosineSlopes(coefficients(),(1,1,2))
    for role,t,s in [(2,Jet(1,order=1),Jet(1,order=1)),(0,Jet(1),Jet(1)),(True,Jet(1,order=1),Jet(1,order=1))]:
        with pytest.raises(ValueError):p.bound(role,t,s)


def test_subdivided_slopes_cover_every_box_and_preserve_chain_and_tiny_terms():
    from tools.opaque_cosine_slopes import subdivided_slopes
    calls=[];tiny=Q(1,2**60)
    def field(t,s):
        calls.append((t[0,0],s[0,0]));return [(1+t)/(2+s),tiny*(t+s)]
    t=Jet.variable(I('.1','.5'),order=3);s=Jet.variable(I('.2','.8'),order=3);s.c[1,0]=I(-2)
    result=subdivided_slopes(field,t,s,parts=(2,4))
    assert len(calls)==8
    area=sum(((I(a.hi)-I(a.lo))*(I(b.hi)-I(b.lo)) for a,b in calls),I(0))
    assert area.lo<=I('.24').lo<=area.hi
    for x in (Q(1,10),Q(3,10),Q(1,2)):
        for y in (Q(1,5),Q(1,2),Q(4,5)):
            expected=(I(2)+I(y)+2*(1+I(x)))/(2+I(y))**2
            assert result[0].lo<=expected.lo<=expected.hi<=result[0].hi
    assert result[1].lo==result[1].hi==I(-tiny).lo

def test_subdivided_slopes_reject_unbounded_work_and_propagate_cancellation():
    from tools.opaque_cosine_slopes import subdivided_slopes
    from tools.opaque_shading_intervals import Unresolved
    args=(Jet.variable(I(0,1),order=1),Jet(I(0,1),order=1))
    for bad in ((4,4),(3,1),[2,4],(True,4)):
        with pytest.raises(ValueError):subdivided_slopes(lambda x,y:[x,y],*args,parts=bad)
    calls=[]
    def stop():
        calls.append(1)
        if len(calls)==3:raise Unresolved('cancel slope proof')
    with pytest.raises(Unresolved):subdivided_slopes(lambda x,y:[x,y],*args,checkpoint=stop)
    assert len(calls)==3
