"""Independent exact and90digit integrals of discontinuous angular tables."""
from fractions import Fraction as Q
import math
import pytest

def step_curve():
    from tools.opaque_shading_intervals import I
    class Curve:
        cn=[I('.25'),I('.5'),I(1)]
        x=[v.ln() for v in cn]
        horizon=[I(1),I(-2)]
        values=[[I(2),I(-1)],[I(4),I(3)],[I(4),I(3)]]
        def power(self,i,ch,anchor):return [I([[2,-1],[4,3]][i][ch]),I(0),I(0),I(0)]
    return Curve()

@pytest.mark.parametrize('order',[1,2,3,4,5,6])
@pytest.mark.parametrize('a,q',[(0,Q(1,8)),(0,Q(3,4)),(Q(3,4),Q(1,8)),(Q(1,4),Q(1,2)),(Q(1,2),Q(1,2))])
def test_primitives_keep_every_step_horizon_and_orientation(order,a,q):
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    f=Primitives(step_curve(),max_order=6);got=f.integral(I(q),I(a),order)
    for ch in [0,1]:
        total=Q(0);lo,hi=sorted([a,q])
        for left,right,co in [(Q(0),Q(1,4),([1,4] if ch==0 else [-2,4])),(Q(1,4),Q(1,2),([2] if ch==0 else [-1])),(Q(1,2),Q(1),([4] if ch==0 else [3]))]:
            l,h=max(lo,left),min(hi,right)
            if l>=h:continue
            for j in range(order):
                for k,c in enumerate(co):total+=Q(math.comb(order-1,j)*(-1)**j,math.factorial(order-1))*q**(order-1-j)*c*(h**(j+k+1)-l**(j+k+1))/Q(j+k+1)
        if q<a:total=-total
        exact=I(total);assert got[ch].lo<=exact.lo<=exact.hi<=got[ch].hi
        assert got[ch].width<I('1e-48').lo

@pytest.mark.parametrize('order',[1,2,3,4,5,6])
@pytest.mark.parametrize('q',[Q(1,3),Q(3,4),Q(1)])
def test_log_cubic_primitives_match_independent_exponential_moments(order,q):
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    from tests.packaging.test_opaque_residual_range import curve
    from decimal import Decimal as D,localcontext
    c=curve(7);got=Primitives(c,max_order=6).integral(I(q),I(Q(1,7)),order)
    with localcontext() as ctx:
        ctx.prec=90;end=D(q.numerator)/D(q.denominator);start=D(1)/7;total=[D(0),D(0)]
        for i in range(7):
            lo=max(start,D(i+1)/8);hi=min(end,D(i+2)/8)
            if lo>=hi:continue
            for ch in [0,1]:
                co=[D(z.numerator)/D(z.denominator) for z in c.global_power(i,ch)]
                for j in range(order):
                    lam=D(j+1)
                    def primitive(x):
                        z=x.ln();s=D(0)
                        for k,a in enumerate(co):
                            for v in range(k+1):s+=a*D((-1)**v*math.factorial(k))/math.factorial(k-v)*(z**(k-v) if k!=v else D(1))/lam**(v+1)
                        return x**(j+1)*s
                    total[ch]+=D(math.comb(order-1,j)*(-1)**j)/math.factorial(order-1)*end**(order-1-j)*(primitive(hi)-primitive(lo))
        for ch in [0,1]:assert got[ch].lo<=total[ch]<=got[ch].hi

def test_uncertain_arguments_enclose_all_points_across_knots():
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    f=Primitives(step_curve(),max_order=6)
    for n in [1,2,3]:
        wide=f.integral(I('.2','.8'),I('.4','.6'),n)
        for q in ['.2','.25','.5','.8']:
            for a in ['.4','.5','.6']:
                narrow=f.integral(I(q),I(a),n)
                assert all(x.lo<=y.lo<=y.hi<=x.hi for x,y in zip(wide,narrow))

def test_primitives_refuse_bad_domains_and_preserve_cancellation_and_snapshot():
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    c=step_curve();f=Primitives(c,max_order=6);before=f.integral(I('.8'),I('.1'),3);c.horizon[0]=I(999);c.cn[0]=I('.01')
    assert [(z.lo,z.hi) for z in before]==[(z.lo,z.hi) for z in f.integral(I('.8'),I('.1'),3)]
    for q,a,n in [(-1,0,1),(1,2,1),(1,0,7),(1,0,True)]:
        with pytest.raises(ValueError):f.integral(I(q),I(a),n)
    def stop():raise RuntimeError('cancel primitive work')
    with pytest.raises(RuntimeError,match='cancel'):Primitives(step_curve(),checkpoint=stop)


@pytest.mark.parametrize('base',[-1,0,1])
@pytest.mark.parametrize('order',[1,2,3,4,5,6])
def test_residual_primitives_retain_exact_same_branch_zero_and_nonzero_changes(base,order):
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    f=Primitives(step_curve(),max_order=6);a,q=[('.05','.2'),('.3','.4'),('.6','.8')][base+1]
    assert all(v.zero for v in f.residual(I(q),I(a),order,base))
    got=f.residual(I('.75'),I('.25'),order,0)
    # Difference from constant2 is0until.5, then2.
    exact=I(2)*I('.25')**order/math.factorial(order)
    assert got[0].lo<=exact.lo<=exact.hi<=got[0].hi


@pytest.mark.parametrize('order',[1,2,3,4,5,6])
def test_small_residual_amplitude_survives_uncertain_endpoints(order):
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    c=step_curve();eps=I('1e-30');c.power=lambda i,ch,anchor:[I(2)+(eps if i else I(0)),I(0),I(0),I(0)]
    c.values=[[I(2),I(2)]]*3;c.horizon=[I(2),I(2)]
    got=Primitives(c,max_order=6).residual(I('.55','.75'),I('.25'),order,0)[0]
    upper=eps*I('.5')**order/math.factorial(order)
    assert got.lo>=0 and got.hi<=upper.hi*I('1.000000000000000001').hi
    exact=eps*I('.25')**order/math.factorial(order)
    assert got.lo<=exact.lo<=exact.hi<=got.hi


def test_centered_primitive_bound_preserves_cancellation_over_many_knots():
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    c=step_curve();eps=I('1e-30');c.power=lambda i,ch,anchor:[I(2)+(eps if i else I(0)),I(0),I(0),I(0)];c.values=[[I(2),I(2)]]*3;c.horizon=[I(2),I(2)]
    got=Primitives(c,max_order=6).residual(I('.7','.70001'),I('.25'),3,0)[0]
    for q in ['.7','.700005','.70001']:
        exact=eps*(I(q)-I('.5'))**3/6
        assert got.lo<=exact.lo<=exact.hi<=got.hi
    assert got.width<I('1e-36').lo


@pytest.mark.parametrize('q',[('.7','.7'),('.7','.70001'),('.55','.75')])
def test_residual_family_preserves_all_endpoints_and_reuses_finite_moments(q,monkeypatch):
    from tools.opaque_angular_primitives import Primitives
    from tools.opaque_shading_intervals import I
    f=Primitives(step_curve(),max_order=6);at=I(*q);a=I('.25')
    expected=[f.residual(at,a,n,0) for n in range(1,7)]
    count=[];original=f.moments
    def counted(v):count.append((v.lo,v.hi));return original(v)
    monkeypatch.setattr(f,'moments',counted)
    got=f.residual_family(at,a,6,0)
    assert [[(v.lo,v.hi) for v in row] for row in got]==[[(v.lo,v.hi) for v in row] for row in expected]
    assert len(count)==len(set(count)) and len(count)<=3
    before=len(count);again=f.residual_family(at,a,6,0)
    assert len(count)>before # call-local lifetime, no accumulating global cache
    got[0][0].lo=I(-999).lo
    assert again[0][0].lo>=0 # no mutable alias across calls
