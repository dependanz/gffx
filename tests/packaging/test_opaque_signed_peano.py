"""Independent exact integrals for signed Peano boundary corrections."""
from fractions import Fraction as Q
import json,math
import pytest
from tools.opaque_shading_intervals import I,Jet


def witness(position,values,required=True):
    return dict(position=I(position),jumps=[I(I(v).bound) for v in values],signed_jumps=list(map(I,values)),required=required,max_crossings=1)


def positive(z,degree):
    if z.hi<0:return I(0)
    if z.lo>0:return z**degree
    return I(0,max(I(1).hi,(z**degree).hi)) if degree==0 else I(0,(I(0,max(z.hi,I(0).hi))**degree).hi)


@pytest.mark.parametrize('degree',[0,1,2])
@pytest.mark.parametrize('orientation',[-1,1])
def test_signed_jump_recovers_exact_truncated_power(degree,orientation,record_property):
    from tools.opaque_peano import residual_integral,kernel_local_ranges
    knot=Q(13,100);scale=Q(1,10**9);box=(I(0,1),I(0,1))
    def field(x,y):return [x.lift(positive((x[0,0]-I(knot))*orientation,degree)*scale)]
    values=[I(0)]*3;values[degree]=I(scale*math.factorial(degree)*orientation**(degree+1))
    def bounds(box,axis):return [dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[] if axis else [witness(knot,values)])]
    def unsigned(box,axis):
        rows=bounds(box,axis)
        for w in rows[0]['localized_jumps']:w.pop('signed_jumps')
        return rows
    got=residual_integral(field,box,bounds)[0];old=residual_integral(field,box,unsigned)[0]
    exact=scale*((1-knot) if orientation==1 else knot)**(degree+1)/(degree+1)
    assert Q(got.lo)<=exact<=Q(got.hi)
    assert got.width<I('1e-50').lo and got.width*1000<old.width
    record_property('signed_exact',json.dumps(dict(degree=degree,orientation=orientation,width=str(got.width),unsigned_width=str(old.width))))


@pytest.mark.parametrize('curved',[False,True])
def test_oblique_and_curved_boundary_enclose_exact_polynomial(curved,record_property):
    from tools.opaque_peano import residual_integral,kernel_local_ranges
    a=Q(13,100);b=Q(1,10000);scale=Q(1,10**9);power=2 if curved else 1;box=(I(0,1),I(0,1))
    def field(x,y):return [x.lift(positive(x[0,0]-I(a)-I(b)*y[0,0]**power,2)*scale)]
    def bounds(box,axis):
        if axis==0:entry=witness(I(I(a).lo,I(a+b).hi),[0,0,2*scale]);third=I(0)
        else:entry=witness(I(0,1),[0,0,I(I(-8*b*b*scale).lo,0) if curved else I(-2*b*b*scale)],False);third=I(24*b*b*scale if curved else 0)
        return [dict(third=third,jumps=[I(0)]*3,localized_jumps=[entry])]
    def unsigned(box,axis):
        rows=bounds(box,axis);rows[0]['localized_jumps'][0].pop('signed_jumps');return rows
    got=residual_integral(field,box,bounds)[0];old=residual_integral(field,box,unsigned)[0]
    exact=scale*sum((Q(math.comb(3,j))*(1-a)**(3-j)*(-b)**j/Q(power*j+1) for j in range(4)),Q(0))/3
    assert Q(got.lo)<=exact<=Q(got.hi)
    assert got.width*20<old.width
    record_property('signed_boundary',json.dumps(dict(curved=curved,width=str(got.width),unsigned_width=str(old.width),exact=str(exact))))


def test_optional_crossing_includes_zero_error():
    from tools.opaque_peano import residual_integral,kernel_local_ranges
    # A conservative producer cannot establish that its potential step exists.
    def bounds(box,axis):return [dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[] if axis else [witness('.13',[I('1e-9'),0,0],False)])]
    got=residual_integral(lambda x,y:[x.lift(0)],(I(0,1),I(0,1)),bounds)[0]
    assert got.lo<=0<=got.hi and got.width>0


def test_signed_kernel_covers_both_sides_of_uncertain_nodes():
    from tools.opaque_peano import kernel_local_ranges,gauss5
    nodes,weights=gauss5()
    for n,w in zip(nodes,weights):
        r=kernel_local_ranges(n)
        assert r[0].width>=w.lo
    assert kernel_local_ranges(I('.13'))[0].width<I('1e-50').lo


@pytest.mark.parametrize('change',[{'max_crossings':2},{'required':'yes'},{'signed_jumps':[I(1)]},{'signed_jumps':[I(2),I(0),I(0)]}])
def test_signed_witness_requires_single_crossing_and_valid_intervals(change):
    from tools.opaque_peano import residual_integral,kernel_local_ranges
    w=witness('.13',[1,0,0]);w.update(change)
    with pytest.raises(ValueError):residual_integral(lambda x,y:[x.lift(0)],(I(0,1),I(0,1)),lambda b,a:[dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[w])])


@pytest.mark.parametrize('degree',[0,1,2])
def test_signed_physical_lengths_and_transverse_mass(degree):
    from tools.opaque_peano import residual_integral
    box=(I(2,4),I(3,6));knot=Q(113,50);scale=Q(1,10**9)
    def field(x,y):return [x.lift(positive(x[0,0]-I(knot),degree)*scale)]
    values=[I(0)]*3;values[degree]=I(scale*math.factorial(degree))
    def bounds(box,axis):return [dict(third=I(0),jumps=[I(0)]*3,localized_jumps=[] if axis else [witness(knot,values)])]
    got=residual_integral(field,box,bounds)[0];exact=3*scale*(4-knot)**(degree+1)/(degree+1)
    assert Q(got.lo)<=exact<=Q(got.hi) and got.width<I('1e-49').lo
