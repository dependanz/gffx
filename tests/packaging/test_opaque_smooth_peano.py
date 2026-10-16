"""Exact controls for centering the ordinary third-derivative Peano remainder."""
import math
from fractions import Fraction as Q
import pytest
from tools.opaque_shading_intervals import I,Jet


@pytest.mark.parametrize('value',[-7,0,13])
def test_constant_third_derivative_has_zero_centered_remainder(value):
    from tools.opaque_smooth_peano import ordinary_bounds
    row=dict(third=I(abs(value)),third_range=I(value))
    b=ordinary_bounds(row,I(2,5),I(4))
    assert b['centered'].zero and b['effective'].zero
    assert b['unanchored'].hi>=0


@pytest.mark.parametrize('value',[I(-7,-6),I(6,7),I(-3,4)])
def test_centering_retains_full_signed_range_and_physical_scale(value):
    from tools.opaque_smooth_peano import ordinary_bounds
    from tools.opaque_peano import kernel_bounds
    row=dict(third=I(value.bound),third_range=value)
    b=ordinary_bounds(row,I(2,4),I(3))
    assert b['effective'].hi<=b['unanchored'].hi
    assert (value-b['anchor']).bound<=b['variation'].hi
    expected=b['variation']*I(2)**4*kernel_bounds()[1][2]*3
    assert b['centered'].hi==expected.hi


@pytest.mark.parametrize('row',[dict(third=I(-1)),dict(third=I(1),third_range=I(-2,1)),dict(third=I(1),third_range=I(0,2))])
def test_smooth_bound_refuses_inconsistent_derivative_evidence(row):
    from tools.opaque_smooth_peano import ordinary_bounds
    with pytest.raises(ValueError):ordinary_bounds(row,I(0,1),I(1))


def test_smooth_bound_preserves_unsigned_fallback():
    from tools.opaque_smooth_peano import ordinary_bounds
    b=ordinary_bounds(dict(third=I(3)),I(0,1),I(1))
    assert b['effective'].lo==b['unanchored'].lo and b['effective'].hi==b['unanchored'].hi
    assert b['anchor'] is None


def test_material_producer_keeps_signed_third_without_extra_inputs():
    from tests.packaging.test_opaque_boundary_diagnostic import fixture
    c=fixture();calls=[0];original=c.inputs
    def counted(*args):calls[0]+=1;return original(*args)
    c.inputs=counted;rows=c.axis_bounds((I(0,1),I(0,1)),1)
    for row in rows:assert row['third_range'].bound<=row['third'].hi
    assert rows[1]['third_range'].width>0 and calls[0]==5


def test_diagnostic_components_reconstruct_both_tensor_radii():
    from tools.opaque_boundary_diagnostic import diagnose
    from tests.packaging.test_opaque_boundary_diagnostic import fixture
    r=diagnose(fixture(),[(I(0,1),I(0,1))])
    assert sum(r['input_calls_by_order'].values())==r['input_calls']
    for a in r['boxes'][0]['orientations']:
        for k,error in enumerate(a['tensor_error']):
            parts=a['components'];radius=sum((parts[direction][k][name] for direction in ('conditioned','integrated') for name in ('ordinary','global_jumps','localized_jumps')),I(0))
            # Independent interval additions may add an outward rounding ulp.
            assert abs(float(error.width)-2*float(radius.hi))<1e-45
            comparison=a['ordinary_comparison'][k]
            assert comparison['centered'].hi<=comparison['unanchored'].hi
        assert a['integrated_seconds']>=0 and a['conditioned_seconds']>=0


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sign',[-1,1])
def test_fixed_subboxes_keep_all_derivative_work_and_exact_cubic(axis,sign):
    from tools.opaque_smooth_peano import partitioned_ordinary
    calls=[]
    def third(box,active):calls.append((box,active));return I(6*sign)
    r=partitioned_ordinary(third,(I(2,4),I(-3,0)),axis)
    assert len(calls)==4 and r['evaluations']==4 and r['effective'].zero
    assert r['unanchored'].hi>0


@pytest.mark.parametrize('axis',[0,1])
def test_partitioned_bound_retains_change_between_inner_halves(axis):
    from tools.opaque_smooth_peano import partitioned_ordinary
    def third(box,active):return I(0) if box[axis].hi<=I('.5').hi else I(6)
    r=partitioned_ordinary(third,(I(0,1),I(0,1)),axis)
    assert r['effective'].hi>0  # Separate anchors per INNER half would wrongly return zero.


def test_partitioned_bound_refuses_bad_scope_and_propagates_stop():
    from tools.opaque_smooth_peano import partitioned_ordinary
    called=[]
    with pytest.raises(ValueError):partitioned_ordinary(lambda *a:called.append(a),(I(0,1),I(0,1)),2)
    assert not called
    def stop():raise InterruptedError('stop')
    with pytest.raises(InterruptedError):partitioned_ordinary(lambda *a:called.append(a),(I(0,1),I(0,1)),0,checkpoint=stop)
    assert not called


@pytest.mark.parametrize('degree',[3,4,6])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('physical',[False,True])
@pytest.mark.parametrize('sign',[-1,1])
def test_partitioned_remainder_encloses_exact_truncated_polynomial(degree,axis,physical,sign):
    from tools.opaque_smooth_peano import partitioned_ordinary
    from tools.opaque_peano import gauss5
    inner=I(2,4) if physical else I(0,1);outer=I(-3,0) if physical else I(0,1)
    box=[None,None];box[axis]=inner;box[1-axis]=outer
    hi=I(inner.hi)-I(inner.lo);ho=I(outer.hi)-I(outer.lo);k=I('.37')
    def third(sub,active):
        x=(sub[axis]-I(inner.lo))/hi;y=(sub[1-axis]-I(outer.lo))/ho;z=x-k
        if z.hi<0:return I(0)
        if degree==3:v=I(0,1) if z.lo<=0 else I(1)
        else:v=I(max(I(0).lo,z.lo),max(I(0).hi,z.hi))**(degree-3)
        return v*(1+y)*(sign*math.factorial(degree)//math.factorial(degree-3))/hi**3
    bound=partitioned_ordinary(third,tuple(box),axis)['effective']
    nodes,weights=gauss5();q=I(0)
    for x,w in zip(nodes,weights):
        z=x-k;v=I(max(I(0).lo,z.lo),max(I(0).hi,z.hi))**degree
        q=q+v*w*I(Q(3,2))*hi*ho*sign
    exact=I(Q(3,2))*I(Q(63,100))**(degree+1)/(degree+1)*hi*ho*sign
    got=q+I(bound.hi.copy_negate(),bound.hi)
    assert got.lo<=exact.lo<=exact.hi<=got.hi


def test_partition_diagnostic_charges_eight_extra_calls_and_keeps_base_result():
    from tools.opaque_boundary_diagnostic import diagnose
    from tests.packaging.test_opaque_boundary_diagnostic import fixture
    before=diagnose(fixture(),[(I(0,1),I(0,1))]);after=diagnose(fixture(),[(I(0,1),I(0,1))],partition_smooth=True)
    assert after['input_calls']==before['input_calls']+8
    for a,b in zip(before['boxes'][0]['orientations'],after['boxes'][0]['orientations']):
        assert str(a['tensor_error'])==str(b['tensor_error'])
        assert b['partitioned_ordinary']['input_calls']==4


@pytest.mark.parametrize('orientation_x',[-1,1])
@pytest.mark.parametrize('orientation_y',[-1,1])
@pytest.mark.parametrize('offset',['.4988','.4992','.4998'])
def test_partitioned_material_retains_exact_log_integral_and_all_jumps(orientation_x,orientation_y,offset):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*orientation_x;c=I('.001')*orientation_y
    a=I(offset)+(I('.001') if orientation_x<0 else I(0))+(I('.001') if orientation_y<0 else I(0))
    def inputs(x,y):return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    old=MaterialCorrection(e,inputs,curved=False,transverse=True)
    new=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True)
    box=(I(0,1),I(0,1));before=old(box)[1];after=new(box)[1];exact=_log_rectangle(e,eps,slope,a,b,c)
    assert after.lo<=exact.lo<=exact.hi<=after.hi
    assert before.lo<=after.lo<=after.hi<=before.hi
    w=new.integrated_bounds(box,1)[1]
    assert 'ordinary_integral' in w and len(w['localized_jumps'][0]['signed_jumps'])==3


def test_partitioned_material_curved_map_retains_exact_measure():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();a=I('.4988');b=I('.001');c=I('.001')
    def inputs(x,y):return (x+x*x)*b+y*c+a,x.lift('.75'),x.lift(1),1+2*x
    correction=MaterialCorrection(e,inputs,curved=True,transverse=True,smooth_partition=True)
    got=correction((I(0,1),I(0,1)))[1];known=_log_rectangle(e,eps,slope,a,b,c,I(2))
    assert got.lo<=known.lo<=known.hi<=got.hi


@pytest.mark.parametrize('offset',['.4988','.4992'])
def test_partitioned_material_work_is_charged_at_unchanged_accuracy(offset,record_property):
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_square_reference import integrate_square
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    import json
    e,eps,slope=material_fixture();a=I(offset);b=c=I('.001');calls=[0]
    def inputs(x,y):calls[0]+=1;return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    results=[]
    for enabled in (False,True):
        calls[0]=0;correction=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=enabled)
        r=integrate_square(lambda x,y:[x.lift(0),x.lift(0)],correction_integrator=correction,tolerance='1e-12',max_cells=256,max_depth=10)
        exact=_log_rectangle(e,eps,slope,a,b,c);got=r['value'][1]
        assert got.lo<=exact.lo<=exact.hi<=got.hi and got.width<=I('2e-12').lo
        results.append(dict(enabled=enabled,cells=r['cells'],input_calls=calls[0],seconds=r['seconds'],width=str(got.width)))
    record_property('smooth_material_work',json.dumps(dict(offset=offset,results=results)))


def test_integrated_ordinary_bound_never_replaces_conditioned_mass():
    from tools.opaque_transverse_peano import _error
    row=dict(third=I(1),jumps=[I(0)]*3,ordinary_integral=I(0))
    assert _error(row,I(0,1),I(1),True)[0].zero
    assert _error(row,I(0,1),I(1),False)[0].width>0
    row['ordinary_integral']=I(-1)
    with pytest.raises(ValueError):_error(row,I(0,1),I(1),True)
