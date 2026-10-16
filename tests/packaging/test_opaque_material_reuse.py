"""Exact controls for shared material inputs and localized jump derivatives."""
import pytest
from fractions import Fraction as Q
from tools.opaque_shading_intervals import I,Jet


def test_axis_bounds_does_not_repeat_identical_input_derivatives():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();calls=[]
    def inputs(x,y):calls.append((x.order,y.order));return x.lift('.6'),x.lift('.75'),x.lift(1),x.lift(1)
    c=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True)
    c.axis_bounds((I(0,1),I(0,1)),0)
    assert calls==[(3,3)]


@pytest.mark.parametrize('ends',[(0,1),('.12','.13'),('.49','.51'),(0,0),(1,1)])
def test_absolute_kernel_bounds_share_existing_signed_cache(ends):
    from tools.opaque_peano import kernel_local_ranges,kernel_local_bounds,_kernel_local_ranges_cached
    p=I(*ends);_kernel_local_ranges_cached.cache_clear();signed=kernel_local_ranges(p);before=_kernel_local_ranges_cached.cache_info()
    absolute=kernel_local_bounds(p);after=_kernel_local_ranges_cached.cache_info()
    assert after.hits==before.hits+1 and after.misses==before.misses
    assert [(v.lo,v.hi) for v in absolute]==[(v.bound,v.bound) for v in signed]
    assert after.currsize<=128


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('position',[I('.3','.4'),I('.5'),I(0,1)])
def test_jump_inputs_use_certified_region_and_preserve_spatial_jet(axis,position):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();calls=[]
    def inputs(x,y):calls.append((x,y));return x*y+x,x-y,x.lift(1),1+x*x
    c=MaterialCorrection(e,inputs,curved=False);box=(I(0,1),I(0,1));args=[Jet(v,order=3) for v in box];args[axis]=Jet.variable(box[axis],order=3)
    original=inputs(*args);calls.clear();got=c._jump_inputs(box,axis,position,original)
    if position.width==box[axis].width:assert got is original and not calls
    else:
        assert len(calls)==1
        assert calls[0][axis][0,0].lo==position.lo and calls[0][axis][0,0].hi==position.hi
        assert calls[0][axis][1,0].lo==calls[0][axis][1,0].hi==1
        assert calls[0][1-axis][1,0].zero
        for v,w in zip(got,original):
            for k in range(4):assert w[k,0].lo<=v[k,0].lo<=v[k,0].hi<=w[k,0].hi


@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
@pytest.mark.parametrize('offset',['.4988','.4992','.4998'])
def test_localized_jump_encloses_exact_log_material(sx,sy,offset,monkeypatch,record_property):
    import json
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy
    a=I(offset)+(I('.001') if sx<0 else I(0))+(I('.001') if sy<0 else I(0))
    def inputs(x,y):return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    correction=MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True)
    actual=correction._jump_inputs
    monkeypatch.setattr(correction,'_jump_inputs',lambda box,axis,position,values:values)
    before=correction((I(0,1),I(0,1)))[1]
    monkeypatch.setattr(correction,'_jump_inputs',actual)
    after=correction((I(0,1),I(0,1)))[1];exact=_log_rectangle(e,eps,slope,a,b,c)
    assert before.lo<=exact.lo<=exact.hi<=before.hi
    assert after.lo<=exact.lo<=exact.hi<=after.hi
    assert after.width<=before.width*I('1.00000000000000000000000001').hi
    record_property('jump_localization',json.dumps(dict(before=str(before.width),after=str(after.width))))


def test_jump_localization_propagates_stop_before_extra_inputs():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();calls=[]
    def stop():raise RuntimeError('stop')
    c=MaterialCorrection(e,lambda *a:calls.append(a),curved=False,checkpoint=stop)
    with pytest.raises(RuntimeError,match='stop'):c._jump_inputs((I(0,1),I(0,1)),0,I('.2','.3'),None)
    assert not calls
