"""Exact reuse must preserve all jet coefficients, isolation and stop checks."""
import pytest
from tools.opaque_shading_intervals import I,Jet

def fixture():
    from tools.opaque_input_reuse import InputReuse
    calls=[];checks=[]
    def inputs(x,y):
        calls.append((x,y));return x*x+y,x-y,x.lift(1),1+x*y
    return InputReuse(inputs,checkpoint=lambda:checks.append(1)),inputs,calls,checks

def key(rows):return [[(k,str(v.lo),str(v.hi)) for k,v in sorted(q.c.items())] for q in rows]

@pytest.mark.parametrize('axis',[0,1])
def test_exact_polynomial_jets_and_returned_objects_are_independent(axis):
    cache,raw,calls,checks=fixture();args=[Jet(I('.2','.3'),order=3),Jet(I('.4','.5'),order=3)];args[axis]=Jet.variable(args[axis][0,0],order=3)
    first=cache(*args);expected=key(first);first[0].c[0,0]=I(999)
    second=cache(*args);assert key(second)==expected and len(calls)==1 and len(checks)==2
    second[1].c[1,0]=I(-99);assert key(cache(*args))==expected and len(calls)==1
    assert key(raw(*args))==expected

@pytest.mark.parametrize('change',['bound','derivative','order','explicit_zero'])
def test_every_input_coefficient_and_order_is_part_of_identity(change):
    cache,raw,calls,checks=fixture();x=Jet.variable(I('.2','.3'),order=3);y=Jet(I('.4'),order=3);cache(x,y)
    z=Jet(order=x.order,coeff=dict(x.c))
    if change=='bound':z.c[0,0]=I('.2','.31')
    if change=='derivative':z.c[1,0]=I(2)
    if change=='order':z=Jet.variable(I('.2','.3'),order=2);y=Jet(I('.4'),order=2)
    if change=='explicit_zero':z.c[0,1]=I(0)
    got=cache(z,y);assert len(calls)==2;assert key(got)==key(raw(z,y))

def test_cache_is_bounded_and_new_call_scope_does_not_reuse_old_state():
    cache,raw,calls,checks=fixture()
    for n in range(20):cache(Jet.variable(I(n),order=3),Jet(I(0),order=3))
    assert cache.size<=16
    cache(Jet.variable(I(19),order=3),Jet(I(0),order=3));assert len(calls)==21
    fresh,_,fresh_calls,_=fixture();fresh(Jet.variable(I(0),order=3),Jet(I(0),order=3));assert len(fresh_calls)==1

def test_cache_hit_still_checks_cancellation_and_failed_evaluation_is_not_cached():
    from tools.opaque_input_reuse import InputReuse
    def fail():raise RuntimeError('cancelled')
    cache,raw,calls,checks=fixture();args=(Jet.variable(I('.2'),order=3),Jet(I('.4'),order=3));cache(*args);cache.checkpoint=fail
    with pytest.raises(RuntimeError,match='cancelled'):cache(*args)
    assert len(calls)==1
    failed=InputReuse(lambda *a:fail(),checkpoint=lambda:None)
    with pytest.raises(RuntimeError):failed(*args)
    assert failed.size==0

@pytest.mark.parametrize('sx,sy',[(-1,-1),(-1,1),(1,-1),(1,1)])
@pytest.mark.parametrize('offset',['.4988','.4992','.4998'])
def test_reused_segmented_log_material_matches_uncached_and_exact_integral(sx,sy,offset,monkeypatch):
    from tools import opaque_peano_material as material
    from tests.packaging.test_opaque_localized_peano import material_fixture
    from tests.packaging.test_opaque_transverse_peano import _log_rectangle
    e,eps,slope=material_fixture();b=I('.001')*sx;c=I('.001')*sy;a=I(offset)+(I('.001') if sx<0 else I(0))+(I('.001') if sy<0 else I(0));calls=[]
    def inputs(x,y):calls.append(1);return x*b+y*c+a,x.lift('.75'),x.lift(1),x.lift(1)
    correction=material.MaterialCorrection(e,inputs,curved=False,transverse=True,smooth_partition=True,segmented_conditioned=True)
    after=correction((I(0,1),I(0,1)))[1];count=len(calls);calls.clear()
    monkeypatch.setattr(material,'InputReuse',lambda inputs,checkpoint:inputs)
    before=correction((I(0,1),I(0,1)))[1]
    assert (before.lo,before.hi)==(after.lo,after.hi) and count<=len(calls)
    exact=_log_rectangle(e,eps,slope,a,b,c);assert after.lo<=exact.lo<=exact.hi<=after.hi
