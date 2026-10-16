"""Material-only inputs keep exact cosine jets and omit only unused half-vector work."""
import pytest
from fractions import Fraction as Q

@pytest.mark.parametrize('order',[0,1,3,6,10])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('positive',[False,True])
def test_material_only_retains_exact_cosine_endpoints_and_derivatives(order,axis,positive):
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_curved_reference import flattened_radial_map
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_horizon_factor import setup
    p,f=setup(True,positive);fn=radial_cosines(f,[1,1,2],horizon_patch=p)
    args=[Jet(I('.51','.52'),order=order),Jet(I('.31','.32'),order=order)]
    if order:args[axis]=Jet.variable(args[axis][0,0],order=order)
    args[0].c[0,1]=I('.25');t,jac,distance=flattened_radial_map(p,*args,power=4)
    full=fn(t,args[1],horizon_distance=distance);lean=fn(t,args[1],horizon_distance=distance,material_only=True)
    assert lean[2] is None and full[2] is not None
    for a,b in zip(full[:2],lean[:2]):
        assert set(a.c)==set(b.c)
        assert all((a[k].lo,a[k].hi)==(b[k].lo,b[k].hi) for k in a.c)

def test_half_vector_sqrt_work_is_absent_only_when_explicitly_selected(monkeypatch):
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_shading_intervals import I,Jet
    from tests.packaging.test_opaque_horizon_factor import setup
    p,f=setup();fn=radial_cosines(f,[1,1,2]);count=[];sqrt=Jet.sqrt
    def called(self):count.append(1);return sqrt(self)
    monkeypatch.setattr(Jet,'sqrt',called)
    full=fn(Jet('.5'),Jet('.3'));assert len(count)==4
    count.clear();lean=fn(Jet('.5'),Jet('.3'),material_only=True);assert len(count)==2
    assert lean[2] is None and full[2] is not None
    with pytest.raises(ValueError):fn(Jet('.5'),Jet('.3'),material_only=1)

def test_material_only_preserves_horizon_mismatch_and_domain_refusals():
    from tools.opaque_peano_material import radial_cosines
    from tools.opaque_shading_intervals import Jet
    from tests.packaging.test_opaque_horizon_factor import setup
    p,f=setup();fn=radial_cosines(f,[1,1,2],horizon_patch=p)
    with pytest.raises(ValueError):fn(Jet('.5'),Jet('.3'),material_only=True)


@pytest.mark.parametrize('power',[4,8])
def test_original_material_profile_separates_specular_calls_and_cache_keys(monkeypatch,power):
    import json,os
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    from tools.opaque_shading_intervals import I,Jet
    calls=[];mapping=m.flattened_radial_map
    def observed(*args,**kw):calls.append(kw["power"]);return mapping(*args,**kw)
    monkeypatch.setattr(m,"flattened_radial_map",observed)
    class Captured(Exception):pass
    def capture(field,**kw):
        prepare=kw['prepare'];c=kw['correction_integrator'];box=(I('.75','.78125'),I('.75','1'));prepare(0,0,box)
        original=prepare.__closure__[prepare.__code__.co_freevars.index('inputs')].cell_contents
        for order in [0,3,10]:
            u=Jet.variable(I('.76'),order=order) if order else Jet('.76');v=Jet('.875',order=order)
            a=original(u,v);b=c.inputs(u,v);again=original(u,v)
            assert b[2] is None and a[2] is not None and again[2] is not None
            for i in [0,1,3]:
                assert set(a[i].c)==set(b[i].c)
                assert all((a[i][k].lo,a[i][k].hi)==(b[i][k].lo,b[i][k].hi) for k in a[i].c)
            assert field(u,v)[0][0,0].lo>0 # full specular field still exists
        raise Captured
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Captured):m.full_pixel(json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes()),selected_piece=8,material_only_inputs=True,horizon_power=power)
    assert calls and set(calls)=={power}
    with pytest.raises(ValueError):m.full_pixel({},material_only_inputs=1)


def test_material_cache_preserves_explicit_missing_half_vector_and_owned_jets():
    from tools.opaque_input_reuse import InputReuse
    from tools.opaque_shading_intervals import I,Jet
    calls=[]
    def inputs(x,y):calls.append(1);return x+y,x-y,None,x*x
    cache=InputReuse(inputs,checkpoint=lambda:None);x=Jet.variable(I('.3'),order=3);y=Jet('.2',order=3)
    a=cache(x,y);expected=(a[0][0,0].lo,a[0][0,0].hi);a[0].c[0,0]=I(999)
    b=cache(x,y);assert len(calls)==1 and b[2] is None
    assert (b[0][0,0].lo,b[0][0,0].hi)==expected


@pytest.mark.parametrize('power',[0,1,16,True,'8'])
def test_original_profile_refuses_unqualified_coordinate_powers(power):
    from tools.opaque_full_material_reference import full_pixel
    with pytest.raises(ValueError):full_pixel({},horizon_power=power)
