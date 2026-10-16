"""One additional certified mean-value inclusion; no sampled root acceptance."""
from fractions import Fraction as Q
from functools import partial
import pytest
from tools.opaque_shading_intervals import I,Jet,Unresolved

@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('transverse',[False,True])
def test_quadratic_crossing_contracts_and_encloses_all_transverse_roots(direction,transverse):
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();calls=[]
    def inputs(x,y):
        calls.append(max(x.order,y.order));q=x*x*I('.1')+x*I('.2')
        q=q+x.lift('.4') if direction==1 else x.lift('.6')-q
        if transverse:q=q+y*I('.0001')
        return q,x.lift('.75'),x.lift(1),x.lift(1)
    box=(I(0,1),I(0,1) if transverse else I('.3'))
    old=MaterialCorrection(e,inputs,curved=False)._axis_bounds(box,0)[1]['localized_jumps']
    before=len(calls)
    new=MaterialCorrection(e,inputs,curved=False,refine_roots=True)._axis_bounds(box,0)[1]['localized_jumps']
    assert len(old)==len(new)==1
    a,b=old[0]['position'],new[0]['position'];assert a.lo<=b.lo<=b.hi<=a.hi and b.width<a.width/10
    for y in ([I(0),I(1)] if transverse else [I('.3')]):
        exact=(I(2)-direction*y*I('.001') if transverse else I(2)).sqrt()-1
        assert b.lo<=exact.lo<=exact.hi<=b.hi
    assert new[0]['max_crossings']==old[0]['max_crossings']==1
    assert new[0]['required']==old[0]['required']
    assert len(calls)-before==before+1 # One midpoint input; original narrow derivative evidence reused.

def test_root_refinement_cancellation_cannot_be_hidden():
    from tools.opaque_peano_material import MaterialCorrection
    from tests.packaging.test_opaque_localized_peano import material_fixture
    e,_,_=material_fixture();calls=[0]
    def inputs(x,y):return x.lift('.4')+x*I('.2')+x*x*I('.1'),x.lift('.75'),x.lift(1),x.lift(1)
    def stopped():raise Unresolved('intentional stop')
    c=MaterialCorrection(e,inputs,curved=False,refine_roots=True,checkpoint=stopped)
    with pytest.raises(Unresolved):c._axis_bounds((I(0,1),I('.3')),0)

@pytest.mark.parametrize('bad',[None,0,1,'yes'])
def test_invalid_refinement_switch_refused(bad):
    from tools.opaque_peano_material import MaterialCorrection
    with pytest.raises(ValueError,match='root refinement'):MaterialCorrection(None,None,curved=False,refine_roots=bad)

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('full',[False,True])
@pytest.mark.parametrize('case',range(15))
def test_refined_roots_keep_independent_material_integrals(axis,full,case,monkeypatch):
    from tools import opaque_peano_material as module
    from tests.packaging.test_opaque_tensor_order import test_both_orders_enclose_independent_material_integrals
    real=module.MaterialCorrection
    monkeypatch.setattr(module,'MaterialCorrection',partial(real,refine_roots=True,smooth_partition=full,segmented_conditioned=full))
    test_both_orders_enclose_independent_material_integrals(axis,case,monkeypatch)



@pytest.mark.parametrize('piece',range(13))
def test_original_piece_guidance_and_optional_root_refinement_binding(piece,monkeypatch):
    import json,os
    from pathlib import Path
    from tools import opaque_full_material_reference as m
    class Stop(Exception):pass
    def capture(field,**kwargs):
        guide=kwargs.get('refinement_field')
        assert guide is None if kwargs['correction_integrator'].curved else callable(guide)
        assert kwargs['correction_integrator'].refine_roots is False
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(Path(os.environ['GFFX_FULL_INPUT']).read_bytes()),selected_piece=piece)

def test_certified_width_priority_keeps_closed_integral_and_refuses_cell_exhaustion():
    from decimal import Decimal as D,localcontext
    from tools.opaque_square_reference import integrate_square
    field=lambda u,v:[1/(u+v+u.lift(2,tangent=1))]
    r=integrate_square(field)
    with localcontext() as c:
        c.prec=90;value=4*D(4).ln()-6*D(3).ln()+2*D(2).ln();tangent=(D(8)/9).ln()
    for label,want in [('value',value),('derivative',tangent)]:assert r[label][0].lo<=want<=r[label][0].hi and r[label][0].width<=D('2e-12')
    with pytest.raises(Unresolved):integrate_square(field,max_cells=1)
