"""Closed material integrals exercise every retained high-order jump."""
import pytest

@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('order',[2,3,4,6,10])
def test_high_material_contains_closed_log_squared_integral(monkeypatch,sx,sy,axis,order):
    from tools.opaque_high_material import HighMaterial
    from tools import opaque_grouped_peano as module
    from tests.packaging.test_opaque_grouped_peano import test_grouped_curvature_encloses_closed_log_squared_integral
    monkeypatch.setattr(module,'GroupedMaterial',lambda c:HighMaterial(c,order=order));test_grouped_curvature_encloses_closed_log_squared_integral(sx,sy,axis)

@pytest.mark.parametrize('case',['steps','simultaneous','log','refusal'])
def test_high_material_retains_product_jumps_and_refusals(monkeypatch,case):
    from tools.opaque_high_material import HighMaterial
    from tools import opaque_aggregate_peano as module
    from tests.packaging import test_opaque_aggregate_peano as controls
    monkeypatch.setattr(module,'AggregateMaterial',HighMaterial)
    if case=='steps':high_steps()
    elif case=='simultaneous':controls.test_simultaneous_view_and_light_steps_preserve_product_jumps()
    elif case=='refusal':controls.test_unknown_monotonicity_and_cancellation_refuse()
    else:
        for sx,sy in [(1,1),(-1,1),(1,-1),(-1,-1)]:controls.test_aggregate_material_bound_contains_closed_log_integral(sx,sy)

from fractions import Fraction as Q
def high_steps():
    from tools.opaque_shading_intervals import I
    from tools.opaque_full_material_reference import HorizonExtension
    from tools.opaque_peano_material import MaterialCorrection
    from tools.opaque_high_material import HighMaterial
    from types import SimpleNamespace
    n=16;cn=[I(Q(1,4)+Q(3*i,4*n)) for i in range(n+1)];x=[v.ln() for v in cn]
    values=[I(Q((i*7)%13,10**8)) for i in range(n)];controls=[[[v]*4]*2 for v in values];e=HorizonExtension(SimpleNamespace(cn=cn,x=x,controls=controls,mean=[I(1)]*2,pi=I(1),horizon=[values[0]]*2,values=[[v]*2 for v in values]+[[values[-1]]*2]));e.indices=[0,n-1]
    lo=cn[0];span=1-lo
    def inputs(u,v):return u.lift(lo)+u*span,u.lift(1),u.lift(1),u.lift(1)
    proof=HighMaterial(MaterialCorrection(e,inputs,curved=True));result=proof((I(0,1),I(0,1)))[1]
    expected=(sum((v for v in values),I(0))/n-values[0])*values[-1]
    assert result.lo<=expected.lo<=expected.hi<=result.hi
    rows=proof.axis_bounds((I(0,1),I(0,1)),0);assert any(not w['values'][0].zero for w in rows[1]['jumps'])
