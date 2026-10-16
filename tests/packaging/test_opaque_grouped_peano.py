"""Independent closed integrals for signed aggregate curvature witnesses."""
import pytest

@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
@pytest.mark.parametrize('axis',[0,1])
def test_grouped_curvature_encloses_closed_log_squared_integral(sx,sy,axis):
    from decimal import Decimal as D,localcontext
    from tests.packaging.test_opaque_aggregate_transport import fixture
    from tools.opaque_grouped_peano import GroupedMaterial
    from tools.opaque_shading_intervals import I
    proof=GroupedMaterial(fixture(sx,sy));got=proof((I(0,1),I(0,1)),integrated_axis=axis)[1]
    with localcontext() as context:
        context.prec=90;k=D('.5');a=D('.3');b=D('.01');c=D('.4')
        def f(q):
            if q<=k:return D(0)
            z=(q/k).ln();return q*q/2*(z*z-3*z+D('3.5'))-2*k*q+k*k/4
        expected=(f(a+b+c)-f(a+b)-f(a+c)+f(a))/(b*c)*(1+D(2).ln()**2)
        assert got.lo<=expected<=got.hi


@pytest.mark.parametrize('case',['steps','simultaneous','log'])
def test_grouped_retains_discontinuities_and_product_jumps(monkeypatch,case):
    from tools.opaque_grouped_peano import GroupedMaterial
    from tools import opaque_aggregate_peano as module
    from tests.packaging import test_opaque_aggregate_peano as controls
    monkeypatch.setattr(module,'AggregateMaterial',GroupedMaterial)
    if case=='steps':controls.test_more_than_eight_discontinuous_steps_keep_every_jump()
    elif case=='simultaneous':controls.test_simultaneous_view_and_light_steps_preserve_product_jumps()
    else:
        for sx,sy in [(1,1),(-1,1),(1,-1),(-1,-1)]:controls.test_aggregate_material_bound_contains_closed_log_integral(sx,sy)

def test_disjoint_signed_jump_evidence_is_rejected():
    from tools.opaque_transverse_peano import _error
    from tools.opaque_shading_intervals import I
    with pytest.raises(ValueError,match='disjoint aggregate signed jump'):_error(dict(third=I(0),jumps=[I(0)]*3,global_signed=I(1)),I(0,1),I(1),False)

def test_grouped_cancellation_and_unknown_monotonicity_refuse(monkeypatch):
    from tools.opaque_grouped_peano import GroupedMaterial
    from tools import opaque_aggregate_peano as module
    from tests.packaging.test_opaque_aggregate_peano import test_unknown_monotonicity_and_cancellation_refuse
    monkeypatch.setattr(module,'AggregateMaterial',GroupedMaterial);test_unknown_monotonicity_and_cancellation_refuse()
