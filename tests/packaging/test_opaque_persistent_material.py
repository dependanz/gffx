"""Closed material integrals qualify the opt-in arithmetic in its intended use."""
import pytest
from tools.opaque_arb_intervals import backend
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('order',[6,10])
def test_persistent_high_material_closed_integral(monkeypatch,sx,sy,axis,order):
    from tests.packaging.test_opaque_high_material import test_high_material_contains_closed_log_squared_integral
    with backend(series=True):test_high_material_contains_closed_log_squared_integral(monkeypatch,sx,sy,axis,order)
@pytest.mark.parametrize('case',['steps','simultaneous','log','refusal'])
def test_persistent_material_jumps_and_refusals(monkeypatch,case):
    from tests.packaging.test_opaque_high_material import test_high_material_retains_product_jumps_and_refusals
    with backend(series=True):test_high_material_retains_product_jumps_and_refusals(monkeypatch,case)
