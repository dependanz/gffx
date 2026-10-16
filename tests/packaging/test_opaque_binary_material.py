"""Closed material integrals qualify the opt-in arithmetic in its intended use."""
import pytest
from tools.opaque_binary_derivatives import binary_derivatives
@pytest.mark.parametrize('sx,sy',[(1,1),(-1,1),(1,-1),(-1,-1)])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('order',[6,10])
def test_binary_high_material_closed_integral(monkeypatch,sx,sy,axis,order):
    from tests.packaging.test_opaque_high_material import test_high_material_contains_closed_log_squared_integral
    with binary_derivatives():test_high_material_contains_closed_log_squared_integral(monkeypatch,sx,sy,axis,order)
@pytest.mark.parametrize('case',['steps','simultaneous','log','refusal'])
def test_binary_material_jumps_and_refusals(monkeypatch,case):
    from tests.packaging.test_opaque_high_material import test_high_material_retains_product_jumps_and_refusals
    with binary_derivatives():test_high_material_retains_product_jumps_and_refusals(monkeypatch,case)
