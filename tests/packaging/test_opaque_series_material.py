"""Whole-material independent controls under persistent compiled dual series."""
import pytest

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sx,sy',[(1,1),(1,-1),(-1,1),(-1,-1)])
@pytest.mark.parametrize('parts',[2,4])
def test_series_localized_material_keeps_closed_integral(monkeypatch,axis,sx,sy,parts):
    from tools.opaque_persistent_series import backend
    from tests.packaging.test_opaque_boundary_jump_localization import test_localized_higher_jumps_keep_independent_closed_integral
    with backend():test_localized_higher_jumps_keep_independent_closed_integral(monkeypatch,axis,sx,sy,parts)

@pytest.mark.parametrize('axis',[0,1])
def test_series_curved_jump_integral_matches_independent_polynomial(axis):
    from tools.opaque_persistent_series import backend
    from tests.packaging.test_opaque_boundary_jump_localization import test_curved_higher_jump_integral_matches_independent_polynomial
    with backend():test_curved_higher_jump_integral_matches_independent_polynomial(axis)

@pytest.mark.parametrize('case',['steps','simultaneous','refusal'])
def test_series_localized_material_keeps_discontinuities(monkeypatch,case):
    from tools.opaque_persistent_series import backend
    from tests.packaging.test_opaque_boundary_jump_localization import test_localized_higher_jumps_keep_discontinuities_and_refusals
    with backend():test_localized_higher_jumps_keep_discontinuities_and_refusals(monkeypatch,case)
