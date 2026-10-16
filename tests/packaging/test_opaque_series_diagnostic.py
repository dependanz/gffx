"""Pinned persistent-series evaluation of the complete saved boundary proof."""
def test_saved_boundary_persistent_series_cost_and_bound(monkeypatch,record_property):
    from tools.opaque_persistent_series import backend
    from tests.packaging.test_opaque_boundary_jump_diagnostic import test_saved_ninth_localized_higher_jumps
    with backend():test_saved_ninth_localized_higher_jumps(monkeypatch,record_property)
