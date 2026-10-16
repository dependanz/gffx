"""Apply qualified value centering to the three saved twelfth turn regions."""
def test_value_centering_on_saved_twelfth_turn_regions(monkeypatch,record_property):
    from tools import opaque_centered_gradient as m
    from tests.packaging.test_opaque_turn_diagnostic import test_saved_twelfth_turn_values_and_derivatives
    monkeypatch.setattr(m,'CenteredBoundaryMaterial',m.ValueCenteredBoundary)
    test_saved_twelfth_turn_values_and_derivatives(monkeypatch,record_property,11)
