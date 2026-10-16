"""Complete independent material controls with the isolated native convolution."""
import pytest
from tests.packaging.test_opaque_algebraic_material import (
 test_algebraic_boundary_keeps_independent_material_integral,
 test_algebraic_boundary_fallback_is_specific_and_cancellation_propagates,
 test_algebraic_boundary_is_opt_in_and_validated,
 test_boundary_does_not_require_unused_specular_half_vector)
@pytest.fixture(autouse=True)
def native_arithmetic():
    from tools.opaque_decimal_convolution import decimal_convolution
    with decimal_convolution():yield
