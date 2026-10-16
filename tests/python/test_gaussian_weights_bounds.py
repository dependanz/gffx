"""Exact-rational certificates for the proposed Gaussian arithmetic, not native evidence."""
from fractions import Fraction as F


def test_gaussian_bound_constants_have_certified_binary_encodings():
    from tools.gaussian_weights_bounds import certificate
    c=certificate()
    assert c['constant_rounding_certified']
    assert c['high_product_significand_bits']<=53
    assert len(c['coefficients_hex'])==19


def test_gaussian_bound_reduction_and_polynomial_fit_original_envelope():
    from tools.gaussian_weights_bounds import certificate
    c=certificate();u=F(1,2**53)
    assert c['reduced_argument_bound']<F(347,1000)
    assert c['reduction_error']<8*u
    assert c['polynomial_remainder']<F(3,10**26)
    assert c['mantissa_relative_error']<100*u
    assert c['input_exponent_error_at_800']<F(1,10**10)
    assert c['input_relative_error_units_per_b']<5


def test_gaussian_bound_uniform_moments_are_certified_without_libm():
    from tools.gaussian_weights_bounds import certificate
    c=certificate()
    assert c['moments_certified']
    assert c['derivative_magnitude_bound']<11


def test_gaussian_bound_propagation_retains_declared_accuracy_budgets():
    from tools.gaussian_weights_bounds import certificate
    c=certificate()
    assert c['weight_error_units']<512
    assert c['derivative_error_units']<8192
    assert c['runtime_derivative_error_units']<1024
    assert c['tail_error']<F(1,2**990)


def test_gaussian_bound_early_zero_covers_derivatives_not_only_weights():
    from tools.gaussian_weights_bounds import certificate
    c=certificate()
    assert c['early_zero_certified']
    assert c['max_exponent_input_error']<1
