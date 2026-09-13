"""Independent Gaussian oracle controls; not native/runtime qualification."""
from decimal import Decimal, localcontext
import pytest


@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_gaussian_exact_ieee_decode_and_roundtrip(dtype):
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,round_bits
    for value in (.1,.83,1.,32.,-0.):
        bits = bits_from_float(value,dtype)
        assert round_bits(decode_bits(bits,dtype),dtype) == bits
    assert round_bits(decode_bits(1,dtype),dtype) == 1
    assert decode_bits(bits_from_float(-0.,dtype),dtype).is_signed()
    if dtype == 'float32':
        assert decode_bits(bits_from_float(.83,dtype),dtype) != Decimal('.83')


@pytest.mark.parametrize('dtype,fraction', [('float32',23),('float64',52)])
def test_gaussian_decimal_rounding_midpoint_subnormal_and_no_double_round(dtype,fraction):
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,round_bits
    with localcontext() as context:
        context.prec = 2500
        midpoint = Decimal(1)+Decimal(2)**(-fraction-1)
        tiny = Decimal(2)**(-fraction-100)
        one = bits_from_float(1.,dtype)
        assert round_bits(midpoint,dtype) == one
        assert round_bits(midpoint+tiny,dtype) == one+1
        assert round_bits(midpoint-tiny,dtype) == one
        minimum = decode_bits(1,dtype)
        assert round_bits(minimum/2,dtype) == 0
        assert round_bits(minimum*Decimal('1.5'),dtype) == 2
        assert round_bits(-minimum/4,dtype) == bits_from_float(-0.,dtype)


@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_gaussian_all_radii_normalization_symmetry_and_radius_zero(dtype):
    from tools.gaussian_weights_oracle import bits_from_float,reference
    for radius in range(33):
        row = reference(bits_from_float(.83,dtype),dtype,radius,precision=100)
        assert len(row['weights']) == 2*radius+1
        assert row['normalizer'] >= 1
        assert row['weights'] == row['weights'][::-1]
        assert row['derivatives'] == row['derivatives'][::-1]
        with localcontext() as context:
            context.prec = 100
            assert abs(sum(row['weights'])-1) < Decimal('1e-95')
            assert abs(sum(row['derivatives'])) < Decimal('1e-94')
        if radius == 0:
            assert row['weights'] == [Decimal(1)]
            assert row['derivatives'] == [Decimal(0)]
            assert not row['derivatives'][0].is_signed()


@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_gaussian_represented_bounds_and_invalid_inputs_fail_closed(dtype):
    from tools.gaussian_weights_oracle import bits_from_float,reference,decode_bits
    low, high = bits_from_float(.1,dtype), bits_from_float(32.,dtype)
    reference(low,dtype,0); reference(high,dtype,0)
    for bits in (low-1,high+1,bits_from_float(-0.,dtype)):
        with pytest.raises(ValueError): reference(bits,dtype,0)
    for radius in (-1,33,True,1.5):
        with pytest.raises(ValueError): reference(low,dtype,radius)
    for bits in (True,-1,1 << (32 if dtype=='float32' else 64)):
        with pytest.raises(ValueError): decode_bits(bits,dtype)
    with pytest.raises(ValueError): reference(low,dtype,1,cotangent_bits=[low])
    with pytest.raises(ValueError): reference(low,dtype,0,precision=20)
    with pytest.raises(ValueError): decode_bits(bits_from_float(float('inf'),dtype),dtype)


@pytest.mark.parametrize('dtype,sigma,radius', [('float32',.138,2),('float64',.1033,4)])
def test_gaussian_underflowed_weight_retains_nonzero_analytic_derivative(dtype,sigma,radius):
    from tools.gaussian_weights_oracle import bits_from_float,reference,round_bits,precision_stability
    bits = bits_from_float(sigma,dtype)
    row = reference(bits,dtype,radius,precision=180)
    assert row['weights'][0] > 0 and row['derivatives'][0] > 0
    assert round_bits(row['weights'][0],dtype) == 0
    assert round_bits(row['derivatives'][0],dtype) != 0
    assert precision_stability(bits,dtype,radius,precisions=(100,180))['stable']


@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_gaussian_precision_stability_includes_signed_cotangent_vjps(dtype):
    from tools.gaussian_weights_oracle import bits_from_float,precision_stability
    cot = [bits_from_float(v,dtype) for v in (-3.,.25,2.,-.5,1.)]
    for sigma in (.1,.83,32.):
        result = precision_stability(bits_from_float(sigma,dtype),dtype,2,
                                     cotangent_bits=cot,precisions=(100,180))
        assert result['stable']
        assert 'contract_vjp' in result['signature']
        assert 'mathematical_vjp' in result['signature']


@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_gaussian_analytic_sigma_derivatives_match_independent_central_difference(dtype):
    from tools.gaussian_weights_oracle import bits_from_float,reference
    middle = bits_from_float(.83,dtype)
    low = reference(middle-64,dtype,3,precision=100)
    high = reference(middle+64,dtype,3,precision=100)
    row = reference(middle,dtype,3,precision=100)
    with localcontext() as context:
        context.prec = 100
        delta = high['sigma']-low['sigma']
        tolerance = Decimal('1e-7') if dtype=='float32' else Decimal('1e-20')
        for a,b,derivative in zip(low['weights'],high['weights'],row['derivatives']):
            assert abs((b-a)/delta-derivative) < tolerance


def test_gaussian_rounded_contract_separates_mathematical_sum_and_fixed_tree():
    from tools.gaussian_weights_oracle import bits_from_float,decode_bits,rounded_contract
    # Synthetic cancellation tests the arithmetic convention, not Gaussian calculus.
    row = dict(dtype='float32',weights=[Decimal(1)]*3,
               derivatives=[Decimal(2)**24,Decimal(1),-(Decimal(2)**24)])
    assert sum(row['derivatives']) == 1
    result = rounded_contract(row,[bits_from_float(1.,'float32')]*3)
    assert decode_bits(result['vjp_bits'],'float32') == 0


def test_gaussian_rounded_contract_reports_overflow_not_clipping():
    from tools.gaussian_weights_oracle import decode_bits,rounded_contract
    row = dict(dtype='float32',weights=[Decimal(1)],derivatives=[Decimal(2)])
    assert decode_bits(0x7f7fffff,'float32').is_finite()
    with pytest.raises(OverflowError): rounded_contract(row,[0x7f7fffff])
