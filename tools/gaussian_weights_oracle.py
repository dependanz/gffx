"""Independent Decimal Gaussian model and a separate ideal IEEE VJP convention.

No torch, native runtime, approximation coefficients or production imports. Inputs
are explicit IEEE bit patterns. Decimal precision is a caller-selected evidence
parameter, not a proof of correctly rounded results. No work occurs at import.
"""
from decimal import Decimal, localcontext
import struct


def _profile(dtype):
    if dtype == 'float32':
        return 32, 23, 8, 127
    if dtype == 'float64':
        return 64, 52, 11, 1023
    raise ValueError('dtype must be float32 or float64')


def bits_from_float(value, dtype):
    """Convenience encoding only; reference() consumes the resulting exact bits."""
    _profile(dtype)
    if isinstance(value, bool):
        raise ValueError('bool is not a floating input')
    code, integer = ('f', 'I') if dtype == 'float32' else ('d', 'Q')
    return struct.unpack('<'+integer, struct.pack('<'+code, value))[0]


def decode_bits(bits, dtype):
    """Decode any finite IEEE value exactly, including signed zero/subnormals."""
    width, fraction, exponent_bits, bias = _profile(dtype)
    if type(bits) is not int or not 0 <= bits < (1 << width):
        raise ValueError('bits must be an unsigned integer of the selected width')
    sign = bits >> (width-1)
    exponent = (bits >> fraction) & ((1 << exponent_bits)-1)
    mantissa = bits & ((1 << fraction)-1)
    if exponent == (1 << exponent_bits)-1:
        raise ValueError('reference inputs must be finite')
    if exponent:
        mantissa += 1 << fraction
        power = exponent-bias-fraction
    else:
        power = 1-bias-fraction
    with localcontext() as context:
        context.prec = 2500
        value = Decimal(mantissa) * (Decimal(2)**power)
    return value.copy_negate() if sign else value


def round_bits(value, dtype):
    """Nearest-even Decimal -> IEEE bits without a float64 double-rounding step.

    Infinite rounded outputs are represented by their IEEE infinity bits; callers
    decide whether that is an expected overflow or an invalid accepted result.
    """
    width, fraction, exponent_bits, bias = _profile(dtype)
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError('round_bits requires a finite Decimal')
    sign = int(value.is_signed()) << (width-1)
    magnitude = value.copy_abs()
    if magnitude.is_zero():
        return sign
    with localcontext() as context:
        context.prec = 2500
        half_subnormal = decode_bits(1, dtype)/2
    if magnitude < half_subnormal:
        return sign
    numerator, denominator = magnitude.as_integer_ratio()
    exponent = numerator.bit_length()-denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent: exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    quantum = max(1-bias-fraction, exponent-fraction)
    if quantum < 0:
        numerator <<= -quantum
    else:
        denominator <<= quantum
    integer, remainder = divmod(numerator, denominator)
    if remainder*2 > denominator or (remainder*2 == denominator and integer & 1):
        integer += 1
    if integer == 0:
        return sign
    if integer >= 1 << (fraction+1):
        integer >>= 1
        quantum += 1
    if quantum+fraction > bias:
        return sign | (((1 << exponent_bits)-1) << fraction)
    if quantum == 1-bias-fraction and integer < 1 << fraction:
        return sign | integer
    stored_exponent = quantum+fraction+bias
    return sign | (stored_exponent << fraction) | (integer-(1 << fraction))


def _validate(sigma_bits, dtype, radius, precision):
    sigma = decode_bits(sigma_bits, dtype)
    if type(radius) is not int or not 0 <= radius <= 32:
        raise ValueError('radius must be an integer in [0,32]')
    if type(precision) is not int or not 40 <= precision <= 2000:
        raise ValueError('precision must be an integer in [40,2000]')
    lower = decode_bits(bits_from_float(.1, dtype), dtype)
    if not lower <= sigma <= Decimal(32):
        raise ValueError('sigma is outside represented [.1,32]')
    return sigma


def reference(sigma_bits, dtype, radius, *, cotangent_bits=None, precision=120):
    """Mathematical normalized taps/analytic derivatives, not runtime arithmetic.

    Exact sigma decoding precedes Decimal calculations at the requested precision.
    Tiny raw weights remain Decimal values; derivatives never use rounded weights.
    """
    sigma = _validate(sigma_bits, dtype, radius, precision)
    if cotangent_bits is not None:
        if len(cotangent_bits) != 2*radius+1:
            raise ValueError('cotangent length must equal 2*radius+1')
        cotangents = [decode_bits(v, dtype) for v in cotangent_bits]
    else:
        cotangents = None
    with localcontext() as context:
        context.prec = precision
        if radius == 0:
            raw, weights, derivatives = [Decimal(1)], [Decimal(1)], [Decimal(0)]
            total, mu = Decimal(1), Decimal(0)
        else:
            indices = range(-radius, radius+1)
            raw = [context.exp(-Decimal(i*i)/(2*sigma*sigma)) for i in indices]
            total = sum(raw, Decimal(0))
            weights = [q/total for q in raw]
            slopes = [Decimal(i*i)/(sigma*sigma*sigma) for i in indices]
            mu = sum((w*a for w,a in zip(weights,slopes)), Decimal(0))
            derivatives = [q*(a-mu)/total for q,a in zip(raw,slopes)]
        vjp = None if cotangents is None else sum(
            (g*d for g,d in zip(cotangents,derivatives)), Decimal(0))
    return dict(sigma=sigma, dtype=dtype, radius=radius, precision=precision,
                raw=raw, normalizer=total, weighted_slope=mu, weights=weights,
                derivatives=derivatives, mathematical_vjp=vjp,
                meaning='Decimal mathematical model, not production evaluation order')


def _rounded_finite(value, dtype):
    bits = round_bits(value, dtype)
    try:
        return decode_bits(bits, dtype)
    except ValueError as error:
        raise OverflowError('ideal contract product or reduction overflow') from error


def rounded_contract(row, cotangent_bits):
    """Ideal derivative-rounded-once/products/tree convention, separate from math VJP.

    Each ideal derivative rounds to the chosen dtype; each product and adjacent-pair
    sum rounds separately. The odd final element is carried unchanged. This is an
    independently specified arithmetic oracle, not a production approximation.
    """
    dtype = row['dtype']
    if len(cotangent_bits) != len(row['derivatives']):
        raise ValueError('cotangent length must equal tap count')
    derivative_bits = [round_bits(v, dtype) for v in row['derivatives']]
    cotangents = [decode_bits(v, dtype) for v in cotangent_bits]
    with localcontext() as context:
        # Exact products/sums of two finite binary64 operands fit this precision.
        context.prec = 2500
        values = [_rounded_finite(decode_bits(d,dtype)*g,dtype)
                  for d,g in zip(derivative_bits,cotangents)]
        while len(values) > 1:
            values = [_rounded_finite(values[i]+values[i+1],dtype)
                      for i in range(0,len(values)-1,2)] + (values[-1:] if len(values)%2 else [])
    # The adopted radius-zero boundary explicitly returns +0 even for a
    # negative cotangent; decoding above still rejects nonfinite cotangents.
    if row.get('radius') == 0:
        values[0] = Decimal(0)
    return dict(weight_bits=[round_bits(v,dtype) for v in row['weights']],
                derivative_bits=derivative_bits, vjp_bits=round_bits(values[0],dtype),
                meaning='ideal rounded derivative/product/fixed-tree contract, not mathematical VJP')


def precision_stability(sigma_bits, dtype, radius, *, cotangent_bits=None,
                        precisions=(100,180)):
    """Require identical final rounded reference bits at two independent precisions."""
    if len(precisions) != 2 or precisions[1] <= precisions[0]:
        raise ValueError('two increasing precisions required')
    rows = [reference(sigma_bits,dtype,radius,cotangent_bits=cotangent_bits,precision=p)
            for p in precisions]
    signatures = []
    for row in rows:
        signature = {name: [round_bits(v,dtype) for v in row[name]]
                     for name in ('weights','derivatives')}
        if cotangent_bits is not None:
            signature['mathematical_vjp'] = round_bits(row['mathematical_vjp'],dtype)
            signature['contract_vjp'] = rounded_contract(row,cotangent_bits)['vjp_bits']
        signatures.append(signature)
    if signatures[0] != signatures[1]:
        raise ArithmeticError('reference rounding is not precision-stable')
    return dict(stable=True, precisions=list(precisions), signature=signatures[1],
                boundary='sampled precision agreement, not exhaustive correct rounding')
