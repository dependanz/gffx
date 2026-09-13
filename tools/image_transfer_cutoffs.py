"""Exact integer overflow-cutoff certificates; no native or floating-power oracle.

This tool only classifies the positive nonlinear decode and negative linear encode
overflow boundaries. It is not a full transfer evaluator or native qualification.
"""
from fractions import Fraction
import struct


def _spec(dtype):
    if dtype=='float32':return ('I','f',0x7f7fffff,127,23)
    if dtype=='float64':return ('Q','d',0x7fefffffffffffff,1023,52)
    raise ValueError('unsupported dtype')


def represented(bits,dtype):
    integer,floating,maximum,_,_=_spec(dtype)
    if isinstance(bits,bool) or not isinstance(bits,int) or not 0<=bits<=maximum:
        raise ValueError('finite positive integer bits required')
    return struct.unpack('<'+floating,struct.pack('<'+integer,bits))[0]


def midpoint(dtype):
    _,_,maximum,exponent,fraction_bits=_spec(dtype)
    power=exponent-fraction_bits-1
    half=Fraction(1<<power) if power>=0 else Fraction(1,1<<(-power))
    return Fraction.from_float(represented(maximum,dtype))+half


def compare_to_limit(bits,dtype,mode,limit=None):
    """Return exact sign(value-limit); equality means overflow at the IEEE midpoint."""
    x=Fraction.from_float(represented(bits,dtype))
    bound=midpoint(dtype) if limit is None else Fraction(limit)
    if bound<=0:raise ValueError('positive limit required')
    if mode=='decode':
        base=(200*x+11)/211
        left=base.numerator**12*bound.denominator**5
        right=base.denominator**12*bound.numerator**5
    elif mode=='negative_encode':
        left=323*x.numerator*bound.denominator
        right=25*x.denominator*bound.numerator
    else:raise ValueError('unsupported boundary')
    return (left>right)-(left<right)


def verify_certificate(certificate):
    dtype,mode=certificate['dtype'],certificate['mode']
    last,first=certificate['last_finite_bits'],certificate['first_overflow_bits']
    if first!=last+1 or compare_to_limit(last,dtype,mode)>=0 or compare_to_limit(first,dtype,mode)<0:
        raise ValueError('certificate does not straddle the adjacent exact overflow cutoff')
    if certificate.get('equality')!='overflow':raise ValueError('incorrect midpoint equality rule')
    return True


def certificate(dtype,mode):
    maximum=_spec(dtype)[2]
    lo,hi=0,maximum;iterations=0
    while lo<hi:
        middle=(lo+hi+1)//2;iterations+=1
        if compare_to_limit(middle,dtype,mode)<0:lo=middle
        else:hi=middle-1
    if lo==maximum:raise ValueError('no finite-input overflow boundary')
    row=dict(dtype=dtype,mode=mode,last_finite_bits=lo,first_overflow_bits=lo+1,
             last_finite_hex=represented(lo,dtype).hex(),first_overflow_hex=represented(lo+1,dtype).hex(),
             equality='overflow',binary_search_iterations=iterations,
             method='Exact integer cross-products of rational powers; monotone positive input bit order')
    # A candidate error-budget comparison, not a proof of the implementation's error.
    relative=Fraction(1,1<<80);bound=midpoint(dtype)
    row['cutoff_neighbours_have_relative_margin_gt_2_pow_minus_80']=(
        compare_to_limit(lo,dtype,mode,bound*(1-relative))<0 and
        compare_to_limit(lo+1,dtype,mode,bound*(1+relative))>0)
    verify_certificate(row)
    return row
