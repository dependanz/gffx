"""Interpreted arithmetic candidate, deliberately separate from the Decimal oracle.

This is a pre-native design experiment, not an installed gffx implementation.
Python operations materialize binary64 intermediates separately; native strict-FP
and device qualifications cannot be inferred from this interpreted model.
"""
import math
import struct
from functools import lru_cache


def _bits(x):return struct.unpack('<Q',struct.pack('<d',x))[0]
def _value(x):return struct.unpack('<d',struct.pack('<Q',x))[0]


def scale(value,exponent):
    """Exact integer significand scaling with nearest-even subnormal rounding."""
    if type(exponent) is not int or not math.isfinite(value):raise ValueError('finite value and integer exponent required')
    bits=_bits(value);sign=bits&(1<<63);stored=(bits>>52)&2047;fraction=bits&((1<<52)-1)
    if not stored and not fraction:return value
    significand=fraction if stored==0 else fraction|(1<<52)
    power=(-1074 if stored==0 else stored-1023-52)+exponent
    leading=significand.bit_length()-1;result_exponent=leading+power
    if result_exponent>1023:raise OverflowError('binary scaling overflow')
    if result_exponent>=-1022:
        normalized=significand<<(52-leading)
        return _value(sign|((result_exponent+1023)<<52)|(normalized-2**52))
    shift=-1074-power
    if shift>significand.bit_length()+1:return _value(sign)
    if shift>0:
        integer=significand>>shift;discard=significand- (integer<<shift);half=1<<(shift-1)
        if discard>half or (discard==half and integer&1):integer+=1
    else:integer=significand<<(-shift)
    # A carry to bit 52 is exactly the minimum normal encoding.
    return _value(sign|integer)


@lru_cache(maxsize=1)
def constants():
    from tools.gaussian_weights_bounds import certificate
    row=certificate()
    return (float.fromhex(row['ln2_hi_hex']),float.fromhex(row['ln2_lo_hex']),
            float.fromhex(row['inv_ln2_hex']),[float.fromhex(v) for v in row['coefficients_hex']])


def exponential(x):
    """Degree-18 split-reduced negative exponential retained as mantissa/exponent."""
    if not math.isfinite(x) or x>0:raise ValueError('nonpositive finite exponent required')
    if x < -800:return 0.,0
    high,low,inverse,coefficients=constants()
    n=round(x*inverse)
    r=(x-n*high)-n*low
    polynomial=coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        product=polynomial*r
        polynomial=coefficient+product
    return polynomial,n


def _tree(values):
    while len(values)>1:
        values=[values[i]+values[i+1] for i in range(0,len(values)-1,2)]+(values[-1:] if len(values)%2 else [])
    return values[0]


def _cast(value,dtype):
    if dtype=='float64':return value
    if dtype=='float32':return struct.unpack('<f',struct.pack('<f',value))[0]
    raise ValueError('unsupported precision')


def weights(sigma,dtype,radius):
    sigma=_cast(sigma,dtype)
    if type(radius) is not int or not 0<=radius<=32:raise ValueError('radius outside [0,32]')
    if not math.isfinite(sigma) or not _cast(.1,dtype)<=sigma<=32:raise ValueError('sigma outside represented domain')
    if radius==0:return [1.],[0.]
    rows=[]
    for j in range(radius+1):
        if j==0:rows.append((1.,0,0.));continue
        t=j/sigma;b=t*t;x=-.5*b;a=b/sigma
        m,n=exponential(x);rows.append((m,n,a))
    indices=list(range(radius,0,-1))+list(range(radius+1))
    z=_tree([scale(rows[j][0],rows[j][1]) for j in indices])
    slope=[]
    for m,n,a in rows:
        product=m*a;normalized=product/z
        slope.append(scale(normalized,n))
    mu=_tree([slope[j] for j in indices])
    values=[];derivatives=[]
    for m,n,a in rows:
        values.append(_cast(scale(m/z,n),dtype))
        coefficient=a-mu
        if coefficient==0:derivatives.append(0.);continue
        product=m*coefficient;normalized=product/z
        derivatives.append(_cast(scale(normalized,n),dtype))
    return [values[j] for j in indices],[derivatives[j] for j in indices]


def vjp(sigma,dtype,radius,cotangents):
    """First VJP with separately rounded dtype products and checked pair sums."""
    _,derivatives=weights(sigma,dtype,radius)
    if len(cotangents)!=2*radius+1:raise ValueError('cotangent length must equal tap count')
    gradients=[_cast(g,dtype) for g in cotangents]
    if not all(math.isfinite(g) for g in gradients):raise ValueError('finite cotangents required')
    if radius==0:return 0.
    def finite(value):
        result=_cast(value,dtype)
        if not math.isfinite(result):raise OverflowError('Gaussian VJP product or reduction overflow')
        return result
    values=[finite(d*g) for d,g in zip(derivatives,gradients)]
    while len(values)>1:
        values=[finite(values[i]+values[i+1]) for i in range(0,len(values)-1,2)]+(values[-1:] if len(values)%2 else [])
    return values[0]
