"""Independent 120-digit decimal sRGB model; no tensor or production-math imports.

Input bits and threshold branch selection match the declared dtype. Within that
branch the decimal rational model is independent of the implementation's rounded
intermediate operations. This module is an oracle, not a proposed runtime path.
"""
from decimal import Decimal,localcontext
import math
import struct
import sys


def rounded(value,dtype):
    if dtype not in ('float32','float64'):
        raise ValueError('oracle dtype must be float32 or float64')
    try:
        return struct.unpack('f',struct.pack('f',value))[0] if dtype=='float32' else float(value)
    except OverflowError:
        return math.copysign(math.inf,value)


def limits(dtype):
    if dtype=='float32':
        return dict(min_subnormal=2.**-149,min_normal=2.**-126,
                    max_finite=struct.unpack('f',struct.pack('I',0x7f7fffff))[0],emin=-149,emax=127,fraction_bits=23)
    if dtype=='float64':
        return dict(min_subnormal=2.**-1074,min_normal=sys.float_info.min,
                    max_finite=sys.float_info.max,emin=-1074,emax=1023,fraction_bits=52)
    raise ValueError('oracle dtype must be float32 or float64')


def overflow_threshold(dtype):
    """Nearest-even overflow boundary: max-finite plus half its final-bin spacing."""
    spec=limits(dtype)
    with localcontext() as context:
        context.prec=2000
        return Decimal.from_float(spec['max_finite'])+Decimal.from_float(
            math.ldexp(1.,spec['emax']-spec['fraction_bits']-1))


def next_up(value,dtype):
    value=rounded(value,dtype)
    if value<0 or not math.isfinite(value):
        raise ValueError('positive finite boundary value required')
    if dtype=='float64':return math.nextafter(value,math.inf)
    bits=struct.unpack('I',struct.pack('f',value))[0]
    return struct.unpack('f',struct.pack('I',bits+1))[0]


def next_down(value,dtype):
    value=rounded(value,dtype)
    if value<=0 or not math.isfinite(value):raise ValueError('positive finite boundary value required')
    if dtype=='float64':return math.nextafter(value,0.)
    bits=struct.unpack('I',struct.pack('f',value))[0]
    return struct.unpack('f',struct.pack('I',bits-1))[0]


def reference(value,dtype,mode):
    value=rounded(value,dtype)
    if mode not in ('decode','encode') or not math.isfinite(value):
        raise ValueError('oracle requires finite input and decode/encode mode')
    threshold=rounded(.04045 if mode=='decode' else .0031308,dtype)
    with localcontext() as context:
        context.prec=120
        x=Decimal.from_float(value);D=Decimal
        if value<=threshold:
            y=x/D('12.92') if mode=='decode' else x*D('12.92')
            derivative=D(1)/D('12.92') if mode=='decode' else D('12.92')
            branch='linear'
        elif mode=='decode':
            base=(x+D('.055'))/D('1.055')
            y=context.power(base,D(12)/D(5))
            derivative=(D(12)/D(5))*context.power(base,D(7)/D(5))/D('1.055')
            branch='nonlinear'
        else:
            y=D('1.055')*context.power(x,D(5)/D(12))-D('.055')
            derivative=D('1.055')*(D(5)/D(12))*context.power(x,-D(7)/D(12))
            branch='nonlinear'
        return dict(value=y,derivative=derivative,branch=branch,input=value,
                    overflow=abs(y)>=overflow_threshold(dtype),overflow_policy='nearest-even finite limit')


def overflow_inputs(dtype):
    """Eight neighbouring input floats on each side of both reachable overflow boundaries."""
    with localcontext() as context:
        context.prec=120;D=Decimal;threshold=overflow_threshold(dtype)
        decode=D('1.055')*context.power(threshold,D(5)/D(12))-D('.055')
        negative_encode=threshold/D('12.92')
    result=[]
    for boundary,sign in ((decode,1.),(negative_encode,-1.)):
        center=rounded(float(boundary),dtype);result.append(sign*center)
        lower=upper=center
        for _ in range(8):
            lower=next_down(lower,dtype);upper=next_up(upper,dtype)
            result.extend((sign*lower,sign*upper))
    return result


def envelope_inputs(dtype):
    """Prospective exponent-bin/mantissa boundaries, including signed linear extremes.

    This returns inputs only. Consuming the full Decimal envelope is a separately
    supervised experiment, not work performed by importing the module.
    """
    spec=limits(dtype)
    below_two=struct.unpack('f',struct.pack('I',0x3fffffff))[0] if dtype=='float32' else math.nextafter(2.,0.)
    seen=set()
    for value in (-0.,0.,-spec['min_subnormal'],-spec['min_normal'],-spec['max_finite']):
        yield value
    for exponent in range(spec['emin'],spec['emax']+1):
        for mantissa in (1.,1.25,1.5,below_two):
            value=rounded(math.ldexp(mantissa,exponent),dtype)
            if math.isfinite(value) and value>0 and value not in seen:
                seen.add(value);yield value;yield -value
    for threshold in (.04045,.0031308):
        value=rounded(threshold,dtype)
        yield next_down(value,dtype);yield value;yield next_up(value,dtype)
    yield from overflow_inputs(dtype)
