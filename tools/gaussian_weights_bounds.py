"""Exact-rational certificates for the conditional Gaussian arithmetic design.

No production imports or libm exponential. This proves inequalities conditional on
nearest-even binary64 primitives, the specified operation order and a correctly
rounded bit scaler. It does not qualify compiled primitives or certify every
subnormal rounding decision of the eventual approximation.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import struct


def exp_interval(x, terms=80):
    """Positive Taylor series with a geometric upper bound on its entire tail."""
    x=F(x)
    if x<0 or x>=terms+2: raise ValueError('positive bounded argument required')
    total=term=F(1)
    for k in range(1,terms+1):
        term*=x/k
        total+=term
    first=term*x/(terms+1)
    return total,total+first/(1-x/F(terms+2))


def ln2_interval(terms=120):
    """ln(2)=2*atanh(1/3); all omitted-term ratios are below 1/9."""
    low=2*sum((F(1,(2*k+1)*3**(2*k+1)) for k in range(terms)),F(0))
    first=F(2,(2*terms+1)*3**(2*terms+1))
    return low,low+first/F(1-F(1,9))


def _bits(value): return struct.unpack('<Q',struct.pack('<d',value))[0]
def _float(bits): return struct.unpack('<d',struct.pack('<Q',bits))[0]
def _exact(value): return F.from_float(value)
def gamma(n):
    u=F(1,2**53)
    return n*u/(1-n*u)


@lru_cache(maxsize=1)
def certificate():
    u=F(1,2**53);lo,hi=ln2_interval()
    # Both rigorous endpoints round to the same constant; no guessed decimal.
    rounded=float(lo)
    constants_ok=rounded==float(hi)
    high=_float(_bits(rounded)&~((1<<32)-1));H=_exact(high)
    low=float(lo-H);L=_exact(low)
    constants_ok &= low==float(hi-H)
    inverse=float(1/hi);I=_exact(inverse)
    constants_ok &= inverse==float(1/lo)
    split_error=max(abs(lo-H-L),abs(hi-H-L))
    inverse_error=max(abs(1/lo-I),abs(1/hi-I))
    coefficients=[float(F(1,factorial(k))) for k in range(19)]
    constants_ok &= all(abs(_exact(v)-F(1,factorial(k)))<=u/F(factorial(k))
                        for k,v in enumerate(coefficients))
    # Independent Decimal precisions check generated encodings, not the proof.
    for precision in (180,240):
        with localcontext() as context:
            context.prec=precision
            logarithm=Decimal(2).ln()
            constants_ok &= float(logarithm)==rounded
            residual=logarithm-(Decimal(H.numerator)/Decimal(H.denominator))
            constants_ok &= float(residual)==low
            constants_ok &= [float(Decimal(1)/Decimal(factorial(k))) for k in range(19)]==coefficients
    # For x in [-800,0], RN(x*I) differs from x/ln2 by this absolute amount.
    selection_error=800*(inverse_error+u*abs(I))
    ideal_remainder=hi*(F(1,2)+selection_error)
    nmax=1155
    assert 800*abs(I)*(1+u)+F(1,2)<nmax
    # high has at most 21 significand bits, |n| needs at most 11: exact product.
    high_bits=21+nmax.bit_length()
    low_product=nmax*abs(L)
    first_sub_error=u*(ideal_remainder+low_product+nmax*split_error)
    low_product_error=u*low_product
    second_sub_error=u*(ideal_remainder+nmax*split_error+first_sub_error+low_product_error)
    reduction_error=nmax*split_error+first_sub_error+low_product_error+second_sub_error
    reduced=ideal_remainder+reduction_error
    r=F(347,1000)
    exponential_upper=exp_interval(r)[1]
    remainder=exponential_upper*r**19/factorial(19)
    # Absolute Horner coefficient polynomial / exp(r_actual) <= exp(2*r).
    poly_relative=(gamma(36)*(1+u)+u)*exp_interval(2*r)[1]+remainder*exponential_upper
    reduction_relative=exp_interval(reduction_error)[1]-1
    mantissa_error=(1+poly_relative)*(1+reduction_relative)-1
    # Ratio error is squared; division+multiply gives at most gamma3 in x.
    exponent_error=gamma(3)*800
    input_relative_per_b=(exp_interval(exponent_error)[1]-1)/(800*u)
    max_input_error=gamma(3)*51200
    H32=sum((F(1,k) for k in range(1,33)),F(0))
    e1=exp_interval(1)[0];e3=exp_interval(3)[0];e5=exp_interval(5)[0]
    wa=F(116,100);wb=F(368,1000);wab=F(23,10)
    mu=F(942,100);E=F(2355,100);Eab=F(1868,100)
    moments=(H32<F(406,100) and 27/e3<wa**2 and F(3125,4)/e5<wab**2
             and 1/e1<wb and 2*wa*H32<mu and 64/e1<E and 2*wab*H32<Eab)
    raw_weight=200+5*(wb+E)
    raw_derivative=200*(wa+2*mu)+5*(wab+mu*wb+E*wa+2*E*mu+Eab)
    # Match native t=RN(j/sigma), b=RN(t*t), a=RN(b/sigma): gamma4.
    # a perturbation; Z depth7; final weight division; mu product/division/tree;
    # final coefficient subtraction and product/division. Values are units of u.
    arithmetic_a=(gamma(4)/u)*(wa+mu)
    arithmetic_Z=(gamma(7)/u)*2*(wa+2*mu)
    arithmetic_mu=(gamma(9)/u)*mu
    arithmetic_final=(wa+mu)*(1+gamma(2)/u)
    runtime=(arithmetic_a+arithmetic_Z+arithmetic_mu+arithmetic_final)*F(101,100)
    derivative_units=(raw_derivative+runtime)*F(101,100)
    weight_units=(raw_weight+gamma(8)/u)*F(101,100)
    # Deliberate global tail allowance; transition-specific representability is
    # a separate gate, so this does NOT allow dropping tiny nonzero derivatives.
    tail=65*F(1,2**1074)*2**20*16
    early=(max_input_error<1 and 799>1100*hi and F(1,2**1080)<F(1,2**1075))
    return dict(constant_rounding_certified=bool(constants_ok),
        ln2_hi_hex=high.hex(),ln2_lo_hex=low.hex(),inv_ln2_hex=inverse.hex(),
        coefficients_hex=[v.hex() for v in coefficients],
        high_product_significand_bits=high_bits,reduced_argument_bound=reduced,
        reduction_error=reduction_error,polynomial_remainder=remainder,
        mantissa_relative_error=mantissa_error,input_exponent_error_at_800=exponent_error,
        input_relative_error_units_per_b=input_relative_per_b,
        max_exponent_input_error=max_input_error,moments_certified=moments,
        derivative_magnitude_bound=wa+mu,weight_error_units=weight_units,
        derivative_error_units=derivative_units,runtime_derivative_error_units=runtime,
        tail_error=tail,early_zero_certified=early,
        assumptions='RN binary64, strict operations, nearest-integer reduction, correctly rounded scaler; compiled and transition gates remain separate')
