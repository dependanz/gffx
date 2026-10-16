"""Private float64 bounded-loss arithmetic with explicit first gradients.

The loss resource contains strictly positive values at most one. Its bounded
log-cubic interpolation lies in [-745,0]. This is not a public generic exp API.
Constants share the independently certified degree-18 split-ln2 polynomial;
first gradients and exact device arithmetic have separate independent controls.
"""
import math
import torch


def _exp_kernel(value):
    n=torch.round(value*float.fromhex('0x1.71547652b82fep+0'))
    remainder=(value-n*float.fromhex('0x1.62e4200000000p-1'))-n*float.fromhex('0x1.fdf473de6af28p-22')
    p=torch.full_like(value,1/math.factorial(18))
    for k in range(17,-1,-1):p=p*remainder+1/math.factorial(k)
    exponent=n.to(torch.int32)
    high=exponent.clamp(min=-1022)
    # Both scaling factors are normal. Underflow happens only on the final
    # product, rather than prematurely materializing 2**-1075 as zero.
    def power_of_two(e):
        return ((e.to(torch.int64)+1023)<<52).view(torch.float64)
    return (p*power_of_two(exponent-high))*power_of_two(high)


def _evaluate(value):
    if value.dtype!=torch.float64 or not bool((torch.isfinite(value)&(value>=-745)&(value<=0)).all()):
        raise ValueError('bounded loss exponential requires finite float64 in [-745,0]')
    return _exp_kernel(value)


class _Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value):
        result=_evaluate(value);ctx.save_for_backward(result);return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gradient):
        result,=ctx.saved_tensors
        # Analytic first VJP. Differentiating every Horner term separately causes
        # repeated subnormal rounding; multiply by the qualified value once.
        return gradient*result


def exp_negative(value):
    return _Exp.apply(value)


def sum_last(value):
    if value.shape[-1]==0:return value.sum(-1)
    while value.shape[-1]>1:
        count=value.shape[-1]//2*2
        paired=value[...,:count:2]+value[...,1:count:2]
        value=torch.cat((paired,value[...,-1:]),-1) if count<value.shape[-1] else paired
    return value[...,0]


class _CubicU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,control,u):
        if control.requires_grad:raise ValueError('roughness coefficients must be immutable')
        ctx.save_for_backward(control,u)
        p=(1-u)*control[0]+u*control[1]
        q=(1-u)*control[1]+u*control[2]
        s=(1-u)*control[2]+u*control[3]
        return (1-u)*((1-u)*p+u*q)+u*((1-u)*q+u*s)

    @staticmethod
    def backward(ctx,gradient):
        b,u=ctx.saved_tensors
        p=(1-u)*(b[1]-b[0])+u*(b[2]-b[1])
        q=(1-u)*(b[2]-b[1])+u*(b[3]-b[2])
        product=gradient*(3*((1-u)*p+u*q))
        if product.shape!=u.shape:product=sum_last(product.reshape(product.shape[0],-1)).reshape(u.shape)
        return None,product


def roughness_interpolate(control,u):
    return exp_negative(_CubicU.apply(control,u))


def _log_evaluate(value):
    if value.dtype != torch.float64 or not bool((torch.isfinite(value) & (value > 0) & (value <= 1)).all()):
        raise ValueError('bounded resource logarithm requires positive float64 at most one')
    small = value < torch.finfo(torch.float64).tiny
    scaled = torch.where(small, value * 2.**52, value)
    bits = scaled.contiguous().view(torch.int64)
    exponent = ((bits >> 52) & 2047) - 1023 - small.to(torch.int64)*52
    mantissa = ((bits & ((1 << 52)-1)) | (1023 << 52)).view(torch.float64)
    upper = mantissa > float.fromhex('0x1.6a09e667f3bcdp+0')
    mantissa = torch.where(upper, mantissa * .5, mantissa)
    exponent = exponent + upper.to(torch.int64)
    z = (mantissa - 1) / (mantissa + 1)
    square = z * z
    polynomial = torch.full_like(z, 1/39)
    for denominator in range(37, 0, -2):
        polynomial = polynomial * square + 1/denominator
    local = (2*z) * polynomial
    k = exponent.to(torch.float64)
    return (k*float.fromhex('0x1.62e4200000000p-1') + local) + k*float.fromhex('0x1.fdf473de6af28p-22')


class _LogUnit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return _log_evaluate(value)

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        return gradient / value


log_unit = _LogUnit.apply
