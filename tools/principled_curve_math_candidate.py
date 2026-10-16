"""Bounded tensor-arithmetic experiment, not installed Principled qualification.

The degree-18 split-ln2 polynomial uses the independently certified Gaussian
coefficients, but this tensor evaluation and its differentiated polynomial need
separate tests. No parameter or computed answer is transferred to the CPU.
"""
import math
import torch


def _evaluate(value):
    if value.dtype!=torch.float64 or not bool((torch.isfinite(value)&(value>=-745)&(value<=0)).all()):
        raise ValueError('candidate requires finite float64 in [-745,0]')
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
