"""Private real-exponent positive-unit power with explicit analytic derivatives.

Inputs are already validated by the lobe. Output dtype/device are preserved.
The shared exponential kernel is unchanged; the -746 floor is independently
certified to round to zero. No correctly-rounded general elementary API claim.
"""
import torch
from ._curve_math import log_unit, _exp_kernel

class _NegativeExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        result = _exp_kernel(value.clamp(min=-746))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        result, = ctx.saved_tensors
        return gradient*result


def _exp(value):
    positive = value > 0
    magnitude = torch.where(positive, -value, value)
    small = _NegativeExp.apply(magnitude)
    denominator = torch.where(positive, small, torch.ones_like(small))
    reciprocal = torch.ones_like(denominator)/denominator
    return torch.where(positive, reciprocal, small)


def _log_positive(value):
    large = value > 1
    denominator = torch.where(large, value, torch.ones_like(value))
    unit = torch.where(large, torch.ones_like(value)/denominator, value)
    answer = log_unit(unit)
    return torch.where(large, -answer, answer)


class _Power(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cosine, exponent):
        c=cosine.to(torch.float64);n=exponent.to(torch.float64)
        value=_exp(n*log_unit(c)).to(cosine.dtype)
        value=torch.where(exponent==2,cosine*cosine,
            torch.where(exponent==1,cosine,
                torch.where(exponent==0,torch.ones_like(cosine),value)))
        ctx.save_for_backward(cosine,exponent)
        return value

    @staticmethod
    def backward(ctx, gradient):
        cosine,exponent=ctx.saved_tensors
        c=cosine.to(torch.float64);n=exponent.to(torch.float64)
        logarithm=log_unit(c)
        safe_n=torch.where(n>0,n,torch.ones_like(n))
        dc_log=_log_positive(safe_n)+(n-1)*logarithm
        dc=_exp(torch.where(n>0,dc_log,torch.zeros_like(dc_log)))
        dc=torch.where(n>0,dc,torch.zeros_like(dc))
        # Combining logarithms preserves scaled derivatives when an intermediate
        # reciprocal would overflow or a power would prematurely underflow.
        magnitude=-logarithm
        safe_magnitude=torch.where(magnitude>0,magnitude,torch.ones_like(magnitude))
        dn=-_exp(n*logarithm+_log_positive(safe_magnitude))
        dn=torch.where(magnitude>0,dn,torch.zeros_like(dn))
        g=gradient.to(torch.float64)
        return (g*dc).to(cosine.dtype),(g*dn).to(exponent.dtype)

power_unit=_Power.apply
