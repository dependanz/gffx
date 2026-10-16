"""Private deterministic soft-coverage logistic value and existing first VJP.

Use the independently certified degree18 negative exponential and a stable
symmetric formula. The -746 floor rounds to zero; it is not a coverage cutoff.
The original rounded-output y*(1-y) derivative convention is retained, including
saturation. No new epsilon, blur model, public API or generic exp claim.
"""
import torch
from ._curve_math import _exp_kernel

class _Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value):
        x=value.to(torch.float64)
        e=_exp_kernel((-x.abs()).clamp(min=-746))
        denominator=1+e
        result=torch.where(x>=0,torch.ones_like(e)/denominator,e/denominator).to(value.dtype)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx,gradient):
        result,=ctx.saved_tensors
        return (gradient*result)*(1-result)

sigmoid=_Sigmoid.apply
