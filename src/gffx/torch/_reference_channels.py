"""Fixed three-channel reductions for eager material/light compositions.

These private operations preserve the original elementwise forward expressions;
only broadcast-gradient sums use an explicit left-to-right three-channel order.
Inputs are already validated RGB/vector fields with matching [...,3] and [...,1].
"""
import torch


def sum3(value):
    return (value[...,0]+value[...,1])+value[...,2]


class _Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,weight):
        ctx.save_for_backward(value,weight)
        return value*weight

    @staticmethod
    def backward(ctx,gradient):
        value,weight=ctx.saved_tensors
        return (gradient*weight if ctx.needs_input_grad[0] else None,
                sum3(gradient*value).unsqueeze(-1) if ctx.needs_input_grad[1] else None)


class _Divide(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,divisor):
        ctx.save_for_backward(value,divisor)
        return value/divisor

    @staticmethod
    def backward(ctx,gradient):
        value,divisor=ctx.saved_tensors
        return (gradient/divisor if ctx.needs_input_grad[0] else None,
                sum3(-(gradient*(value/divisor))/divisor).unsqueeze(-1) if ctx.needs_input_grad[1] else None)


multiply=_Multiply.apply
divide=_Divide.apply


def _sum_last(value):
    if value.shape[-1]==0:return value.sum(-1)
    while value.shape[-1]>1:
        count=value.shape[-1]//2*2
        paired=value[...,:count:2]+value[...,1:count:2]
        value=torch.cat((paired,value[...,-1:]),-1) if count<value.shape[-1] else paired
    return value[...,0]


class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,shape):
        ctx.shape=value.shape
        return value.expand(shape)

    @staticmethod
    def backward(ctx,gradient):
        original=(1,)*(gradient.ndim-len(ctx.shape))+tuple(ctx.shape)
        reduced=[i for i,(a,b) in enumerate(zip(original,gradient.shape)) if a==1 and b!=1]
        if reduced:
            kept=[i for i in range(gradient.ndim) if i not in reduced]
            extent=1
            for i in reduced:extent*=gradient.shape[i]
            arranged=gradient.permute(kept+reduced).reshape(*(gradient.shape[i] for i in kept),extent)
            gradient=_sum_last(arranged)
        return gradient.reshape(ctx.shape),None


expand=_Expand.apply


class _UnitSqrt(torch.autograd.Function):
    """Fixed-rounding square root for prevalidated scaled lengths in [1, 3].

    Six float64 Newton steps have the independent error bound documented in
    docs/REFERENCE_NORMALIZATION.md. This is not a general sqrt replacement.
    """
    @staticmethod
    def forward(ctx, value):
        x = value.to(torch.float64)
        if x.numel()<=1048576 and x.device.type in ('cpu','cuda'):
            y=torch.ops.gffx.shading_unit_newton(x.contiguous().reshape(-1)).reshape(x.shape)
        else:
            y = torch.full_like(x, 1.5)
            for _ in range(6):
                quotient = x / y
                y = (y + quotient) * 0.5
        result = y.to(value.dtype)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        result, = ctx.saved_tensors
        return gradient / (2 * result)


sqrt_unit = _UnitSqrt.apply


def unit_forward(value):
    """Exact fixed-input normalization; None preserves differentiated inputs."""
    if torch.is_grad_enabled():
        from ._principled_groups import _fixed_replay_constants
        if not _fixed_replay_constants(value):return None
    if value.dtype!=torch.float64 or value.device.type not in ('cpu','cuda') or value.layout!=torch.strided or value.ndim<1 or value.shape[-1]!=3 or value.numel()>3*1048576:
        return None
    from ._common import translate_native_error
    try:return torch.ops.gffx.shading_unit_vectors(value.contiguous().reshape(-1,3)).reshape(value.shape)
    except RuntimeError as error:raise translate_native_error(error) from None
