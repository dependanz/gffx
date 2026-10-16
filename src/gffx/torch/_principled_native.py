"""Bounded native Principled material call, first derivatives only.

The public material owns admission/layout/resource limits. No graph or resource
cache is retained between calls. Empty layouts stay with the explicit reference.
"""
import torch
from ._common import materialize,translate_native_error


def _error(error):
    converted=translate_native_error(error)
    # The existing public material reports unrepresentable arithmetic as a
    # ValueError; the raw C interface retains its typed OVERFLOW status.
    return ValueError(str(converted)) if isinstance(converted,OverflowError) else converted


class _Principled(torch.autograd.Function):
    @staticmethod
    def forward(ctx,budget,normal,view,light,albedo,roughness,f0,two_sided,derivatives):
        normal,view,light,albedo,roughness,f0=(materialize(x) for x in (normal,view,light,albedo,roughness,f0))
        cn,coefficients,u,width=budget._prepare_curve(roughness,with_width=True)
        d=budget._data
        try:
            curve,partial=torch.ops.gffx.shading_loss_curve(d['_clog'],coefficients,u)
            _,mean=torch.ops.gffx.shading_loss_query(cn[1:],d['_clog'],d['_weights'],curve,partial,
                roughness.new_empty((0,)),torch.empty((0,),dtype=torch.int32,device=roughness.device))
            value,jacobian=torch.ops.gffx.shading_principled_pairs(cn[1:],d['_clog'],curve,partial,mean,width,
                normal,view,light,albedo,roughness,f0,two_sided,derivatives)
        except RuntimeError as error:raise _error(error) from error
        if derivatives:ctx.save_for_backward(jacobian)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gradient):
        jacobian,=ctx.saved_tensors
        try:point,light=torch.ops.gffx.shading_principled_pairs_backward(jacobian,materialize(gradient))
        except RuntimeError as error:raise _error(error) from error
        return (None,point[:,:3].contiguous(),point[:,3:6].contiguous(),light,
                point[:,9:12].contiguous(),point[:,12].contiguous(),point[:,13:16].contiguous(),None,None)


evaluate=_Principled.apply
