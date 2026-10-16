"""Existing native lighting, scoped to explicit fixed-albedo replay only."""
import torch
from ._common import translate_native_error,materialize


def _native(n,b,d,i,g):
    try:return torch.ops.gffx.shading_delta_brdf(*(materialize(x) for x in (n,b,d,i,g)))
    except RuntimeError as error:raise translate_native_error(error) from None


class _Delta(torch.autograd.Function):
    @staticmethod
    def forward(ctx,n,b,d,i):
        rows=[]
        for j in range(0,len(n),128):
            value,_=_native(n[j:j+128],b[j:j+128],d[j:j+128],i[j:j+128],n.new_zeros((min(128,len(n)-j),3)))
            rows.append(value)
        ctx.save_for_backward(n,b,d,i)
        return torch.cat(rows)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,g):
        n,b,d,i=ctx.saved_tensors
        rows=[]
        for j in range(0,len(n),128):
            _,gradient=_native(n[j:j+128],b[j:j+128],d[j:j+128],i[j:j+128],g[j:j+128])
            rows.append(gradient)
        return None,torch.cat(rows),None,None


def integrate(n,b,d,i,valid,omega):
    from ._principled_groups import _fixed_replay_constants
    if not _fixed_replay_constants(n,d,i) or omega is not None or n.ndim!=2 or d.shape!=(len(n),1,3) or len(n)>800 or not bool(valid.all()):return None
    # One custom VJP assembles disjoint native chunks directly. Separate autograd
    # slice branches would add zero-filled gradients and lose negative zeros.
    # The material VJP retains its original four-point grouping and carry order.
    if not len(n):return b[:,0,:]
    return _Delta.apply(n,b,d,i)
