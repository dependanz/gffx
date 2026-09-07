"""PyTorch binding for the shared eager camera composition."""
import torch
from .._camera import Camera as _Camera, ProjectedPoints
from . import transforms
from ._batching import check_offsets
from ._common import materialize


class _Backend:
    xp=torch
    eps=transforms.DEFAULT_EPS
    transform=staticmethod(transforms.transform_points)
    divide=staticmethod(transforms.perspective_divide)
    is_array=staticmethod(lambda v:isinstance(v,torch.Tensor))
    dense=staticmethod(materialize)

    @staticmethod
    def check_float(value,like=None,*,cast_dtype=False):
        if not isinstance(value,torch.Tensor): raise TypeError('camera arguments must be PyTorch tensors or real constructor scalars')
        if value.dtype not in (torch.float32,torch.float64): raise TypeError('camera tensors must be float32 or float64')
        if value.device.type not in ('cpu','cuda'): raise ValueError('camera device must be CPU or CUDA')
        if like is not None:
            if value.device!=like.device: raise ValueError('camera inputs must be on the same device; move data explicitly')
            if not cast_dtype and value.dtype!=like.dtype: raise TypeError('camera input dtypes must agree')

    @staticmethod
    def intrinsic(value,like):
        if isinstance(value,torch.Tensor):return value.reshape(1) if value.ndim==0 else value
        from numbers import Real
        if isinstance(value,bool) or not isinstance(value,Real):raise TypeError('intrinsics must be real scalars or PyTorch tensors')
        return torch.tensor([value],dtype=like.dtype if like is not None else torch.get_default_dtype(),
            device=like.device if like is not None else 'cpu')

    @staticmethod
    def expand_scalar(value,batch):return value.expand(batch)

    @staticmethod
    def identity(batch,like):return torch.eye(4,dtype=like.dtype,device=like.device).repeat(batch,1,1)

    @staticmethod
    def inverse(matrix):
        try:return torch.linalg.inv(matrix)
        except torch.linalg.LinAlgError as error:raise ValueError('camera pose/projection must be invertible') from error

    @staticmethod
    def cast(value,like):return value.to(dtype=like.dtype)

    @staticmethod
    def offsets(batch,count,like):return torch.arange(batch+1,dtype=torch.int32,device=like.device)*count

    @staticmethod
    def check_offsets(offsets,total,batch,like):
        check_offsets(offsets,total,'point_offsets')
        if offsets.device!=like.device:raise ValueError('point_offsets must share the points device')
        if offsets.numel()!=batch+1 or bool(torch.any(offsets[1:]<offsets[:-1])):
            raise ValueError('point_offsets must declare B monotone slices')

    @staticmethod
    def check_mask(mask,depth):
        if not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool:raise TypeError('valid mask must be a boolean tensor')
        if mask.shape!=depth.shape:raise ValueError('valid mask shape must equal depth shape')
        if mask.device!=depth.device:raise ValueError('valid mask must share the depth device')

    @staticmethod
    def checked_grid_mask(mask,depth):
        _Backend.check_mask(mask,depth);return mask

    @staticmethod
    def pixel_grid(depth):
        y,x=torch.meshgrid(torch.arange(depth.shape[1],dtype=depth.dtype,device=depth.device)+.5,
            torch.arange(depth.shape[2],dtype=depth.dtype,device=depth.device)+.5,indexing='ij')
        return torch.stack((x,y),dim=-1).unsqueeze(0).expand(depth.shape[0],-1,-1,-1)


class Camera(_Camera):
    _backend=_Backend


__all__=['Camera','ProjectedPoints']
