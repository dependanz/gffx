"""Image compositions and shared native transfers for the explicit PyTorch adapter.

Linear-light premultiplied RGBA is the default. This module performs no image I/O,
automatic display conversion, caching, optimizer scheduling or GPU probing. See
IMAGE_ACCEPTANCE_V0_1.md for branch conventions and unqualified device/export profiles.
"""
from __future__ import annotations

import numbers

import torch
from torch.autograd.function import once_differentiable
from ._common import materialize, translate_native_error

__all__ = ["convert_alpha", "srgb_to_linear", "linear_to_srgb", "apply_mask",
           "composite", "reconstruct", "affine_matrix", "warp_affine", "adjust",
           "gaussian_blur", "capabilities"]


def _tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} must be on CPU or CUDA")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _image(value, name="image", *, rgba=True):
    _tensor(value, name)
    if value.ndim not in (3, 4) or any(s <= 0 for s in value.shape[-3:]):
        raise ValueError(f"{name} must be HWC or BHWC with positive H/W/C")
    if rgba and value.shape[-1] != 4:
        raise ValueError(f"{name} must have four RGBA channels")
    if rgba and bool(((value[..., 3] < 0) | (value[..., 3] > 1)).any()):
        raise ValueError(f"{name} alpha must be in [0,1]")
    return value.contiguous()


def _enum(value, choices, name):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")


def _alpha(value):
    _enum(value, ("straight", "premultiplied"), "alpha convention")


def _same(value, other, name, *, shape=True):
    if value.dtype != other.dtype:
        raise TypeError(f"{name} must match image dtype")
    if value.device != other.device:
        raise ValueError(f"{name} must match image device")
    if shape and value.shape != other.shape:
        raise ValueError(f"{name} must match image shape; no implicit batch broadcasting")


def _parameter(value, image, name, *, lower=None, upper=None):
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = image.new_tensor(value)
    _tensor(value, name)
    _same(value, image, name, shape=False)
    allowed = ((), (image.shape[0],)) if image.ndim == 4 else ((),)
    if tuple(value.shape) not in allowed:
        raise ValueError(f"{name} must be scalar or an explicit [B] for batched images")
    if lower is not None and bool((value < lower).any()):
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and bool((value > upper).any()):
        raise ValueError(f"{name} must be <= {upper}")
    return value.reshape((-1, 1, 1, 1)) if value.ndim else value


def _finish(value, name):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} produced nonfinite arithmetic")
    return value.contiguous()


def _canonical(value):
    alpha = value[..., 3:4]
    return torch.cat((torch.where(alpha > 0, value[..., :3], torch.zeros_like(value[..., :3])), alpha), -1)


def _premultiply(value, alpha):
    _alpha(alpha)
    value = _canonical(_image(value))
    if alpha == "straight":
        return torch.cat((value[..., :3]*value[..., 3:4], value[..., 3:4]), -1)
    return value


def _represent(value, alpha):
    if alpha == "straight":
        a = value[..., 3:4]
        safe = torch.where(a > 0, a, torch.ones_like(a))
        rgb = torch.where(a > 0, value[..., :3]/safe, torch.zeros_like(value[..., :3]))
        return torch.cat((rgb, a), -1)
    return _canonical(value)


def convert_alpha(image, *, source="straight", target="premultiplied"):
    """Explicitly change alpha representation; zero-alpha colour/colour derivatives are zero."""
    _alpha(target)
    return _finish(_represent(_premultiply(image, source), target), "convert_alpha")


class _Transfer(torch.autograd.Function):
    """Shared native transfer arithmetic and explicit first VJP, not a generic power op."""
    @staticmethod
    def forward(ctx,rgb,mode):
        ctx.save_for_backward(rgb);ctx.mode=mode
        try:
            return torch.ops.gffx.image_transfer(rgb,mode)
        except RuntimeError as error:
            raise translate_native_error(error) from None

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        rgb,=ctx.saved_tensors
        try:
            return torch.ops.gffx.image_transfer_backward(rgb,materialize(gradient),ctx.mode),None
        except RuntimeError as error:
            raise translate_native_error(error) from None


def srgb_to_linear(rgb):
    """Decode channel values, not RGBA implicitly; signed/HDR values are not clipped."""
    _tensor(rgb, "rgb")
    return _finish(_Transfer.apply(materialize(rgb),0), "srgb_to_linear")


def linear_to_srgb(rgb):
    """Encode channel values with the low linear derivative at the exact threshold."""
    _tensor(rgb, "rgb")
    return _finish(_Transfer.apply(materialize(rgb),1), "linear_to_srgb")


def apply_mask(image, mask=None, *, opacity=1.0, alpha="premultiplied"):
    """Multiply source-space mask, existing alpha and opacity in premultiplied space."""
    pm = _premultiply(image, alpha)
    q = _parameter(opacity, pm, "opacity", lower=0, upper=1)
    if mask is not None:
        _tensor(mask, "mask")
        _same(mask, pm, "mask", shape=False)
        if mask.shape != pm.shape[:-1]+(1,):
            raise ValueError("mask must match image layout with exactly one channel")
        if bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError("mask must be in [0,1]")
        q = q*mask
    return _finish(_represent(pm*q, alpha), "apply_mask")


def composite(layers, background, *, alpha="premultiplied"):
    """Alpha over, with layers ordered bottom-to-top above the explicit RGBA background."""
    if not isinstance(layers, (tuple, list)):
        raise TypeError("layers must be an explicit finite tuple or list")
    value = _premultiply(background, alpha)
    prepared = []
    for layer in layers:
        src = _premultiply(layer, alpha)
        _same(src, value, "layer")
        prepared.append(src)
    for src in prepared:
        value = src+(1-src[..., 3:4])*value
    return _finish(_represent(value, alpha), "composite")


def reconstruct(image, factor):
    """Resolve exact nonoverlapping boxes in row-major order; channels are uninterpreted."""
    value = _image(image, rgba=False)
    if type(factor) is not int or factor < 1:
        raise ValueError("factor must be a positive integer, not bool")
    if value.shape[-3] % factor or value.shape[-2] % factor:
        raise ValueError("image H/W must be exactly divisible by factor")
    if factor == 1:
        return value.clone()
    result = torch.zeros_like(value[..., ::factor, ::factor, :])
    for dy in range(factor):
        for dx in range(factor):
            result = result+value[..., dy::factor, dx::factor, :]
    return _finish(result/(factor*factor), "reconstruct")


def affine_matrix(translation, rotation, scale, pivot):
    """Source-to-destination 2x3 matrix; clockwise radians in y-down pixel-centre coordinates."""
    _tensor(translation, "translation")
    if translation.ndim not in (1, 2) or translation.shape[-1] != 2:
        raise ValueError("translation must be [2] or [B,2]")
    for name, value in (("scale", scale), ("pivot", pivot)):
        _tensor(value, name)
        _same(value, translation, name)
    if isinstance(rotation, numbers.Real) and not isinstance(rotation, bool):
        rotation = translation.new_tensor(rotation)
    _tensor(rotation, "rotation")
    _same(rotation, translation, "rotation", shape=False)
    if rotation.shape != translation.shape[:-1]:
        raise ValueError("rotation shape must match vector batch shape")
    if bool((scale == 0).any()):
        raise ValueError("scale components must be nonzero")
    c, s = torch.cos(rotation), torch.sin(rotation)
    a, b = c*scale[..., 0], -s*scale[..., 1]
    d, e = s*scale[..., 0], c*scale[..., 1]
    tx = pivot[..., 0]+translation[..., 0]-(a*pivot[..., 0]+b*pivot[..., 1])
    ty = pivot[..., 1]+translation[..., 1]-(d*pivot[..., 0]+e*pivot[..., 1])
    return _finish(torch.stack((torch.stack((a,b,tx), -1), torch.stack((d,e,ty), -1)), -2), "affine_matrix")


def _fixed_sum(value):
    """Adjacent-pair tree, preserving an odd tail, with identical ordering on each device."""
    flat=value.reshape(-1)
    while flat.numel()>1:
        paired=flat.numel()//2*2
        reduced=flat[:paired:2]+flat[1:paired:2]
        flat=torch.cat((reduced,flat[-1:])) if paired<flat.numel() else reduced
    return flat[0]


class _InverseGrid(torch.autograd.Function):
    """Analytical inverse-map VJP avoids device-dependent broadcast-reduction trees.

    First derivatives are the advertised scope. The frame retains sx, sy (2*N
    floating values), the six-value matrix and determinant. Backward forms two
    N-value coordinate cotangents, reduces one matrix component at a time, and
    uses geometric-size adjacent-pair buffers rather than an N-by-six Jacobian.
    This is an O(N) storage estimate, not whole-process/allocator peak evidence.
    """
    @staticmethod
    def forward(ctx,matrix,xx,yy,height,width):
        a,b,tx=matrix[0].unbind();d,e,ty=matrix[1].unbind()
        det=a*e-b*d
        qx,qy=xx-tx,yy-ty
        sx,sy=(e*qx-b*qy)/det,(a*qy-d*qx)/det
        ctx.save_for_backward(matrix,sx,sy,det)
        ctx.extents=(height,width)
        # Tensor denominators retain the same division on CPU and CUDA; wrapped
        # Python scalars can instead select differently rounded reciprocals.
        extent = matrix.new_tensor((width, height))
        return ((torch.stack((sx,sy),-1)+.5)/extent).reshape(-1,2)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_uv):
        matrix,sx,sy,det=ctx.saved_tensors
        height,width=ctx.extents
        a,b,_=matrix[0].unbind();d,e,_=matrix[1].unbind()
        extent = matrix.new_tensor((width, height))
        grad_xy = grad_uv/extent
        gu=grad_xy[:,0].reshape(sx.shape)
        gv=grad_xy[:,1].reshape(sy.shape)
        # For s=A^-1(p-t), dL/dt=-A^-T dL/ds and dL/dA=(dL/dt) s^T.
        rx=(d*gv-e*gu)/det
        ry=(b*gu-a*gv)/det
        del gu,gv,grad_xy
        first=torch.stack((_fixed_sum(rx*sx),_fixed_sum(rx*sy),_fixed_sum(rx)))
        second=torch.stack((_fixed_sum(ry*sx),_fixed_sum(ry*sy),_fixed_sum(ry)))
        return torch.stack((first,second)),None,None,None,None


def warp_affine(image, matrix, output_size, *, filter="bilinear", border="transparent", alpha="premultiplied"):
    """Inverse-map output pixel centres through the native sampler; no automatic minification."""
    from . import render
    pm = _premultiply(image, alpha)
    _tensor(matrix, "matrix")
    _same(matrix, pm, "matrix", shape=False)
    expected = (2,3) if pm.ndim == 3 else (pm.shape[0],2,3)
    if tuple(matrix.shape) != expected:
        raise ValueError("matrix must be [2,3] or matching [B,2,3]")
    if not isinstance(output_size, (tuple, list)) or len(output_size) != 2 or any(type(v) is not int or v <= 0 for v in output_size):
        raise ValueError("output_size must contain positive integer H,W")
    _enum(filter, ("bilinear", "nearest"), "filter")
    _enum(border, ("transparent", "clamp"), "border")
    batched = pm.ndim == 4
    frames = pm if batched else pm.unsqueeze(0)
    matrices = matrix if batched else matrix.unsqueeze(0)
    oh, ow = output_size
    height, width = pm.shape[-3:-1]
    if len(frames) == 0:
        return pm.new_empty((0,oh,ow,4))+(pm.sum()+matrix.sum())*0
    yy, xx = torch.meshgrid(torch.arange(oh,dtype=pm.dtype,device=pm.device),
                           torch.arange(ow,dtype=pm.dtype,device=pm.device), indexing="ij")
    offsets = torch.tensor([0,height*width*4], dtype=torch.int32, device=pm.device)
    wrap = render.WRAP_BORDER if border == "transparent" else render.WRAP_CLAMP
    filtering = render.FILTER_BILINEAR if filter == "bilinear" else render.FILTER_NEAREST
    results = []
    for frame, transform in zip(frames, matrices):
        a,b,tx = transform[0].unbind(); d,e,ty = transform[1].unbind()
        det = a*e-b*d
        _finish(det, "warp_affine determinant")
        if bool(det == 0):
            raise ValueError("matrix linear part is singular")
        uv = _InverseGrid.apply(transform,xx,yy,height,width)
        _finish(uv, "warp_affine inverse coordinates")
        samples = render.texture(frame.reshape(-1), offsets, height, width, uv,
                                 channel_count=4, filter=filtering, wrap_u=wrap, wrap_v=wrap,
                                 border=pm.new_zeros(4))
        results.append(samples.reshape(oh,ow,4))
    result = torch.stack(results)
    return _finish(_represent(result if batched else result[0], alpha), "warp_affine")


def adjust(image, *, exposure=0.0, brightness=0.0, alpha="premultiplied"):
    """Apply exposure in stops, then additive linear brightness; alpha remains unchanged."""
    pm = _premultiply(image, alpha)
    ev = _parameter(exposure, pm, "exposure")
    lift = _parameter(brightness, pm, "brightness")
    gain = _finish(torch.pow(pm.new_tensor(2), ev), "adjust exposure scale")
    result = torch.cat((pm[..., :3]*gain+lift*pm[..., 3:4],pm[..., 3:4]), -1)
    return _finish(_represent(result, alpha), "adjust")


class _WeightedMultiply(torch.autograd.Function):
    """Keep each blur tap's broadcast-weight VJP on the explicit pairwise tree.

    Forward multiplication and image derivatives are unchanged. Only a requested
    weight derivative retains the shifted pixels; its product buffer is reduced
    one image at a time for per-batch weights, without a tap-by-pixel Jacobian.
    """
    @staticmethod
    def forward(ctx,pixels,weight):
        ctx.save_for_backward(pixels if ctx.needs_input_grad[1] else pixels.new_empty(0),weight)
        return pixels*weight

    @staticmethod
    def backward(ctx,gradient):
        pixels,weight=ctx.saved_tensors
        pixel_gradient=gradient*weight if ctx.needs_input_grad[0] else None
        weight_gradient=None
        if ctx.needs_input_grad[1]:
            product=gradient*pixels
            if product.numel()==0:
                weight_gradient=torch.zeros_like(weight)
            elif weight.ndim==0:
                weight_gradient=_fixed_sum(product)
            else:
                weight_gradient=torch.stack([_fixed_sum(frame) for frame in product]).reshape(weight.shape)
        return pixel_gradient,weight_gradient


def gaussian_blur(image, sigma, *, radius, border="transparent", alpha="premultiplied"):
    """Fixed-support separable Gaussian on premultiplied RGBA, horizontal then vertical."""
    pm = _premultiply(image, alpha)
    if type(radius) is not int or not 0 <= radius <= 32:
        raise ValueError("radius must be an integer in [0,32]")
    _enum(border, ("transparent", "clamp"), "border")
    width = _parameter(sigma, pm, "sigma", lower=.1, upper=32)
    if radius == 0:
        return _finish(_represent(pm+width*0, alpha), "gaussian_blur")
    weights = [torch.exp(-.5*(i/width).square()) for i in range(-radius,radius+1)]
    total = weights[0]
    for weight in weights[1:]:
        total = total+weight
    weights = [weight/total for weight in weights]
    result = pm
    for axis in (-2,-3):
        indices = torch.arange(result.shape[axis],device=pm.device)
        value = torch.zeros_like(result)
        for offset, weight in zip(range(-radius,radius+1), weights):
            raw = indices+offset
            shifted = torch.index_select(result, axis, raw.clamp(0,result.shape[axis]-1))
            if border == "transparent":
                shape = [1]*result.ndim; shape[axis] = len(indices)
                shifted = shifted*((raw >= 0)&(raw < result.shape[axis])).reshape(shape)
            value = value+_WeightedMultiply.apply(shifted,weight)
        result = value
    return _finish(_represent(result, alpha), "gaussian_blur")


def capabilities():
    """Describe the implemented surface without probing a driver or inventing qualification."""
    def scalar(units, bounds=None):
        return dict(type="float or matching tensor", shape="scalar or [B] for BHWC",
                    finite=True, range=bounds, units=units, differentiable=True)
    def choice(values, default):
        return dict(type="enum", choices=values, default=default, differentiable=False)
    def alpha():
        return choice(["premultiplied", "straight"], "premultiplied")
    def pixels():
        return dict(type="tensor", layouts=["HWC", "BHWC"], channels=4,
                    colour="linear signed/HDR", alpha_range=[0,1], differentiable=True)
    def vector(units):
        return dict(type="tensor", shape="[2] or matching [B,2]", finite=True,
                    units=units, differentiable=True)
    parameters = {
        "convert_alpha": dict(image=pixels(), source=choice(["straight","premultiplied"],"straight"),
                              target=choice(["premultiplied","straight"],"premultiplied")),
        "srgb_to_linear": dict(rgb=dict(type="tensor", shape="arbitrary", finite=True,
                                        units="sRGB encoded channel values", differentiable=True)),
        "linear_to_srgb": dict(rgb=dict(type="tensor", shape="arbitrary", finite=True,
                                        units="linear channel values", differentiable=True)),
        "apply_mask": dict(image=pixels(), mask=dict(type="tensor or None", shape="image layout, C=1",
                          range=[0,1], differentiable=True), opacity=scalar("unitless",[0,1]), alpha=alpha()),
        "composite": dict(layers=dict(type="list or tuple of matching RGBA tensors", order="bottom-to-top",
                           differentiable="values yes; count/order no"), background=pixels(), alpha=alpha()),
        "reconstruct": dict(image=dict(type="tensor", layouts=["HWC","BHWC"], channels="positive",
                             differentiable=True), factor=dict(type="positive integer, not bool", minimum=1,
                             exact_divisibility_required=True, differentiable=False)),
        "affine_matrix": dict(translation=vector("pixels"), pivot=vector("pixels"),
                              scale={**vector("unitless"),"nonzero_components":True},
                              rotation=dict(type="float or tensor", shape="scalar or matching [B]",
                                            units="clockwise radians, y down", differentiable=True)),
        "warp_affine": dict(image=pixels(), matrix=dict(type="tensor",shape="[2,3] or matching [B,2,3]",
                            mapping="source-to-destination pixel centres", nonsingular=True,
                            differentiable="bilinear cell branch; nearest coordinate zero"),
                            output_size=dict(type="(H,W) positive integers",differentiable=False),
                            filter=choice(["bilinear","nearest"],"bilinear"),
                            border=choice(["transparent","clamp"],"transparent"), alpha=alpha()),
        "adjust": dict(image=pixels(), exposure=scalar("stops"), brightness=scalar("linear RGB units"), alpha=alpha()),
        "gaussian_blur": dict(image=pixels(), sigma=scalar("pixels",[.1,32]),
                              radius=dict(type="integer, not bool", range=[0,32], required=True,
                                          differentiable=False),
                              border=choice(["transparent","clamp"],"transparent"), alpha=alpha()),
    }
    gradients = {
        "convert_alpha": ["RGB", "alpha except canonicalized zero-alpha RGB branch"],
        "srgb_to_linear": ["channel values; low branch at threshold"],
        "linear_to_srgb": ["channel values; low branch at threshold"],
        "apply_mask": ["image", "mask", "opacity"],
        "composite": ["every layer", "background"],
        "reconstruct": ["image"],
        "affine_matrix": ["translation", "rotation", "scale", "pivot"],
        "warp_affine": ["image", "matrix within bilinear cell; zero for nearest coordinates"],
        "adjust": ["image", "exposure", "brightness"],
        "gaussian_blur": ["image", "sigma with fixed radius"],
    }
    return dict(schema_version=1, backend="torch", gpu_probed=False,
                working_colour="linear signed/HDR", default_alpha="premultiplied",
                layouts=["HWC","BHWC"], dtypes=["float32","float64"],
                device_execution=["cpu","cuda via tensor/native operations; qualification separate"],
                cross_device_reference_qualified=False, compilation_export_qualified=False,
                qualification="not yet published; see exact-artifact test evidence",
                ownership="caller inputs unchanged; fresh outputs/graphs; no cache or I/O",
                cancellation="consumer checkpoints between calls; no in-flight preemption",
                operations={name: dict(parameters=parameters[name], first_derivatives=fields,
                                       higher_derivatives="unsupported: first-order custom VJP" if name in ("warp_affine","srgb_to_linear","linear_to_srgb") else "unqualified",
                                       implementation="native C11/CUDA transfer" if name in ("srgb_to_linear","linear_to_srgb") else
                                       "native texture composition" if name == "warp_affine" else "tensor composition")
                            for name,fields in gradients.items()})
