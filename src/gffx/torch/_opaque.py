"""Native exact opaque alpha; experimental CPU reference, no colour approximation."""
import math
import torch
from ._common import check_vertices,check_faces,check_same_device,materialize,translate_native_error

class _Coverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx,vertices,faces,height,width,ox,oy,limit):
        try:result=torch.ops.gffx.opaque_coverage(vertices,faces,height,width,ox,oy,limit)
        except RuntimeError as e:raise translate_native_error(e) from None
        ctx.save_for_backward(vertices,faces);ctx.options=(height,width,ox,oy,limit)
        return result
    @staticmethod
    def backward(ctx,gradient):
        from torch.autograd.function import once_differentiable
        if torch.is_grad_enabled():raise RuntimeError('opaque coverage supports first derivatives only')
        v,f=ctx.saved_tensors
        try:result=torch.ops.gffx.opaque_coverage_backward(v,f,materialize(gradient),*ctx.options)
        except RuntimeError as e:raise translate_native_error(e) from None
        return result,None,None,None,None,None,None

def opaque_coverage(pixel_vertices,faces,height,width,*,origin=(0.,0.),workspace_limit_bytes=67108864):
    """Area of projected triangle union in each unit pixel, with an ordered VJP.

    Pixel space: x right, y down, integer corners. z does not affect opaque alpha.
    CPU float32/float64 only. No culling, depth clipping or colour filtering.
    A maximum of32 conservative AABB candidates per pixel is checked, not dropped.
    Coordinates/axes are bounded by2**20. Native polygon/work overflow raises.
    Gradients use the selected clipping topology at nonsmooth branch events.
    Scratch plus output must fit the explicit budget, at most64MiB on CPU.
    """
    check_vertices(pixel_vertices,'pixel_vertices');check_faces(faces);check_same_device(pixel_vertices,faces)
    for name,value in [('height',height),('width',width)]:
        if type(value)!=int:raise TypeError(name+' must be an integer')
        if not 1<=value<=1048576:raise ValueError(name+' outside supported range')
    if type(workspace_limit_bytes)!=int:raise TypeError('workspace budget must be integer bytes')
    if not 1<=workspace_limit_bytes<=67108864:raise ValueError('workspace budget must be in (0,64MiB]')
    if not isinstance(origin,(tuple,list)) or len(origin)!=2:raise ValueError('origin requires two coordinates')
    ox,oy=map(float,origin)
    if not all(math.isfinite(a) and abs(a)<=1048576 for a in (ox,oy)):raise ValueError('origin outside supported range')
    # Include retained inputs and both forward/backward outputs in host admission.
    reserve=1048576+pixel_vertices.numel()*pixel_vertices.element_size()*2+faces.numel()*faces.element_size()+height*width*pixel_vertices.element_size()*2
    if reserve>workspace_limit_bytes:raise ValueError('opaque conservative workspace budget exceeded')
    return _Coverage.apply(materialize(pixel_vertices),materialize(faces),height,width,ox,oy,workspace_limit_bytes)
