"""Native depth-ordered integration of a declared screen-affine RGB field."""
import math
import torch
from ._common import check_vertices,check_faces,check_same_device,materialize,translate_native_error

class _Integrate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,v,f,c,h,w,ox,oy,limit):
        try:result=torch.ops.gffx.opaque_integrate(v,f,c,h,w,ox,oy,limit)
        except RuntimeError as e:raise translate_native_error(e) from None
        ctx.save_for_backward(v,f,c);ctx.options=(h,w,ox,oy,limit)
        return result
    @staticmethod
    def backward(ctx,g):
        if torch.is_grad_enabled():raise RuntimeError('opaque integration supports first derivatives only')
        v,f,c=ctx.saved_tensors
        try:gv,gc=torch.ops.gffx.opaque_integrate_backward(v,f,c,materialize(g),*ctx.options)
        except RuntimeError as e:raise translate_native_error(e) from None
        return gv,None,gc,None,None,None,None,None

def opaque_integrate(pixel_vertices,faces,corner_radiance,height,width,*,origin=(0.,0.),workspace_limit_bytes=67108864):
    """Integrate the nearest screen-affine radiance field over each unit pixel.

    CPU float32/64. Vertices[V,3] store pixel x/y and affine depth (smaller wins).
    RGB values[F,3,3] are linearly interpolated in SCREEN coordinates. Returns
    premultiplied RGBA[H,W,4]; compose a background with RGB+(1-alpha)*background.
    Identical depth planes select earlier faces. First VJP follows the selected
    topology; depth ties/topology changes need not have an ordinary derivative.
    Fixed32 candidate cap; no hidden geometry truncation or CPU transfer.
    This does not approximate a nonlinear material or perspective-varying shader.
    """
    check_vertices(pixel_vertices,'pixel_vertices');check_faces(faces)
    if not isinstance(corner_radiance,torch.Tensor):raise TypeError('corner_radiance must be a tensor')
    if corner_radiance.shape!=(faces.shape[0],3,3):raise ValueError('corner_radiance must have shape [F,3,3]')
    if corner_radiance.dtype!=pixel_vertices.dtype:raise ValueError('radiance dtype must match vertices')
    check_same_device(pixel_vertices,faces,corner_radiance)
    for name,q in [('height',height),('width',width)]:
        if type(q)!=int:raise TypeError(name+' must be an integer')
        if not 1<=q<=1048576:raise ValueError(name+' outside supported range')
    if type(workspace_limit_bytes)!=int:raise TypeError('workspace budget must be integer bytes')
    if not 1<=workspace_limit_bytes<=67108864:raise ValueError('workspace budget must be in (0,64MiB]')
    if not isinstance(origin,(tuple,list)) or len(origin)!=2:raise ValueError('origin requires two coordinates')
    ox,oy=map(float,origin)
    if not all(math.isfinite(q) and abs(q)<=1048576 for q in (ox,oy)):raise ValueError('origin outside supported range')
    reserve=1048576+2*(pixel_vertices.numel()*pixel_vertices.element_size()+corner_radiance.numel()*corner_radiance.element_size())+faces.numel()*faces.element_size()+8*height*width*pixel_vertices.element_size()
    if reserve>workspace_limit_bytes:raise ValueError('opaque colour conservative workspace budget exceeded')
    return _Integrate.apply(materialize(pixel_vertices),materialize(faces),materialize(corner_radiance),height,width,ox,oy,workspace_limit_bytes)
