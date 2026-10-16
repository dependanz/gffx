"""Finite linear-texture / polynomial-SH image formation.

This is an explicit image model, NOT a physical classic-material albedo.
Signed/out-of-range finite texture values remain unchanged. Classic material
validation and all default render paths retain their original contracts.
Visibility is produced by the public deferred scene integrator.
"""
import math
import torch
from . import mesh,scene,render


def clamped_area_normals(vertices,faces,eps=1e-6):
    """Area-weighted vertex normals divided by max(norm,eps), including zero."""
    if not math.isfinite(eps) or eps<=0:raise ValueError('positive normal epsilon')
    tri=mesh.gather_faces(vertices.contiguous(),faces.contiguous())
    cross=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1)
    total=torch.zeros_like(vertices)
    for corner in range(3):total.index_add_(0,faces[:,corner].long(),cross)
    return total/torch.linalg.vector_norm(total,dim=-1,keepdim=True).clamp_min(eps)


def polynomial_sh(normals,coefficients):
    """RGB irradiance basis: 1,x,y,z,xy,xz,yz,x²-y²,3z²-1.

    DC uses Y00; first/second-order factors are 2pi/3 and pi/4.
    No implicit normal normalization, light clamp, BRDF division or colour map.
    """
    if coefficients.shape!=(9,3):raise ValueError('nine RGB SH coefficients')
    x,y,z=normals.unbind(-1)
    basis=torch.stack((torch.ones_like(x),x,y,z,x*y,x*z,y*z,x*x-y*y,3*z*z-1),-1)
    pi=math.pi
    factors=normals.new_tensor([1/math.sqrt(4*pi),*([2*pi/3*math.sqrt(3/(4*pi))]*3),
        *([pi/4*math.sqrt(15/(4*pi))]*3),pi/4*math.sqrt(15/(16*pi)),pi/4*math.sqrt(5/(16*pi))])
    return ((basis*factors)[...,None]*coefficients).sum(-2)


def linear_texture_sh(vertices,faces,face_uv,texture,coefficients,camera,*,
                      stop_normal_gradient=False,normal_epsilon=1e-6,
                      workspace_limit_bytes=67108864,cpu_workspace_allowance_bytes=67108864,
                      normal_override=None,affine_denominator_offset=0.,visibility_depth=None):
    """One mesh, hard nearest visibility, no culling, affine UV interpolation.

    Texture is HWC finite linear image-model data with zero border, bilinear
    half-texel sampling. stop_normal_gradient detaches AFTER interpolation;
    no derivative is taken through the binary coverage mask. normal_override
    is an explicit fixed field for testing the conditional derivative contract.
    Default normal derivatives remain enabled in this operation. An explicit
    affine_denominator_offset and per-vertex visibility_depth opt into signed
    NDC-area offset interpolation/depth. Defaults preserve exact affine math.
    """
    if type(stop_normal_gradient)!=bool:raise TypeError('explicit normal gradient policy')
    for value in (vertices,face_uv,texture,coefficients):
        if value.dtype!=vertices.dtype or value.device!=vertices.device or not bool(torch.isfinite(value).all()):
            raise ValueError('finite matched image-model inputs')
    if texture.ndim!=3 or texture.shape[-1]!=3:raise ValueError('HWC RGB texture')
    if faces.ndim!=2 or faces.shape[1]!=3 or face_uv.shape!=(len(faces),3,2):raise ValueError('triangle UV layout')
    # Unit attributes satisfy the existing scene-buffer normal contract; the
    # image-model field below separately preserves clamped vertex magnitudes.
    unit=mesh.vertex_normals(vertices.contiguous(),faces.contiguous(),eps=0.)
    obj=scene.Scene([scene.Mesh('surface',vertices,faces,attributes={
        'uv':scene.Attribute(face_uv,'face_varying'),'normals':scene.Attribute(unit,'vertex')})])
    if affine_denominator_offset:
        from ._affine_image_buffers import affine_image_buffers
        buffers=affine_image_buffers(obj,camera,vertices,faces,face_uv,
            denominator_offset=affine_denominator_offset,visibility_depth=visibility_depth,
            workspace_limit_bytes=workspace_limit_bytes,cpu_workspace_allowance_bytes=cpu_workspace_allowance_bytes)
    else:
        if visibility_depth is not None:raise ValueError('visibility_depth requires explicit affine denominator offset')
        buffers=render(obj,camera,outputs=('mask','face_index','barycentric','uv'),
                   cull_mode=render.CULL_NONE,workspace_limit_bytes=workspace_limit_bytes,
                   cpu_workspace_allowance_bytes=cpu_workspace_allowance_bytes)
    fi=buffers['face_index'].unsqueeze(-1).contiguous()
    bc=buffers['barycentric'].unsqueeze(-2).contiguous()
    normals=clamped_area_normals(vertices,faces,normal_epsilon)
    corners=normals[faces.long()].contiguous()
    field=render.interpolate(fi,bc,corners)[...,0,:]
    if normal_override is not None:
        if normal_override.shape!=field.shape:raise ValueError('fixed normal field shape')
        field=normal_override
    if stop_normal_gradient:field=field.detach()
    uv=buffers['uv']
    pyramid,offsets=render.texture_pyramid(texture.contiguous(),levels=1)
    sampled=render.texture(pyramid,offsets,texture.shape[0],texture.shape[1],uv.reshape(-1,2).contiguous(),
                           channel_count=3,wrap_u=render.WRAP_BORDER,wrap_v=render.WRAP_BORDER,
                           border=texture.new_zeros(3)).reshape(*uv.shape[:-1],3)
    shade=polynomial_sh(field,coefficients)
    coverage=buffers['mask'].detach()
    colour=sampled*shade*coverage[...,None]
    if not bool(torch.isfinite(colour).all()):raise ArithmeticError('nonfinite image model')
    return dict(colour=colour,mask=coverage,normal=field,albedo=sampled,shading=shade,**{
        n:buffers[n] for n in ('face_index','barycentric','uv')})
