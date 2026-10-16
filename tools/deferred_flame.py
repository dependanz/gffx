"""Strict test-only bindings for a separate nondegenerate 5002V/10000F scale head."""
import math

def validate_profile(data,kind,*,albedo='constant',device='cpu'):
    from tools.synthetic_flame_fixture import generate,encode
    if kind not in ('classic','principled') or albedo not in ('constant','vertex','gradient','checker') or device not in ('cpu','cuda'):
        raise ValueError('unsupported fixed scale-head profile')
    # Exact fixed source recipe comparison rejects topology/scene edits, not only
    # count changes. Loading separately verifies the corpus source/license hashes.
    if encode(data)!=encode(generate()):raise ValueError('fixed scale-head recipe differs')

def create_scene(data,kind,*,albedo="constant",device="cpu"):
    """Explicit fixed corpus bindings; no renderer invocation or hidden assets."""
    validate_profile(data,kind,albedo=albedo,device=device)
    import torch
    from gffx.torch import scene,mesh,material,light,render
    from gffx.torch.camera import Camera
    def t(value):return torch.tensor(value,dtype=torch.float64,device=device)
    vertices=t(data['vertices']);faces=torch.tensor(data['triangles'],dtype=torch.int32,device=device)
    normals=mesh.vertex_normals(vertices,faces,weighting=mesh.WEIGHTING_AREA)
    attributes={'normals':scene.Attribute(normals,'vertex')}
    colour=t([.2,.4,.6])
    if albedo=='vertex':
        attributes['colour']=scene.Attribute(t(data['vertex_rgb']),'vertex');colour=render.AttributeSource('colour')
    elif albedo in ('gradient','checker'):
        attributes['uv']=scene.Attribute(t(data['corner_uvs']),'face_varying')
        pyramid,offsets=render.texture_pyramid(t(data['textures'][albedo]),levels=1)
        colour=render.TextureSource(pyramid,offsets,32,32,3)
    parameters=dict(albedo=colour)
    if kind=='classic':parameters.update(specular=t(.1),exponent=t(2.))
    else:
        parameters.update(roughness=t(1.),f0=t(.04),
            budget=material.PrincipledBudget.bundled('ggx-smith-loss-'+device+'-f64-v1',device=device,dtype=torch.float64))
    mat=scene.Material('skin',getattr(material,kind),parameters)
    lamp=scene.Light('key',light.directional,dict(directions=t([[1.,1.,2.]]),irradiance=t([[1.,1.,1.]])))
    model=scene.Scene([scene.Mesh('fixed-scale-head',vertices,faces,material='skin',
        attributes=attributes),mat,lamp])
    focal=64/math.tan(math.radians(20));pose=torch.eye(4,dtype=torch.float64,device=device)[None];pose[0,2,3]=3.
    camera=Camera.from_intrinsics(t(focal),t(focal),t(64.),t(64.),128,128,.1,10.,pose=pose)
    return model,camera
