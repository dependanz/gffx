"""Small end-to-end renderer controls with independent algebraic references."""
import math
import torch
from gffx.torch import scene, material, light, render, transforms, mesh
from gffx.torch.camera import Camera
from test_deferred import api, fixture


def test_deferred_matches_manual_native_chain_bit_exactly():
    d=api();model,camera,vertices,faces,albedo=fixture()
    # Deliberately compose the public primitives, never integrator helpers.
    clip=transforms.transform_points(vertices,camera.world_to_clip().contiguous())
    ndc,_=transforms.perspective_divide(clip)
    index,beta,depth,_=render.rasterize(ndc.contiguous(),faces,4,4,cull_mode=render.CULL_BACK)
    corner_w=clip[:,3][faces.long()][index.long()]
    weighted=beta/corner_w
    corrected=(weighted/weighted.sum(-1,keepdim=True)).contiguous()
    world=render.interpolate(index,corrected,vertices[faces.long()].contiguous())[...,0,:]
    normal,_,_=mesh.face_geometry(vertices,faces)
    normal=render.interpolate(index,corrected,normal[:,None].expand(-1,3,-1).contiguous())[...,0,:]
    normal=normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True)
    points=world.reshape(-1,3);n=normal.reshape(-1,3)
    view=camera.pose[0,:3,3]-points
    lamp=light.directional(vertices.new_tensor([[0.,0.,1.]]),vertices.new_ones((1,3)))
    directions=lamp['direction'][None].expand(len(points),-1,-1)
    packet=dict(measure='delta',direction=directions,irradiance=lamp['irradiance'][None].expand_as(directions),
                valid=torch.ones(directions.shape[:-1],dtype=torch.bool))
    brdf=material.classic(n,view,directions,albedo=albedo,specular=albedo.new_tensor(0.),exponent=albedo.new_tensor(4.))
    colour=light.integrate(n,brdf,packet).reshape(1,4,4,3)
    actual=d.render(model,camera,outputs=('colour','world_position','normal','barycentric','face_index','depth'))
    for name,expected in (('colour',colour),('world_position',world),('normal',normal),
                          ('barycentric',corrected[...,0,:]),('face_index',index[...,0]),('depth',depth[...,0])):
        assert torch.equal(actual[name],expected),name


def test_deferred_point_lights_match_independent_inverse_square_image_and_partials():
    d=api();model,camera,vertices,faces,albedo=fixture()
    positions=vertices.new_tensor([[.3,-.2,.4],[-.4,.3,-.2]]).requires_grad_()
    intensity=vertices.new_tensor([[1.,.8,.6],[.4,.7,.9]]).requires_grad_()
    mat=next(p for p in model.prims if type(p)==scene.Material)
    lamp=scene.Light('points',light.point,dict(positions=positions,intensity=intensity))
    actual=d.render(scene.Scene([scene.Mesh('q',vertices,faces,material='mat'),mat,lamp]),camera)['colour']
    # Known camera rays intersect the plane z=-2; do not reuse a renderer buffer.
    axis=(torch.arange(4,dtype=vertices.dtype)+.5-2)/2
    y,x=torch.meshgrid(-axis,axis,indexing='ij')
    points=torch.stack((x,y,torch.full_like(x,-2)),dim=-1)
    offset=positions[None,None]-points[...,None,:]
    distance_squared=(offset*offset).sum(-1)
    expected=albedo/math.pi*(intensity[None,None]*offset[...,2,None]/distance_squared.pow(1.5)[...,None]).sum(-2)
    torch.testing.assert_close(actual[0],expected,atol=1e-13,rtol=1e-12)
    weights=torch.arange(1,49,dtype=vertices.dtype).reshape(4,4,3)/50
    observed=torch.autograd.grad((actual[0]*weights).sum(),(positions,intensity))
    independent=torch.autograd.grad((expected*weights).sum(),(positions,intensity))
    for got,want in zip(observed,independent):
        torch.testing.assert_close(got,want,atol=1e-12,rtol=1e-11)


def test_deferred_named_camera_and_point_light_hierarchy_matches_world_configuration():
    d=api();model,camera,vertices,faces,_=fixture()
    shift=torch.eye(4,dtype=vertices.dtype);shift[:3,3]=vertices.new_tensor([.5,-.25,.75])
    positions=vertices.new_tensor([[0.,0.,0.]])
    intensity=vertices.new_tensor([[1.,.8,.6]])
    mat=next(p for p in model.prims if type(p)==scene.Material)
    hierarchical=scene.Scene([scene.Rig('root',transform=shift),
        scene.Mesh('q',vertices,faces,parent='root',material='mat'),mat,
        scene.Light('p',light.point,dict(positions=positions,intensity=intensity),parent='root'),
        scene.Camera('eye',camera,parent='root')])
    direct_camera=Camera.from_intrinsics(camera.fx,camera.fy,camera.cx,camera.cy,4,4,.1,10.,pose=shift[None])
    explicit=scene.Scene([scene.Mesh('q',vertices+shift[:3,3],faces,material='mat'),mat,
        scene.Light('p',light.point,dict(positions=positions+shift[:3,3],intensity=intensity))])
    outputs=('colour','mask','world_position','normal','face_index')
    got=d.render(hierarchical,'eye',outputs=outputs)
    want=d.render(explicit,direct_camera,outputs=outputs)
    for name in outputs:assert torch.equal(got[name],want[name]),name


def test_deferred_two_sided_shading_flips_brdf_and_cosine_together():
    d=api();model,camera,vertices,faces,albedo=fixture()
    reversed_faces=faces[:,[0,2,1]].contiguous()
    lamp=next(p for p in model.prims if type(p)==scene.Light)
    def image(two_sided):
        mat=scene.Material('mat',material.classic,dict(albedo=albedo,specular=albedo.new_tensor(0.),
            exponent=albedo.new_tensor(4.),two_sided=two_sided))
        return d.render(scene.Scene([scene.Mesh('q',vertices,reversed_faces,material='mat'),mat,lamp]),
                        camera,cull_mode=render.CULL_NONE)['colour']
    assert torch.equal(image(False),torch.zeros((1,4,4,3),dtype=vertices.dtype))
    torch.testing.assert_close(image(True),(albedo/math.pi).expand(1,4,4,3),atol=1e-13,rtol=1e-13)


def test_deferred_diffuse_sh_coefficients_match_independent_basis_and_gradient():
    d=api();model,camera,vertices,faces,albedo=fixture()
    coefficients=torch.zeros((9,3),dtype=vertices.dtype)
    coefficients[0]=vertices.new_tensor([1.,.8,.6]);coefficients.requires_grad_()
    mat=next(p for p in model.prims if type(p)==scene.Material)
    lamp=scene.Light('sh',light.spherical_harmonics,dict(coefficients=coefficients))
    actual=d.render(scene.Scene([scene.Mesh('q',vertices,faces,material='mat'),mat,lamp]),camera)['colour']
    # Real SH at n=(0,0,1), with radiance-to-irradiance band factors.
    weights=vertices.new_tensor([1/math.sqrt(4*math.pi),0,math.sqrt(3/(4*math.pi))*2/3,
                                0,0,0,math.sqrt(5/(16*math.pi))*2/4,0,0])
    expected=albedo*(weights[:,None]*coefficients).sum(0)
    torch.testing.assert_close(actual,expected.expand_as(actual),atol=1e-13,rtol=1e-13)
    derivative,=torch.autograd.grad(actual.sum(),coefficients)
    torch.testing.assert_close(derivative,16*weights[:,None]*albedo,atol=1e-13,rtol=1e-13)


def test_deferred_multiple_materials_keep_pixel_identity_and_separate_gradients():
    d=api();_,camera,vertices,faces,_=fixture()
    left=vertices.clone();left[:,0]=(left[:,0]-1)/2
    right=vertices.clone();right[:,0]=(right[:,0]+1)/2
    colours=[vertices.new_tensor([.2,.4,.6]).requires_grad_(),vertices.new_tensor([.7,.3,.1]).requires_grad_()]
    materials=[scene.Material(name,material.classic,dict(albedo=colour,specular=colour.new_tensor(0.),
        exponent=colour.new_tensor(4.))) for name,colour in zip(('left','right'),colours)]
    lamp=scene.Light('key',light.directional,dict(directions=vertices.new_tensor([[0.,0.,1.]]),irradiance=vertices.new_ones((1,3))))
    model=scene.Scene([scene.Mesh('l',left,faces,material='left'),scene.Mesh('r',right,faces,material='right'),*materials,lamp])
    actual=d.render(model,camera,outputs=('colour','material_index'))
    assert actual['material_index'][0].tolist()==[[0,0,1,1]]*4
    for start,colour in zip((0,2),colours):
        torch.testing.assert_close(actual['colour'][0,:,start:start+2],(colour/math.pi).expand(4,2,3),atol=1e-13,rtol=1e-13)
    for gradient in torch.autograd.grad(actual['colour'].sum(),colours):
        torch.testing.assert_close(gradient,torch.full_like(gradient,8/math.pi),atol=1e-13,rtol=1e-13)
