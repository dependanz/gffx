"""Tiny exact CPU/CUDA compositor parity; no priming or tolerant fallback."""
import pytest
import torch
from gffx.torch import scene,material,light,render
from gffx.torch.camera import Camera


@pytest.fixture
def cuda_required():
    if not torch.cuda.is_available():pytest.skip('CUDA qualification requires a CUDA worker')


def fixture(device,dtype,*,principled=False):
    def t(value):return torch.tensor(value,dtype=dtype,device=device)
    vertices=t([[-1.,-1.,-2.],[1.,-1.,-2.],[0.,1.,-2.]])
    faces=torch.tensor([[0,1,2]],dtype=torch.int32,device=device)
    albedo=t([.25,.5,.75]).requires_grad_()
    params=dict(albedo=albedo)
    if principled:
        roughness=t(.613).requires_grad_();f0=t(.071).requires_grad_()
        identifier='ggx-smith-loss-cuda-f64-v1' if device=='cuda' else 'ggx-smith-loss-cpu-f64-v1'
        budget=material.PrincipledBudget.bundled(identifier,device=device,dtype=dtype)
        params.update(roughness=roughness,f0=f0,budget=budget)
        parameters=(albedo,roughness,f0)
    else:
        params.update(specular=t(0.),exponent=t(4.));parameters=(albedo,)
    mat=scene.Material('surface',material.principled if principled else material.classic,params)
    lamp=scene.Light('key',light.directional,dict(directions=t([[0.,0.,1.]]),irradiance=t([[1.,1.,1.]])))
    attributes={'uv':scene.Attribute(t([[0.,0.],[1.,0.],[.5,1.]]),'vertex'),
                'colour':scene.Attribute(t([[.25,.5,.75]]).expand(3,3).clone(),'vertex')}
    model=scene.Scene([scene.Mesh('triangle',vertices,faces,material='surface',attributes=attributes),mat,lamp])
    camera=Camera.from_intrinsics(t(1.),t(1.),t(.5),t(.5),1,1,.1,10.)
    return model,camera,parameters


def exact(cpu,cuda,label):
    value=cuda.detach().cpu()
    assert cpu.dtype==value.dtype and cpu.shape==value.shape,label+' layout'
    # A singleton can be contiguous while retaining a nonunit stride. Copy
    # logical values into explicitly flat storage before reinterpreting bytes.
    # Keep the original shape/dtype guard and distinguish signed zero.
    left=torch.empty(cpu.numel(),dtype=cpu.dtype,device='cpu')
    right=torch.empty(value.numel(),dtype=value.dtype,device='cpu')
    left.copy_(cpu.detach().reshape(-1))
    right.copy_(value.reshape(-1))
    left=left.view(torch.uint8)
    right=right.view(torch.uint8)
    assert torch.equal(left,right),label+' CPU/CUDA bytes are not exactly equal'


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
def test_deferred_cuda_hard_geometry_and_identity_match_cpu(cuda_required,dtype):
    outputs=('mask','depth','world_position','normal','uv','uv_present','vertex_colour',
             'colour_present','face_index','material_index','barycentric','visibility')
    pairs=[]
    for device in ('cpu','cuda'):
        model,camera,_=fixture(device,dtype)
        with torch.no_grad():pairs.append(render(model,camera,outputs=outputs))
    for name in outputs:exact(pairs[0][name],pairs[1][name],name)


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
def test_deferred_cuda_classic_colour_and_matched_albedo_vjp(cuda_required,dtype):
    pairs=[];gradients=[]
    for device in ('cpu','cuda'):
        model,camera,parameters=fixture(device,dtype)
        result=render(model,camera)['colour'];pairs.append(result)
        weights=result.new_tensor([[[[.25,.5,.75]]]])
        gradients.append(torch.autograd.grad(result,parameters,weights)[0])
    exact(pairs[0],pairs[1],'classic colour');exact(gradients[0],gradients[1],'albedo VJP')
    assert bool((gradients[0]!=0).all())


def test_deferred_cuda_soft_silhouette_and_vertex_vjp_match_cpu(cuda_required):
    images=[];gradients=[]
    for device in ('cpu','cuda'):
        def t(value):return torch.tensor(value,dtype=torch.float64,device=device)
        vertices=t([[-1.,-.8,-2.],[.9,-1.1,-2.],[.2,1.,-2.]]).requires_grad_()
        faces=torch.tensor([[0,1,2]],dtype=torch.int32,device=device)
        camera=Camera.from_intrinsics(t(1.),t(1.),t(.45),t(.55),1,1,.1,10.)
        value=render(scene.Scene([scene.Mesh('q',vertices,faces)]),camera,soft=.7,outputs=('mask',))['mask']
        images.append(value);gradients.append(torch.autograd.grad(value,vertices,torch.ones_like(value))[0])
    exact(images[0],images[1],'soft coverage');exact(gradients[0],gradients[1],'soft vertex VJP')
    assert bool((gradients[0]!=0).any())


def test_deferred_cuda_principled_same_resource_table_and_fitted_vjps(cuda_required):
    outputs=[];gradients=[];payloads=[]
    for device in ('cpu','cuda'):
        model,camera,parameters=fixture(device,torch.float64,principled=True)
        mat=next(p for p in model.prims if type(p)==scene.Material)
        payloads.append(mat.parameters['budget'].to_envelope()['payload'])
        value=render(model,camera)['colour'];outputs.append(value)
        weights=value.new_tensor([[[[.25,.5,.75]]]])
        gradients.append(torch.autograd.grad(value,parameters,weights))
    assert payloads[0]==payloads[1],'CPU/CUDA resources must represent the same mathematical table'
    exact(outputs[0],outputs[1],'Principled colour')
    for index,(cpu,cuda) in enumerate(zip(*gradients)):
        exact(cpu,cuda,'Principled VJP'+str(index))
        assert torch.isfinite(cpu).all() and bool((cpu!=0).any())
