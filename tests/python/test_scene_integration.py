"""Existing native render integration, smooth fitting and bounded scene costs."""
import json
import os
from pathlib import Path
import time
import pytest
import torch
from gffx.torch import render,mesh
from test_scene import api,triangle,affine


def picture(x,device='cpu'):
    s=api();v,f=triangle(device=device)
    # Parameterize the affine translation; never perturb the fixed last row.
    t=torch.eye(4,dtype=v.dtype,device=device)+x*torch.tensor(
        [[0.,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=v.dtype,device=device)
    p=s.Scene([s.Mesh('triangle',v,f,transform=t,attributes={'colour':s.Attribute(
        torch.tensor([[1.,.1,.2],[.2,1.,.3],[.1,.3,1.]],dtype=v.dtype,device=device),'vertex')})]).flatten()
    fragments=render.rasterize(p.vertices,p.faces,8,8,cull_mode=render.CULL_NONE,
                               vertex_offsets=p.vertex_offsets,face_offsets=p.face_offsets)
    return render.interpolate(fragments[0],fragments[1],p.attributes['colour'].values)


def test_scene_integration_raster_matches_manual_and_frames():
    s=api();v,f=triangle();p=s.Scene([s.Mesh('a',v,f)]).flatten(batch_size=3)
    actual=render.rasterize(p.vertices,p.faces,8,8,cull_mode=render.CULL_NONE,vertex_offsets=p.vertex_offsets,face_offsets=p.face_offsets)
    manual=render.rasterize(v,f,8,8,cull_mode=render.CULL_NONE)
    for b in range(3):
        assert torch.equal(actual[0][b],torch.where(manual[0][0]>=0,manual[0][0]+b,manual[0][0]))
        for a,e in zip(actual[1:],manual[1:]):torch.testing.assert_close(a[b],e[0],rtol=0,atol=0)


def test_scene_integration_image_gradient_and_fresh_descent():
    x=torch.tensor(.013,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(picture,(x,),eps=1e-6,atol=1e-5,rtol=1e-4)
    target=picture(torch.tensor(.014,dtype=x.dtype)).detach()
    loss=(picture(x)-target).square().sum();g=torch.autograd.grad(loss,x)[0]
    next_x=(x-.001*g).detach().requires_grad_()
    next_loss=(picture(next_x)-target).square().sum()
    assert 0<=next_loss<loss and torch.isfinite(torch.autograd.grad(next_loss,next_x)[0])


def test_scene_integration_parent_and_vertex_gradcheck():
    s=api();v,f=triangle();v.requires_grad_();x=torch.tensor(.1,dtype=v.dtype,requires_grad=True)
    def function(vertices,shift):
        t=affine()+shift*torch.tensor([[0,0,0,1.],[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=v.dtype)
        return s.Scene([s.Rig('r',transform=t),s.Mesh('m',vertices,f,parent='r')]).flatten().vertices
    assert torch.autograd.gradcheck(function,(v,x))


def test_scene_cuda_inherited_packing_and_gradients():
    assert torch.cuda.is_available(),'CUDA qualification requires a device'
    s=api();results=[]
    for device in ('cpu','cuda'):
        v,f=triangle(device=device);v.requires_grad_();t=affine(.1,device=device).requires_grad_()
        p=s.Scene([s.Rig('r',transform=t),s.Mesh('m',v,f,parent='r')]).flatten(batch_size=2)
        assert p.vertices.device==v.device and p.faces.device==v.device
        if device=='cpu':
            assert mesh.validate(p.vertices,p.faces,vertex_offsets=p.vertex_offsets,face_offsets=p.face_offsets).is_clean
        else:
            # Existing mesh.validate is CPU-only; scene must not silently copy to it.
            assert p.vertex_offsets.tolist()==[0,3,6] and p.faces.tolist()==[[0,1,2],[3,4,5]]
        out=render.rasterize(p.vertices,p.faces,8,8,cull_mode=render.CULL_NONE,vertex_offsets=p.vertex_offsets,face_offsets=p.face_offsets)
        grads=torch.autograd.grad(p.vertices.square().sum(),(v,t))
        results.append([p.vertices.cpu(),out[0].cpu(),*[g.cpu() for g in grads]])
    for a,b in zip(*results):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    x=torch.tensor(.013,dtype=torch.float64,device='cuda',requires_grad=True)
    cpu=picture(x.detach().cpu().requires_grad_());gpu=picture(x,'cuda')
    torch.testing.assert_close(cpu,gpu.cpu(),rtol=1e-12,atol=1e-12)
    assert torch.isfinite(torch.autograd.grad(gpu.square().sum(),x)[0])


def test_scene_budget_synthetic_flame_scale():
    s=api();rows=[]
    for frames in (1,60):
        # Explicitly synthetic cardinalities: no FLAME model or licensed data.
        v=torch.arange(5023*3,dtype=torch.float32).reshape(5023,3)/20000
        f=(torch.arange(9976*3,dtype=torch.int32)%5023).reshape(9976,3)
        scene=s.Scene([s.Mesh('synthetic',v,f)]);times=[]
        with torch.no_grad():
            scene.flatten(batch_size=frames)
            for _ in range(5):
                start=time.perf_counter();p=scene.flatten(batch_size=frames);times.append(time.perf_counter()-start)
        arrays=(p.vertices,p.faces,p.vertex_offsets,p.face_offsets,p.object_vertex_offsets,p.object_face_offsets,p.face_material,*p.world_matrices.values())
        rows.append(dict(frames=frames,vertices=5023,faces=9976,seconds=times,output_tensor_bytes=sum(t.numel()*t.element_size() for t in arrays)))
        assert p.vertices.shape==(frames*5023,3)
    destination=os.environ.get('GFFX_SCENE_BUDGET')
    if destination:
        path=Path(destination)
        with path.open('x',encoding='utf-8') as stream:json.dump(rows,stream,indent=2)
